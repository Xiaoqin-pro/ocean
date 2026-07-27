"""Evaluate the frozen FT-Reliability v1.2 SegFormer pilot exactly once.

This entry point has no split, threshold, condition, model, or checkpoint
selection arguments.  It admits only the five final v1.2 checkpoints and the
fixed 231-image method_development role after training is frozen.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import tempfile
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as functional

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from datasets.suim_dataset import SUIMDataset, build_eval_transform  # noqa: E402
from degradations.registry import build_image_degradation, load_conditions  # noqa: E402
from metrics.calibration import brier_score, classwise_ece, expected_calibration_error, nll  # noqa: E402
from metrics.segmentation import segmentation_metrics  # noqa: E402
from metrics.uncertainty_ranking import ranking_metrics_by_region  # noqa: E402
from scripts.train_ft_reliability_pilot import (  # noqa: E402
    CHECKPOINT_FORMAT, PROTOCOL_COMMIT, _boundary, _build_segformer, load_config, sha256,
)

VARIANTS = ("A", "B", "C", "D", "E")
REGIONS = ("full", "boundary", "interior")
ITERATIONS = 1000
SEED = 20260725


def _atomic_csv(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile("w", suffix=".csv", dir=path.parent, delete=False, encoding="utf-8", newline="") as handle:
        temporary = Path(handle.name)
        frame.to_csv(handle, index=False)
    os.replace(temporary, path)


def _atomic_json(value: Mapping[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile("w", suffix=".json", dir=path.parent, delete=False, encoding="utf-8") as handle:
        temporary = Path(handle.name)
        json.dump(dict(value), handle, ensure_ascii=False, indent=2, sort_keys=True)
        handle.write("\n")
    os.replace(temporary, path)


def _reject_barred_path(path: Path) -> None:
    forbidden = {"val.csv", "calibration.csv", "test.csv", "risk_head_train.csv"}
    if path.name.lower() in forbidden or any(part.lower() in {"test", "calibration", "uiis", "dut-useg", "usis10k"} for part in path.parts):
        raise PermissionError("FT evaluation may access only the frozen method_development CSV.")


def _checkpoint_path(config: Mapping[str, Any], variant: str) -> Path:
    return ROOT / str(config["experiment"]["output_dir"]) / "segformer" / "formal" / variant / "checkpoints" / "final.pt"


def preflight(config: Mapping[str, Any]) -> tuple[Path, dict[str, Path]]:
    access = config["access_control"]
    if any(bool(access[key]) for key in ("validation_evaluated", "calibration_evaluated", "official_suim_test_evaluated")):
        raise ValueError("Formal SUIM splits must remain locked.")
    development = (ROOT / str(access["method_development_csv"])).resolve()
    _reject_barred_path(development)
    if not development.is_file() or sha256(development).upper() != str(access["method_development_csv_sha256"]).upper():
        raise ValueError("The frozen method_development CSV is missing or has changed.")
    checkpoints = {variant: _checkpoint_path(config, variant) for variant in VARIANTS}
    if not all(path.is_file() for path in checkpoints.values()):
        raise FileNotFoundError("All five final v1.2 checkpoints are required before development access.")
    return development, checkpoints


def load_final_model(config: Mapping[str, Any], checkpoint_path: Path, *, device: torch.device) -> torch.nn.Module:
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    required = {
        "checkpoint_format": CHECKPOINT_FORMAT, "protocol_commit": PROTOCOL_COMMIT,
        "model_name": "segformer", "run_kind": "formal",
        "checkpoint_selection": "final_epoch", "epoch": 100, "epoch_completed": True,
    }
    for key, expected in required.items():
        if checkpoint.get(key) != expected:
            raise ValueError(f"Invalid final checkpoint field {key}: {checkpoint_path}")
    if any(bool(checkpoint.get(key, True)) for key in ("method_development_evaluated", "validation_evaluated", "calibration_evaluated", "official_suim_test_evaluated")):
        raise ValueError("A final checkpoint records prohibited evaluation access.")
    model = _build_segformer(config, device)
    model.load_state_dict(checkpoint["model_state_dict"])
    return model.eval()


def _masked_segmentation(prediction: torch.Tensor, labels: torch.Tensor, region: torch.Tensor) -> dict[str, float]:
    masked = labels.clone()
    masked[~region] = 255
    metrics = segmentation_metrics(prediction, masked, num_classes=8, ignore_index=255)
    return {"miou": float(metrics["miou"]), "pixel_accuracy": float(metrics["pixel_accuracy"]), "mean_dice": float(metrics["mean_dice"])}


def error_auroc_diagnostic(value: float, errors: np.ndarray) -> tuple[float | None, bool, str | None]:
    """Allow only the mathematically undefined one-class error target."""
    if np.isfinite(value):
        return float(value), True, None
    flattened = np.asarray(errors, dtype=bool).reshape(-1)
    if len(flattened) and np.unique(flattened).size == 1:
        return None, False, "single_class_error_target"
    raise AssertionError("error_auroc is non-finite despite a two-class error target.")


def validate_finite_table(table: pd.DataFrame) -> None:
    """error_auroc is the sole permitted undefined diagnostic field."""
    numeric = table.select_dtypes(include=[np.number]).drop(columns=["error_auroc"], errors="ignore")
    if not np.isfinite(numeric.to_numpy()).all():
        raise AssertionError("A non-AUROC evaluation metric is non-finite.")


def rows_for_batch(logits: torch.Tensor, labels: torch.Tensor, sample_ids: list[str], *, variant: str, condition: Any) -> list[dict[str, object]]:
    probabilities = logits.float().softmax(dim=1)
    prediction = probabilities.argmax(dim=1)
    valid = labels.ge(0) & labels.lt(8)
    boundary = valid & _boundary(labels)
    regions = {"full": valid, "boundary": boundary, "interior": valid & ~boundary}
    uncertainty = 1.0 - probabilities.max(dim=1).values
    rows: list[dict[str, object]] = []
    for index, sample_id in enumerate(sample_ids):
        for region_name, region in regions.items():
            mask = region[index]
            if not mask.any():
                raise ValueError(f"Empty {region_name} region for {sample_id}.")
            error_values = prediction[index][valid[index]].ne(labels[index][valid[index]]).detach().cpu().numpy()
            ranking = ranking_metrics_by_region(
                uncertainty[index][valid[index]].detach().cpu().numpy(),
                error_values,
                {region_name: mask[valid[index]].detach().cpu().numpy()},
                coverages=(0.9, 0.8, 0.7), top_fractions=(0.1,),
            )[region_name]
            region_errors = error_values[mask[valid[index]].detach().cpu().numpy()]
            auroc, auroc_defined, auroc_reason = error_auroc_diagnostic(float(ranking["error_auroc"]), region_errors)
            ranking["error_auroc"] = auroc
            selected_probs = probabilities[index:index + 1]
            selected_labels = labels[index:index + 1].clone()
            selected_labels[:, ~mask] = 255
            row = {
                "variant": variant, "sample_id": str(sample_id), "condition": condition.name,
                "degradation_type": condition.degradation_type, "severity": condition.severity,
                "region": region_name, **_masked_segmentation(prediction[index:index + 1], labels[index:index + 1], mask.unsqueeze(0)),
                "nll": nll(selected_probs, selected_labels), "brier": brier_score(selected_probs, selected_labels),
                "ece": expected_calibration_error(selected_probs, selected_labels, bins=15),
                "error_auroc_defined": auroc_defined, "error_auroc_undefined_reason": auroc_reason,
                **ranking,
            }
            for class_id, value in enumerate(classwise_ece(selected_probs, selected_labels, bins=15)):
                row[f"classwise_ece_{class_id}"] = value
            rows.append(row)
    return rows


def paired_bootstrap(per_image: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Use original sample IDs as clusters containing all thirteen conditions."""
    rows: list[dict[str, object]] = []
    generator = np.random.default_rng(SEED)
    for baseline, candidate, comparison in (("B", "E", "E_vs_B"), ("C", "E", "E_vs_C")):
        for region in REGIONS:
            left = per_image[(per_image.variant == baseline) & (per_image.region == region)].groupby("sample_id").mean(numeric_only=True)
            right = per_image[(per_image.variant == candidate) & (per_image.region == region)].groupby("sample_id").mean(numeric_only=True)
            if not left.index.equals(right.index) or len(left) != 231:
                raise ValueError("Paired bootstrap clusters are incomplete.")
            for metric, sign in (("eaurc", 1.0), ("error_auprc", -1.0), ("miou", -1.0)):
                values = sign * (left[metric] - right[metric]).to_numpy(dtype=np.float64)
                draws = values[generator.integers(0, len(values), size=(ITERATIONS, len(values)))].mean(axis=1)
                rows.append({"comparison": comparison, "region": region, "metric": metric,
                             "mean_improvement": float(values.mean()), "ci95_low": float(np.quantile(draws, .025)),
                             "ci95_high": float(np.quantile(draws, .975)), "iterations": ITERATIONS,
                             "cluster_unit": "sample_id_with_all_13_conditions", "clusters": len(values)})
    table = pd.DataFrame(rows)
    primary = table[(table.comparison == "E_vs_C") & (table.region == "full") & (table.metric == "eaurc")].iloc[0]
    return table, {"primary_endpoint": "full_eaurc_C_minus_E", "ci95_lower_bound": float(primary.ci95_low), "passes": bool(primary.ci95_low > 0)}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.parse_args()
    config = load_config(ROOT)
    development_csv, checkpoints = preflight(config)
    frame = pd.read_csv(development_csv)
    if len(frame) != 231 or frame.sample_id.duplicated().any():
        raise ValueError("The frozen method_development role must contain exactly 231 unique images.")
    conditions = load_conditions(ROOT / str(config["degradations"]["registry_config"]))
    if len(conditions) != 13:
        raise ValueError("Evaluation requires exactly the registered 13 conditions.")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for fixed-protocol evaluation.")
    device = torch.device("cuda")
    rows: list[dict[str, object]] = []
    for variant in VARIANTS:
        model = load_final_model(config, checkpoints[variant], device=device)
        for condition in conditions:
            dataset = SUIMDataset(development_csv, transform=build_eval_transform(384), image_degradation=build_image_degradation(condition))
            loader = torch.utils.data.DataLoader(dataset, batch_size=4, shuffle=False, num_workers=0, pin_memory=True)
            for batch in loader:
                labels = batch["labels"].to(device, non_blocking=True)
                with torch.no_grad(), torch.amp.autocast("cuda", enabled=True):
                    logits = functional.interpolate(model(pixel_values=batch["pixel_values"].to(device, non_blocking=True)).logits, size=labels.shape[-2:], mode="bilinear", align_corners=False)
                rows.extend(rows_for_batch(logits, labels, [str(value) for value in batch["sample_id"]], variant=variant, condition=condition))
            print(f"evaluated {variant}/{condition.name}", flush=True)
        del model
        torch.cuda.empty_cache()
    table = pd.DataFrame(rows)
    expected = len(VARIANTS) * len(conditions) * len(frame) * len(REGIONS)
    keys = ["variant", "condition", "sample_id", "region"]
    if len(table) != expected or table.duplicated(keys).any():
        raise AssertionError("FT development result table is incomplete or non-finite.")
    validate_finite_table(table)
    bootstrap, primary = paired_bootstrap(table)
    group_columns = ["variant", "condition", "degradation_type", "severity", "region"]
    aggregate = table.groupby(group_columns, as_index=False).mean(numeric_only=True)
    auroc_counts = table.groupby(group_columns, as_index=False).agg(
        error_auroc_defined_scenes=("error_auroc_defined", "sum"),
        error_auroc_undefined_scenes=("error_auroc_defined", lambda values: int((~values).sum())),
        error_auroc_defined_fraction=("error_auroc_defined", "mean"),
    )
    aggregate = aggregate.merge(auroc_counts, on=group_columns, validate="one_to_one")
    output = ROOT / str(config["experiment"]["output_dir"]) / "development_evaluation"
    _atomic_csv(table, output / "per_image_metrics.csv")
    _atomic_csv(aggregate, output / "aggregate_metrics.csv")
    _atomic_csv(bootstrap, output / "paired_cluster_bootstrap.csv")
    _atomic_json({
        "protocol": PROTOCOL_COMMIT, "split_evaluated": "method_development", "samples": 231,
        "conditions": [item.name for item in conditions], "variants": list(VARIANTS),
        "checkpoint_sha256": {variant: sha256(path) for variant, path in checkpoints.items()},
        "bootstrap_iterations": ITERATIONS, "primary_endpoint": primary,
        "validation_evaluated": False, "calibration_evaluated": False,
        "official_suim_test_evaluated": False, "model_retrained": False,
    }, output / "metadata.json")


if __name__ == "__main__":
    main()
