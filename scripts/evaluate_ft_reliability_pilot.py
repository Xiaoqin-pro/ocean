"""Cache, audit, and aggregate the frozen FT-Reliability v1.2 evaluation.

Inference and metric aggregation are deliberately separate.  The only source
split remains the frozen 231-image ``method_development`` role; phases cannot
select models, checkpoints, conditions, or statistical thresholds.
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
from metrics.segmentation import confusion_matrix, metrics_from_confusion_matrix, segmentation_metrics  # noqa: E402
from metrics.uncertainty_ranking import ranking_metrics, ranking_metrics_by_region  # noqa: E402
from scripts.train_ft_reliability_pilot import (  # noqa: E402
    CHECKPOINT_FORMAT, PROTOCOL_COMMIT, _boundary, _build_segformer, load_config, sha256,
)

VARIANTS = ("A", "B", "C", "D", "E")
REGIONS = ("full", "boundary", "interior")
ITERATIONS = 1000
SEED = 20260725
CACHE_SCHEMA = "ft_reliability_development_prediction_cache_v1"
PER_IMAGE_DIAGNOSTICS = {
    "error_auroc": "single_class_error_target",
    "error_auprc": "no_positive_errors",
    "top_10_uncertainty_recall": "no_positive_errors",
}


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


def _atomic_torch(value: Mapping[str, Any], path: Path) -> None:
    """Write a cache atomically so a power loss never creates a valid-looking file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, raw_path = tempfile.mkstemp(suffix=".pt", dir=path.parent)
    os.close(descriptor)
    temporary = Path(raw_path)
    try:
        torch.save(dict(value), temporary)
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _cache_root(config: Mapping[str, Any]) -> Path:
    return ROOT / str(config["experiment"]["output_dir"]) / "development_evaluation" / "prediction_cache"


def _cache_path(config: Mapping[str, Any], variant: str, condition_name: str) -> Path:
    return _cache_root(config) / variant / f"{condition_name}.pt"


def _tensor_sha256(value: torch.Tensor) -> str:
    array = value.detach().contiguous().cpu().numpy()
    return hashlib.sha256(array.tobytes()).hexdigest().upper()


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


def diagnostic_value(name: str, value: float, errors: np.ndarray) -> tuple[float | None, bool, str | None]:
    """Return an explicit null only for a declared mathematical domain boundary."""
    if name not in PER_IMAGE_DIAGNOSTICS:
        raise KeyError(f"Unknown diagnostic metric: {name}")
    if np.isfinite(value):
        return float(value), True, None
    flattened = np.asarray(errors, dtype=bool).reshape(-1)
    reason = PER_IMAGE_DIAGNOSTICS[name]
    valid_undefined = (
        (name == "error_auroc" and len(flattened) and np.unique(flattened).size == 1)
        or (name in {"error_auprc", "top_10_uncertainty_recall"} and len(flattened) and not flattened.any())
    )
    if valid_undefined:
        return None, False, reason
    raise AssertionError(f"{name} is non-finite outside its declared mathematical domain.")


def error_auroc_diagnostic(value: float, errors: np.ndarray) -> tuple[float | None, bool, str | None]:
    """Backward-compatible public helper for the frozen AUROC diagnostic rule."""
    return diagnostic_value("error_auroc", value, errors)


def validate_finite_table(table: pd.DataFrame) -> None:
    """Reject every non-finite value except declared per-image diagnostic nulls."""
    for _, row in table.iterrows():
        if not bool(row.get("region_defined", True)):
            if row.get("region_undefined_reason") != "empty_region":
                raise AssertionError("An undefined region lacks the declared empty_region reason.")
            continue
        for metric, declared_reason in PER_IMAGE_DIAGNOSTICS.items():
            value = row.get(metric)
            defined = bool(row.get(f"{metric}_defined", False))
            reason = row.get(f"{metric}_undefined_reason")
            if defined:
                if not np.isfinite(value):
                    raise AssertionError(f"Defined diagnostic {metric} is non-finite.")
            elif not (pd.isna(value) and reason == declared_reason):
                raise AssertionError(f"Undefined diagnostic {metric} violates its declared domain rule.")
        for name, value in row.items():
            if name in PER_IMAGE_DIAGNOSTICS or name.endswith("_defined") or name.endswith("_undefined_reason"):
                continue
            if isinstance(value, (float, np.floating, int, np.integer)) and not isinstance(value, (bool, np.bool_)):
                if not np.isfinite(value):
                    raise AssertionError(f"A required evaluation metric is non-finite: {name}.")


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
                row: dict[str, object] = {
                    "variant": variant, "sample_id": str(sample_id), "condition": condition.name,
                    "degradation_type": condition.degradation_type, "severity": condition.severity,
                    "region": region_name, "region_defined": False, "region_undefined_reason": "empty_region",
                }
                for metric, reason in PER_IMAGE_DIAGNOSTICS.items():
                    row[metric] = None
                    row[f"{metric}_defined"] = False
                    row[f"{metric}_undefined_reason"] = reason
                rows.append(row)
                continue
            error_values = prediction[index][valid[index]].ne(labels[index][valid[index]]).detach().cpu().numpy()
            ranking = ranking_metrics_by_region(
                uncertainty[index][valid[index]].detach().cpu().numpy(),
                error_values,
                {region_name: mask[valid[index]].detach().cpu().numpy()},
                coverages=(0.9, 0.8, 0.7), top_fractions=(0.1,),
            )[region_name]
            region_errors = error_values[mask[valid[index]].detach().cpu().numpy()]
            diagnostic_fields: dict[str, object] = {}
            for metric in PER_IMAGE_DIAGNOSTICS:
                value, defined, reason = diagnostic_value(metric, float(ranking[metric]), region_errors)
                ranking[metric] = value
                diagnostic_fields[f"{metric}_defined"] = defined
                diagnostic_fields[f"{metric}_undefined_reason"] = reason
            selected_probs = probabilities[index:index + 1]
            selected_labels = labels[index:index + 1].clone()
            selected_labels[:, ~mask] = 255
            row = {
                "variant": variant, "sample_id": str(sample_id), "condition": condition.name,
                "degradation_type": condition.degradation_type, "severity": condition.severity,
                "region": region_name, "region_defined": True, "region_undefined_reason": None,
                **_masked_segmentation(prediction[index:index + 1], labels[index:index + 1], mask.unsqueeze(0)),
                "nll": nll(selected_probs, selected_labels), "brier": brier_score(selected_probs, selected_labels),
                "ece": expected_calibration_error(selected_probs, selected_labels, bins=15),
                **diagnostic_fields,
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
                if not np.isfinite(values).all():
                    rows.append({"comparison": comparison, "region": region, "metric": metric,
                                 "defined": False, "undefined_reason": "one_or_more_clusters_metric_undefined",
                                 "mean_improvement": None, "ci95_low": None, "ci95_high": None,
                                 "iterations": ITERATIONS, "cluster_unit": "sample_id_with_all_13_conditions",
                                 "clusters": len(values)})
                    continue
                draws = values[generator.integers(0, len(values), size=(ITERATIONS, len(values)))].mean(axis=1)
                rows.append({"comparison": comparison, "region": region, "metric": metric, "defined": True,
                             "undefined_reason": None,
                             "mean_improvement": float(values.mean()), "ci95_low": float(np.quantile(draws, .025)),
                             "ci95_high": float(np.quantile(draws, .975)), "iterations": ITERATIONS,
                             "cluster_unit": "sample_id_with_all_13_conditions", "clusters": len(values)})
    table = pd.DataFrame(rows)
    primary = table[(table.comparison == "E_vs_C") & (table.region == "full") & (table.metric == "eaurc")].iloc[0]
    if not bool(primary.defined) or not np.isfinite(primary.ci95_low):
        raise AssertionError("The frozen full eAURC primary endpoint is non-finite.")
    return table, {"primary_endpoint": "full_eaurc_C_minus_E", "ci95_lower_bound": float(primary.ci95_low), "passes": bool(primary.ci95_low > 0)}


def _cache_expected_metadata(*, variant: str, condition: Any, checkpoint: Path, development_csv: Path) -> dict[str, str]:
    return {
        "schema": CACHE_SCHEMA,
        "variant": variant,
        "condition": condition.name,
        "checkpoint_sha256": sha256(checkpoint).upper(),
        "development_csv_sha256": sha256(development_csv).upper(),
    }


def _validate_cache_payload(payload: Mapping[str, Any], *, expected: Mapping[str, str], sample_ids: list[str]) -> None:
    for key, value in expected.items():
        if payload.get(key) != value:
            raise ValueError(f"Prediction cache provenance mismatch: {key}")
    if list(payload.get("sample_ids", ())) != sample_ids:
        raise ValueError("Prediction cache sample IDs do not match the frozen development CSV.")
    logits = payload.get("logits")
    labels = payload.get("labels")
    if not isinstance(logits, torch.Tensor) or logits.dtype != torch.float16 or logits.ndim != 4 or logits.shape[:2] != (len(sample_ids), 8):
        raise ValueError("Prediction cache logits must be finite float16 [231,8,h,w].")
    if not isinstance(labels, torch.Tensor) or labels.dtype != torch.uint8 or tuple(labels.shape) != (len(sample_ids), 384, 384):
        raise ValueError("Prediction cache labels must be uint8 [231,384,384].")
    if not torch.isfinite(logits).all() or labels.max().item() > 7:
        raise ValueError("Prediction cache contains non-finite logits or invalid labels.")
    if payload.get("labels_sha256") != _tensor_sha256(labels):
        raise ValueError("Prediction cache label checksum mismatch.")


def _cache_inference(config: Mapping[str, Any], development_csv: Path, checkpoints: Mapping[str, Path], conditions: list[Any]) -> None:
    """Run the expensive model forward pass once and atomically persist low-resolution logits."""
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for fixed-protocol inference caching.")
    frame = pd.read_csv(development_csv)
    sample_ids = frame["sample_id"].astype(str).tolist()
    if len(sample_ids) != 231 or len(set(sample_ids)) != len(sample_ids):
        raise ValueError("The frozen method_development role must contain exactly 231 unique images.")
    device = torch.device("cuda")
    root = _cache_root(config)
    manifest: dict[str, Any] = {"schema": CACHE_SCHEMA, "split": "method_development", "samples": len(sample_ids),
                                "conditions": [item.name for item in conditions], "variants": list(VARIANTS), "files": {}}
    for variant in VARIANTS:
        model = load_final_model(config, checkpoints[variant], device=device)
        for condition in conditions:
            expected = _cache_expected_metadata(variant=variant, condition=condition, checkpoint=checkpoints[variant], development_csv=development_csv)
            path = _cache_path(config, variant, condition.name)
            if path.is_file():
                payload = torch.load(path, map_location="cpu", weights_only=False)
                _validate_cache_payload(payload, expected=expected, sample_ids=sample_ids)
            else:
                dataset = SUIMDataset(development_csv, transform=build_eval_transform(384), image_degradation=build_image_degradation(condition))
                loader = torch.utils.data.DataLoader(dataset, batch_size=4, shuffle=False, num_workers=0, pin_memory=True)
                cached_logits: list[torch.Tensor] = []
                cached_labels: list[torch.Tensor] = []
                observed_ids: list[str] = []
                for batch in loader:
                    with torch.no_grad(), torch.amp.autocast("cuda", enabled=True):
                        logits = model(pixel_values=batch["pixel_values"].to(device, non_blocking=True)).logits
                    cached_logits.append(logits.detach().cpu().to(dtype=torch.float16))
                    cached_labels.append(batch["labels"].detach().cpu().to(dtype=torch.uint8))
                    observed_ids.extend(str(value) for value in batch["sample_id"])
                payload = {**expected, "sample_ids": observed_ids,
                           "logits": torch.cat(cached_logits, dim=0), "labels": torch.cat(cached_labels, dim=0)}
                payload["labels_sha256"] = _tensor_sha256(payload["labels"])
                _validate_cache_payload(payload, expected=expected, sample_ids=sample_ids)
                _atomic_torch(payload, path)
            manifest["files"][f"{variant}/{condition.name}"] = {"path": str(path.relative_to(ROOT)), "sha256": sha256(path).upper()}
            print(f"cached {variant}/{condition.name}", flush=True)
        del model
        torch.cuda.empty_cache()
    _atomic_json(manifest, root / "cache_manifest.json")
    _atomic_json({"schema": CACHE_SCHEMA, "status": "inference_complete", "files": len(manifest["files"]),
                  "validation_evaluated": False, "calibration_evaluated": False,
                  "official_suim_test_evaluated": False, "model_retrained": False}, root / "inference_complete.json")


def _rows_from_cache(config: Mapping[str, Any], development_csv: Path, checkpoints: Mapping[str, Path], conditions: list[Any]) -> pd.DataFrame:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required to replay frozen interpolation from the prediction cache.")
    frame = pd.read_csv(development_csv)
    sample_ids = frame["sample_id"].astype(str).tolist()
    rows: list[dict[str, object]] = []
    device = torch.device("cuda")
    for variant in VARIANTS:
        for condition in conditions:
            expected = _cache_expected_metadata(variant=variant, condition=condition, checkpoint=checkpoints[variant], development_csv=development_csv)
            path = _cache_path(config, variant, condition.name)
            if not path.is_file():
                raise FileNotFoundError(f"Missing immutable prediction cache: {path}")
            payload = torch.load(path, map_location="cpu", weights_only=False)
            _validate_cache_payload(payload, expected=expected, sample_ids=sample_ids)
            for start in range(0, len(sample_ids), 4):
                end = min(start + 4, len(sample_ids))
                with torch.no_grad(), torch.amp.autocast("cuda", enabled=True):
                    logits = functional.interpolate(payload["logits"][start:end].to(device, non_blocking=True), size=(384, 384), mode="bilinear", align_corners=False)
                rows.extend(rows_for_batch(logits, payload["labels"][start:end].to(device, non_blocking=True).long(),
                                           sample_ids[start:end], variant=variant, condition=condition))
    table = pd.DataFrame(rows)
    expected_rows = len(VARIANTS) * len(conditions) * len(frame) * len(REGIONS)
    keys = ["variant", "condition", "sample_id", "region"]
    if len(table) != expected_rows or table.duplicated(keys).any():
        raise AssertionError("Cached FT development result table is incomplete or duplicated.")
    validate_finite_table(table)
    return table


def _metric_domain_audit(table: pd.DataFrame) -> dict[str, Any]:
    """Return only domain counts/reasons: never metric values, deltas, or decisions."""
    diagnostics: dict[str, Any] = {}
    for metric, reason in PER_IMAGE_DIAGNOSTICS.items():
        defined = table[f"{metric}_defined"].astype(bool)
        reasons = table.loc[~defined, f"{metric}_undefined_reason"].fillna("missing_reason").value_counts().to_dict()
        diagnostics[metric] = {"defined_count": int(defined.sum()), "undefined_count": int((~defined).sum()),
                               "undefined_reasons": {str(key): int(value) for key, value in reasons.items()},
                               "declared_reason": reason}
    bootstrap, _ = paired_bootstrap(table)
    bootstrap_status = [{"comparison": str(row.comparison), "region": str(row.region), "metric": str(row.metric),
                         "defined": bool(row.defined), "undefined_reason": None if pd.isna(row.undefined_reason) else str(row.undefined_reason),
                         "iterations_executed": int(row.iterations) if bool(row.defined) else 0}
                        for row in bootstrap.itertuples(index=False)]
    required_columns = ["miou", "pixel_accuracy", "mean_dice", "nll", "brier", "ece", "eaurc"]
    required_nonfinite = {name: int((~np.isfinite(table.loc[table.region_defined.astype(bool), name].to_numpy(dtype=float))).sum())
                          for name in required_columns}
    return {"status": "metric_domain_audit_complete", "rows_scanned": int(len(table)),
            "expected_rows": int(len(VARIANTS) * 13 * 231 * len(REGIONS)),
            "region_defined_count": int(table.region_defined.astype(bool).sum()),
            "region_undefined_count": int((~table.region_defined.astype(bool)).sum()),
            "diagnostics": diagnostics, "required_nonfinite_counts": required_nonfinite,
            "bootstrap_domain": bootstrap_status,
            "scientific_metric_values_emitted": False, "model_comparisons_emitted": False,
            "pass_fail_emitted": False}


def _empty_calibration_sums() -> dict[str, Any]:
    return {"pixels": 0, "nll_sum": 0.0, "brier_sum": 0.0,
            "ece_count": np.zeros(15, dtype=np.int64), "ece_confidence": np.zeros(15, dtype=np.float64),
            "ece_correct": np.zeros(15, dtype=np.float64),
            "class_count": np.zeros((8, 15), dtype=np.int64),
            "class_probability": np.zeros((8, 15), dtype=np.float64),
            "class_event": np.zeros((8, 15), dtype=np.float64)}


def _update_calibration_sums(sums: dict[str, Any], probabilities: torch.Tensor, labels: torch.Tensor, mask: torch.Tensor) -> None:
    """Accumulate sufficient statistics; final calibration values never average per-image metrics."""
    selected = probabilities.movedim(1, -1)[mask].float()
    target = labels[mask].long()
    if selected.numel() == 0:
        return
    count = int(target.numel())
    epsilon = torch.finfo(selected.dtype).eps
    true_probability = selected[torch.arange(count, device=selected.device), target].clamp_min(epsilon)
    sums["pixels"] += count
    sums["nll_sum"] += float((-true_probability.log()).sum().item())
    sums["brier_sum"] += float((selected.square().sum(dim=1) - 2.0 * true_probability + 1.0).sum().item())
    confidence, prediction = selected.max(dim=1)
    correct = prediction.eq(target)
    bin_index = torch.clamp((confidence * 15).to(torch.long), max=14)
    for bin_id in range(15):
        chosen = bin_index.eq(bin_id)
        if chosen.any():
            sums["ece_count"][bin_id] += int(chosen.sum().item())
            sums["ece_confidence"][bin_id] += float(confidence[chosen].sum().item())
            sums["ece_correct"][bin_id] += float(correct[chosen].sum().item())
    for class_id in range(8):
        values = selected[:, class_id]
        event = target.eq(class_id)
        class_bin = torch.clamp((values * 15).to(torch.long), max=14)
        for bin_id in range(15):
            chosen = class_bin.eq(bin_id)
            if chosen.any():
                sums["class_count"][class_id, bin_id] += int(chosen.sum().item())
                sums["class_probability"][class_id, bin_id] += float(values[chosen].sum().item())
                sums["class_event"][class_id, bin_id] += float(event[chosen].sum().item())


def _ece_from_sums(counts: np.ndarray, prediction_sums: np.ndarray, event_sums: np.ndarray) -> float:
    total = int(counts.sum())
    if total == 0:
        raise ValueError("A required aggregate region contains no valid pixels.")
    chosen = counts > 0
    weights = counts[chosen].astype(np.float64) / total
    calibration_gap = np.abs(event_sums[chosen] / counts[chosen] - prediction_sums[chosen] / counts[chosen])
    return float(np.sum(weights * calibration_gap))


def _direct_condition_aggregate(
    *, config: Mapping[str, Any], development_csv: Path, checkpoint: Path, variant: str, condition: Any,
) -> list[dict[str, object]]:
    """Recompute condition metrics from cached predictions, not from per-image means."""
    frame = pd.read_csv(development_csv)
    sample_ids = frame["sample_id"].astype(str).tolist()
    expected = _cache_expected_metadata(variant=variant, condition=condition, checkpoint=checkpoint, development_csv=development_csv)
    payload = torch.load(_cache_path(config, variant, condition.name), map_location="cpu", weights_only=False)
    _validate_cache_payload(payload, expected=expected, sample_ids=sample_ids)
    device = torch.device("cuda")
    matrices = {region: torch.zeros((8, 8), dtype=torch.long) for region in REGIONS}
    calibration_sums = {region: _empty_calibration_sums() for region in REGIONS}
    score_chunks: dict[str, list[np.ndarray]] = {region: [] for region in REGIONS}
    error_chunks: dict[str, list[np.ndarray]] = {region: [] for region in REGIONS}
    for start in range(0, len(sample_ids), 4):
        end = min(start + 4, len(sample_ids))
        labels = payload["labels"][start:end].to(device, non_blocking=True).long()
        with torch.no_grad(), torch.amp.autocast("cuda", enabled=True):
            logits = functional.interpolate(payload["logits"][start:end].to(device, non_blocking=True), size=(384, 384), mode="bilinear", align_corners=False)
        probabilities = logits.float().softmax(dim=1)
        prediction = probabilities.argmax(dim=1)
        valid = labels.ge(0) & labels.lt(8)
        boundary = valid & _boundary(labels)
        regions = {"full": valid, "boundary": boundary, "interior": valid & ~boundary}
        uncertainty = 1.0 - probabilities.max(dim=1).values
        errors = prediction.ne(labels)
        for region, mask in regions.items():
            if not mask.any():
                continue
            masked_labels = labels.clone()
            masked_labels[~mask] = 255
            matrices[region] += confusion_matrix(prediction, masked_labels, num_classes=8, ignore_index=255).cpu()
            _update_calibration_sums(calibration_sums[region], probabilities, labels, mask)
            score_chunks[region].append(uncertainty[mask].detach().cpu().numpy().astype(np.float32, copy=False))
            error_chunks[region].append(errors[mask].detach().cpu().numpy().astype(bool, copy=False))
    rows: list[dict[str, object]] = []
    for region in REGIONS:
        sums = calibration_sums[region]
        if sums["pixels"] == 0:
            raise ValueError(f"The aggregate {region} region contains no valid pixels.")
        segmentation = metrics_from_confusion_matrix(matrices[region])
        scores = np.concatenate(score_chunks[region])
        errors = np.concatenate(error_chunks[region])
        ranking = ranking_metrics(scores, errors, coverages=(0.9, 0.8, 0.7), top_fractions=(0.1,))
        diagnostics: dict[str, object] = {}
        for metric in PER_IMAGE_DIAGNOSTICS:
            value, defined, reason = diagnostic_value(metric, float(ranking[metric]), errors)
            ranking[metric] = value
            diagnostics[f"{metric}_defined"] = defined
            diagnostics[f"{metric}_undefined_reason"] = reason
        if not diagnostics["error_auprc_defined"]:
            raise AssertionError("Aggregate Error AUPRC is a required final metric and must be finite.")
        classwise = [_ece_from_sums(sums["class_count"][class_id], sums["class_probability"][class_id], sums["class_event"][class_id])
                     for class_id in range(8)]
        row: dict[str, object] = {
            "variant": variant, "condition": condition.name, "degradation_type": condition.degradation_type,
            "severity": condition.severity, "region": region, "pixels": sums["pixels"],
            "miou": float(segmentation["miou"]), "pixel_accuracy": float(segmentation["pixel_accuracy"]),
            "mean_dice": float(segmentation["mean_dice"]), "nll": sums["nll_sum"] / sums["pixels"],
            "brier": sums["brier_sum"] / sums["pixels"],
            "ece": _ece_from_sums(sums["ece_count"], sums["ece_confidence"], sums["ece_correct"]),
            **diagnostics, **ranking,
        }
        for class_id, value in enumerate(classwise):
            row[f"classwise_ece_{class_id}"] = value
        rows.append(row)
    return rows


def _aggregate_from_cache(config: Mapping[str, Any], development_csv: Path, checkpoints: Mapping[str, Path], conditions: list[Any]) -> pd.DataFrame:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required to replay frozen interpolation from the prediction cache.")
    rows: list[dict[str, object]] = []
    for variant in VARIANTS:
        for condition in conditions:
            rows.extend(_direct_condition_aggregate(config=config, development_csv=development_csv,
                                                     checkpoint=checkpoints[variant], variant=variant, condition=condition))
    table = pd.DataFrame(rows)
    if len(table) != len(VARIANTS) * len(conditions) * len(REGIONS):
        raise AssertionError("Direct aggregate table is incomplete.")
    required = ["miou", "pixel_accuracy", "mean_dice", "nll", "brier", "ece", "eaurc", "error_auprc"]
    if not np.isfinite(table[required].to_numpy(dtype=float)).all():
        raise AssertionError("A required direct aggregate metric is non-finite.")
    return table


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--phase", required=True, choices=("inference", "audit", "aggregate"),
                        help="Run immutable inference caching, a value-free audit, or the final frozen aggregation.")
    args = parser.parse_args()
    config = load_config(ROOT)
    development_csv, checkpoints = preflight(config)
    frame = pd.read_csv(development_csv)
    if len(frame) != 231 or frame.sample_id.duplicated().any():
        raise ValueError("The frozen method_development role must contain exactly 231 unique images.")
    conditions = load_conditions(ROOT / str(config["degradations"]["registry_config"]))
    if len(conditions) != 13:
        raise ValueError("Evaluation requires exactly the registered 13 conditions.")
    output = ROOT / str(config["experiment"]["output_dir"]) / "development_evaluation"
    if args.phase == "inference":
        _cache_inference(config, development_csv, checkpoints, conditions)
        return
    table = _rows_from_cache(config, development_csv, checkpoints, conditions)
    if args.phase == "audit":
        _atomic_json(_metric_domain_audit(table), output / "metric_domain_audit.json")
        return
    audit_path = output / "metric_domain_audit.json"
    if not audit_path.is_file():
        raise FileNotFoundError("Final aggregation requires the completed value-free metric-domain audit.")
    audit = json.loads(audit_path.read_text(encoding="utf-8"))
    if audit.get("status") != "metric_domain_audit_complete" or any(int(value) for value in audit.get("required_nonfinite_counts", {}).values()):
        raise AssertionError("Final aggregation is blocked until the metric-domain audit is clean.")
    bootstrap, primary = paired_bootstrap(table)
    aggregate = _aggregate_from_cache(config, development_csv, checkpoints, conditions)
    _atomic_csv(table, output / "per_image_metrics.csv")
    _atomic_csv(aggregate, output / "aggregate_metrics.csv")
    _atomic_csv(bootstrap, output / "paired_cluster_bootstrap.csv")
    metadata = {
        "protocol": PROTOCOL_COMMIT, "split_evaluated": "method_development", "samples": 231,
        "conditions": [item.name for item in conditions], "variants": list(VARIANTS),
        "checkpoint_sha256": {variant: sha256(path) for variant, path in checkpoints.items()},
        "bootstrap_iterations": ITERATIONS, "primary_endpoint": primary,
        "validation_evaluated": False, "calibration_evaluated": False,
        "official_suim_test_evaluated": False, "model_retrained": False,
    }
    _atomic_json(metadata, output / "metadata.json")
    _atomic_json({"per_image_metrics.csv": sha256(output / "per_image_metrics.csv"),
                  "aggregate_metrics.csv": sha256(output / "aggregate_metrics.csv"),
                  "paired_cluster_bootstrap.csv": sha256(output / "paired_cluster_bootstrap.csv"),
                  "metadata.json": sha256(output / "metadata.json")}, output / "result_file_sha256.json")


if __name__ == "__main__":
    main()
