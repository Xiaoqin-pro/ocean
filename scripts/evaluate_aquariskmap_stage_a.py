"""Run the preregistered SUIM Stage-A ranking evaluation on validation only.

This evaluator deliberately has no split, condition, or threshold arguments.
It consumes all thirteen existing frozen validation-logit caches, and refuses
calibration and official-TEST paths.  CRC calibration is a separate later
stage and is not read here.
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
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from datasets.suim_dataset import SUIMDataset, build_eval_transform  # noqa: E402
from degradations.registry import build_image_degradation, load_conditions  # noqa: E402
from metrics.uncertainty_ranking import SCORE_NAMES, ranking_metrics_by_region, uncertainty_scores  # noqa: E402
from reliability.aquariskmap import CONDITIONS, IGNORE_INDEX, AquaRiskMap, build_features, full_resolution_risk, gt_boundary  # noqa: E402
from scripts.cache_temperature_logits import sha256  # noqa: E402
from scripts.evaluate_temperature_scaling import validate_cache_payload  # noqa: E402


ALLOWED_MODELS = ("segformer", "deeplab")
SCORES = (*SCORE_NAMES, "aquariskmap")
PRIMARY_SCORES = ("raw_msp", "aquariskmap")
REGIONS = ("full", "boundary", "interior")
BOOTSTRAP_ITERATIONS = 1000
BOOTSTRAP_SEED = 20260725


def load_config(root: Path = ROOT) -> dict[str, Any]:
    return yaml.safe_load((root / "configs" / "aquariskmap_pilot.yaml").read_text(encoding="utf-8"))


def _atomic_csv(table: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile("w", suffix=".csv", dir=path.parent, delete=False, encoding="utf-8", newline="") as handle:
        temporary = Path(handle.name)
        table.to_csv(handle, index=False)
    try:
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def _atomic_json(value: Mapping[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile("w", suffix=".json", dir=path.parent, delete=False, encoding="utf-8") as handle:
        temporary = Path(handle.name)
        json.dump(dict(value), handle, ensure_ascii=False, indent=2, sort_keys=True)
        handle.write("\n")
    try:
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def contains_barred_split_component(path: Path) -> bool:
    """Reject a real calibration/TEST directory, not the project name itself."""
    return any(part.lower() in {"calibration", "test"} for part in path.parts)


def resolve_stage_a_context(root: Path, config: Mapping[str, Any], *, model_name: str) -> tuple[Path, Path, Path, Path, Path, Path]:
    """Resolve only frozen validation caches and final risk-head checkpoint."""
    if model_name not in ALLOWED_MODELS:
        raise ValueError("Stage A permits only the two preregistered base models.")
    if not config["protocol"]["official_suim_test_locked"] or config["protocol"]["official_suim_test_evaluated"]:
        raise ValueError("Official SUIM TEST must remain locked.")
    val_csv = (root / str(config["protocol"]["validation_split"])).resolve()
    if val_csv.name != "val.csv" or not val_csv.is_file():
        raise ValueError("Stage A may read only the frozen validation CSV.")
    if contains_barred_split_component(val_csv):
        raise ValueError("Stage A validation path must not resolve to calibration or TEST.")
    temperature_config = root / ("configs/temperature_scaling.yaml" if model_name == "segformer" else "configs/deeplabv3_temperature_scaling.yaml")
    cache_config = yaml.safe_load(temperature_config.read_text(encoding="utf-8"))
    cache_root = (root / str(cache_config["experiment"]["output_dir"]) / "cache" / "val").resolve()
    if not cache_root.is_dir() or contains_barred_split_component(cache_root):
        raise ValueError("Stage A requires the existing frozen validation-logit cache only.")
    risk_checkpoint = (root / str(config["experiment"]["output_dir"]) / model_name / "checkpoints" / "last.pt").resolve()
    if not risk_checkpoint.is_file():
        raise FileNotFoundError(risk_checkpoint)
    base_checkpoint = (root / str(config["base_models"][model_name]["checkpoint"])).resolve()
    degradation = (root / str(config["degradations"]["config"])).resolve()
    temperature_path = (root / str(cache_config["experiment"]["output_dir"]) / "temperatures.json").resolve()
    baseline_path = (root / str(config["base_models"][model_name]["baseline_config"])).resolve()
    if not all(path.is_file() for path in (base_checkpoint, degradation, temperature_path, baseline_path)):
        raise FileNotFoundError("Frozen base checkpoint or degradation registry is missing.")
    return val_csv, cache_root, risk_checkpoint, degradation, temperature_path, baseline_path


def load_risk_head(path: Path, *, model_name: str, device: torch.device) -> AquaRiskMap:
    checkpoint = torch.load(path, map_location=device, weights_only=False)
    required_false = ("validation_evaluated", "calibration_evaluated", "official_suim_test_evaluated")
    if checkpoint.get("checkpoint_format") != "aquariskmap_pilot_v1" or checkpoint.get("epoch") != 20 or checkpoint.get("model_name") != model_name:
        raise ValueError("Stage A requires the final fixed-protocol AquaRiskMap checkpoint.")
    if checkpoint.get("checkpoint_selection") != "final_epoch" or any(bool(checkpoint.get(key, True)) for key in required_false):
        raise ValueError("Risk-head checkpoint violates the locked Stage-A protocol.")
    model = AquaRiskMap().to(device).eval()
    model.load_state_dict(checkpoint["model_state_dict"])
    return model


def _score_arrays(logits: torch.Tensor, pixels: torch.Tensor, risk_head: AquaRiskMap, labels: torch.Tensor, *, clean_temperature: float) -> tuple[dict[str, torch.Tensor], torch.Tensor, dict[str, torch.Tensor]]:
    full_logits = functional.interpolate(logits.float(), size=labels.shape[-2:], mode="bilinear", align_corners=False)
    prediction = full_logits.argmax(dim=1)
    risk = full_resolution_risk(risk_head(build_features(logits, pixels)), size=tuple(labels.shape[-2:]))[:, 0].sigmoid()
    valid = labels.ne(IGNORE_INDEX) & labels.ge(0) & labels.lt(full_logits.shape[1])
    boundary = valid & gt_boundary(labels, radius=3)
    regions = {"full": valid, "boundary": boundary, "interior": valid & ~boundary}
    return {**uncertainty_scores(full_logits, temperature=clean_temperature), "aquariskmap": risk}, prediction.ne(labels), regions


def rows_for_condition(payload: Mapping[str, Any], dataset: SUIMDataset, risk_head: AquaRiskMap, *, device: torch.device, clean_temperature: float) -> list[dict[str, object]]:
    labels = payload["labels"]
    logits = payload["logits"]
    sample_ids = [str(value) for value in payload["sample_id"]]
    rows: list[dict[str, object]] = []
    if len(dataset) != len(sample_ids):
        raise ValueError("Frozen-logit cache and validation dataset have different sample counts.")
    for start in range(0, len(sample_ids), 4):
        batch_logits = logits[start:start + 4].to(device=device, dtype=torch.float32, non_blocking=True)
        samples = [dataset[index] for index in range(start, min(start + 4, len(sample_ids)))]
        if [item["sample_id"] for item in samples] != sample_ids[start:start + len(samples)]:
            raise ValueError("Validation image order differs from the frozen-logit cache.")
        batch_pixels = torch.stack([item["pixel_values"] for item in samples]).to(device=device, dtype=torch.float32, non_blocking=True)
        batch_labels = torch.stack([item["labels"] for item in samples])
        if not torch.equal(batch_labels, labels[start:start + len(samples)]):
            raise ValueError("Validation labels differ from the frozen-logit cache.")
        batch_labels = batch_labels.to(device=device, dtype=torch.long, non_blocking=True)
        with torch.no_grad():
            scores, errors, regions = _score_arrays(batch_logits, batch_pixels, risk_head, batch_labels, clean_temperature=clean_temperature)
        for offset, sample_id in enumerate(sample_ids[start:start + len(samples)]):
            valid_count = int(regions["full"][offset].sum())
            if valid_count < 1:
                raise ValueError("A Stage-A validation image has no valid pixels.")
            for score_name, values in scores.items():
                image_scores = values[offset][regions["full"][offset]].detach().cpu().numpy()
                image_errors = errors[offset][regions["full"][offset]].detach().cpu().numpy()
                image_regions = {
                    "full": np.ones(valid_count, dtype=bool),
                    "boundary": regions["boundary"][offset][regions["full"][offset]].detach().cpu().numpy(),
                    "interior": regions["interior"][offset][regions["full"][offset]].detach().cpu().numpy(),
                }
                for region_name, metrics in ranking_metrics_by_region(image_scores, image_errors, image_regions, discrete_histogram=score_name == "local_disagreement", coverages=(0.9, 0.8, 0.7), top_fractions=(0.05, 0.1, 0.2)).items():
                    rows.append({"sample_id": sample_id, "score": score_name, "region": region_name, **metrics})
    return rows


def aggregate_and_bootstrap(per_image: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    required = {"sample_id", "condition", "score", "region", "eaurc", "error_auprc", "top_10_uncertainty_recall"}
    if missing := required.difference(per_image.columns):
        raise ValueError(f"Missing Stage-A metric columns: {sorted(missing)}")
    expected = len(CONDITIONS) * len(SCORES) * len(REGIONS)
    counts = per_image.groupby("sample_id").size()
    if not (counts == expected).all():
        raise ValueError("Each Stage-A image cluster must retain all 13 conditions, scores, and regions.")
    aggregate = per_image.groupby(["score", "region"], as_index=False).mean(numeric_only=True)
    rows: list[dict[str, object]] = []
    generator = np.random.default_rng(BOOTSTRAP_SEED)
    metrics = (("eaurc", -1.0), ("error_auprc", 1.0), ("top_10_uncertainty_recall", 1.0))
    for region in REGIONS:
        raw = per_image[(per_image.score == "raw_msp") & (per_image.region == region)].groupby("sample_id").mean(numeric_only=True)
        candidate = per_image[(per_image.score == "aquariskmap") & (per_image.region == region)].groupby("sample_id").mean(numeric_only=True)
        if not raw.index.equals(candidate.index):
            raise ValueError("Paired Stage-A score clusters do not align.")
        for metric, direction in metrics:
            values = direction * (candidate[metric] - raw[metric]).to_numpy(dtype=np.float64)
            draws = values[generator.integers(0, len(values), size=(BOOTSTRAP_ITERATIONS, len(values)))].mean(axis=1)
            rows.append({"region": region, "metric": metric, "direction": "positive_is_better", "mean_improvement": float(values.mean()), "ci95_low": float(np.quantile(draws, 0.025)), "ci95_high": float(np.quantile(draws, 0.975)), "iterations": BOOTSTRAP_ITERATIONS, "cluster_unit": "sample_id_with_all_13_conditions", "clusters": len(values)})
    bootstrap = pd.DataFrame(rows)
    lookup = aggregate.set_index(["score", "region"])
    gate = {
        "full_eaurc_relative_decrease": float((lookup.loc[("raw_msp", "full"), "eaurc"] - lookup.loc[("aquariskmap", "full"), "eaurc"]) / lookup.loc[("raw_msp", "full"), "eaurc"]),
        "boundary_eaurc_relative_decrease": float((lookup.loc[("raw_msp", "boundary"), "eaurc"] - lookup.loc[("aquariskmap", "boundary"), "eaurc"]) / lookup.loc[("raw_msp", "boundary"), "eaurc"]),
        "full_error_auprc_increase": float(lookup.loc[("aquariskmap", "full"), "error_auprc"] - lookup.loc[("raw_msp", "full"), "error_auprc"]),
        "full_top10_recall_increase": float(lookup.loc[("aquariskmap", "full"), "top_10_uncertainty_recall"] - lookup.loc[("raw_msp", "full"), "top_10_uncertainty_recall"]),
    }
    gate["ranking_gate_pass"] = bool(gate["full_eaurc_relative_decrease"] >= 0.10 and gate["boundary_eaurc_relative_decrease"] >= 0.10 and gate["full_error_auprc_increase"] >= 0.03 and gate["full_top10_recall_increase"] >= 0.05)
    return aggregate, bootstrap, gate


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True, choices=ALLOWED_MODELS)
    args = parser.parse_args()
    config = load_config(ROOT)
    val_csv, cache_root, risk_checkpoint, degradation_path, temperature_path, baseline_path = resolve_stage_a_context(ROOT, config, model_name=args.model)
    conditions = load_conditions(degradation_path)
    if tuple(item.name for item in conditions) != CONDITIONS:
        raise ValueError("Stage-A requires the frozen 13-condition degradation registry.")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for Stage-A AquaRiskMap inference.")
    baseline = yaml.safe_load(baseline_path.read_text(encoding="utf-8"))
    temperatures = json.loads(temperature_path.read_text(encoding="utf-8"))
    clean_temperature = float(temperatures["clean_global"])
    device = torch.device("cuda")
    risk_head = load_risk_head(risk_checkpoint, model_name=args.model, device=device)
    base_checkpoint = ROOT / str(config["base_models"][args.model]["checkpoint"])
    expected_checkpoint_hash = sha256(base_checkpoint)
    degradation_hash = sha256(degradation_path)
    expected_ids = pd.read_csv(val_csv)["sample_id"].astype(str).tolist()
    if len(expected_ids) != 146 or len(set(expected_ids)) != len(expected_ids):
        raise ValueError("Stage A expects exactly the frozen 146-image SUIM validation partition.")
    all_rows: list[dict[str, object]] = []
    for condition in conditions:
        payload = torch.load(cache_root / f"{condition.name}.pt", map_location="cpu", weights_only=False)
        validate_cache_payload(payload, split="val", condition=condition.name, checkpoint_sha256=expected_checkpoint_hash, degradation_config_sha256=degradation_hash)
        if [str(value) for value in payload["sample_id"]] != expected_ids:
            raise ValueError("Frozen validation-logit cache sample IDs differ from the protocol CSV.")
        dataset = SUIMDataset(val_csv, transform=build_eval_transform(int(baseline["data"]["image_size"])), image_degradation=build_image_degradation(condition))
        current = rows_for_condition(payload, dataset, risk_head, device=device, clean_temperature=clean_temperature)
        all_rows.extend({"model": args.model, "condition": condition.name, "degradation_type": condition.degradation_type, "severity": condition.severity, **row} for row in current)
        print(f"evaluated Stage-A {args.model}/{condition.name}", flush=True)
    table = pd.DataFrame(all_rows)
    expected_rows = len(expected_ids) * len(CONDITIONS) * len(SCORES) * len(REGIONS)
    if len(table) != expected_rows or table.duplicated(["model", "condition", "sample_id", "score", "region"]).any():
        raise AssertionError("Stage-A result table is incomplete or has duplicate keys.")
    aggregate, bootstrap, gate = aggregate_and_bootstrap(table)
    output = ROOT / str(config["experiment"]["output_dir"]) / "stage_a" / args.model
    _atomic_csv(table, output / "per_image_metrics.csv")
    _atomic_csv(aggregate, output / "aggregate_metrics.csv")
    _atomic_csv(bootstrap, output / "clustered_bootstrap.csv")
    metadata = {
        "stage": "A_ranking_validation_only", "model": args.model, "split_evaluated": "val", "conditions": list(CONDITIONS),
        "scores": list(SCORES), "boundary_radius": 3, "bootstrap_iterations": BOOTSTRAP_ITERATIONS,
        "cluster_unit": "sample_id_with_all_13_conditions", "risk_checkpoint_sha256": sha256(risk_checkpoint),
        "base_checkpoint_sha256": expected_checkpoint_hash, "degradation_config_sha256": degradation_hash,
        "validation_csv_sha256": sha256(val_csv), "temperature_file_sha256": sha256(temperature_path),
        "clean_global_temperature": clean_temperature, "ranking_gate": gate,
        "calibration_evaluated": False, "official_suim_test_evaluated": False, "crc_pending": True,
    }
    _atomic_json(metadata, output / "metadata.json")


if __name__ == "__main__":
    main()
