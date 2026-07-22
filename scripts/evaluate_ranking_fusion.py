"""Fit calibration-only logistic uncertainty fusion and evaluate it on frozen val logits."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as functional

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from calibration.ranking_fusion import FEATURE_NAMES, fit_logistic_fusion, fusion_score  # noqa: E402
from metrics.uncertainty_ranking import ranking_metrics_by_region, uncertainty_scores  # noqa: E402
from scripts.analyze_boundary_residual import boundary_mask  # noqa: E402
from scripts.evaluate_temperature_scaling import CONDITIONS, load_yaml, sha256, validate_cache_payload  # noqa: E402
from scripts.evaluate_uncertainty_ranking import _atomic_csv, _atomic_json, _bootstrap  # noqa: E402


def _cached_feature_tensors(low_resolution: torch.Tensor, labels: torch.Tensor, temperature: float) -> tuple[dict[str, torch.Tensor], torch.Tensor, torch.Tensor]:
    device = labels.device
    with torch.amp.autocast("cuda", enabled=device.type == "cuda"):
        amp_logits = functional.interpolate(low_resolution, size=labels.shape[-2:], mode="bilinear", align_corners=False)
        frozen_prediction = amp_logits.argmax(dim=1)
    logits = amp_logits.float()
    prediction = (logits / temperature).argmax(dim=1)
    if not torch.equal(frozen_prediction, prediction):
        raise AssertionError("Float32 fusion features changed frozen AMP predictions.")
    return uncertainty_scores(logits, temperature=temperature), prediction, labels.ge(0) & labels.lt(logits.shape[1])


def fit_from_calibration(cache_root: Path, *, conditions: tuple[str, ...], checkpoint_sha256: str, degradation_sha256: str, temperature: float, samples_per_image: int, c: float, seed: int, limit_samples: int | None) -> tuple[Any, Any, dict[str, object]]:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for frozen-cache feature reproduction.")
    device, rng = torch.device("cuda"), np.random.default_rng(seed)
    features: list[np.ndarray] = []
    targets: list[np.ndarray] = []
    for condition in conditions:
        payload = torch.load(cache_root / "calibration" / f"{condition}.pt", map_location="cpu", weights_only=False)
        validate_cache_payload(payload, split="calibration", condition=condition, checkpoint_sha256=checkpoint_sha256, degradation_config_sha256=degradation_sha256)
        count = len(payload["labels"]) if limit_samples is None else limit_samples
        for start in range(0, count, 4):
            labels = payload["labels"][start:start + 4].long().to(device)
            scores, prediction, valid = _cached_feature_tensors(payload["logits"][start:start + 4].to(device), labels, temperature)
            for offset in range(len(labels)):
                ids = np.flatnonzero(valid[offset].detach().cpu().numpy().reshape(-1))
                take = min(samples_per_image, len(ids))
                selected = rng.choice(ids, size=take, replace=False)
                columns = [scores[name][offset].detach().cpu().numpy().reshape(-1)[selected] for name in FEATURE_NAMES]
                features.append(np.stack(columns, axis=1))
                targets.append(prediction[offset].ne(labels[offset]).detach().cpu().numpy().reshape(-1)[selected])
    scaler, classifier, parameters = fit_logistic_fusion(np.concatenate(features), np.concatenate(targets), c=c, seed=seed)
    return scaler, classifier, parameters.to_dict()


def evaluate_condition(payload: dict[str, Any], *, temperature: float, scaler: Any, classifier: Any, radius: int, coverages: tuple[float, ...], fractions: tuple[float, ...]) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    device = torch.device("cuda")
    parts: list[np.ndarray] = []
    error_parts: list[np.ndarray] = []
    boundary_parts: list[np.ndarray] = []
    image_rows: list[dict[str, object]] = []
    for start in range(0, len(payload["labels"]), 4):
        labels = payload["labels"][start:start + 4].long().to(device)
        scores, prediction, valid = _cached_feature_tensors(payload["logits"][start:start + 4].to(device), labels, temperature)
        boundary = valid & boundary_mask(labels, radius)
        errors = prediction.ne(labels)
        values = np.stack([scores[name].detach().cpu().numpy().reshape(-1) for name in FEATURE_NAMES], axis=1)
        valid_flat = valid.detach().cpu().numpy().reshape(-1)
        parts.append(fusion_score(values[valid_flat], scaler, classifier))
        error_parts.append(errors[valid].detach().cpu().numpy())
        boundary_parts.append(boundary[valid].detach().cpu().numpy())
        for offset, sample_id in enumerate(payload["sample_id"][start:start + len(labels)]):
            image_valid = valid[offset].detach().cpu().numpy()
            image_boundary = boundary[offset][valid[offset]].detach().cpu().numpy()
            regions = {"full": np.ones(image_valid.sum(), dtype=bool), "boundary": image_boundary, "interior": ~image_boundary}
            image_values = np.stack([scores[name][offset][valid[offset]].detach().cpu().numpy() for name in FEATURE_NAMES], axis=1)
            fusion = fusion_score(image_values, scaler, classifier)
            for region, metrics in ranking_metrics_by_region(fusion, errors[offset][valid[offset]].detach().cpu().numpy(), regions, coverages=coverages, top_fractions=fractions).items():
                image_rows.append({"sample_id": sample_id, "score": "logistic_fusion", "region": region, **metrics})
    errors_all, boundary_all = np.concatenate(error_parts), np.concatenate(boundary_parts)
    regions_all = {"full": np.ones_like(boundary_all, dtype=bool), "boundary": boundary_all, "interior": ~boundary_all}
    fusion_all = np.concatenate(parts)
    global_rows = [{"score": "logistic_fusion", "region": region, **metrics} for region, metrics in ranking_metrics_by_region(fusion_all, errors_all, regions_all, coverages=coverages, top_fractions=fractions).items()]
    return global_rows, image_rows


def main() -> None:
    parser = argparse.ArgumentParser(description="Fit calibration-only logistic ranking fusion; official TEST is locked.")
    parser.add_argument("--config", type=Path, default=ROOT / "configs" / "ranking_fusion.yaml")
    parser.add_argument("--limit-samples", type=int, default=None, help="Smoke-test only.")
    args = parser.parse_args()
    config_path = args.config.resolve()
    config = load_yaml(config_path)
    experiment, fit, metrics = config["experiment"], config["fit"], config["metrics"]
    if experiment["fit_split"] != "calibration" or experiment["evaluation_split"] != "val":
        raise ValueError("Fusion may fit only calibration and evaluate only val; official TEST is locked.")
    conditions = tuple(experiment["conditions"])
    if set(conditions) != set(CONDITIONS):
        raise ValueError("Fusion protocol requires exactly the 13 registered conditions.")
    output = ROOT / experiment["output_dir"]
    output.mkdir(parents=True, exist_ok=True)
    temperature = float(json.loads((ROOT / experiment["temperature_file"]).read_text(encoding="utf-8"))["clean_global"])
    checkpoint_sha256, degradation_sha256 = sha256(ROOT / experiment["checkpoint"]), sha256(ROOT / experiment["degradation_config"])
    cache_root = ROOT / experiment["cache_dir"]
    scaler, classifier, fit_metadata = fit_from_calibration(cache_root, conditions=conditions, checkpoint_sha256=checkpoint_sha256, degradation_sha256=degradation_sha256, temperature=temperature, samples_per_image=int(fit["samples_per_image"]), c=float(fit["regularization_c"]), seed=int(fit["seed"]), limit_samples=args.limit_samples)
    _atomic_json(fit_metadata, output / "fusion_parameters.json")
    rows: list[dict[str, object]] = []
    images: list[dict[str, object]] = []
    for condition in conditions:
        payload = torch.load(cache_root / "val" / f"{condition}.pt", map_location="cpu", weights_only=False)
        validate_cache_payload(payload, split="val", condition=condition, checkpoint_sha256=checkpoint_sha256, degradation_config_sha256=degradation_sha256)
        if args.limit_samples is not None:
            payload = {**payload, "sample_id": payload["sample_id"][:args.limit_samples], "logits": payload["logits"][:args.limit_samples], "labels": payload["labels"][:args.limit_samples]}
        global_rows, image_rows = evaluate_condition(payload, temperature=temperature, scaler=scaler, classifier=classifier, radius=int(metrics["boundary_radius"]), coverages=tuple(metrics["coverages"]), fractions=tuple(metrics["top_uncertainty_fractions"]))
        common = {"condition": condition, "split": "val", "degradation_type": payload["degradation_type"], "severity": payload["severity"], "boundary_radius": int(metrics["boundary_radius"])}
        rows.extend([{**common, **row} for row in global_rows])
        images.extend([{**common, **row} for row in image_rows])
        print(f"processed {condition}")
    table, image_table = pd.DataFrame(rows), pd.DataFrame(images)
    _atomic_csv(table, output / "metrics.csv")
    _atomic_csv(image_table, output / "per_image_metrics.csv")
    baseline = pd.read_csv(ROOT / "outputs" / "uncertainty_ranking" / "per_image_metrics.csv")
    bootstrap_rows: list[dict[str, object]] = []
    for region in ("full", "boundary", "interior"):
        left = image_table.loc[image_table.region.eq(region)].groupby("sample_id").mean(numeric_only=True)
        right = baseline.loc[(baseline.score == "calibrated_msp") & (baseline.region == region)].groupby("sample_id").mean(numeric_only=True)
        for name, direction in (("eaurc", -1.0), ("error_auprc", 1.0), ("top_10_uncertainty_recall", 1.0)):
            mean, low, high = _bootstrap(direction * (left[name] - right[name]).to_numpy(), iterations=int(config["bootstrap"]["iterations"]), seed=int(config["bootstrap"]["seed"]) + len(region) + len(name))
            bootstrap_rows.append({"region": region, "metric": name, "mean_improvement": mean, "ci95_low": low, "ci95_high": high, "cluster_unit": "sample_id_with_all_13_conditions"})
    _atomic_csv(pd.DataFrame(bootstrap_rows), output / "clustered_bootstrap.csv")
    _atomic_json({"official_test_evaluated": False, "model_retrained": False, "fit_split": "calibration", "evaluation_split": "val", "checkpoint_sha256": checkpoint_sha256, "degradation_config_sha256": degradation_sha256, "temperature": temperature, "conditions": list(conditions)}, output / "metadata.json")


if __name__ == "__main__":
    main()
