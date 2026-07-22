"""Benchmark frozen-logit uncertainty scores without model retraining or TEST access."""
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

from metrics.uncertainty_ranking import SCORE_NAMES, ranking_metrics_by_region, uncertainty_scores  # noqa: E402
from scripts.analyze_boundary_residual import boundary_mask  # noqa: E402
from scripts.evaluate_temperature_scaling import CONDITIONS, load_yaml, sha256, validate_cache_payload  # noqa: E402


def validate_benchmark_protocol(config: dict[str, Any]) -> None:
    experiment = config["experiment"]
    if list(experiment["splits"]) != ["val"]:
        raise ValueError("Uncertainty-ranking benchmark permits validation caches only; official TEST is locked.")
    unknown = set(experiment["conditions"]) - set(CONDITIONS)
    if unknown:
        raise ValueError(f"Unknown registered degradation conditions: {sorted(unknown)}")


def _bootstrap(values: np.ndarray, *, iterations: int, seed: int) -> tuple[float, float, float]:
    generator = np.random.default_rng(seed)
    count = len(values)
    means = values[generator.integers(0, count, size=(iterations, count))].mean(axis=1)
    return float(values.mean()), float(np.quantile(means, 0.025)), float(np.quantile(means, 0.975))


def clustered_comparisons(per_image: pd.DataFrame, *, iterations: int, seed: int) -> pd.DataFrame:
    """Compare each score with calibrated MSP while retaining all 13 conditions per image."""
    rows: list[dict[str, object]] = []
    metrics = (("eaurc", -1.0), ("error_auprc", 1.0), ("top_10_uncertainty_recall", 1.0))
    for region in sorted(per_image["region"].unique()):
        baseline = per_image.loc[(per_image["score"] == "calibrated_msp") & (per_image["region"] == region)].groupby("sample_id").mean(numeric_only=True)
        for score in SCORE_NAMES:
            candidate = per_image.loc[(per_image["score"] == score) & (per_image["region"] == region)].groupby("sample_id").mean(numeric_only=True)
            if not candidate.index.equals(baseline.index) or len(candidate) != 146:
                raise ValueError("Every image cluster must contain every registered condition and score.")
            for metric, direction in metrics:
                # Positive means the candidate is better, including for eAURC where lower is better.
                values = direction * (candidate[metric] - baseline[metric]).to_numpy()
                mean, low, high = _bootstrap(values, iterations=iterations, seed=seed + len(score) * 37 + len(metric) * 11 + len(region))
                rows.append({"region": region, "score": score, "baseline": "calibrated_msp", "metric": metric, "direction": "positive_is_better", "mean_improvement": mean, "ci95_low": low, "ci95_high": high, "iterations": iterations, "cluster_unit": "sample_id_with_all_13_conditions", "images": len(candidate)})
    return pd.DataFrame(rows)


def _condition_summary(table: pd.DataFrame) -> pd.DataFrame:
    return table.groupby(["score", "region", "degradation_type", "severity"], dropna=False).mean(numeric_only=True).reset_index()


def per_image_bootstrap_metrics(scores: torch.Tensor, errors: torch.Tensor, regions: dict[str, torch.Tensor], *, top_fraction: float = 0.1) -> dict[str, dict[str, torch.Tensor]]:
    """GPU-vectorized per-image summaries used only for clustered bootstrap.

    Global condition metrics remain tie-aware and exact.  This helper resolves
    equal per-image scores by deterministic raster order so all candidate
    scores use the same ranking rule in the image-clustered comparison.
    """
    batch = scores.shape[0]
    flat_scores, flat_errors = scores.float().reshape(batch, -1), errors.reshape(batch, -1).to(torch.float32)
    ranks = torch.arange(1, flat_scores.shape[1] + 1, device=scores.device, dtype=torch.float32).unsqueeze(0)
    result: dict[str, dict[str, torch.Tensor]] = {}
    for region_name, region in regions.items():
        flat_region = region.reshape(batch, -1)
        count = flat_region.sum(dim=1)
        if bool((count == 0).any()):
            raise ValueError(f"An image has no valid {region_name} pixels.")
        ascending_order = torch.argsort(flat_scores.masked_fill(~flat_region, float("inf")), dim=1, stable=True)
        ascending_errors = flat_errors.gather(1, ascending_order)
        cumulative = torch.cumsum(ascending_errors, dim=1)
        risk = cumulative / ranks
        pair_valid = ranks[:, 1:] <= count.unsqueeze(1)
        aurc = (((risk[:, :-1] + risk[:, 1:]) * 0.5) * pair_valid).sum(dim=1) / count
        error_count = flat_errors.masked_fill(~flat_region, 0.0).sum(dim=1)
        oracle_errors = (ranks > (count - error_count).unsqueeze(1)).to(torch.float32)
        oracle_risk = torch.cumsum(oracle_errors, dim=1) / ranks
        oracle_aurc = (((oracle_risk[:, :-1] + oracle_risk[:, 1:]) * 0.5) * pair_valid).sum(dim=1) / count
        descending_order = torch.argsort((-flat_scores).masked_fill(~flat_region, float("inf")), dim=1, stable=True)
        descending_errors = flat_errors.gather(1, descending_order)
        descending_cumulative = torch.cumsum(descending_errors, dim=1)
        average_precision = ((descending_cumulative / ranks) * descending_errors * (ranks <= count.unsqueeze(1))).sum(dim=1) / error_count.clamp_min(1.0)
        top_count = torch.ceil(count * top_fraction).to(torch.long).clamp_min(1)
        top_index = torch.arange(flat_scores.shape[1], device=scores.device).unsqueeze(0) < top_count.unsqueeze(1)
        top_errors = (descending_errors * top_index).sum(dim=1)
        result[region_name] = {
            "eaurc": aurc - oracle_aurc,
            "error_auprc": average_precision,
            "top_10_uncertainty_recall": top_errors / error_count.clamp_min(1.0),
        }
    return result


def evaluate_condition(
    payload: dict[str, Any],
    *,
    temperature: float,
    radius: int,
    coverages: tuple[float, ...],
    fractions: tuple[float, ...],
) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    """Evaluate one cache; global storage is bounded to one condition's score arrays."""
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required to reproduce the frozen AMP cache protocol.")
    device = torch.device("cuda")
    score_parts: dict[str, list[np.ndarray]] = {name: [] for name in SCORE_NAMES}
    errors_parts: list[np.ndarray] = []
    boundary_parts: list[np.ndarray] = []
    per_image: list[dict[str, object]] = []
    for start in range(0, len(payload["labels"]), 4):
        labels = payload["labels"][start:start + 4].long().to(device, non_blocking=True)
        with torch.amp.autocast("cuda", enabled=True):
            logits = functional.interpolate(payload["logits"][start:start + 4].to(device, non_blocking=True), size=labels.shape[-2:], mode="bilinear", align_corners=False)
            raw_prediction = logits.argmax(dim=1)
            prediction = (logits / temperature).argmax(dim=1)
            if not torch.equal(raw_prediction, prediction):
                raise AssertionError("Positive temperature changed a segmentation prediction.")
            scores = uncertainty_scores(logits, temperature=temperature)
        valid = labels.ge(0) & labels.lt(logits.shape[1])
        boundary = valid & boundary_mask(labels, radius)
        regions = {"full": valid, "boundary": boundary, "interior": valid & ~boundary}
        errors = prediction.ne(labels)
        errors_parts.append(errors[valid].detach().cpu().numpy())
        boundary_parts.append(boundary[valid].detach().cpu().numpy())
        for name, value in scores.items():
            score_parts[name].append(value[valid].detach().float().cpu().numpy())
        for score_name, value in scores.items():
            bootstrap_values = per_image_bootstrap_metrics(value, errors, regions)
            for offset, sample_id in enumerate(payload["sample_id"][start:start + len(labels)]):
                for region_name, metrics in bootstrap_values.items():
                    per_image.append({"sample_id": sample_id, "score": score_name, "region": region_name, **{name: float(metric[offset].item()) for name, metric in metrics.items()}})
    errors_all = np.concatenate(errors_parts)
    boundary_all = np.concatenate(boundary_parts)
    regions_all = {"full": np.ones_like(boundary_all, dtype=bool), "boundary": boundary_all, "interior": ~boundary_all}
    global_rows: list[dict[str, object]] = []
    for score_name, parts in score_parts.items():
        values = np.concatenate(parts)
        metrics_by_region = ranking_metrics_by_region(values, errors_all, regions_all, coverages=coverages, top_fractions=fractions)
        for region_name, metrics in metrics_by_region.items():
            global_rows.append({"score": score_name, "region": region_name, **metrics})
    return global_rows, per_image


def main() -> None:
    parser = argparse.ArgumentParser(description="Benchmark frozen-logit uncertainty rankings on validation caches only.")
    parser.add_argument("--config", type=Path, default=ROOT / "configs" / "uncertainty_ranking.yaml")
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--limit-samples", type=int, default=None, help="Smoke-test only; omit for formal evaluation.")
    parser.add_argument("--conditions", nargs="+", default=None, help="Optional registered subset for a smoke run.")
    args = parser.parse_args()
    config_path = args.config.resolve()
    config = load_yaml(config_path)
    validate_benchmark_protocol(config)
    experiment, metric_config = config["experiment"], config["metrics"]
    output = (ROOT / experiment["output_dir"] if args.output_dir is None else args.output_dir.resolve())
    output.mkdir(parents=True, exist_ok=True)
    temperatures = json.loads((ROOT / experiment["temperature_file"]).read_text(encoding="utf-8"))
    temperature = float(temperatures["clean_global"])
    checkpoint_sha256 = sha256(ROOT / experiment["checkpoint"])
    degradation_sha256 = sha256(ROOT / experiment["degradation_config"])
    cache_root = ROOT / experiment["cache_dir"] / "val"
    global_rows: list[dict[str, object]] = []
    image_rows: list[dict[str, object]] = []
    conditions = tuple(experiment["conditions"] if args.conditions is None else args.conditions)
    if not conditions or set(conditions) - set(experiment["conditions"]):
        raise ValueError("Requested conditions must be a non-empty registered subset.")
    for condition in conditions:
        payload = torch.load(cache_root / f"{condition}.pt", map_location="cpu", weights_only=False)
        validate_cache_payload(payload, split="val", condition=condition, checkpoint_sha256=checkpoint_sha256, degradation_config_sha256=degradation_sha256)
        if args.limit_samples is not None:
            payload = {**payload, "sample_id": payload["sample_id"][:args.limit_samples], "logits": payload["logits"][:args.limit_samples], "labels": payload["labels"][:args.limit_samples]}
        global_values, image_values = evaluate_condition(payload, temperature=temperature, radius=int(metric_config["boundary_radii"][0]), coverages=tuple(metric_config["coverages"]), fractions=tuple(metric_config["top_uncertainty_fractions"]))
        common = {"split": "val", "condition": condition, "degradation_type": payload["degradation_type"], "severity": payload["severity"], "boundary_radius": int(metric_config["boundary_radii"][0])}
        global_rows.extend([{**common, **row} for row in global_values])
        image_rows.extend([{**common, **row} for row in image_values])
        print(f"processed {condition}")
    global_table, image_table = pd.DataFrame(global_rows), pd.DataFrame(image_rows)
    expected_rows = len(conditions) * len(SCORE_NAMES) * 3
    if len(global_table) != expected_rows or global_table.duplicated(["condition", "score", "region"]).any():
        raise AssertionError("Unexpected uncertainty-ranking result shape.")
    global_table.to_csv(output / "metrics.csv", index=False)
    image_table.to_csv(output / "per_image_metrics.csv", index=False)
    _condition_summary(global_table).to_csv(output / "condition_summary.csv", index=False)
    if args.limit_samples is None:
        clustered_comparisons(image_table, iterations=int(config["bootstrap"]["iterations"]), seed=int(config["bootstrap"]["seed"])).to_csv(output / "clustered_bootstrap.csv", index=False)
    metadata = {
        "config": str(config_path.relative_to(ROOT)), "config_sha256": sha256(config_path),
        "checkpoint_sha256": checkpoint_sha256, "degradation_config_sha256": degradation_sha256,
        "temperature": temperature, "conditions": list(conditions), "scores": list(SCORE_NAMES),
        "split_evaluated": "val", "official_test_evaluated": False, "model_retrained": False,
        "boundary_is_evaluation_stratum_only": True, "formal_run": args.limit_samples is None,
        "result_rows": len(global_table), "per_image_rows": len(image_table),
    }
    (output / "metadata.json").write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
