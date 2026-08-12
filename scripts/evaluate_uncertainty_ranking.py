"""Benchmark frozen-logit uncertainty scores without model retraining or TEST access."""
from __future__ import annotations

import argparse
import json
import os
import sys
import tempfile
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as functional
import psutil

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from metrics.uncertainty_ranking import SCORE_NAMES, ranking_metrics_by_region, uncertainty_scores  # noqa: E402
from scripts.analyze_boundary_residual import boundary_mask  # noqa: E402
from scripts.evaluate_temperature_scaling import CONDITIONS, load_yaml, sha256, validate_cache_payload  # noqa: E402


def validate_benchmark_protocol(config: dict[str, Any]) -> None:
    experiment = config["experiment"]
    if list(experiment["splits"]) not in (["val"], ["confirmation"]):
        raise ValueError("Uncertainty-ranking benchmark permits val/confirmation caches only; official TEST is locked.")
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
            if not candidate.index.equals(baseline.index) or len(candidate) != len(baseline):
                raise ValueError("Every image cluster must contain every registered condition and score.")
            for metric, direction in metrics:
                # Positive means the candidate is better, including for eAURC where lower is better.
                values = direction * (candidate[metric] - baseline[metric]).to_numpy()
                mean, low, high = _bootstrap(values, iterations=iterations, seed=seed + len(score) * 37 + len(metric) * 11 + len(region))
                rows.append({"region": region, "score": score, "baseline": "calibrated_msp", "metric": metric, "direction": "positive_is_better", "mean_improvement": mean, "ci95_low": low, "ci95_high": high, "iterations": iterations, "cluster_unit": "sample_id_with_all_13_conditions", "images": len(candidate)})
    return pd.DataFrame(rows)


def _condition_summary(table: pd.DataFrame) -> pd.DataFrame:
    return table.groupby(["score", "region", "degradation_type", "severity"], dropna=False).mean(numeric_only=True).reset_index()


def _atomic_csv(table: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile("w", suffix=".csv", dir=path.parent, delete=False, encoding="utf-8", newline="") as handle:
        temporary = Path(handle.name)
        table.to_csv(handle, index=False)
    os.replace(temporary, path)


def _atomic_json(value: dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile("w", suffix=".json", dir=path.parent, delete=False, encoding="utf-8") as handle:
        temporary = Path(handle.name)
        json.dump(value, handle, ensure_ascii=False, indent=2)
    os.replace(temporary, path)


def _peak_process_rss_mb() -> float:
    return float(psutil.Process().memory_info().peak_wset / (1024 * 1024))


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
            amp_logits = functional.interpolate(payload["logits"][start:start + 4].to(device, non_blocking=True), size=labels.shape[-2:], mode="bilinear", align_corners=False)
            frozen_prediction = amp_logits.argmax(dim=1)
        score_logits = amp_logits.float()
        prediction = (score_logits / temperature).argmax(dim=1)
        if not torch.equal(frozen_prediction, prediction):
            raise AssertionError("Float32 scoring logits changed a frozen AMP prediction.")
        scores = uncertainty_scores(score_logits, temperature=temperature)
        valid = labels.ge(0) & labels.lt(score_logits.shape[1])
        boundary = valid & boundary_mask(labels, radius)
        regions = {"full": valid, "boundary": boundary, "interior": valid & ~boundary}
        errors = prediction.ne(labels)
        errors_parts.append(errors[valid].detach().cpu().numpy())
        boundary_parts.append(boundary[valid].detach().cpu().numpy())
        for name, value in scores.items():
            score_parts[name].append(value[valid].detach().float().cpu().numpy())
        for offset, sample_id in enumerate(payload["sample_id"][start:start + len(labels)]):
            image_valid = valid[offset].detach().cpu().numpy()
            image_boundary = boundary[offset][valid[offset]].detach().cpu().numpy()
            image_regions = {"full": np.ones(image_valid.sum(), dtype=bool), "boundary": image_boundary, "interior": ~image_boundary}
            image_errors = errors[offset][valid[offset]].detach().cpu().numpy()
            for score_name, value in scores.items():
                image_metrics = ranking_metrics_by_region(value[offset][valid[offset]].detach().cpu().numpy(), image_errors, image_regions, discrete_histogram=score_name == "local_disagreement", coverages=coverages, top_fractions=fractions)
                for region_name, metrics in image_metrics.items():
                    per_image.append({"sample_id": sample_id, "score": score_name, "region": region_name, **metrics})
    errors_all = np.concatenate(errors_parts)
    boundary_all = np.concatenate(boundary_parts)
    regions_all = {"full": np.ones_like(boundary_all, dtype=bool), "boundary": boundary_all, "interior": ~boundary_all}
    global_rows: list[dict[str, object]] = []
    for score_name, parts in score_parts.items():
        values = np.concatenate(parts)
        metrics_by_region = ranking_metrics_by_region(values, errors_all, regions_all, discrete_histogram=score_name == "local_disagreement", coverages=coverages, top_fractions=fractions)
        for region_name, metrics in metrics_by_region.items():
            global_rows.append({"score": score_name, "region": region_name, **metrics})
    return global_rows, per_image


def main() -> None:
    parser = argparse.ArgumentParser(description="Benchmark frozen-logit uncertainty rankings on validation caches only.")
    parser.add_argument("--config", type=Path, default=ROOT / "configs" / "uncertainty_ranking.yaml")
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--limit-samples", type=int, default=None, help="Smoke-test only; omit for formal evaluation.")
    parser.add_argument("--conditions", nargs="+", default=None, help="Optional registered subset for a smoke run.")
    parser.add_argument("--resume", action="store_true", help="Reuse only complete per-condition files matching the registered row counts.")
    args = parser.parse_args()
    config_path = args.config.resolve()
    config = load_yaml(config_path)
    validate_benchmark_protocol(config)
    experiment, metric_config = config["experiment"], config["metrics"]
    output = (ROOT / experiment["output_dir"] if args.output_dir is None else args.output_dir.resolve())
    output.mkdir(parents=True, exist_ok=True)
    conditions_output = output / "conditions"
    temperatures = json.loads((ROOT / experiment["temperature_file"]).read_text(encoding="utf-8"))
    temperature = float(temperatures["clean_global"])
    checkpoint_sha256 = sha256(ROOT / experiment["checkpoint"])
    degradation_sha256 = sha256(ROOT / experiment["degradation_config"])
    evaluation_split = str(experiment.get("evaluation_split", experiment["splits"][0]))
    cache_root = ROOT / experiment["cache_dir"] / evaluation_split
    global_rows: list[dict[str, object]] = []
    image_rows: list[dict[str, object]] = []
    conditions = tuple(experiment["conditions"] if args.conditions is None else args.conditions)
    if not conditions or set(conditions) - set(experiment["conditions"]):
        raise ValueError("Requested conditions must be a non-empty registered subset.")
    for condition in conditions:
        condition_path = conditions_output / f"{condition}.csv"
        image_path = conditions_output / f"{condition}_per_image.csv"
        expected_condition_rows = len(SCORE_NAMES) * 3
        expected_image_rows = (args.limit_samples or int(experiment.get("expected_samples", 146))) * len(SCORE_NAMES) * 3
        if args.resume and condition_path.is_file() and image_path.is_file():
            stored_global, stored_image = pd.read_csv(condition_path), pd.read_csv(image_path)
            if len(stored_global) == expected_condition_rows and len(stored_image) == expected_image_rows and not stored_global.duplicated(["condition", "score", "region"]).any():
                global_rows.extend(stored_global.to_dict("records"))
                image_rows.extend(stored_image.to_dict("records"))
                print(f"resumed {condition}")
                continue
        payload = torch.load(cache_root / f"{condition}.pt", map_location="cpu", weights_only=False)
        validate_cache_payload(payload, split=evaluation_split, condition=condition, checkpoint_sha256=checkpoint_sha256, degradation_config_sha256=degradation_sha256)
        if args.limit_samples is not None:
            payload = {**payload, "sample_id": payload["sample_id"][:args.limit_samples], "logits": payload["logits"][:args.limit_samples], "labels": payload["labels"][:args.limit_samples]}
        global_values, image_values = evaluate_condition(payload, temperature=temperature, radius=int(metric_config["boundary_radii"][0]), coverages=tuple(metric_config["coverages"]), fractions=tuple(metric_config["top_uncertainty_fractions"]))
        common = {"split": evaluation_split, "condition": condition, "degradation_type": payload["degradation_type"], "severity": payload["severity"], "boundary_radius": int(metric_config["boundary_radii"][0])}
        condition_global = pd.DataFrame([{**common, **row} for row in global_values])
        condition_image = pd.DataFrame([{**common, **row} for row in image_values])
        if len(condition_global) != expected_condition_rows or len(condition_image) != expected_image_rows:
            raise AssertionError(f"Incomplete condition result for {condition}.")
        condition_global["peak_process_rss_mb"] = _peak_process_rss_mb()
        _atomic_csv(condition_global, condition_path)
        _atomic_csv(condition_image, image_path)
        global_rows.extend(condition_global.to_dict("records"))
        image_rows.extend(condition_image.to_dict("records"))
        manifest = {"completed_conditions": list(dict.fromkeys(row["condition"] for row in global_rows)), "official_test_evaluated": False, "model_retrained": False}
        _atomic_json(manifest, output / "manifest.json")
        score_timing = condition_global.groupby("score")[["sort_seconds", "metric_seconds"]].max().to_dict("index")
        print(f"processed {condition}; peak_process_rss_mb={_peak_process_rss_mb():.1f}; score_timing={json.dumps(score_timing)}")
        del payload, condition_global, condition_image
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    global_table, image_table = pd.DataFrame(global_rows), pd.DataFrame(image_rows)
    expected_rows = len(conditions) * len(SCORE_NAMES) * 3
    if len(global_table) != expected_rows or global_table.duplicated(["condition", "score", "region"]).any():
        raise AssertionError("Unexpected uncertainty-ranking result shape.")
    _atomic_csv(global_table, output / "metrics.csv")
    _atomic_csv(image_table, output / "per_image_metrics.csv")
    _atomic_csv(_condition_summary(global_table), output / "condition_summary.csv")
    if args.limit_samples is None:
        _atomic_csv(clustered_comparisons(image_table, iterations=int(config["bootstrap"]["iterations"]), seed=int(config["bootstrap"]["seed"])), output / "clustered_bootstrap.csv")
    metadata = {
        "config": str(config_path.relative_to(ROOT)), "config_sha256": sha256(config_path),
        "checkpoint_sha256": checkpoint_sha256, "degradation_config_sha256": degradation_sha256,
        "temperature": temperature, "conditions": list(conditions), "scores": list(SCORE_NAMES),
        "split_evaluated": evaluation_split, "official_test_evaluated": False, "model_retrained": False,
        "boundary_is_evaluation_stratum_only": True, "formal_run": args.limit_samples is None,
        "result_rows": len(global_table), "per_image_rows": len(image_table),
    }
    _atomic_json(metadata, output / "metadata.json")


if __name__ == "__main__":
    main()
