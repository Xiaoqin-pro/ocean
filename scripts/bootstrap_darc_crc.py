"""Clustered paired bootstrap for DARC-Seg controller comparisons."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.fit_darc_crc import _coverage_indices_global, _coverage_indices_oracle, _coverage_indices_quality, load_curves  # noqa: E402


def load_yaml(path: Path) -> dict[str, Any]:
    return yaml.safe_load(path.read_text(encoding="utf-8"))


def _interval(values: np.ndarray, *, iterations: int, seed: int) -> tuple[float, float, float]:
    generator = np.random.default_rng(seed)
    count = len(values)
    means = values[generator.integers(0, count, size=(iterations, count))].mean(axis=1)
    return float(values.mean()), float(np.quantile(means, 0.025)), float(np.quantile(means, 0.975))


def _cluster_statistics(actual: np.ndarray, coverages: np.ndarray, indices: np.ndarray, *, lowlight_index: int, blur_index: int, alpha: float) -> dict[str, np.ndarray]:
    """One metric value per independent sample_id, retaining all conditions."""
    condition_count, sample_count, _ = actual.shape
    selected_risk = np.empty((condition_count, sample_count), dtype=np.float64)
    selected_coverage = coverages[indices]
    for condition in range(condition_count):
        selected_risk[condition] = actual[condition, np.arange(sample_count), indices[condition]]
    risk_excess = selected_risk - alpha
    return {
        "coverage": selected_coverage.mean(axis=0),
        "risk_excess": risk_excess.mean(axis=0),
        "worst_condition_risk_excess": risk_excess.max(axis=0),
        "lowlight_s3_risk_excess": risk_excess[lowlight_index],
        "blur_s3_risk_excess": risk_excess[blur_index],
    }


def _comparison_rows(*, baseline: dict[str, np.ndarray], candidate: dict[str, np.ndarray], alpha: float, method: str, seed: int | None, iterations: int, bootstrap_seed: int) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    definitions = (
        ("coverage_improvement", candidate["coverage"] - baseline["coverage"]),
        ("risk_excess_reduction", baseline["risk_excess"] - candidate["risk_excess"]),
        ("worst_condition_risk_excess_reduction", baseline["worst_condition_risk_excess"] - candidate["worst_condition_risk_excess"]),
        ("lowlight_s3_risk_excess_reduction", baseline["lowlight_s3_risk_excess"] - candidate["lowlight_s3_risk_excess"]),
        ("blur_s3_risk_excess_reduction", baseline["blur_s3_risk_excess"] - candidate["blur_s3_risk_excess"]),
    )
    for offset, (metric, values) in enumerate(definitions):
        mean, low, high = _interval(values, iterations=iterations, seed=bootstrap_seed + offset * 101 + int(alpha * 1000) + (0 if seed is None else seed % 1000))
        rows.append({"comparison": f"{method} - global_crc", "method": method, "target_alpha": alpha, "seed": seed, "metric": metric, "mean_improvement": mean, "ci95_low": low, "ci95_high": high, "iterations": iterations, "cluster_unit": "sample_id_with_all_13_conditions", "clusters": len(values)})
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description="Bootstrap DARC CRC comparisons by original sample_id cluster.")
    parser.add_argument("--config", type=Path, default=ROOT / "configs" / "darc_crc_pilot.yaml")
    args = parser.parse_args()
    config = load_yaml(args.config.resolve())
    if config["experiment"]["evaluation_split"] != "val" or not config["experiment"]["official_test_locked"]:
        raise ValueError("Only validation curves may be bootstrapped; TEST is locked.")
    output = ROOT / config["experiment"]["output_dir"]
    curves = load_curves(output / "risk_curves" / "val.npz")
    parameters = json.loads((output / "parameters.json").read_text(encoding="utf-8"))
    primary = list(curves["score_names"].astype(str)).index(config["ranking"]["primary_score"])
    full = list(curves["region_names"].astype(str)).index("full")
    conditions = curves["conditions"].astype(str).tolist()
    sample_ids = curves["sample_ids"].astype(str).tolist()
    actual = curves["actual_risk"][primary, full]
    grid = curves["coverages"].astype(np.float64)
    lowlight_index, blur_index = conditions.index("lowlight_s3"), conditions.index("blur_s3")
    rows: list[dict[str, object]] = []
    for alpha in config["risk"]["targets"]:
        alpha_key, alpha = str(float(alpha)), float(alpha)
        global_index = int(parameters["global"][alpha_key]["crc"]["index"])
        global_stats = _cluster_statistics(actual, grid, _coverage_indices_global(type("S", (), {"index": global_index})(), len(conditions), len(sample_ids)), lowlight_index=lowlight_index, blur_index=blur_index, alpha=alpha)
        oracle_indices = np.asarray([[int(parameters["oracle"][alpha_key][condition]["index"])] * len(sample_ids) for condition in conditions], dtype=np.int64)
        oracle_stats = _cluster_statistics(actual, grid, oracle_indices, lowlight_index=lowlight_index, blur_index=blur_index, alpha=alpha)
        rows.extend(_comparison_rows(baseline=global_stats, candidate=oracle_stats, alpha=alpha, method="oracle_condition_crc", seed=None, iterations=int(config["bootstrap"]["iterations"]), bootstrap_seed=int(config["bootstrap"]["seed"])))
        for seed in config["quality"]["seeds"]:
            selected = {int(group): type("S", (), {"index": int(data["index"])})() for group, data in parameters["quality_group"][alpha_key][str(seed)]["groups"].items()}
            fallback = type("S", (), {"index": global_index})()
            assignments = pd.read_csv(output / "descriptors" / f"val_assignments_seed_{seed}.csv")
            indices = _coverage_indices_quality(assignments, conditions=conditions, sample_ids=sample_ids, selections=selected, fallback=fallback)
            quality_stats = _cluster_statistics(actual, grid, indices, lowlight_index=lowlight_index, blur_index=blur_index, alpha=alpha)
            rows.extend(_comparison_rows(baseline=global_stats, candidate=quality_stats, alpha=alpha, method="quality_group_crc_global_fallback", seed=int(seed), iterations=int(config["bootstrap"]["iterations"]), bootstrap_seed=int(config["bootstrap"]["seed"])))
    result = pd.DataFrame(rows)
    if len(result) != len(config["risk"]["targets"]) * (1 + len(config["quality"]["seeds"])) * 5:
        raise AssertionError("Unexpected clustered bootstrap result shape.")
    result.to_csv(output / "clustered_bootstrap.csv", index=False)
    print(f"wrote {len(result)} clustered bootstrap rows")


if __name__ == "__main__":
    main()
