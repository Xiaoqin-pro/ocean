"""Evaluate the fixed UIIS CRC protocol at three prespecified risk targets."""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from reliability.conformal_risk import CRCSelection, select_crc_coverage  # noqa: E402
from scripts.fit_uiis_alpha010_crc import quality_group_losses  # noqa: E402


def load_yaml(path: Path) -> dict[str, Any]:
    return yaml.safe_load(path.read_text(encoding="utf-8"))


def load_curves(path: Path) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as values:
        return {key: values[key] for key in values.files}


def atomic_csv(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    frame.to_csv(temporary, index=False)
    os.replace(temporary, path)


def atomic_json(value: dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def validate_protocol(config: dict[str, Any]) -> None:
    experiment, risk, quality, protocol = config["experiment"], config["risk"], config["quality"], config["protocol"]
    if not experiment["official_suim_test_locked"] or protocol["official_suim_test_evaluated"]:
        raise ValueError("SUIM official TEST must remain locked.")
    if not protocol["external_benchmark_extension"] or not protocol["confirmation_previously_opened_for_darc_negative_control"]:
        raise ValueError("This evaluation must be labelled as a fixed external benchmark extension.")
    if protocol["confirmation_used_for_fitting"] or protocol["model_retrained"]:
        raise ValueError("Confirmation fitting and model retraining are forbidden.")
    if [float(value) for value in risk["targets"]] != [0.05, 0.10, 0.15]:
        raise ValueError("Only the fixed three-target sensitivity analysis is allowed.")
    if int(quality["groups"]) != 3 or list(quality["seeds"]) != [20260722, 20260723, 20260724]:
        raise ValueError("Quality grouping is frozen to the preregistered three seeds and three groups.")


def validate_curve_pair(calibration: dict[str, np.ndarray], confirmation: dict[str, np.ndarray], config: dict[str, Any]) -> tuple[list[str], list[str], np.ndarray]:
    for key in ("coverages", "conditions"):
        if key not in calibration or key not in confirmation or not np.array_equal(calibration[key], confirmation[key]):
            raise ValueError(f"Calibration and confirmation curves disagree on {key}.")
    conditions = calibration["conditions"].astype(str).tolist()
    if conditions != list(config["experiment"]["conditions"]):
        raise ValueError("Risk curves must use exactly the frozen 13-condition registry.")
    if calibration["actual_risk"].shape[:2] != (13, 508) or confirmation["actual_risk"].shape[:2] != (13, 511):
        raise ValueError("UIIS sensitivity requires 508 calibration and 511 confirmation image clusters.")
    if calibration["sample_ids"].shape[0] != 508 or confirmation["sample_ids"].shape[0] != 511:
        raise ValueError("Curve sample IDs do not match the frozen UIIS split.")
    return conditions, confirmation["sample_ids"].astype(str).tolist(), calibration["coverages"].astype(np.float64)


def global_indices(selection: CRCSelection, conditions: int, samples: int) -> np.ndarray:
    return np.full((conditions, samples), selection.index, dtype=np.int64)


def oracle_indices(selections: list[CRCSelection], samples: int) -> np.ndarray:
    return np.asarray([[selection.index] * samples for selection in selections], dtype=np.int64)


def quality_indices(assignments: pd.DataFrame, conditions: list[str], sample_ids: list[str], selected: dict[int, CRCSelection], fallback: CRCSelection) -> np.ndarray:
    lookup = assignments.set_index(["sample_id", "condition"])["quality_group"]
    indices = np.empty((len(conditions), len(sample_ids)), dtype=np.int64)
    for condition_index, condition in enumerate(conditions):
        for sample_index, sample_id in enumerate(sample_ids):
            group = int(lookup.loc[(sample_id, condition)])
            indices[condition_index, sample_index] = selected.get(group, fallback).index
    return indices


def cluster_statistics(actual: np.ndarray, coverages: np.ndarray, indices: np.ndarray, alpha: float, lowlight_index: int, blur_index: int) -> dict[str, np.ndarray]:
    selected_risk = actual[np.arange(actual.shape[0])[:, None], np.arange(actual.shape[1])[None, :], indices]
    risk_excess = selected_risk - alpha
    return {
        "coverage": coverages[indices].mean(axis=0),
        "risk_excess": risk_excess.mean(axis=0),
        "worst_condition_risk_excess": risk_excess.max(axis=0),
        "lowlight_s3_risk_excess": risk_excess[lowlight_index],
        "blur_s3_risk_excess": risk_excess[blur_index],
    }


def bootstrap_interval(values: np.ndarray, iterations: int, seed: int) -> tuple[float, float, float]:
    generator = np.random.default_rng(seed)
    means = values[generator.integers(0, len(values), size=(iterations, len(values)))].mean(axis=1)
    return float(values.mean()), float(np.quantile(means, 0.025)), float(np.quantile(means, 0.975))


def comparison_rows(baseline: dict[str, np.ndarray], candidate: dict[str, np.ndarray], alpha: float, method: str, seed: int | None, iterations: int, bootstrap_seed: int) -> list[dict[str, object]]:
    definitions = (
        ("coverage_improvement", candidate["coverage"] - baseline["coverage"]),
        ("risk_excess_reduction", baseline["risk_excess"] - candidate["risk_excess"]),
        ("worst_condition_risk_excess_reduction", baseline["worst_condition_risk_excess"] - candidate["worst_condition_risk_excess"]),
        ("lowlight_s3_risk_excess_reduction", baseline["lowlight_s3_risk_excess"] - candidate["lowlight_s3_risk_excess"]),
        ("blur_s3_risk_excess_reduction", baseline["blur_s3_risk_excess"] - candidate["blur_s3_risk_excess"]),
    )
    return [
        {
            "comparison": f"{method} - global_crc", "method": method, "target_alpha": alpha, "seed": seed,
            "metric": metric, "mean_improvement": bootstrap_interval(values, iterations, bootstrap_seed + offset * 101 + int(alpha * 1000) + (0 if seed is None else seed % 1000))[0],
            "ci95_low": bootstrap_interval(values, iterations, bootstrap_seed + offset * 101 + int(alpha * 1000) + (0 if seed is None else seed % 1000))[1],
            "ci95_high": bootstrap_interval(values, iterations, bootstrap_seed + offset * 101 + int(alpha * 1000) + (0 if seed is None else seed % 1000))[2],
            "iterations": iterations, "cluster_unit": "original_sample_id_with_all_13_conditions", "clusters": len(values),
        }
        for offset, (metric, values) in enumerate(definitions)
    ]


def condition_rows(actual: np.ndarray, coverages: np.ndarray, indices: np.ndarray, alpha: float, conditions: list[str], method: str, seed: int | None, fit_scope: str) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for condition_index, condition in enumerate(conditions):
        risks = actual[condition_index, np.arange(actual.shape[1]), indices[condition_index]]
        coverage = coverages[indices[condition_index]]
        rows.append({"method": method, "fit_scope": fit_scope, "seed": seed, "target_alpha": alpha, "condition": condition, "coverage": float(coverage.mean()), "selective_risk": float(risks.mean()), "risk_excess": float(risks.mean() - alpha), "worst_condition_risk": float(risks.mean())})
    return rows


def fit_quality_selections(calibration_envelope: np.ndarray, assignments: pd.DataFrame, conditions: list[str], sample_ids: list[str], alpha: float, grid: np.ndarray, minimum: int, bound: float) -> dict[int, CRCSelection]:
    selected: dict[int, CRCSelection] = {}
    for group in sorted(assignments["quality_group"].unique()):
        losses = quality_group_losses(calibration_envelope, assignments, conditions, sample_ids, int(group))
        if len(losses) >= minimum:
            selected[int(group)] = select_crc_coverage(losses, grid, alpha, bound=bound)
    return selected


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=ROOT / "configs" / "uiis_crc_sensitivity.yaml")
    args = parser.parse_args()
    config = load_yaml(args.config.resolve())
    validate_protocol(config)
    experiment, risk, quality, bootstrap = config["experiment"], config["risk"], config["quality"], config["bootstrap"]
    calibration = load_curves(ROOT / experiment["calibration_curves"])
    confirmation = load_curves(ROOT / experiment["confirmation_curves"])
    conditions, confirmation_ids, grid = validate_curve_pair(calibration, confirmation, config)
    calibration_ids = calibration["sample_ids"].astype(str).tolist()
    calibration_envelope, confirmation_actual = calibration["monotone_envelope"].astype(np.float64), confirmation["actual_risk"].astype(np.float64)
    lowlight_index, blur_index = conditions.index("lowlight_s3"), conditions.index("blur_s3")
    metrics: list[dict[str, object]] = []
    comparisons: list[dict[str, object]] = []
    parameters: dict[str, Any] = {"fit_split": "calibration", "evaluation_split": "confirmation", "targets": {}}
    descriptor_dir = ROOT / experiment["descriptors_dir"]
    for alpha in [float(value) for value in risk["targets"]]:
        global_selection = select_crc_coverage(calibration_envelope.mean(axis=0), grid, alpha, bound=float(risk["crc_bound"]))
        oracle_selection = [select_crc_coverage(calibration_envelope[index], grid, alpha, bound=float(risk["crc_bound"])) for index in range(len(conditions))]
        global_index = global_indices(global_selection, len(conditions), len(confirmation_ids))
        oracle_index = oracle_indices(oracle_selection, len(confirmation_ids))
        global_stats = cluster_statistics(confirmation_actual, grid, global_index, alpha, lowlight_index, blur_index)
        oracle_stats = cluster_statistics(confirmation_actual, grid, oracle_index, alpha, lowlight_index, blur_index)
        metrics.extend(condition_rows(confirmation_actual, grid, global_index, alpha, conditions, "global_crc", None, "calibration_cluster_mean"))
        metrics.extend(condition_rows(confirmation_actual, grid, oracle_index, alpha, conditions, "oracle_condition_crc", None, "condition_label_diagnostic_only"))
        comparisons.extend(comparison_rows(global_stats, oracle_stats, alpha, "oracle_condition_crc", None, int(bootstrap["iterations"]), int(bootstrap["seed"])))
        parameters["targets"][str(alpha)] = {"global": global_selection.__dict__, "oracle": {condition: selection.__dict__ for condition, selection in zip(conditions, oracle_selection, strict=True)}, "quality": {}}
        for seed in quality["seeds"]:
            calibration_assignments = pd.read_csv(descriptor_dir / f"calibration_assignments_seed_{seed}.csv")
            confirmation_assignments = pd.read_csv(descriptor_dir / f"confirmation_assignments_seed_{seed}.csv")
            selected = fit_quality_selections(calibration_envelope, calibration_assignments, conditions, calibration_ids, alpha, grid, int(quality["minimum_calibration_clusters"]), float(risk["crc_bound"]))
            quality_index = quality_indices(confirmation_assignments, conditions, confirmation_ids, selected, global_selection)
            quality_stats = cluster_statistics(confirmation_actual, grid, quality_index, alpha, lowlight_index, blur_index)
            metrics.extend(condition_rows(confirmation_actual, grid, quality_index, alpha, conditions, "quality_group_crc_global_fallback", int(seed), "frozen_train_quality_groups_calibration_crc"))
            comparisons.extend(comparison_rows(global_stats, quality_stats, alpha, "quality_group_crc_global_fallback", int(seed), int(bootstrap["iterations"]), int(bootstrap["seed"])))
            parameters["targets"][str(alpha)]["quality"][str(seed)] = {"groups": {str(group): value.__dict__ for group, value in selected.items()}, "fallback": global_selection.__dict__}
    metric_table, comparison_table = pd.DataFrame(metrics), pd.DataFrame(comparisons)
    if len(metric_table) != 195 or len(comparison_table) != 60 or metric_table.duplicated(["method", "target_alpha", "condition", "seed"]).any():
        raise AssertionError("Unexpected UIIS CRC sensitivity result shape.")
    output = ROOT / experiment["output_dir"]
    atomic_csv(metric_table, output / "condition_metrics.csv")
    atomic_csv(comparison_table, output / "clustered_bootstrap.csv")
    atomic_json(parameters, output / "calibration_only_parameters.json")
    atomic_json({"external_benchmark_extension": True, "split_evaluated": "confirmation", "fit_split": "calibration", "targets": risk["targets"], "conditions": conditions, "cluster_unit": "original_sample_id_with_all_13_conditions", "official_suim_test_evaluated": False, "model_retrained": False, "metrics_rows": len(metric_table), "bootstrap_rows": len(comparison_table)}, output / "metadata.json")


if __name__ == "__main__":
    main()
