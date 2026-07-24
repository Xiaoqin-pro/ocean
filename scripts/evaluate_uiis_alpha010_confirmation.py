"""Evaluate the single preregistered UIIS confirmation using frozen CRC parameters."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as functional
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from degradations.registry import load_conditions  # noqa: E402
from metrics.uncertainty_ranking import uncertainty_scores  # noqa: E402
from reliability.quality_grouping import FrozenQualityGrouping  # noqa: E402
from reliability.selective_risk import coverage_grid, curve_summary  # noqa: E402
from scripts.fit_uiis_alpha010_crc import DESCRIPTOR_NAMES, descriptor_table  # noqa: E402


def load_yaml(path: Path) -> dict[str, Any]:
    return yaml.safe_load(path.read_text(encoding="utf-8"))


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest().upper()


def atomic_json(value: dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def atomic_npz(path: Path, **arrays: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("wb") as handle:
        np.savez_compressed(handle, **arrays)
    os.replace(temporary, path)


def validate_protocol(config: dict[str, Any], parameters: dict[str, Any]) -> None:
    experiment, protocol = config["experiment"], config["protocol"]
    if experiment["fit_split"] != "calibration" or experiment["evaluation_split"] != "confirmation":
        raise ValueError("UIIS confirmation must use the frozen calibration-to-confirmation protocol.")
    if not protocol["confirmation_opened"] or protocol["confirmation_used_for_fitting"]:
        raise ValueError("Confirmation may be opened for evaluation only.")
    if not experiment["official_suim_test_locked"] or protocol["official_suim_test_evaluated"]:
        raise ValueError("SUIM official TEST must remain locked.")
    if float(config["risk"]["target_alpha"]) != 0.10 or float(parameters["target_alpha"]) != 0.10:
        raise ValueError("The preregistered alpha=0.10 parameter file is required.")
    if parameters["fit_split"] != "calibration" or parameters["confirmation_opened"]:
        raise ValueError("CRC parameters must have been frozen before opening confirmation.")


def load_grouping(path: Path) -> FrozenQualityGrouping:
    with np.load(path, allow_pickle=False) as values:
        return FrozenQualityGrouping(values["mean"], values["scale"], values["centers"], int(values["seed"]))


def build_confirmation_curves(config: dict[str, Any]) -> tuple[dict[str, np.ndarray], list[str]]:
    experiment, risk = config["experiment"], config["risk"]
    conditions = list(experiment["conditions"])
    cache_dir = ROOT / experiment["cache_dir"] / "confirmation"
    grid = coverage_grid(float(risk["coverage_grid_start"]), float(risk["coverage_grid_stop"]), float(risk["coverage_grid_step"]))
    checkpoint_hash = sha256(ROOT / experiment["checkpoint"])
    degradation_hash = sha256(ROOT / experiment["degradation_config"])
    size = int(risk["measurement_size"])
    actual: np.ndarray | None = None
    envelope: np.ndarray | None = None
    counts: np.ndarray | None = None
    sample_ids: list[str] | None = None
    for condition_index, condition in enumerate(conditions):
        payload = torch.load(cache_dir / f"{condition}.pt", map_location="cpu", weights_only=False)
        if payload["split"] != "confirmation" or payload["condition"] != condition:
            raise ValueError("Confirmation cache metadata mismatch.")
        if payload["checkpoint_sha256"] != checkpoint_hash or payload["degradation_config_sha256"] != degradation_hash:
            raise ValueError("Confirmation cache provenance mismatch.")
        current_ids = [str(value) for value in payload["sample_id"]]
        if sample_ids is None:
            sample_ids = current_ids
            actual = np.empty((len(conditions), len(sample_ids), len(grid)), dtype=np.float32)
            envelope = np.empty_like(actual)
            counts = np.empty((len(conditions), len(sample_ids)), dtype=np.int64)
        elif current_ids != sample_ids:
            raise ValueError("Confirmation cache conditions do not preserve sample order.")
        for index in range(len(current_ids)):
            logits = functional.interpolate(payload["logits"][index:index + 1].to(torch.float32), size=(size, size), mode="bilinear", align_corners=False)
            labels = functional.interpolate(payload["labels"][index:index + 1].to(torch.float32).unsqueeze(1), size=(size, size), mode="nearest").squeeze(1).to(torch.long)
            score = uncertainty_scores(logits, temperature=1.0)["raw_msp"][0].reshape(-1).numpy()
            errors = logits.argmax(dim=1)[0].ne(labels[0]).reshape(-1).numpy()
            actual[condition_index, index], envelope[condition_index, index] = curve_summary(score, errors, grid)
            counts[condition_index, index] = len(errors)
        print(f"built confirmation risk curves: {condition}")
    assert actual is not None and envelope is not None and counts is not None and sample_ids is not None
    curves = {
        "actual_risk": actual,
        "monotone_envelope": envelope,
        "pixel_counts": counts,
        "coverages": grid,
        "conditions": np.asarray(conditions),
        "sample_ids": np.asarray(sample_ids),
        "measurement_size": np.asarray(size),
    }
    atomic_npz(ROOT / experiment["output_dir"] / "risk_curves" / "confirmation.npz", **curves)
    return curves, sample_ids


def confirmation_assignments(config: dict[str, Any], conditions: list[Any]) -> dict[int, pd.DataFrame]:
    experiment, quality = config["experiment"], config["quality"]
    training = load_yaml(ROOT / experiment["training_config"])
    split_csv = ROOT / training["data"]["split_dir"] / "confirmation.csv"
    descriptors = descriptor_table(split_csv, conditions, int(quality["descriptor_long_side"]))
    output = ROOT / experiment["output_dir"] / "descriptors"
    descriptors.to_csv(output / "confirmation_descriptors.csv", index=False)
    result: dict[int, pd.DataFrame] = {}
    for seed in quality["seeds"]:
        grouping = load_grouping(output / f"grouping_seed_{seed}.npz")
        assigned = descriptors.copy()
        assigned["quality_group"] = grouping.predict(assigned.loc[:, DESCRIPTOR_NAMES].to_numpy(dtype=np.float64))
        assigned.to_csv(output / f"confirmation_assignments_seed_{seed}.csv", index=False)
        result[int(seed)] = assigned
    return result


def indices_for_quality(assignments: pd.DataFrame, conditions: list[str], sample_ids: list[str], parameters: dict[str, Any], grid: np.ndarray) -> np.ndarray:
    lookup = assignments.set_index(["sample_id", "condition"])["quality_group"]
    groups = parameters["groups"]
    fallback_index = int(parameters["fallback"]["index"])
    indices = np.empty((len(conditions), len(sample_ids)), dtype=np.int64)
    for condition_index, condition in enumerate(conditions):
        for sample_index, sample_id in enumerate(sample_ids):
            group = str(int(lookup.loc[(sample_id, condition)]))
            indices[condition_index, sample_index] = int(groups[group]["index"]) if group in groups else fallback_index
    return indices


def selected_statistics(actual: np.ndarray, indices: np.ndarray, grid: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    selected = actual[np.arange(actual.shape[0])[:, None], np.arange(actual.shape[1])[None, :], indices]
    return grid[indices].mean(axis=0), selected.mean(axis=0)


def bootstrap(values: np.ndarray, iterations: int, seed: int) -> tuple[float, float, float]:
    generator = np.random.default_rng(seed)
    samples = values[generator.integers(0, len(values), size=(iterations, len(values)))].mean(axis=1)
    return float(values.mean()), float(np.quantile(samples, 0.025)), float(np.quantile(samples, 0.975))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=ROOT / "configs" / "uiis_alpha010_confirmation_evaluation.yaml")
    args = parser.parse_args()
    config_path = args.config.resolve()
    config = load_yaml(config_path)
    parameter_path = ROOT / config["experiment"]["parameters_file"]
    parameters = json.loads(parameter_path.read_text(encoding="utf-8"))
    validate_protocol(config, parameters)
    conditions = load_conditions(ROOT / config["experiment"]["degradation_config"])
    if [condition.name for condition in conditions] != list(config["experiment"]["conditions"]):
        raise ValueError("Condition registry differs from the frozen calibration protocol.")
    curves, sample_ids = build_confirmation_curves(config)
    assignments = confirmation_assignments(config, conditions)
    actual, grid = curves["actual_risk"], curves["coverages"]
    condition_names = curves["conditions"].astype(str).tolist()
    global_index = int(parameters["global_crc"]["index"])
    global_indices = np.full((len(condition_names), len(sample_ids)), global_index, dtype=np.int64)
    global_coverage, global_risk = selected_statistics(actual, global_indices, grid)
    rows: list[dict[str, object]] = []
    comparison_rows: list[dict[str, object]] = []
    alpha = float(config["risk"]["target_alpha"])
    for seed in config["quality"]["seeds"]:
        quality_indices = indices_for_quality(assignments[int(seed)], condition_names, sample_ids, parameters["quality_group_crc"][str(seed)], grid)
        quality_coverage, quality_risk = selected_statistics(actual, quality_indices, grid)
        difference = quality_coverage - global_coverage
        mean, lower, upper = bootstrap(difference, int(config["bootstrap"]["iterations"]), int(config["bootstrap"]["seed"]) + int(seed) % 1000)
        lowlight = condition_names.index("lowlight_s3")
        global_lowlight = actual[lowlight, np.arange(len(sample_ids)), global_indices[lowlight]].mean()
        quality_lowlight = actual[lowlight, np.arange(len(sample_ids)), quality_indices[lowlight]].mean()
        lookup = assignments[int(seed)].set_index(["sample_id", "condition"])["quality_group"]
        group_ids = np.asarray(
            [[int(lookup.loc[(sample_id, condition)]) for sample_id in sample_ids] for condition in condition_names],
            dtype=np.int64,
        )
        condition_index = np.arange(len(condition_names))[:, None]
        sample_index = np.arange(len(sample_ids))[None, :]
        global_selected = actual[condition_index, sample_index, global_indices]
        quality_selected = actual[condition_index, sample_index, quality_indices]
        group_global_excess = []
        group_quality_excess = []
        for group in np.unique(group_ids):
            group_global_excess.append(float(global_selected[group_ids == group].mean() - alpha))
            group_quality_excess.append(float(quality_selected[group_ids == group].mean() - alpha))
        worst_group_risk_excess_delta = max(group_quality_excess) - max(group_global_excess)
        rows.extend(
            [
                {"method": "global_crc", "seed": int(seed), "condition": condition, "coverage": float(grid[global_indices[index]].mean()), "selective_risk": float(actual[index, np.arange(len(sample_ids)), global_indices[index]].mean())}
                for index, condition in enumerate(condition_names)
            ]
        )
        rows.extend(
            [
                {"method": "quality_group_crc_global_fallback", "seed": int(seed), "condition": condition, "coverage": float(grid[quality_indices[index]].mean()), "selective_risk": float(actual[index, np.arange(len(sample_ids)), quality_indices[index]].mean())}
                for index, condition in enumerate(condition_names)
            ]
        )
        comparison_rows.append(
            {
                "seed": int(seed),
                "coverage_improvement": mean,
                "coverage_ci95_low": lower,
                "coverage_ci95_high": upper,
                "global_aggregate_risk": float(global_risk.mean()),
                "quality_aggregate_risk": float(quality_risk.mean()),
                "global_lowlight_s3_risk": float(global_lowlight),
                "quality_lowlight_s3_risk": float(quality_lowlight),
                "worst_group_risk_excess_delta": float(worst_group_risk_excess_delta),
                "clusters": len(sample_ids),
            }
        )
    output = ROOT / config["experiment"]["output_dir"] / "confirmation"
    output.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(output / "condition_metrics.csv", index=False)
    comparisons = pd.DataFrame(comparison_rows)
    comparisons.to_csv(output / "comparison.csv", index=False)
    passes = {
        "coverage_at_least_3pp_all_seeds": bool((comparisons["coverage_improvement"] >= 0.03).all()),
        "coverage_ci_lower_above_zero_all_seeds": bool((comparisons["coverage_ci95_low"] > 0.0).all()),
        "aggregate_risk_at_most_alpha_all_seeds": bool((comparisons["quality_aggregate_risk"] <= alpha).all()),
        "lowlight_not_worse_than_global_all_seeds": bool((comparisons["quality_lowlight_s3_risk"] <= comparisons["global_lowlight_s3_risk"]).all()),
        "worst_quality_group_not_worse_than_global_all_seeds": bool((comparisons["worst_group_risk_excess_delta"] <= 0.0).all()),
        "consistent_positive_direction_all_seeds": bool((comparisons["coverage_improvement"] > 0.0).all()),
    }
    decision = {
        "target_alpha": alpha,
        "passes": passes,
        "protocol_pass": bool(all(passes.values())),
        "confirmation_evaluated": True,
        "official_suim_test_evaluated": False,
        "checkpoint_sha256": sha256(ROOT / config["experiment"]["checkpoint"]),
        "parameters_sha256": sha256(parameter_path),
        "config_sha256": sha256(config_path),
    }
    atomic_json(decision, output / "decision.json")
    print(json.dumps(decision, indent=2))


if __name__ == "__main__":
    main()
