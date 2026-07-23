"""Fit and evaluate zero-training DARC-Seg CRC controllers.

Fitting reads calibration curves only.  Validation curves are opened only
after all coverages have been frozen.  Registered degradation names are used
solely by the oracle diagnostic, never by quality-group controllers.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from reliability.conformal_risk import CRCSelection, select_crc_coverage, select_naive_coverage  # noqa: E402


def load_yaml(path: Path) -> dict[str, Any]:
    return yaml.safe_load(path.read_text(encoding="utf-8"))


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest().upper()


def load_curves(path: Path) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as values:
        return {key: values[key] for key in values.files}


def validate_protocol(config: dict[str, Any]) -> None:
    exp, protocol = config["experiment"], config["protocol"]
    if exp["fit_split"] != "calibration" or exp["evaluation_split"] != "val" or not exp["official_test_locked"]:
        raise ValueError("DARC fitting requires calibration->val only; official TEST is locked.")
    if protocol["validation_used_for_fitting"] or protocol["official_test_evaluated"] or protocol["model_retrained"]:
        raise ValueError("DARC protocol forbids validation fitting, TEST access, and retraining.")


def validate_curve_pair(calibration: dict[str, np.ndarray], validation: dict[str, np.ndarray], config: dict[str, Any]) -> tuple[int, int, int]:
    for key in ("coverages", "score_names", "region_names", "conditions"):
        if not np.array_equal(calibration[key], validation[key]):
            raise ValueError(f"Calibration and validation curves disagree on {key}.")
    conditions = np.asarray(config["experiment"]["conditions"])
    if not np.array_equal(calibration["conditions"], conditions):
        raise ValueError("Risk curves must use exactly the registered 13 conditions.")
    primary = config["ranking"]["primary_score"]
    scores = calibration["score_names"].tolist()
    regions = calibration["region_names"].tolist()
    if primary not in scores or "full" not in regions:
        raise ValueError("Risk curves are missing the primary score or full region.")
    if len(calibration["sample_ids"]) != 146 or len(validation["sample_ids"]) != 146:
        raise ValueError("DARC pilot requires exactly 146 independent calibration and validation sample_id clusters.")
    return scores.index(primary), regions.index("full"), len(calibration["coverages"])


def cluster_mean_losses(envelope: np.ndarray) -> np.ndarray:
    """Average all 13 registered conditions per independent sample_id cluster."""
    if envelope.ndim != 3 or envelope.shape[0] != 13:
        raise ValueError("Expected [13 conditions, sample_id clusters, coverage] envelope curves.")
    return envelope.mean(axis=0)


def grouped_losses(envelope: np.ndarray, assignments: pd.DataFrame, *, conditions: list[str], sample_ids: list[str], group: int) -> np.ndarray:
    """One loss curve per sample_id with one or more versions assigned to group."""
    required = {"sample_id", "condition", "quality_group"}
    if required - set(assignments.columns) or assignments.duplicated(["sample_id", "condition"]).any():
        raise ValueError("Quality assignments must contain one label per sample_id-condition pair.")
    lookup = assignments.set_index(["sample_id", "condition"])["quality_group"]
    curves: list[np.ndarray] = []
    for sample_index, sample_id in enumerate(sample_ids):
        chosen = [condition_index for condition_index, condition in enumerate(conditions) if int(lookup.loc[(sample_id, condition)]) == group]
        if chosen:
            curves.append(envelope[chosen, sample_index].mean(axis=0))
    if not curves:
        return np.empty((0, envelope.shape[-1]), dtype=np.float64)
    return np.asarray(curves, dtype=np.float64)


def _coverage_indices_global(selection: CRCSelection, conditions: int, samples: int) -> np.ndarray:
    return np.full((conditions, samples), selection.index, dtype=np.int64)


def _coverage_indices_oracle(selections: list[CRCSelection], samples: int) -> np.ndarray:
    return np.asarray([[selection.index] * samples for selection in selections], dtype=np.int64)


def _coverage_indices_quality(assignments: pd.DataFrame, *, conditions: list[str], sample_ids: list[str], selections: dict[int, CRCSelection], fallback: CRCSelection | None) -> np.ndarray:
    lookup = assignments.set_index(["sample_id", "condition"])["quality_group"]
    indices = np.empty((len(conditions), len(sample_ids)), dtype=np.int64)
    for condition_index, condition in enumerate(conditions):
        for sample_index, sample_id in enumerate(sample_ids):
            group = int(lookup.loc[(sample_id, condition)])
            if group in selections:
                indices[condition_index, sample_index] = selections[group].index
            elif fallback is not None:
                indices[condition_index, sample_index] = fallback.index
            else:
                raise ValueError("Quality-group CRC is unavailable because a group lacks enough calibration clusters.")
    return indices


def _image_aurc(curve: np.ndarray, coverages: np.ndarray) -> float:
    return float(np.trapezoid(curve, x=coverages))


def _oracle_curve(error_rate: float, coverages: np.ndarray) -> np.ndarray:
    correct = 1.0 - error_rate
    return np.maximum(coverages - correct, 0.0) / coverages


def evaluate_indices(*, actual: np.ndarray, envelope: np.ndarray, counts: np.ndarray, coverages: np.ndarray, indices: np.ndarray, method: str, alpha: float, conditions: list[str], regions: list[str], seed: int | None, fit_scope: str) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for condition_index, condition in enumerate(conditions):
        for region_index, region in enumerate(regions):
            risks = np.asarray([actual[region_index, condition_index, image, indices[condition_index, image]] for image in range(indices.shape[1])], dtype=np.float64)
            bounds = np.asarray([envelope[region_index, condition_index, image, indices[condition_index, image]] for image in range(indices.shape[1])], dtype=np.float64)
            selected = coverages[indices[condition_index]]
            pixels = counts[region_index, condition_index].astype(np.float64)
            accepted = np.ceil(pixels * selected)
            expected_errors = risks * accepted
            total_pixels = pixels.sum()
            pixel_weighted_risk = expected_errors.sum() / accepted.sum()
            per_image_aurc = np.asarray([_image_aurc(actual[region_index, condition_index, image], coverages) for image in range(indices.shape[1])])
            full_risks = actual[region_index, condition_index, :, -1]
            oracle = np.asarray([_image_aurc(_oracle_curve(float(rate), coverages), coverages) for rate in full_risks])
            rows.append({
                "method": method, "fit_scope": fit_scope, "seed": seed, "target_alpha": alpha,
                "condition": condition, "region": region, "selected_coverage": float(selected.mean()),
                "coverage": float(accepted.sum() / total_pixels), "rejected_pixel_fraction": float(1.0 - accepted.sum() / total_pixels),
                "actual_selective_risk": float(risks.mean()), "pixel_weighted_selective_risk": float(pixel_weighted_risk),
                "monotone_envelope_risk": float(bounds.mean()), "risk_excess": float(risks.mean() - alpha),
                "absolute_risk_violation": float(max(risks.mean() - alpha, 0.0)),
                "accepted_correct_pixel_fraction": float((accepted - expected_errors).sum() / total_pixels),
                "accepted_error_count": float(expected_errors.sum()), "cluster_count": int(len(risks)),
                "aurc": float(per_image_aurc.mean()), "eaurc": float((per_image_aurc - oracle).mean()),
            })
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description="Fit image-clustered DARC CRC controllers; official TEST is locked.")
    parser.add_argument("--config", type=Path, default=ROOT / "configs" / "darc_crc_pilot.yaml")
    args = parser.parse_args()
    config_path = args.config.resolve()
    config = load_yaml(config_path)
    validate_protocol(config)
    output = ROOT / config["experiment"]["output_dir"]
    calibration = load_curves(output / "risk_curves" / "calibration.npz")
    validation = load_curves(output / "risk_curves" / "val.npz")
    score_index, full_region, _ = validate_curve_pair(calibration, validation, config)
    conditions = calibration["conditions"].astype(str).tolist()
    regions = calibration["region_names"].astype(str).tolist()
    grid = calibration["coverages"].astype(np.float64)
    calibration_envelope = calibration["monotone_envelope"][score_index]
    calibration_actual = calibration["actual_risk"][score_index]
    validation_envelope = validation["monotone_envelope"][score_index]
    validation_actual = validation["actual_risk"][score_index]
    calibration_full = calibration_envelope[full_region]
    global_losses = cluster_mean_losses(calibration_full)
    parameters: dict[str, object] = {"global": {}, "oracle": {}, "quality_group": {}, "official_test_evaluated": False, "model_retrained": False}
    all_rows: list[dict[str, object]] = []
    for alpha in config["risk"]["targets"]:
        alpha = float(alpha)
        global_crc = select_crc_coverage(global_losses, grid, alpha, bound=float(config["risk"]["crc_bound"]))
        naive = select_naive_coverage(global_losses, grid, alpha)
        oracle = [select_crc_coverage(calibration_full[condition_index], grid, alpha, bound=float(config["risk"]["crc_bound"])) for condition_index in range(len(conditions))]
        parameters["global"][str(alpha)] = {"crc": global_crc.__dict__, "naive": naive.__dict__}
        parameters["oracle"][str(alpha)] = {condition: selection.__dict__ for condition, selection in zip(conditions, oracle, strict=True)}
        all_rows.extend(evaluate_indices(actual=validation_actual, envelope=validation_envelope, counts=validation["pixel_counts"], coverages=grid, indices=np.full((len(conditions), len(validation["sample_ids"])), len(grid) - 1), method="no_rejection", alpha=alpha, conditions=conditions, regions=regions, seed=None, fit_scope="none"))
        all_rows.extend(evaluate_indices(actual=validation_actual, envelope=validation_envelope, counts=validation["pixel_counts"], coverages=grid, indices=_coverage_indices_global(naive, len(conditions), len(validation["sample_ids"])), method="naive_empirical", alpha=alpha, conditions=conditions, regions=regions, seed=None, fit_scope="calibration_cluster_mean"))
        all_rows.extend(evaluate_indices(actual=validation_actual, envelope=validation_envelope, counts=validation["pixel_counts"], coverages=grid, indices=_coverage_indices_global(global_crc, len(conditions), len(validation["sample_ids"])), method="global_crc", alpha=alpha, conditions=conditions, regions=regions, seed=None, fit_scope="calibration_cluster_mean"))
        all_rows.extend(evaluate_indices(actual=validation_actual, envelope=validation_envelope, counts=validation["pixel_counts"], coverages=grid, indices=_coverage_indices_oracle(oracle, len(validation["sample_ids"])), method="oracle_condition_crc", alpha=alpha, conditions=conditions, regions=regions, seed=None, fit_scope="condition_label_diagnostic_only"))
        for seed in config["quality"]["seeds"]:
            assignments = pd.read_csv(output / "descriptors" / f"calibration_assignments_seed_{seed}.csv")
            selected: dict[int, CRCSelection] = {}
            group_counts: dict[str, int] = {}
            for group in sorted(assignments["quality_group"].unique()):
                losses = grouped_losses(calibration_full, assignments, conditions=conditions, sample_ids=calibration["sample_ids"].astype(str).tolist(), group=int(group))
                group_counts[str(int(group))] = len(losses)
                if len(losses) >= int(config["quality"]["minimum_calibration_clusters"]):
                    selected[int(group)] = select_crc_coverage(losses, grid, alpha, bound=float(config["risk"]["crc_bound"]))
            parameters["quality_group"].setdefault(str(alpha), {})[str(seed)] = {"groups": {str(group): value.__dict__ for group, value in selected.items()}, "cluster_counts": group_counts, "fallback": global_crc.__dict__}
            val_assignments = pd.read_csv(output / "descriptors" / f"val_assignments_seed_{seed}.csv")
            fallback_indices = _coverage_indices_quality(val_assignments, conditions=conditions, sample_ids=validation["sample_ids"].astype(str).tolist(), selections=selected, fallback=global_crc)
            all_rows.extend(evaluate_indices(actual=validation_actual, envelope=validation_envelope, counts=validation["pixel_counts"], coverages=grid, indices=fallback_indices, method="quality_group_crc_global_fallback", alpha=alpha, conditions=conditions, regions=regions, seed=int(seed), fit_scope="calibration_quality_group"))
            if len(selected) == len(assignments["quality_group"].unique()):
                direct_indices = _coverage_indices_quality(val_assignments, conditions=conditions, sample_ids=validation["sample_ids"].astype(str).tolist(), selections=selected, fallback=None)
                all_rows.extend(evaluate_indices(actual=validation_actual, envelope=validation_envelope, counts=validation["pixel_counts"], coverages=grid, indices=direct_indices, method="quality_group_crc", alpha=alpha, conditions=conditions, regions=regions, seed=int(seed), fit_scope="calibration_quality_group"))
    table = pd.DataFrame(all_rows)
    required = {"no_rejection", "naive_empirical", "global_crc", "oracle_condition_crc", "quality_group_crc_global_fallback"}
    if required - set(table["method"]) or table.duplicated(["method", "target_alpha", "condition", "region", "seed"]).any():
        raise AssertionError("DARC CRC result table is incomplete or has duplicate keys.")
    table.to_csv(output / "metrics.csv", index=False)
    table.groupby(["method", "target_alpha", "region", "seed"], dropna=False).mean(numeric_only=True).reset_index().to_csv(output / "condition_summary.csv", index=False)
    (output / "parameters.json").write_text(json.dumps(parameters, ensure_ascii=False, indent=2), encoding="utf-8")
    metadata = {"config_sha256": sha256(config_path), "checkpoint_sha256": sha256(ROOT / config["experiment"]["checkpoint"]), "degradation_config_sha256": sha256(ROOT / config["experiment"]["degradation_config"]), "fit_split": "calibration", "evaluation_split": "val", "cluster_unit": "sample_id_with_all_13_conditions", "primary_score": config["ranking"]["primary_score"], "oracle_uses_condition_label_for_diagnostic_only": True, "ground_truth_boundary_as_input": False, "official_test_evaluated": False, "model_retrained": False, "rows": len(table)}
    (output / "metadata.json").write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"wrote {len(table)} DARC CRC rows")


if __name__ == "__main__":
    main()
