"""Freeze train/calibration-only UIIS quality CRC parameters at alpha=0.10."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as functional
import yaml
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from degradations.registry import build_image_degradation, load_conditions  # noqa: E402
from metrics.uncertainty_ranking import uncertainty_scores  # noqa: E402
from reliability.conformal_risk import select_crc_coverage  # noqa: E402
from reliability.quality_descriptors import DESCRIPTOR_NAMES, image_quality_descriptors  # noqa: E402
from reliability.quality_grouping import FrozenQualityGrouping, fit_quality_grouping  # noqa: E402
from reliability.selective_risk import coverage_grid, curve_summary  # noqa: E402


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


def validate_protocol(config: dict[str, Any]) -> None:
    experiment, protocol = config["experiment"], config["protocol"]
    if experiment["train_split"] != "train" or experiment["fit_split"] != "calibration":
        raise ValueError("UIIS CRC may fit only train-derived groups and calibration risk curves.")
    if experiment["evaluation_split"] != "confirmation" or protocol["confirmation_opened"]:
        raise ValueError("Confirmation must remain unopened while CRC parameters are fitted.")
    if protocol["confirmation_used_for_fitting"] or protocol["official_suim_test_evaluated"]:
        raise ValueError("The frozen UIIS protocol forbids confirmation fitting and SUIM TEST evaluation.")
    if float(config["risk"]["target_alpha"]) != 0.10 or int(config["quality"]["groups"]) != 3:
        raise ValueError("Only the preregistered alpha=0.10, three-group protocol is allowed.")


def resize_descriptor_image(image: np.ndarray, long_side: int) -> np.ndarray:
    height, width = image.shape[:2]
    scale = min(1.0, long_side / max(height, width))
    if scale == 1.0:
        return image
    return cv2.resize(image, (max(1, round(width * scale)), max(1, round(height * scale))), interpolation=cv2.INTER_AREA)


def descriptor_table(split_csv: Path, conditions: list[Any], long_side: int) -> pd.DataFrame:
    frame = pd.read_csv(split_csv)
    required = {"sample_id", "image_path"}
    if frame.empty or required - set(frame.columns) or frame["sample_id"].duplicated().any():
        raise ValueError(f"Invalid UIIS split: {split_csv}")
    rows: list[dict[str, object]] = []
    for sample in frame.itertuples(index=False):
        with Image.open(ROOT / str(sample.image_path)) as handle:
            image = resize_descriptor_image(np.asarray(handle.convert("RGB"), dtype=np.uint8), long_side)
        for condition in conditions:
            values = image_quality_descriptors(build_image_degradation(condition)(image, str(sample.sample_id)))
            rows.append(
                {
                    "sample_id": str(sample.sample_id),
                    "condition": condition.name,
                    "degradation_type": condition.degradation_type,
                    "severity": condition.severity,
                    **dict(zip(DESCRIPTOR_NAMES, values, strict=True)),
                }
            )
    return pd.DataFrame(rows)


def save_grouping(grouping: FrozenQualityGrouping, path: Path) -> None:
    np.savez(
        path,
        mean=grouping.mean,
        scale=grouping.scale,
        centers=grouping.centers,
        seed=np.asarray(grouping.seed),
        descriptor_names=np.asarray(DESCRIPTOR_NAMES),
    )


def quality_group_losses(envelope: np.ndarray, assignments: pd.DataFrame, conditions: list[str], sample_ids: list[str], group: int) -> np.ndarray:
    lookup = assignments.set_index(["sample_id", "condition"])["quality_group"]
    losses: list[np.ndarray] = []
    for sample_index, sample_id in enumerate(sample_ids):
        matching = [index for index, condition in enumerate(conditions) if int(lookup.loc[(sample_id, condition)]) == group]
        if matching:
            losses.append(envelope[matching, sample_index].mean(axis=0))
    return np.asarray(losses, dtype=np.float64)


def fit_quality_groups(config: dict[str, Any], config_path: Path, conditions: list[Any]) -> dict[str, Any]:
    experiment, quality = config["experiment"], config["quality"]
    training = load_yaml(ROOT / experiment["training_config"])
    split_dir = ROOT / training["data"]["split_dir"]
    output = ROOT / experiment["output_dir"] / "descriptors"
    output.mkdir(parents=True, exist_ok=True)
    train = descriptor_table(split_dir / "train.csv", conditions, int(quality["descriptor_long_side"]))
    calibration = descriptor_table(split_dir / "calibration.csv", conditions, int(quality["descriptor_long_side"]))
    train.to_csv(output / "train_descriptors.csv", index=False)
    calibration.to_csv(output / "calibration_descriptors.csv", index=False)
    records: list[dict[str, Any]] = []
    for seed in quality["seeds"]:
        grouping = fit_quality_grouping(train.loc[:, DESCRIPTOR_NAMES].to_numpy(dtype=np.float64), groups=int(quality["groups"]), seed=int(seed))
        assigned = calibration.copy()
        assigned["quality_group"] = grouping.predict(assigned.loc[:, DESCRIPTOR_NAMES].to_numpy(dtype=np.float64))
        counts = assigned.groupby("quality_group")["sample_id"].nunique().sort_index()
        effective_groups = int(quality["groups"])
        if len(counts) != effective_groups or int(counts.min()) < int(quality["minimum_calibration_clusters"]):
            effective_groups = 2
            grouping = fit_quality_grouping(train.loc[:, DESCRIPTOR_NAMES].to_numpy(dtype=np.float64), groups=effective_groups, seed=int(seed))
            assigned["quality_group"] = grouping.predict(assigned.loc[:, DESCRIPTOR_NAMES].to_numpy(dtype=np.float64))
            counts = assigned.groupby("quality_group")["sample_id"].nunique().sort_index()
        save_grouping(grouping, output / f"grouping_seed_{seed}.npz")
        assigned.to_csv(output / f"calibration_assignments_seed_{seed}.csv", index=False)
        records.append(
            {
                "seed": int(seed),
                "requested_groups": int(quality["groups"]),
                "effective_groups": effective_groups,
                "calibration_cluster_counts": {str(key): int(value) for key, value in counts.items()},
                "minimum_calibration_clusters": int(quality["minimum_calibration_clusters"]),
            }
        )
    metadata = {
        "config_sha256": sha256(config_path),
        "degradation_config_sha256": sha256(ROOT / experiment["degradation_config"]),
        "descriptor_names": list(DESCRIPTOR_NAMES),
        "train_rows": len(train),
        "calibration_rows": len(calibration),
        "labels_used": False,
        "confirmation_opened": False,
        "groupings": records,
    }
    atomic_json(metadata, output / "metadata.json")
    return metadata


def build_calibration_curves(config: dict[str, Any], config_path: Path) -> dict[str, Any]:
    experiment, risk = config["experiment"], config["risk"]
    cache_dir = ROOT / experiment["cache_dir"] / "calibration"
    conditions = list(experiment["conditions"])
    grid = coverage_grid(float(risk["coverage_grid_start"]), float(risk["coverage_grid_stop"]), float(risk["coverage_grid_step"]))
    measurement_size = int(risk["measurement_size"])
    checkpoint_hash = sha256(ROOT / experiment["checkpoint"])
    degradation_hash = sha256(ROOT / experiment["degradation_config"])
    actual: np.ndarray | None = None
    envelope: np.ndarray | None = None
    counts: np.ndarray | None = None
    sample_ids: list[str] | None = None
    for condition_index, condition in enumerate(conditions):
        payload = torch.load(cache_dir / f"{condition}.pt", map_location="cpu", weights_only=False)
        if payload["split"] != "calibration" or payload["condition"] != condition:
            raise ValueError("Calibration cache condition metadata mismatch.")
        if payload["checkpoint_sha256"] != checkpoint_hash or payload["degradation_config_sha256"] != degradation_hash:
            raise ValueError("Calibration cache provenance mismatch.")
        labels = payload["labels"]
        if labels.dtype != torch.uint8 or int(labels.max()) > 7:
            raise ValueError("Calibration cache labels must be compact semantic IDs.")
        current_ids = [str(value) for value in payload["sample_id"]]
        if sample_ids is None:
            sample_ids = current_ids
            actual = np.empty((len(conditions), len(sample_ids), len(grid)), dtype=np.float32)
            envelope = np.empty_like(actual)
            counts = np.empty((len(conditions), len(sample_ids)), dtype=np.int64)
        elif current_ids != sample_ids:
            raise ValueError("Calibration cache conditions must preserve exact sample order.")
        for index in range(len(current_ids)):
            logits = payload["logits"][index:index + 1].to(dtype=torch.float32)
            target = labels[index:index + 1].to(dtype=torch.float32).unsqueeze(1)
            logits = functional.interpolate(logits, size=(measurement_size, measurement_size), mode="bilinear", align_corners=False)
            target = functional.interpolate(target, size=(measurement_size, measurement_size), mode="nearest").squeeze(1).to(dtype=torch.long)
            score = uncertainty_scores(logits, temperature=1.0)["raw_msp"][0].reshape(-1).numpy()
            error = logits.argmax(dim=1)[0].ne(target[0]).reshape(-1).numpy()
            values, monotone = curve_summary(score, error, grid)
            actual[condition_index, index] = values
            envelope[condition_index, index] = monotone
            counts[condition_index, index] = len(error)
        print(f"built calibration risk curves: {condition}")
    assert actual is not None and envelope is not None and counts is not None and sample_ids is not None
    output = ROOT / experiment["output_dir"] / "risk_curves"
    atomic_npz(
        output / "calibration.npz",
        actual_risk=actual,
        monotone_envelope=envelope,
        pixel_counts=counts,
        coverages=grid,
        conditions=np.asarray(conditions),
        sample_ids=np.asarray(sample_ids),
        measurement_size=np.asarray(measurement_size),
    )
    metadata = {
        "config_sha256": sha256(config_path),
        "checkpoint_sha256": checkpoint_hash,
        "degradation_config_sha256": degradation_hash,
        "split": "calibration",
        "measurement_size": measurement_size,
        "conditions": conditions,
        "samples": len(sample_ids),
        "confirmation_opened": False,
        "official_suim_test_evaluated": False,
    }
    atomic_json(metadata, output / "calibration_metadata.json")
    return metadata


def fit_crc_parameters(config: dict[str, Any], config_path: Path) -> dict[str, Any]:
    experiment, risk, quality = config["experiment"], config["risk"], config["quality"]
    curves_path = ROOT / experiment["output_dir"] / "risk_curves" / "calibration.npz"
    with np.load(curves_path, allow_pickle=False) as loaded:
        curves = {key: loaded[key] for key in loaded.files}
    conditions = curves["conditions"].astype(str).tolist()
    sample_ids = curves["sample_ids"].astype(str).tolist()
    if conditions != list(experiment["conditions"]) or len(sample_ids) != 508:
        raise ValueError("Calibration risk curves do not match the frozen UIIS protocol.")
    global_losses = curves["monotone_envelope"].mean(axis=0)
    alpha = float(risk["target_alpha"])
    global_crc = select_crc_coverage(global_losses, curves["coverages"], alpha, bound=float(risk["crc_bound"]))
    parameters: dict[str, Any] = {
        "target_alpha": alpha,
        "global_crc": global_crc.__dict__,
        "quality_group_crc": {},
        "fit_split": "calibration",
        "evaluation_split": "confirmation",
        "confirmation_opened": False,
        "official_suim_test_evaluated": False,
    }
    descriptor_dir = ROOT / experiment["output_dir"] / "descriptors"
    for seed in quality["seeds"]:
        assignments = pd.read_csv(descriptor_dir / f"calibration_assignments_seed_{seed}.csv")
        selected: dict[str, Any] = {}
        counts: dict[str, int] = {}
        for group in sorted(assignments["quality_group"].unique()):
            losses = quality_group_losses(curves["monotone_envelope"], assignments, conditions, sample_ids, int(group))
            counts[str(int(group))] = len(losses)
            if len(losses) >= int(quality["minimum_calibration_clusters"]):
                selected[str(int(group))] = select_crc_coverage(losses, curves["coverages"], alpha, bound=float(risk["crc_bound"])).__dict__
        parameters["quality_group_crc"][str(seed)] = {
            "groups": selected,
            "cluster_counts": counts,
            "fallback": global_crc.__dict__,
        }
    output = ROOT / experiment["output_dir"]
    atomic_json(parameters, output / "parameters_calibration_only.json")
    metadata = {
        "config_sha256": sha256(config_path),
        "checkpoint_sha256": sha256(ROOT / experiment["checkpoint"]),
        "calibration_samples": len(sample_ids),
        "conditions": conditions,
        "parameter_status": "frozen_before_confirmation",
        "confirmation_opened": False,
        "official_suim_test_evaluated": False,
    }
    atomic_json(metadata, output / "calibration_fit_metadata.json")
    return parameters


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=ROOT / "configs" / "uiis_alpha010_crc.yaml")
    parser.add_argument("--stage", choices=("descriptors", "curves", "fit", "all"), default="all")
    args = parser.parse_args()
    config_path = args.config.resolve()
    config = load_yaml(config_path)
    validate_protocol(config)
    conditions = load_conditions(ROOT / config["experiment"]["degradation_config"])
    if [item.name for item in conditions] != list(config["experiment"]["conditions"]):
        raise ValueError("UIIS CRC must use the frozen condition registry.")
    if args.stage in {"descriptors", "all"}:
        fit_quality_groups(config, config_path, conditions)
        print("fitted train-only quality groups")
    if args.stage in {"curves", "all"}:
        build_calibration_curves(config, config_path)
        print("built calibration-only risk curves")
    if args.stage in {"fit", "all"}:
        fit_crc_parameters(config, config_path)
        print("froze calibration-only CRC parameters")


if __name__ == "__main__":
    main()
