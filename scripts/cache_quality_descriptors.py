"""Fit train-only DARC quality groups and assign frozen groups to cal/val.

No labels, segmentation predictions, registered condition names, validation
statistics, or official TEST files enter the grouping fit.
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
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from degradations.registry import build_image_degradation, load_conditions  # noqa: E402
from reliability.quality_descriptors import DESCRIPTOR_NAMES, image_quality_descriptors  # noqa: E402
from reliability.quality_grouping import FrozenQualityGrouping, fit_quality_grouping  # noqa: E402


def load_yaml(path: Path) -> dict[str, Any]:
    return yaml.safe_load(path.read_text(encoding="utf-8"))


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest().upper()


def _load_rows(split_csv: Path, conditions: list[Any]) -> pd.DataFrame:
    rows = pd.read_csv(split_csv)
    required = {"sample_id", "image_path", "mask_path"}
    if rows.empty or required - set(rows.columns) or rows["sample_id"].duplicated().any():
        raise ValueError(f"Invalid split for quality descriptors: {split_csv}")
    output: list[dict[str, object]] = []
    for sample in rows.itertuples(index=False):
        image_path = ROOT / str(sample.image_path)
        if not image_path.is_file():
            raise FileNotFoundError(image_path)
        with Image.open(image_path) as handle:
            image = np.asarray(handle.convert("RGB"), dtype=np.uint8)
        for condition in conditions:
            degraded = build_image_degradation(condition)(image, str(sample.sample_id))
            values = image_quality_descriptors(degraded)
            output.append({
                "sample_id": str(sample.sample_id), "condition": condition.name,
                "degradation_type": condition.degradation_type, "severity": condition.severity,
                **dict(zip(DESCRIPTOR_NAMES, values, strict=True)),
            })
    return pd.DataFrame(output)


def _save_grouping(grouping: FrozenQualityGrouping, path: Path) -> None:
    np.savez(path, mean=grouping.mean, scale=grouping.scale, centers=grouping.centers, seed=np.asarray(grouping.seed), descriptor_names=np.asarray(DESCRIPTOR_NAMES))


def _assign(table: pd.DataFrame, grouping: FrozenQualityGrouping) -> pd.DataFrame:
    assigned = table.copy()
    assigned["quality_group"] = grouping.predict(assigned.loc[:, DESCRIPTOR_NAMES].to_numpy(dtype=np.float64))
    return assigned


def _cluster_counts(table: pd.DataFrame) -> pd.Series:
    return table.groupby("quality_group")["sample_id"].nunique().sort_index()


def main() -> None:
    parser = argparse.ArgumentParser(description="Fit DARC image-quality groups using train images only.")
    parser.add_argument("--config", type=Path, default=ROOT / "configs" / "darc_crc_pilot.yaml")
    args = parser.parse_args()
    config_path = args.config.resolve()
    config = load_yaml(config_path)
    experiment, quality = config["experiment"], config["quality"]
    if experiment["train_split"] != "train" or experiment["fit_split"] != "calibration" or experiment["evaluation_split"] != "val":
        raise ValueError("DARC protocol requires train/calibration/val roles and locks TEST.")
    if "test" in str(experiment).lower() and not bool(experiment["official_test_locked"]):
        raise ValueError("Official TEST must remain locked.")
    baseline = load_yaml(ROOT / experiment["baseline_config"])
    split_dir = ROOT / baseline["data"]["split_dir"]
    conditions = load_conditions(ROOT / experiment["degradation_config"])
    if [item.name for item in conditions] != list(experiment["conditions"]):
        raise ValueError("DARC conditions must exactly match the frozen 13-condition registry.")
    output = ROOT / experiment["output_dir"] / "descriptors"
    output.mkdir(parents=True, exist_ok=True)
    train = _load_rows(split_dir / "train.csv", conditions)
    calibration = _load_rows(split_dir / "calibration.csv", conditions)
    validation = _load_rows(split_dir / "val.csv", conditions)
    train.to_csv(output / "train_descriptors.csv", index=False)
    calibration.to_csv(output / "calibration_descriptors.csv", index=False)
    validation.to_csv(output / "val_descriptors.csv", index=False)
    records: list[dict[str, object]] = []
    for seed in quality["seeds"]:
        grouping = fit_quality_grouping(train.loc[:, DESCRIPTOR_NAMES].to_numpy(dtype=np.float64), groups=int(quality["groups"]), seed=int(seed))
        calibration_assigned = _assign(calibration, grouping)
        calibration_counts = _cluster_counts(calibration_assigned)
        effective_groups = int(quality["groups"])
        # The fallback is determined from calibration availability only, never
        # from validation performance.  It is deterministic and recorded.
        if len(calibration_counts) != effective_groups or int(calibration_counts.min()) < int(quality["minimum_calibration_clusters"]):
            effective_groups = 2
            grouping = fit_quality_grouping(train.loc[:, DESCRIPTOR_NAMES].to_numpy(dtype=np.float64), groups=effective_groups, seed=int(seed))
            calibration_assigned = _assign(calibration, grouping)
            calibration_counts = _cluster_counts(calibration_assigned)
        validation_assigned = _assign(validation, grouping)
        _save_grouping(grouping, output / f"grouping_seed_{seed}.npz")
        calibration_assigned.to_csv(output / f"calibration_assignments_seed_{seed}.csv", index=False)
        validation_assigned.to_csv(output / f"val_assignments_seed_{seed}.csv", index=False)
        records.append({"seed": int(seed), "requested_groups": int(quality["groups"]), "effective_groups": effective_groups, "calibration_cluster_counts": {str(key): int(value) for key, value in calibration_counts.items()}, "minimum_calibration_clusters": int(quality["minimum_calibration_clusters"]), "fallback_required": bool(int(calibration_counts.min()) < int(quality["minimum_calibration_clusters"])), "grouping_path": str((output / f"grouping_seed_{seed}.npz").relative_to(ROOT))})
        print(f"seed={seed}; effective_groups={effective_groups}; calibration_clusters={calibration_counts.to_dict()}")
    metadata = {"config_sha256": sha256(config_path), "degradation_config_sha256": sha256(ROOT / experiment["degradation_config"]), "descriptor_names": list(DESCRIPTOR_NAMES), "train_rows": len(train), "calibration_rows": len(calibration), "val_rows": len(validation), "groupings": records, "labels_used": False, "validation_used_for_fitting": False, "official_test_evaluated": False, "model_retrained": False}
    (output / "metadata.json").write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
