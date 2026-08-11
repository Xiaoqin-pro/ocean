"""Create the preregistered, scene-group-safe AquaRiskMap inner split.

This script partitions only the already admitted formal SUIM training split.
It never reads validation, calibration, or official TEST records.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import pandas as pd
from sklearn.model_selection import GroupShuffleSplit


ROOT = Path(__file__).resolve().parents[1]
SEED = 20260725


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest().upper()


def split_frame(frame: pd.DataFrame, *, seed: int = SEED) -> tuple[pd.DataFrame, pd.DataFrame]:
    required = {"sample_id", "scene_group_id"}
    missing = required.difference(frame.columns)
    if missing:
        raise ValueError(f"Formal train split is missing columns: {sorted(missing)}")
    if frame.empty or frame["scene_group_id"].isna().any():
        raise ValueError("Formal train split must contain non-empty scene groups.")
    splitter = GroupShuffleSplit(n_splits=1, train_size=0.8, random_state=seed)
    train_index, development_index = next(splitter.split(frame, groups=frame["scene_group_id"]))
    return (
        frame.iloc[train_index].sort_values("sample_id").reset_index(drop=True),
        frame.iloc[development_index].sort_values("sample_id").reset_index(drop=True),
    )


def validate_split(train: pd.DataFrame, development: pd.DataFrame, source: pd.DataFrame) -> None:
    if len(train) + len(development) != len(source):
        raise AssertionError("Inner split does not preserve the formal-train sample count.")
    if set(train["sample_id"]).intersection(development["sample_id"]):
        raise AssertionError("Sample leakage across risk-head train/development.")
    if set(train["scene_group_id"]).intersection(development["scene_group_id"]):
        raise AssertionError("Scene-group leakage across risk-head train/development.")
    if set(pd.concat([train["sample_id"], development["sample_id"]])) != set(source["sample_id"]):
        raise AssertionError("Inner split does not preserve formal-train membership.")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--formal-train-csv",
        type=Path,
        default=ROOT / "data" / "suim_processed" / "splits" / "v2_scene_grouped_deduplicated" / "train.csv",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=ROOT / "data" / "suim_processed" / "splits" / "aquariskmap_risk_head_v1",
    )
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    source = pd.read_csv(args.formal_train_csv)
    train, development = split_frame(source, seed=args.seed)
    validate_split(train, development, source)
    if args.output_dir.exists() and any(args.output_dir.iterdir()) and not args.overwrite:
        raise FileExistsError(f"{args.output_dir} exists; use --overwrite only for deterministic regeneration.")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    train_path = args.output_dir / "risk_head_train.csv"
    development_path = args.output_dir / "risk_head_development.csv"
    train.to_csv(train_path, index=False)
    development.to_csv(development_path, index=False)
    summary = {
        "split_version": "aquariskmap_risk_head_v1",
        "parent_split": "v2_scene_grouped_deduplicated/train.csv",
        "parent_split_sha256": sha256(args.formal_train_csv),
        "seed": args.seed,
        "group_splitter": "GroupShuffleSplit(train_size=0.8)",
        "counts": {"risk_head_train": len(train), "risk_head_development": len(development)},
        "csv_sha256": {"risk_head_train": sha256(train_path), "risk_head_development": sha256(development_path)},
        "sample_leakage": False,
        "scene_group_leakage": False,
        "formal_validation_read": False,
        "formal_calibration_read": False,
        "official_suim_test_evaluated": False,
    }
    (args.output_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
