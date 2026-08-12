"""Select the preregistered UIIS train/calibration/confirmation split."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image
from sklearn.model_selection import GroupShuffleSplit

ROOT = Path(__file__).resolve().parents[1]
SPLIT_NAMES = ("train", "calibration", "confirmation")
TARGET_RATIOS = (0.70, 0.15, 0.15)


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def candidate_split(frame: pd.DataFrame, seed: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    first = GroupShuffleSplit(n_splits=1, train_size=TARGET_RATIOS[0], random_state=seed)
    train_index, holdout_index = next(first.split(frame, groups=frame["scene_group_id"]))
    holdout = frame.iloc[holdout_index]
    second = GroupShuffleSplit(n_splits=1, train_size=0.5, random_state=seed + 1)
    calibration_relative, confirmation_relative = next(second.split(holdout, groups=holdout["scene_group_id"]))
    return train_index, holdout_index[calibration_relative], holdout_index[confirmation_relative]


def mask_histograms(frame: pd.DataFrame, root: Path) -> np.ndarray:
    values = np.zeros((len(frame), 8), dtype=np.int64)
    for index, relative_path in enumerate(frame["mask_path"]):
        with Image.open(root / relative_path) as image:
            labels = np.asarray(image, dtype=np.uint8)
        if labels.ndim != 2 or labels.max(initial=0) > 7:
            raise ValueError(f"Invalid UIIS semantic mask: {relative_path}")
        values[index] = np.bincount(labels.reshape(-1), minlength=8)[:8]
    return values


def score_split(indices: tuple[np.ndarray, np.ndarray, np.ndarray], histograms: np.ndarray) -> float:
    all_pixels = histograms.sum(axis=0)
    global_ratio = all_pixels / all_pixels.sum()
    total = len(histograms)
    score = 0.0
    for index, target in zip(indices, TARGET_RATIOS):
        pixels = histograms[index].sum(axis=0)
        ratio = pixels / pixels.sum()
        score += 3.0 * abs(len(index) / total - target)
        score += float(np.abs(ratio - global_ratio).mean())
        # A split with a missing semantic class is unsuitable for calibration or confirmation.
        score += 1.0 if (pixels[1:] == 0).any() else 0.0
    return score


def assert_protocol(splits: dict[str, pd.DataFrame]) -> None:
    owners: dict[str, str] = {}
    for name, frame in splits.items():
        if frame["sample_id"].duplicated().any():
            raise AssertionError(f"Duplicated sample in {name}.")
        for group in frame["scene_group_id"].unique():
            previous = owners.setdefault(group, name)
            if previous != name:
                raise AssertionError("Scene-group leakage across UIIS splits.")
    if len(set().union(*(set(frame["sample_id"]) for frame in splits.values()))) != sum(len(frame) for frame in splits.values()):
        raise AssertionError("Sample overlap across UIIS splits.")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--members", type=Path, default=ROOT / "data" / "uiis_processed" / "scene_groups" / "scene_group_members.csv")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "data" / "uiis_processed" / "splits" / "uiis_alpha010_confirmation")
    parser.add_argument("--candidate-count", type=int, default=300)
    parser.add_argument("--seed", type=int, default=20260723)
    args = parser.parse_args()
    if args.candidate_count <= 0:
        raise ValueError("candidate-count must be positive.")
    frame = pd.read_csv(args.members).sort_values("sample_id", kind="stable").reset_index(drop=True)
    required = {"sample_id", "mask_path", "scene_group_id"}
    if missing := required - set(frame.columns):
        raise ValueError(f"Scene group members missing columns: {sorted(missing)}")
    if frame["scene_group_id"].isna().any():
        raise ValueError("All admitted UIIS samples need a scene_group_id.")
    values = mask_histograms(frame, ROOT)
    best: tuple[np.ndarray, np.ndarray, np.ndarray] | None = None
    best_seed, best_score = None, float("inf")
    for offset in range(args.candidate_count):
        indices = candidate_split(frame, args.seed + offset)
        value = score_split(indices, values)
        if value < best_score:
            best, best_seed, best_score = indices, args.seed + offset, value
    assert best is not None and best_seed is not None
    splits = {
        name: frame.iloc[index].sort_values("sample_id", kind="stable").reset_index(drop=True)
        for name, index in zip(SPLIT_NAMES, best)
    }
    assert_protocol(splits)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    summary: dict[str, object] = {
        "split_version": "uiis_alpha010_confirmation_v1",
        "protocol": "Fresh UIIS scene-group-aware train/calibration/confirmation split for preregistered alpha=0.10 confirmation.",
        "official_suim_test_evaluated": False,
        "selection_seed": best_seed,
        "candidate_count": args.candidate_count,
        "balance_score": best_score,
        "target_ratios": dict(zip(SPLIT_NAMES, TARGET_RATIOS)),
        "split_counts": {},
        "split_sha256": {},
        "scene_group_leakage": False,
    }
    for name, split in splits.items():
        path = args.output_dir / f"{name}.csv"
        split.to_csv(path, index=False)
        summary["split_counts"][name] = len(split)
        summary["split_sha256"][name] = file_sha256(path)
        pixels = values[best[SPLIT_NAMES.index(name)]].sum(axis=0)
        summary[f"{name}_class_pixels"] = pixels.astype(int).tolist()
    (args.output_dir / "split_summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
