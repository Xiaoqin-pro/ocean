"""Select a class-balanced, scene-group-aware formal v2 split after review completion."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image
from sklearn.model_selection import GroupShuffleSplit

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def histograms(frame: pd.DataFrame) -> np.ndarray:
    result = np.zeros((len(frame), 8), dtype=np.int64)
    for index, relative_path in enumerate(frame["mask_path"]):
        with Image.open(PROJECT_ROOT / relative_path) as image:
            values = np.asarray(image, dtype=np.uint8)
        result[index] = np.bincount(values.reshape(-1), minlength=8)[:8]
    return result


def candidate_split(frame: pd.DataFrame, seed: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    first = GroupShuffleSplit(n_splits=1, train_size=0.8, random_state=seed)
    train, holdout = next(first.split(frame, groups=frame["scene_group_id"]))
    second = GroupShuffleSplit(n_splits=1, train_size=0.5, random_state=seed + 1)
    val_relative, calibration_relative = next(second.split(frame.iloc[holdout], groups=frame.iloc[holdout]["scene_group_id"]))
    return train, holdout[val_relative], holdout[calibration_relative]


def score(indices: tuple[np.ndarray, np.ndarray, np.ndarray], values: np.ndarray) -> float:
    all_pixels = values.sum(axis=0)
    global_ratio = all_pixels / all_pixels.sum()
    expected = (0.8, 0.1, 0.1)
    total = len(values)
    total_score = 0.0
    for split_indices, expected_size in zip(indices, expected):
        ratio = values[split_indices].sum(axis=0)
        ratio = ratio / ratio.sum()
        total_score += 3.0 * abs(len(split_indices) / total - expected_size)
        total_score += float(np.abs(ratio - global_ratio).mean())
    return total_score


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, default=PROJECT_ROOT / "data" / "suim_processed" / "manifest.csv")
    parser.add_argument("--report-dir", type=Path, default=PROJECT_ROOT / "data" / "suim_processed" / "quality_reports")
    parser.add_argument("--output-dir", type=Path, default=PROJECT_ROOT / "data" / "suim_processed" / "splits" / "v2_scene_grouped_deduplicated")
    parser.add_argument("--candidate-count", type=int, default=300)
    parser.add_argument("--seed", type=int, default=20260721)
    parser.add_argument("--allow-pending-review", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    scene_summary = json.loads((args.report_dir / "scene_group_summary.json").read_text(encoding="utf-8"))
    if scene_summary["protocol_status"] != "review_complete" and not args.allow_pending_review:
        raise RuntimeError("Near-duplicate review is still pending; resolve review_decision values before creating the formal split.")
    manifest = pd.read_csv(args.manifest)
    groups = pd.read_csv(args.report_dir / "scene_group_members.csv")[["sample_id", "scene_group_id"]]
    manifest = manifest.merge(groups, on="sample_id", validate="one_to_one")
    exclusions = pd.read_csv(args.report_dir / "v2_excluded_samples.csv")
    development = manifest.loc[manifest["partition"].eq("train_val")].copy()
    official_test = manifest.loc[manifest["partition"].eq("test")].copy()
    excluded_ids = set(exclusions["sample_id"])
    test_scene_groups = set(official_test["scene_group_id"])
    reviewed_test_neighbors = development.loc[development["scene_group_id"].isin(test_scene_groups) & ~development["sample_id"].isin(excluded_ids)]
    if len(reviewed_test_neighbors):
        additions = reviewed_test_neighbors.assign(image_sha256="", reason="reviewed_near_duplicate_scene_of_official_test")[exclusions.columns]
        exclusions = pd.concat([exclusions, additions], ignore_index=True)
        excluded_ids.update(additions["sample_id"])
    retained = development.loc[~development["sample_id"].isin(excluded_ids)].reset_index(drop=True)
    values = histograms(retained)
    best_indices: tuple[np.ndarray, np.ndarray, np.ndarray] | None = None
    best_seed, best_score = None, float("inf")
    for offset in range(args.candidate_count):
        indices = candidate_split(retained, args.seed + offset)
        value = score(indices, values)
        if value < best_score:
            best_indices, best_seed, best_score = indices, args.seed + offset, value
    assert best_indices is not None and best_seed is not None
    names = ("train", "val", "calibration")
    splits = {name: retained.iloc[index].sort_values("sample_id").reset_index(drop=True) for name, index in zip(names, best_indices)}
    owner: dict[str, str] = {}
    for name, frame in splits.items():
        for group_id in frame["scene_group_id"].unique():
            previous = owner.setdefault(str(group_id), name)
            if previous != name:
                raise AssertionError("Scene group leakage across development splits.")
    if set().union(*(set(frame["scene_group_id"]) for frame in splits.values())).intersection(test_scene_groups):
        raise AssertionError("Development scene group leaks to official test.")
    if args.output_dir.exists() and any(args.output_dir.iterdir()) and not args.overwrite:
        raise FileExistsError(f"{args.output_dir} exists; use --overwrite.")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    summary = {"split_version": "v2_scene_grouped_deduplicated", "selection_seed": best_seed, "candidate_count": args.candidate_count, "balance_score": best_score, "near_duplicate_review_status": scene_summary["protocol_status"], "split_counts": {}, "split_sha256": {}, "development_to_official_test_scene_leakage": False}
    for name, frame in {**splits, "test": official_test.sort_values("sample_id").reset_index(drop=True)}.items():
        path = args.output_dir / f"{name}.csv"
        frame.to_csv(path, index=False)
        summary["split_counts"][name] = len(frame)
        summary["split_sha256"][name] = digest(path)
    exclusions.to_csv(args.report_dir / "v2_scene_grouped_excluded_samples.csv", index=False)
    (args.report_dir / "v2_scene_grouped_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
