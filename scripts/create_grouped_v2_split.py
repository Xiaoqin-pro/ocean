"""Create a leakage-safe SUIM v2 development split.

The official test partition is immutable. Any train/val image with the same
SHA-256 as an official-test image is excluded; unresolved size-repaired masks
are also excluded by default. Remaining images are assigned by exact-image
group so no group crosses train, validation, or calibration.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
from PIL import Image
from sklearn.model_selection import GroupShuffleSplit

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SEED = 20260721


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def split_hash(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def pixel_histogram(paths: Iterable[str]) -> np.ndarray:
    histogram = np.zeros(8, dtype=np.int64)
    for relative_path in paths:
        with Image.open(PROJECT_ROOT / relative_path) as image:
            values = np.asarray(image, dtype=np.uint8)
        histogram += np.bincount(values.reshape(-1), minlength=8)[:8]
    return histogram


def mask_pixels_sha256(path: Path) -> str:
    with Image.open(path) as image:
        array = np.asarray(image, dtype=np.uint8)
    return hashlib.sha256(array.tobytes()).hexdigest()


def validate_group_isolation(splits: dict[str, pd.DataFrame]) -> None:
    seen: dict[str, str] = {}
    for split_name, frame in splits.items():
        for group_id in frame["group_id"].unique():
            previous = seen.setdefault(str(group_id), split_name)
            if previous != split_name:
                raise ValueError(f"Group {group_id} leaks between {previous} and {split_name}.")


def grouped_partition(frame: pd.DataFrame, *, seed: int) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    first = GroupShuffleSplit(n_splits=1, train_size=0.80, random_state=seed)
    train_index, holdout_index = next(first.split(frame, groups=frame["group_id"]))
    train, holdout = frame.iloc[train_index].copy(), frame.iloc[holdout_index].copy()
    second = GroupShuffleSplit(n_splits=1, train_size=0.50, random_state=seed + 1)
    validation_index, calibration_index = next(second.split(holdout, groups=holdout["group_id"]))
    return train, holdout.iloc[validation_index].copy(), holdout.iloc[calibration_index].copy()


def main() -> None:
    parser = argparse.ArgumentParser(description="Create the grouped and deduplicated SUIM v2 split.")
    parser.add_argument("--manifest", type=Path, default=PROJECT_ROOT / "data" / "suim_processed" / "manifest.csv")
    parser.add_argument("--output-dir", type=Path, default=PROJECT_ROOT / "data" / "suim_processed" / "splits" / "v2_grouped_deduplicated")
    parser.add_argument("--report-dir", type=Path, default=PROJECT_ROOT / "data" / "suim_processed" / "quality_reports")
    parser.add_argument("--markdown-report", type=Path, default=PROJECT_ROOT / "reports" / "suim_v2_split_report.md")
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--include-size-repaired", action="store_true", help="Do not exclude the 37 unresolved mask-size repairs.")
    parser.add_argument("--include-conflicting-duplicate-labels", action="store_true", help="Do not exclude exact duplicate images with conflicting processed masks.")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    manifest = pd.read_csv(args.manifest)
    if manifest["sample_id"].duplicated().any():
        raise ValueError("Manifest has duplicate sample IDs.")
    manifest["image_sha256"] = [sha256_file(PROJECT_ROOT / path) for path in manifest["image_path"]]
    manifest["mask_pixels_sha256"] = [mask_pixels_sha256(PROJECT_ROOT / path) for path in manifest["mask_path"]]
    manifest["group_id"] = "sha256:" + manifest["image_sha256"]
    development = manifest.loc[manifest["partition"] == "train_val"].copy()
    official_test = manifest.loc[manifest["partition"] == "test"].copy()
    test_hashes = set(official_test["image_sha256"])
    conflicting_development_hashes = set(
        development.groupby("image_sha256")["mask_pixels_sha256"].nunique().loc[lambda values: values > 1].index
    )
    exclusion_rows: list[dict[str, object]] = []
    keep = np.ones(len(development), dtype=bool)
    for position, row in enumerate(development.itertuples(index=False)):
        reasons: list[str] = []
        if row.image_sha256 in test_hashes:
            reasons.append("exact_duplicate_of_official_test")
        if not args.include_size_repaired and bool(row.mask_resized_nearest):
            reasons.append("unresolved_mask_size_repair")
        if not args.include_conflicting_duplicate_labels and row.image_sha256 in conflicting_development_hashes:
            reasons.append("conflicting_labels_for_exact_duplicate_image")
        if reasons:
            keep[position] = False
            exclusion_rows.append({
                "sample_id": row.sample_id,
                "partition": row.partition,
                "image_path": row.image_path,
                "mask_path": row.mask_path,
                "image_sha256": row.image_sha256,
                "reason": ";".join(reasons),
            })
    retained = development.loc[keep].copy()
    if retained.empty:
        raise RuntimeError("All development samples were excluded.")
    train, validation, calibration = grouped_partition(retained, seed=args.seed)
    splits = {"train": train, "val": validation, "calibration": calibration, "test": official_test}
    validate_group_isolation(splits)
    development_hashes = set(pd.concat([train, validation, calibration])["image_sha256"])
    leaked_test_hashes = development_hashes.intersection(test_hashes)
    if leaked_test_hashes:
        raise ValueError(f"{len(leaked_test_hashes)} official-test image groups remain in development.")
    if args.output_dir.exists() and any(args.output_dir.iterdir()) and not args.overwrite:
        raise FileExistsError(f"{args.output_dir} exists; use --overwrite to regenerate it.")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    hashes: dict[str, str] = {}
    counts: dict[str, int] = {}
    class_rows: list[dict[str, object]] = []
    for split_name, frame in splits.items():
        ordered = frame.sort_values("sample_id").reset_index(drop=True)
        path = args.output_dir / f"{split_name}.csv"
        ordered.to_csv(path, index=False)
        hashes[split_name] = split_hash(path)
        counts[split_name] = len(ordered)
        histogram = pixel_histogram(ordered["mask_path"])
        class_rows.extend({"split": split_name, "class_id": class_id, "pixels": int(value)} for class_id, value in enumerate(histogram))
    exclusions = pd.DataFrame(exclusion_rows, columns=["sample_id", "partition", "image_path", "mask_path", "image_sha256", "reason"])
    args.report_dir.mkdir(parents=True, exist_ok=True)
    exclusions.to_csv(args.report_dir / "v2_excluded_samples.csv", index=False)
    pd.DataFrame(class_rows).to_csv(args.report_dir / "v2_class_pixel_distribution.csv", index=False)
    summary = {
        "split_version": "v2_grouped_deduplicated",
        "seed": args.seed,
        "official_test_immutable": True,
        "development_before_exclusion": int(len(development)),
        "development_after_exclusion": int(len(retained)),
        "excluded_samples": int(len(exclusions)),
        "excluded_by_reason": {str(key): int(value) for key, value in exclusions["reason"].value_counts().items()},
        "split_counts": counts,
        "split_sha256": hashes,
        "cross_split_group_leakage": False,
        "development_to_official_test_exact_duplicates": 0,
        "mask_size_repaired_included": bool(args.include_size_repaired),
        "conflicting_duplicate_labels_included": bool(args.include_conflicting_duplicate_labels),
    }
    (args.report_dir / "v2_grouped_deduplicated_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    (args.output_dir / "README.txt").write_text(
        "\n".join([
            "split_version=v2_grouped_deduplicated",
            f"seed={args.seed}",
            "Official TEST is immutable and never used for training, tuning, or calibration.",
            "Development samples with exact image duplicates in official TEST were excluded.",
            "Unresolved mask-size repairs were excluded by default.",
            "Exact duplicate development images with conflicting processed masks were excluded by default.",
            *(f"{name}={count}" for name, count in counts.items()),
            *(f"{name}_sha256={digest}" for name, digest in hashes.items()),
            "",
        ]), encoding="utf-8")
    args.markdown_report.parent.mkdir(parents=True, exist_ok=True)
    args.markdown_report.write_text(
        "\n".join([
            "# SUIM v2 grouped and deduplicated split", "",
            "This protocol keeps the official TEST partition immutable.",
            "Development samples with an exact SHA-256 image duplicate in official TEST are excluded.",
            "The unresolved 55-pixel mask-height repairs are excluded by default.",
            "Exact duplicate development images with conflicting processed masks are excluded by default.",
            "Remaining exact-image groups are assigned as indivisible units to train, validation, or calibration.", "",
            f"- Seed: `{args.seed}`", f"- Development before exclusion: {len(development)}",
            f"- Development after exclusion: {len(retained)}", f"- Excluded: {len(exclusions)}",
            *(f"- {name}: {count} (`{hashes[name]}`)" for name, count in counts.items()), "",
            "See `data/suim_processed/quality_reports/v2_excluded_samples.csv` and the duplicate-mask consistency audit for the complete evidence trail.", "",
        ]), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
