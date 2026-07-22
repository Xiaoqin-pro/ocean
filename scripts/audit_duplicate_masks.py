"""Audit mask consistency for byte-identical SUIM RGB images."""
from __future__ import annotations

import argparse
import hashlib
import itertools
import json
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def mask_comparison(first: Path, second: Path) -> tuple[bool, int | None]:
    with Image.open(first) as image:
        first_array = np.asarray(image, dtype=np.uint8)
    with Image.open(second) as image:
        second_array = np.asarray(image, dtype=np.uint8)
    if first_array.shape != second_array.shape:
        return False, None
    return bool(np.array_equal(first_array, second_array)), int(np.count_nonzero(first_array != second_array))


def main() -> None:
    parser = argparse.ArgumentParser(description="Audit masks attached to byte-identical images.")
    parser.add_argument("--manifest", type=Path, default=PROJECT_ROOT / "data" / "suim_processed" / "manifest.csv")
    parser.add_argument("--output-dir", type=Path, default=PROJECT_ROOT / "data" / "suim_processed" / "quality_reports")
    parser.add_argument("--v1-split-dir", type=Path, default=PROJECT_ROOT / "data" / "suim_processed" / "splits" / "v1_seed_20260721")
    args = parser.parse_args()
    manifest = pd.read_csv(args.manifest)
    v1_lookup: dict[str, str] = {}
    for split_name in ("train", "val", "calibration", "test"):
        split_path = args.v1_split_dir / f"{split_name}.csv"
        if split_path.is_file():
            for sample_id in pd.read_csv(split_path)["sample_id"]:
                v1_lookup[str(sample_id)] = split_name
    manifest["v1_split"] = manifest["sample_id"].map(v1_lookup).fillna("not_assigned")
    manifest["image_sha256"] = [sha256_file(PROJECT_ROOT / path) for path in manifest["image_path"]]
    rows: list[dict[str, object]] = []
    for image_hash, group in manifest.groupby("image_sha256", sort=True):
        if len(group) < 2:
            continue
        for first, second in itertools.combinations(group.itertuples(index=False), 2):
            masks_identical, changed_pixels = mask_comparison(
                PROJECT_ROOT / first.mask_path, PROJECT_ROOT / second.mask_path
            )
            rows.append({
                "image_sha256": image_hash,
                "sample_id_a": first.sample_id,
                "partition_a": first.partition,
                "v1_split_a": first.v1_split,
                "mask_path_a": first.mask_path,
                "sample_id_b": second.sample_id,
                "partition_b": second.partition,
                "v1_split_b": second.v1_split,
                "mask_path_b": second.mask_path,
                "same_partition": first.partition == second.partition,
                "cross_v1_split": first.v1_split != second.v1_split,
                "mask_sha256_a": sha256_file(PROJECT_ROOT / first.mask_path),
                "mask_sha256_b": sha256_file(PROJECT_ROOT / second.mask_path),
                "masks_identical": masks_identical,
                "changed_mask_pixels": changed_pixels,
            })
    report = pd.DataFrame(rows)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    report.to_csv(args.output_dir / "duplicate_image_mask_consistency.csv", index=False)
    cross_partition = report.loc[~report["same_partition"]] if not report.empty else report
    cross_v1 = report.loc[report["cross_v1_split"]] if not report.empty else report
    summary = {
        "duplicate_image_pairs": int(len(report)),
        "cross_partition_duplicate_image_pairs": int(len(cross_partition)),
        "cross_partition_mask_identical_pairs": int(cross_partition["masks_identical"].sum()) if len(cross_partition) else 0,
        "cross_partition_mask_inconsistent_pairs": int((~cross_partition["masks_identical"]).sum()) if len(cross_partition) else 0,
        "cross_v1_split_duplicate_image_pairs": int(len(cross_v1)),
        "cross_v1_split_mask_identical_pairs": int(cross_v1["masks_identical"].sum()) if len(cross_v1) else 0,
        "cross_v1_split_mask_inconsistent_pairs": int((~cross_v1["masks_identical"]).sum()) if len(cross_v1) else 0,
    }
    (args.output_dir / "duplicate_image_mask_consistency_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
