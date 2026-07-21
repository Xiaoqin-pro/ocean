"""Make the label-repair and RGB-quantization decisions auditable."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--processed-dir", type=Path, default=PROJECT_ROOT / "data" / "suim_processed")
    args = parser.parse_args()
    processed_dir = args.processed_dir
    manifest_path = processed_dir / "manifest.csv"
    manifest = pd.read_csv(manifest_path)
    manifest["total_pixels"] = manifest["original_mask_width"] * manifest["original_mask_height"]
    manifest["quantized_pixels"] = manifest["non_exact_rgb_pixels"]
    manifest["quantized_ratio"] = manifest["quantized_pixels"] / manifest["total_pixels"]
    manifest["size_repaired"] = manifest["mask_resized_nearest"].astype(bool)
    manifest["repair_method"] = manifest["size_repaired"].map({True: "class_index_nearest_resize", False: "none"})
    manifest.to_csv(manifest_path, index=False)
    split_dir = processed_dir / "splits" / "v1_seed_20260721"
    split_lookup: dict[str, str] = {}
    for split_name in ("train", "val", "calibration", "test"):
        for sample_id in pd.read_csv(split_dir / f"{split_name}.csv")["sample_id"]:
            split_lookup[str(sample_id)] = split_name
    manifest["split"] = manifest["sample_id"].map(split_lookup)
    report_dir = processed_dir / "quality_reports"
    report_dir.mkdir(parents=True, exist_ok=True)
    repaired = manifest.loc[manifest["size_repaired"], [
        "sample_id", "partition", "split", "width", "height", "original_mask_width", "original_mask_height", "repair_method",
    ]].sort_values(["split", "sample_id"])
    repaired.to_csv(report_dir / "size_repaired_samples.csv", index=False)
    quantized = manifest[["sample_id", "partition", "split", "total_pixels", "quantized_pixels", "quantized_ratio"]].sort_values("quantized_ratio", ascending=False)
    quantized.head(20).to_csv(report_dir / "quantized_ratio_top20.csv", index=False)
    stats = {
        "samples": int(len(manifest)),
        "quantized_pixels": int(manifest["quantized_pixels"].sum()),
        "total_pixels": int(manifest["total_pixels"].sum()),
        "overall_quantized_ratio": float(manifest["quantized_pixels"].sum() / manifest["total_pixels"].sum()),
        "mean_quantized_ratio": float(manifest["quantized_ratio"].mean()),
        "median_quantized_ratio": float(manifest["quantized_ratio"].median()),
        "p95_quantized_ratio": float(manifest["quantized_ratio"].quantile(0.95)),
        "max_quantized_ratio": float(manifest["quantized_ratio"].max()),
        "size_repaired_samples": int(len(repaired)),
        "size_repaired_by_split": {str(key): int(value) for key, value in repaired["split"].value_counts().sort_index().items()},
    }
    with (report_dir / "quantization_summary.json").open("w", encoding="utf-8") as handle:
        json.dump(stats, handle, ensure_ascii=False, indent=2)
    print(json.dumps(stats, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
