"""Render the most similar cross-split pHash candidates for human review."""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
from PIL import Image

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--processed-dir", type=Path, default=PROJECT_ROOT / "data" / "suim_processed")
    parser.add_argument("--per-comparison", type=int, default=5)
    args = parser.parse_args()
    report_dir = args.processed_dir / "quality_reports"
    pairs = pd.read_csv(report_dir / "cross_split_near_duplicate_pairs.csv")
    selected = pairs.groupby("comparison", group_keys=False).head(args.per_comparison).reset_index(drop=True)
    rows = len(selected)
    figure, axes = plt.subplots(rows, 2, figsize=(12, max(3, rows * 3.3)))
    if rows == 1:
        axes = [axes]
    for axis_pair, row in zip(axes, selected.itertuples(index=False)):
        for axis, relative, split, sample_id in (
            (axis_pair[0], row.image_path_a, row.split_a, row.sample_id_a),
            (axis_pair[1], row.image_path_b, row.split_b, row.sample_id_b),
        ):
            with Image.open(PROJECT_ROOT / relative) as image:
                axis.imshow(image.convert("RGB"))
            axis.set_title(f"{split}: {sample_id}\npHash distance={row.phash_hamming_distance}")
            axis.axis("off")
    figure.suptitle("Cross-split pHash candidates: visual review required", y=1.0)
    figure.tight_layout()
    figure.savefig(report_dir / "cross_split_near_duplicate_review.png", dpi=180, bbox_inches="tight")
    plt.close(figure)


if __name__ == "__main__":
    main()
