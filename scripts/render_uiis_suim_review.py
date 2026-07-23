"""Render side-by-side UIIS/SUIM pHash=4 review candidates."""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    parser = argparse.ArgumentParser(description="Render strict UIIS/SUIM near-duplicate review pairs.")
    parser.add_argument("--review-csv", type=Path, default=ROOT / "data" / "uiis_processed" / "quality_reports" / "uiis_suim_phash4_review.csv")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "data" / "uiis_processed" / "quality_reports" / "manual_review" / "uiis_suim_phash4")
    args = parser.parse_args()
    table = pd.read_csv(args.review_csv)
    if table.empty or set(table["phash_distance"]) != {4}:
        raise ValueError("Expected only pending pHash=4 review pairs.")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    for index, row in enumerate(table.itertuples(index=False), start=1):
        with Image.open(ROOT / row.uiis_image_path) as image:
            left = image.convert("RGB")
        with Image.open(ROOT / row.suim_image_path) as image:
            right = image.convert("RGB")
        figure, axes = plt.subplots(1, 2, figsize=(10, 5))
        axes[0].imshow(left); axes[0].set_title(f"UIIS: {row.uiis_sample_id}"); axes[0].axis("off")
        axes[1].imshow(right); axes[1].set_title(f"SUIM {row.suim_partition}: {row.suim_sample_id}\npHash distance=4"); axes[1].axis("off")
        figure.tight_layout()
        figure.savefig(args.output_dir / f"{index:03d}_{row.uiis_sample_id}__{row.suim_sample_id}.png", dpi=150)
        plt.close(figure)
    print(args.output_dir)


if __name__ == "__main__":
    main()
