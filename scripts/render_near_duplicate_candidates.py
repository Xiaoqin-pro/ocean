"""Render side-by-side panels for manual near-duplicate review."""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
from PIL import Image

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--review-csv", type=Path, default=PROJECT_ROOT / "data" / "suim_processed" / "quality_reports" / "near_duplicate_review.csv")
    parser.add_argument("--output-dir", type=Path, default=PROJECT_ROOT / "data" / "suim_processed" / "quality_reports" / "manual_review" / "near_duplicates")
    parser.add_argument("--limit", type=int, default=100)
    args = parser.parse_args()
    candidates = pd.read_csv(args.review_csv)
    candidates = candidates.loc[candidates["review_decision"].fillna("").astype(str).str.strip().eq("pending")].head(args.limit)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    for rank, row in enumerate(candidates.itertuples(index=False), start=1):
        figure, axes = plt.subplots(1, 2, figsize=(12, 5))
        for axis, relative, partition, sample_id in ((axes[0], row.image_path_a, row.partition_a, row.sample_id_a), (axes[1], row.image_path_b, row.partition_b, row.sample_id_b)):
            with Image.open(PROJECT_ROOT / relative) as image:
                axis.imshow(image.convert("RGB"))
            axis.set_title(f"{partition}: {sample_id}")
            axis.axis("off")
        cosine = "n/a" if pd.isna(row.embedding_cosine) else f"{row.embedding_cosine:.4f}"
        figure.suptitle(f"#{rank:03d} | pHash={row.phash_distance}, dHash={row.dhash_distance}, cosine={cosine} | {row.candidate_sources}")
        figure.tight_layout()
        figure.savefig(args.output_dir / f"{rank:03d}_{row.pair_key}.png", dpi=160, bbox_inches="tight")
        plt.close(figure)


if __name__ == "__main__":
    main()
