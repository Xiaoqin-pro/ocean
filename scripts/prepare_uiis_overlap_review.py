"""Apply conservative automatic UIIS/SUIM exclusions and export the tiny review set."""
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare a conservative UIIS/SUIM overlap review without creating a split.")
    parser.add_argument("--report-dir", type=Path, default=ROOT / "data" / "uiis_processed" / "quality_reports")
    args = parser.parse_args()
    exact = pd.read_csv(args.report_dir / "uiis_vs_suim_exact_duplicates.csv")
    near = pd.read_csv(args.report_dir / "uiis_vs_suim_near_duplicate_review.csv")
    if set(near["phash_distance"].unique()) - {0, 2, 4}:
        raise ValueError("Unexpected pHash distances; do not silently change the review policy.")
    exact_ids = set(exact["uiis_sample_id"])
    conservative = near.loc[near["phash_distance"] <= 2].copy()
    conservative["decision"] = "exclude"
    conservative["reason"] = conservative.apply(lambda row: "exact_sha256_overlap_with_suim" if row.uiis_sample_id in exact_ids else "conservative_phash_le_2_overlap_screen", axis=1)
    exclusions = conservative[["uiis_sample_id", "decision", "reason"]].drop_duplicates("uiis_sample_id").sort_values("uiis_sample_id", kind="stable")
    review = near.loc[near["phash_distance"] == 4].copy()
    review["review_decision"] = "pending"
    review["reviewer"] = ""
    review["review_date"] = ""
    review["notes"] = "pHash=4 requires visual confirmation before UIIS admission."
    review = review.sort_values(["suim_partition", "uiis_sample_id", "suim_sample_id"], kind="stable")
    exclusions.to_csv(args.report_dir / "uiis_automatic_suim_exclusions.csv", index=False)
    review.to_csv(args.report_dir / "uiis_suim_phash4_review.csv", index=False)
    print(f"automatic_excluded_uiis_images={len(exclusions)}")
    print(f"manual_review_pairs={len(review)}")


if __name__ == "__main__":
    main()
