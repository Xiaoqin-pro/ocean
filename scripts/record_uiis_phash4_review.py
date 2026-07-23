"""Record the completed visual review of the strict UIIS/SUIM pHash=4 set."""
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_REVIEWER = "Codex-GPT-5.6-visual-review"
DEFAULT_DATE = "2026-07-23"
DEFAULT_NOTE = (
    "Visual review: same underwater scene or adjacent frame; exclude from the "
    "independent UIIS confirmation pool."
)


def record_completed_review(
    review: pd.DataFrame,
    reviewer: str = DEFAULT_REVIEWER,
    review_date: str = DEFAULT_DATE,
) -> pd.DataFrame:
    required = {
        "uiis_sample_id",
        "suim_sample_id",
        "phash_distance",
        "review_decision",
        "reviewer",
        "review_date",
        "notes",
    }
    missing = required - set(review.columns)
    if missing:
        raise ValueError(f"Review CSV is missing required columns: {sorted(missing)}")
    if review.empty or set(review["phash_distance"].unique()) != {4}:
        raise ValueError("This recorder only accepts the non-empty pHash=4 review set.")

    completed = review.copy()
    completed["review_decision"] = "same_scene"
    completed["reviewer"] = reviewer
    completed["review_date"] = review_date
    completed["notes"] = DEFAULT_NOTE
    return completed


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--review-csv",
        type=Path,
        default=ROOT / "data" / "uiis_processed" / "quality_reports" / "uiis_suim_phash4_review.csv",
    )
    parser.add_argument("--reviewer", default=DEFAULT_REVIEWER)
    parser.add_argument("--review-date", default=DEFAULT_DATE)
    args = parser.parse_args()

    completed = record_completed_review(pd.read_csv(args.review_csv), args.reviewer, args.review_date)
    completed.to_csv(args.review_csv, index=False)
    print(f"recorded_same_scene_pairs={len(completed)}")


if __name__ == "__main__":
    main()
