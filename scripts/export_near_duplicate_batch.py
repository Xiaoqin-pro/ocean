"""Export a non-overlapping batch of pending near-duplicate pairs for review."""
from __future__ import annotations

import argparse
import shutil
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
ALLOWED_DECISIONS = {"pending", "same_scene", "different_scene"}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--review-csv", type=Path, default=PROJECT_ROOT / "data" / "suim_processed" / "quality_reports" / "near_duplicate_review.csv")
    parser.add_argument("--batch-size", type=int, default=100)
    parser.add_argument("--batch-index", type=int, default=1, help="One-based pending-pair batch number.")
    parser.add_argument("--output-dir", type=Path, default=PROJECT_ROOT / "data" / "suim_processed" / "quality_reports" / "review_batches")
    args = parser.parse_args()
    if args.batch_index < 1 or args.batch_size < 1:
        raise ValueError("batch-index and batch-size must be positive.")
    review = pd.read_csv(args.review_csv)
    decisions = review["review_decision"].fillna("").astype(str).str.strip()
    invalid = sorted(set(decisions).difference(ALLOWED_DECISIONS))
    if invalid:
        raise ValueError(f"Invalid review decisions: {invalid}")
    pending = review.loc[decisions.eq("pending")].copy()
    start = (args.batch_index - 1) * args.batch_size
    batch = pending.iloc[start:start + args.batch_size].copy()
    if batch.empty:
        raise ValueError(f"No pending candidates in batch {args.batch_index}.")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = args.output_dir / f"near_duplicate_batch_{args.batch_index:03d}.csv"
    batch.to_csv(csv_path, index=False)
    print(f"batch_csv={csv_path}")
    print(f"pending_total={len(pending)} batch_count={len(batch)} range={start + 1}-{start + len(batch)}")
    print("Render panels with:")
    print(f"  python scripts/render_near_duplicate_candidates.py --review-csv {csv_path} --output-dir {args.output_dir / f'batch_{args.batch_index:03d}_images'} --limit {len(batch)}")


if __name__ == "__main__":
    main()
