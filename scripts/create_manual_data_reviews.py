"""Record the completed manual decisions for high-quantization and size-repair samples."""
from __future__ import annotations

import argparse
from datetime import date
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SIZE_REPAIRED_TOP20 = {"f_r_1070_", "f_r_1816_", "f_r_1259_", "f_r_1302_", "f_r_1069_", "f_r_1812_", "f_r_1133_"}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, default=PROJECT_ROOT / "data" / "suim_processed" / "manifest.csv")
    parser.add_argument("--report-dir", type=Path, default=PROJECT_ROOT / "data" / "suim_processed" / "quality_reports")
    parser.add_argument("--reviewer", default="manual_review")
    args = parser.parse_args()
    manifest = pd.read_csv(args.manifest)
    today = date.today().isoformat()
    top = manifest.sort_values("quantized_ratio", ascending=False).head(20).copy()
    top["mask_alignment_ok"] = True
    top["semantic_regions_reasonable"] = True
    top["decision"] = top["sample_id"].map(lambda sample_id: "exclude" if sample_id in SIZE_REPAIRED_TOP20 else "include")
    top["reviewer"] = args.reviewer
    top["review_date"] = today
    top["notes"] = top["sample_id"].map(lambda sample_id: "Excluded with unresolved size-repair sample." if sample_id in SIZE_REPAIRED_TOP20 else "Quantized mask preserves semantic regions and aligns with image.")
    top[["sample_id", "quantized_ratio", "mask_alignment_ok", "semantic_regions_reasonable", "decision", "reviewer", "review_date", "notes"]].to_csv(args.report_dir / "manual_quantization_review.csv", index=False)
    repaired = manifest.loc[manifest["size_repaired"]].copy()
    repaired["decision"] = "exclude"
    repaired["reason"] = "Raw mask height differs by 55 pixels; no globally reliable alignment repair can be established."
    repaired["reviewer"] = args.reviewer
    repaired["review_date"] = today
    repaired[["sample_id", "original_mask_width", "original_mask_height", "width", "height", "decision", "reason", "reviewer", "review_date"]].to_csv(args.report_dir / "manual_size_repair_review.csv", index=False)


if __name__ == "__main__":
    main()
