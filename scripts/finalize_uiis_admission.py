"""Create a leakage-screened UIIS admission manifest before any split or training."""
from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
ALLOWED_REVIEW_DECISIONS = {"same_scene", "different_scene"}


def _external_exclusions(automatic: pd.DataFrame, review: pd.DataFrame) -> pd.DataFrame:
    required_auto = {"uiis_sample_id", "decision", "reason"}
    required_review = {"uiis_sample_id", "suim_sample_id", "review_decision", "reviewer", "review_date"}
    if missing := required_auto - set(automatic.columns):
        raise ValueError(f"Automatic exclusion CSV missing columns: {sorted(missing)}")
    if missing := required_review - set(review.columns):
        raise ValueError(f"Review CSV missing columns: {sorted(missing)}")
    if set(automatic["decision"].unique()) - {"exclude"}:
        raise ValueError("Automatic exclusion decisions must all be 'exclude'.")
    decisions = set(review["review_decision"].dropna().unique())
    invalid = decisions - ALLOWED_REVIEW_DECISIONS
    if invalid:
        raise ValueError(f"Unexpected review decision(s): {sorted(invalid)}")
    if review["review_decision"].isna().any() or (review["review_decision"] == "pending").any():
        raise ValueError("UIIS admission cannot proceed while a visual review remains pending.")
    same_scene = review.loc[review["review_decision"] == "same_scene"].copy()
    if ((same_scene["reviewer"].fillna("") == "") | (same_scene["review_date"].fillna("") == "")).any():
        raise ValueError("Completed same_scene reviews require reviewer and review_date.")

    auto = automatic[["uiis_sample_id", "reason"]].copy()
    auto["exclusion_type"] = "suim_overlap_automatic"
    auto["evidence"] = "exact SHA-256 or conservative pHash distance <= 2"
    auto["related_sample_id"] = ""

    manual = same_scene[["uiis_sample_id", "suim_sample_id"]].copy()
    manual = manual.rename(columns={"suim_sample_id": "related_sample_id"})
    manual["reason"] = "visual_phash4_same_scene_with_suim"
    manual["exclusion_type"] = "suim_overlap_visual_review"
    manual["evidence"] = "pHash distance = 4; completed visual review"
    combined = pd.concat([auto, manual], ignore_index=True)
    return combined.sort_values(["uiis_sample_id", "exclusion_type"], kind="stable").drop_duplicates("uiis_sample_id", keep="first")


def build_admission_manifest(manifest: pd.DataFrame, automatic: pd.DataFrame, review: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    required_manifest = {"sample_id", "source_partition", "image_sha256"}
    if missing := required_manifest - set(manifest.columns):
        raise ValueError(f"UIIS manifest missing columns: {sorted(missing)}")
    if manifest["sample_id"].duplicated().any():
        raise ValueError("UIIS manifest contains duplicated sample_id values.")

    exclusions = _external_exclusions(automatic, review)
    internal_rows: list[dict[str, str]] = []
    duplicate_groups = 0
    for _, group in manifest.groupby("image_sha256", sort=True):
        if len(group) < 2:
            continue
        duplicate_groups += 1
        ordered = group.sort_values("sample_id", kind="stable")
        keep_id = ordered.iloc[0]["sample_id"]
        for sample_id in ordered.iloc[1:]["sample_id"]:
            internal_rows.append(
                {
                    "uiis_sample_id": sample_id,
                    "reason": "internal_exact_sha256_duplicate",
                    "exclusion_type": "uiis_internal_exact_duplicate",
                    "evidence": "identical raw-image SHA-256; deterministic representative retained",
                    "related_sample_id": keep_id,
                }
            )
    if internal_rows:
        exclusions = pd.concat([exclusions, pd.DataFrame(internal_rows)], ignore_index=True)
    exclusions = exclusions.sort_values(["uiis_sample_id", "exclusion_type"], kind="stable").drop_duplicates("uiis_sample_id", keep="first")

    unknown = set(exclusions["uiis_sample_id"]) - set(manifest["sample_id"])
    if unknown:
        raise ValueError(f"Exclusions name UIIS samples absent from manifest: {sorted(unknown)[:5]}")
    admitted = manifest.loc[~manifest["sample_id"].isin(exclusions["uiis_sample_id"])].copy()
    admitted["admission_status"] = "include"
    admitted["admission_reason"] = "cleared_suim_overlap_and_exact_duplicate_screen"
    metadata = {
        "protocol_status": "admission_finalized_before_split",
        "official_suim_test_evaluated": False,
        "official_suim_test_used_for_duplicate_audit_only": True,
        "source_images": int(len(manifest)),
        "admitted_images": int(len(admitted)),
        "excluded_images": int(len(exclusions)),
        "exclusions_by_type": dict(Counter(exclusions["exclusion_type"])),
        "admitted_by_source_partition": dict(Counter(admitted["source_partition"])),
        "internal_exact_duplicate_groups": duplicate_groups,
        "manual_phash4_review_status": "complete",
        "manual_same_scene_pairs": int((review["review_decision"] == "same_scene").sum()),
        "manual_different_scene_pairs": int((review["review_decision"] == "different_scene").sum()),
    }
    return admitted.sort_values("sample_id", kind="stable"), exclusions, metadata


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=ROOT / "data" / "uiis_processed" / "manifest.csv")
    parser.add_argument("--report-dir", type=Path, default=ROOT / "data" / "uiis_processed" / "quality_reports")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "data" / "uiis_processed" / "admission")
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    admitted, exclusions, metadata = build_admission_manifest(
        pd.read_csv(args.manifest),
        pd.read_csv(args.report_dir / "uiis_automatic_suim_exclusions.csv"),
        pd.read_csv(args.report_dir / "uiis_suim_phash4_review.csv"),
    )
    admitted.to_csv(args.output_dir / "admitted_manifest.csv", index=False)
    exclusions.to_csv(args.output_dir / "admission_exclusions.csv", index=False)
    (args.output_dir / "admission_metadata.json").write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(metadata, indent=2))


if __name__ == "__main__":
    main()
