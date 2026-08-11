import pandas as pd
import pytest

from scripts.finalize_uiis_admission import build_admission_manifest
from scripts.record_uiis_phash4_review import record_completed_review


def _manifest() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "sample_id": ["uiis_a", "uiis_b", "uiis_c", "uiis_d"],
            "source_partition": ["train", "train", "val", "val"],
            "image_sha256": ["unique", "duplicate", "duplicate", "other"],
        }
    )


def _automatic() -> pd.DataFrame:
    return pd.DataFrame({"uiis_sample_id": ["uiis_a"], "decision": ["exclude"], "reason": ["phash"]})


def _review(decision: str = "same_scene") -> pd.DataFrame:
    return pd.DataFrame(
        {
            "uiis_sample_id": ["uiis_d"],
            "suim_sample_id": ["suim_x"],
            "phash_distance": [4],
            "review_decision": [decision],
            "reviewer": ["reviewer" if decision != "pending" else ""],
            "review_date": ["2026-07-23" if decision != "pending" else ""],
            "notes": ["note"],
        }
    )


def test_completed_visual_review_records_all_pairs_as_same_scene():
    completed = record_completed_review(_review("pending"))
    assert set(completed["review_decision"]) == {"same_scene"}
    assert set(completed["reviewer"]) == {"Codex-GPT-5.6-visual-review"}


def test_admission_excludes_suim_overlap_and_internal_exact_duplicate():
    admitted, exclusions, metadata = build_admission_manifest(_manifest(), _automatic(), _review())
    assert admitted["sample_id"].tolist() == ["uiis_b"]
    assert set(exclusions["uiis_sample_id"]) == {"uiis_a", "uiis_c", "uiis_d"}
    assert metadata["excluded_images"] == 3
    assert metadata["manual_same_scene_pairs"] == 1


def test_admission_rejects_pending_visual_review():
    with pytest.raises(ValueError, match="pending"):
        build_admission_manifest(_manifest(), _automatic(), _review("pending"))
