import pandas as pd
import pytest

from scripts.build_scene_groups import validate_reviews


def review(decision: str, reviewer: str = "reviewer", review_date: str = "2026-07-22") -> pd.DataFrame:
    return pd.DataFrame([{"pair_key": "a__b", "review_decision": decision, "reviewer": reviewer, "review_date": review_date, "notes": ""}])


def test_pending_review_is_valid_without_reviewer() -> None:
    validate_reviews(review("pending", "", ""))


def test_decided_review_requires_reviewer_and_date() -> None:
    with pytest.raises(ValueError, match="reviewer"):
        validate_reviews(review("same_scene", "", ""))


def test_unknown_review_decision_is_rejected() -> None:
    with pytest.raises(ValueError, match="Invalid review_decision"):
        validate_reviews(review("uncertain"))
