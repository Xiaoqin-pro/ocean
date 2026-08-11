import pandas as pd
import pytest

from scripts.create_aquariskmap_risk_head_split import split_frame, validate_split


def scene_frame() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "sample_id": [f"sample_{index}" for index in range(24)],
            "scene_group_id": [f"scene_{index // 3}" for index in range(24)],
        }
    )


def test_risk_head_split_is_deterministic_and_scene_safe():
    source = scene_frame()
    train_a, development_a = split_frame(source, seed=20260725)
    train_b, development_b = split_frame(source, seed=20260725)
    validate_split(train_a, development_a, source)
    assert train_a.equals(train_b)
    assert development_a.equals(development_b)
    assert set(train_a["scene_group_id"]).isdisjoint(development_a["scene_group_id"])


def test_risk_head_split_requires_scene_groups():
    with pytest.raises(ValueError, match="scene_group_id"):
        split_frame(pd.DataFrame({"sample_id": ["only"]}))
