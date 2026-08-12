import numpy as np
import pandas as pd

from scripts.select_uiis_alpha010_split import assert_protocol, candidate_split, score_split


def test_candidate_split_is_scene_group_safe():
    frame = pd.DataFrame(
        {
            "sample_id": [f"sample_{index}" for index in range(20)],
            "scene_group_id": [f"group_{index // 2}" for index in range(20)],
        }
    )
    train, calibration, confirmation = candidate_split(frame, seed=7)
    splits = {
        "train": frame.iloc[train],
        "calibration": frame.iloc[calibration],
        "confirmation": frame.iloc[confirmation],
    }
    assert_protocol(splits)
    assert sum(len(split) for split in splits.values()) == len(frame)


def test_score_penalizes_missing_class():
    values = np.ones((6, 8), dtype=np.int64)
    values[:2, 7] = 0
    complete = (np.array([0, 2, 4]), np.array([1, 3]), np.array([5]))
    missing = (np.array([0, 1, 2]), np.array([3, 4]), np.array([5]))
    assert score_split(missing, values) > score_split(complete, values)
