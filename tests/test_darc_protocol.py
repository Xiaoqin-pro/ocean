import numpy as np
import pandas as pd
import pytest

from scripts.fit_darc_crc import cluster_mean_losses, grouped_losses


def test_cluster_mean_binds_all_13_conditions_to_one_independent_sample():
    curves = np.arange(13 * 2 * 3, dtype=float).reshape(13, 2, 3) / 100
    result = cluster_mean_losses(curves)
    assert result.shape == (2, 3)
    assert np.allclose(result, curves.mean(axis=0))


def test_quality_group_losses_count_a_sample_once_even_with_many_versions():
    conditions = ["clean", "color_s1", "color_s2"]
    sample_ids = ["a", "b"]
    curves = np.array([
        [[0.1, 0.2], [0.2, 0.3]],
        [[0.3, 0.4], [0.4, 0.5]],
        [[0.5, 0.6], [0.6, 0.7]],
    ])
    assignments = pd.DataFrame([
        {"sample_id": "a", "condition": "clean", "quality_group": 0},
        {"sample_id": "a", "condition": "color_s1", "quality_group": 0},
        {"sample_id": "a", "condition": "color_s2", "quality_group": 1},
        {"sample_id": "b", "condition": "clean", "quality_group": 1},
        {"sample_id": "b", "condition": "color_s1", "quality_group": 1},
        {"sample_id": "b", "condition": "color_s2", "quality_group": 1},
    ])
    group_zero = grouped_losses(curves, assignments, conditions=conditions, sample_ids=sample_ids, group=0)
    assert group_zero.shape == (1, 2)
    assert np.allclose(group_zero[0], (curves[0, 0] + curves[1, 0]) / 2)


def test_grouped_losses_rejects_duplicate_condition_assignments():
    with pytest.raises(ValueError):
        grouped_losses(np.ones((13, 1, 2)) * 0.1, pd.DataFrame([{"sample_id": "a", "condition": "x", "quality_group": 0}, {"sample_id": "a", "condition": "x", "quality_group": 0}]), conditions=["x"] * 13, sample_ids=["a"], group=0)

