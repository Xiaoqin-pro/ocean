import numpy as np
import pytest

from reliability.conformal_risk import select_crc_coverage, select_naive_coverage


def test_crc_matches_official_finite_sample_rule_when_a_violation_exists():
    losses = np.array([[0.01, 0.08, 0.20], [0.03, 0.10, 0.30]])
    grid = np.array([0.2, 0.5, 1.0])
    selected = select_crc_coverage(losses, grid, 0.15)
    # n=2: corrected risks are 0.353..., 0.393..., 0.5, so first point is
    # already too risky and the official algorithm clamps to index zero.
    assert selected.index == 0
    assert selected.coverage == 0.2
    assert selected.corrected_risk == pytest.approx((2 / 3) * 0.02 + 1 / 3)


def test_crc_all_low_risk_selects_largest_coverage():
    selected = select_crc_coverage(np.full((100, 3), 0.01), np.array([0.2, 0.5, 1.0]), 0.10)
    assert selected.index == 2
    assert selected.coverage == 1.0


def test_naive_can_select_more_coverage_than_crc():
    losses = np.array([[0.01, 0.08, 0.20], [0.03, 0.10, 0.30]])
    grid = np.array([0.2, 0.5, 1.0])
    assert select_naive_coverage(losses, grid, 0.15).coverage >= select_crc_coverage(losses, grid, 0.15).coverage


def test_crc_rejects_bad_direction_and_non_monotone_loss():
    with pytest.raises(ValueError):
        select_crc_coverage(np.array([[0.2, 0.1]]), np.array([0.5, 1.0]), 0.1)
    with pytest.raises(ValueError):
        select_crc_coverage(np.ones((2, 2)) * 0.1, np.array([0.5, 1.0]), 0.1, monotonic_direction="decreasing")

