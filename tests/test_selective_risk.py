import numpy as np

from reliability.selective_risk import curve_summary, selective_risk_curve


def test_selective_risk_handles_extreme_predictions_and_full_coverage():
    grid = np.array([0.25, 0.5, 1.0])
    assert np.allclose(selective_risk_curve(np.arange(4), np.zeros(4, dtype=bool), grid), 0.0)
    assert np.allclose(selective_risk_curve(np.arange(4), np.ones(4, dtype=bool), grid), 1.0)


def test_selective_risk_is_raster_order_and_tie_invariant():
    scores = np.array([0.1, 0.1, 0.4, 0.4])
    errors = np.array([False, True, False, True])
    grid = np.array([0.25, 0.5, 0.75, 1.0])
    baseline = selective_risk_curve(scores, errors, grid)
    permutation = np.array([3, 2, 1, 0])
    assert np.allclose(baseline, selective_risk_curve(scores[permutation], errors[permutation], grid))
    # A 50% tie-safe selection contains one expected error out of two pixels.
    assert baseline[1] == 0.5


def test_monotone_envelope_dominates_and_is_monotone():
    risks, envelope = curve_summary(np.arange(5), np.array([False, True, False, False, True]), np.array([0.2, 0.4, 0.6, 0.8, 1.0]))
    assert np.all(envelope >= risks)
    assert np.all(np.diff(envelope) >= 0)

