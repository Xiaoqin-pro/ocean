import itertools

import numpy as np
import pytest
import torch
from sklearn.metrics import average_precision_score

from calibration.ranking_fusion import FEATURE_NAMES, fit_logistic_fusion, fusion_score
from metrics.uncertainty_ranking import SCORE_NAMES, ranking_metrics, uncertainty_scores
from scripts.evaluate_uncertainty_ranking import validate_benchmark_protocol


def test_uncertainty_scores_are_float32_finite_and_aligned() -> None:
    logits = torch.randn((2, 3, 4, 5), dtype=torch.float16)
    scores = uncertainty_scores(logits, temperature=2.0)
    assert tuple(scores) == SCORE_NAMES
    assert all(value.shape == (2, 4, 5) and value.dtype is torch.float32 and torch.isfinite(value).all() for value in scores.values())


def test_perfect_error_ranking_has_zero_excess_aurc() -> None:
    errors = np.array([False, False, True, True])
    metrics = ranking_metrics(np.array([0.1, 0.2, 0.8, 0.9]), errors)
    assert metrics["error_auroc"] == 1.0
    assert abs(metrics["eaurc"]) < 1e-12


@pytest.mark.parametrize(
    "scores,errors",
    [
        (np.array([0.1, 0.2, 0.8, 0.9]), np.array([0, 0, 1, 1])),
        (np.array([0.1, 0.1, 0.8, 0.8]), np.array([0, 1, 0, 1])),
        (np.ones(5), np.array([0, 0, 1, 0, 1])),
        (np.array([0.1, 0.2, 0.3, 0.4, 0.5]), np.array([0, 0, 0, 0, 1])),
        (np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6]), np.array([0, 1, 0, 1, 0, 1])),
    ],
)
def test_tie_group_auprc_matches_sklearn(scores: np.ndarray, errors: np.ndarray) -> None:
    actual = ranking_metrics(scores, errors)["error_auprc"]
    expected = average_precision_score(errors, scores)
    assert actual == pytest.approx(expected, abs=1e-12)


def test_tie_aware_aurc_matches_average_over_tie_permutations() -> None:
    scores = np.array([0.0, 1.0, 1.0, 2.0])
    errors = np.array([0, 1, 0, 1], dtype=bool)
    observed = ranking_metrics(scores, errors)["aurc"]
    values = []
    for tied_errors in itertools.permutations(errors[1:3]):
        sequence = np.array([errors[0], *tied_errors, errors[3]], dtype=float)
        risk = np.cumsum(sequence) / np.arange(1, 5)
        values.append(np.trapezoid(risk, np.arange(1, 5) / 4))
    assert observed == pytest.approx(float(np.mean(values)), abs=1e-12)


def test_tie_aware_cutoffs_use_expected_selection() -> None:
    scores = np.array([0.0, 1.0, 1.0, 1.0])
    errors = np.array([0, 1, 0, 1], dtype=bool)
    metrics = ranking_metrics(scores, errors, coverages=(0.5,), top_fractions=(0.5,))
    assert metrics["risk_at_50_coverage"] == pytest.approx(1.0 / 3.0)
    assert metrics["top_50_uncertainty_precision"] == pytest.approx(2.0 / 3.0)
    assert metrics["top_50_uncertainty_recall"] == pytest.approx(2.0 / 3.0)


def test_local_disagreement_histogram_matches_grouped_sort() -> None:
    scores = np.array([0.0, 1 / 9, 1 / 9, 5 / 9, 5 / 9, 5 / 9], dtype=np.float32)
    errors = np.array([0, 0, 1, 1, 0, 1], dtype=bool)
    histogram = ranking_metrics(scores, errors, discrete_histogram=True)
    sorted_metrics = ranking_metrics(scores, errors)
    for name in ("error_auroc", "error_auprc", "aurc", "eaurc", "top_10_uncertainty_recall"):
        assert histogram[name] == pytest.approx(sorted_metrics[name], abs=1e-12)


def test_ranking_metrics_reject_invalid_scores_and_test_protocol() -> None:
    with pytest.raises(ValueError):
        ranking_metrics(np.array([0.1, np.nan]), np.array([False, True]))
    with pytest.raises(ValueError, match="official TEST"):
        validate_benchmark_protocol({"experiment": {"splits": ["val", "test"], "conditions": ["clean"]}})


def test_logistic_fusion_is_calibration_only_and_scores_errors_higher() -> None:
    features = np.array([[0.05, 0.0, 0.1], [0.10, 0.0, 0.2], [0.8, 0.7, 0.9], [0.9, 0.8, 0.95]], dtype=np.float32)
    scaler, classifier, parameters = fit_logistic_fusion(features, np.array([0, 0, 1, 1]), c=1.0)
    scores = fusion_score(features, scaler, classifier)
    assert parameters.feature_names == FEATURE_NAMES
    assert scores[2:].mean() > scores[:2].mean()
