import numpy as np
import torch

from metrics.uncertainty_ranking import SCORE_NAMES, ranking_metrics, uncertainty_scores
from scripts.evaluate_uncertainty_ranking import per_image_bootstrap_metrics


def test_uncertainty_scores_are_finite_and_aligned() -> None:
    logits = torch.randn((2, 3, 4, 5))
    scores = uncertainty_scores(logits, temperature=2.0)
    assert tuple(scores) == SCORE_NAMES
    assert all(value.shape == (2, 4, 5) and torch.isfinite(value).all() for value in scores.values())


def test_perfect_error_ranking_has_zero_excess_aurc() -> None:
    errors = np.array([False, False, True, True])
    metrics = ranking_metrics(np.array([0.1, 0.2, 0.8, 0.9]), errors)
    assert metrics["error_auroc"] == 1.0
    assert metrics["eaurc"] == 0.0
    assert metrics["top_10_uncertainty_precision"] == 1.0


def test_ranking_metrics_reject_invalid_scores() -> None:
    try:
        ranking_metrics(np.array([0.1, np.nan]), np.array([False, True]))
    except ValueError:
        return
    raise AssertionError("Expected non-finite uncertainty scores to be rejected.")


def test_vectorized_per_image_bootstrap_metrics_returns_each_region() -> None:
    scores = torch.tensor([[[0.1, 0.2], [0.8, 0.9]]])
    errors = torch.tensor([[[False, False], [True, True]]])
    regions = {"full": torch.ones_like(errors), "boundary": torch.tensor([[[True, False], [True, False]]]), "interior": torch.tensor([[[False, True], [False, True]]])}
    result = per_image_bootstrap_metrics(scores, errors, regions)
    assert set(result) == {"full", "boundary", "interior"}
    assert result["full"]["top_10_uncertainty_recall"].item() > 0.0
