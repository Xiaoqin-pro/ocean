import math

import torch

from metrics.calibration import aurc, brier_score, error_detection_auroc, expected_calibration_error, nll


def test_perfect_probabilities_have_zero_nll_brier_and_ece() -> None:
    probabilities = torch.tensor([[[[1.0, 0.0]], [[0.0, 1.0]]]])
    target = torch.tensor([[[0, 1]]])
    assert nll(probabilities, target) < 1e-12
    assert brier_score(probabilities, target) < 1e-12
    assert expected_calibration_error(probabilities, target) < 1e-12


def test_ignore_index_and_error_ranking() -> None:
    probabilities = torch.tensor([[[[0.9, 0.4, 0.1]], [[0.1, 0.6, 0.9]]]])
    target = torch.tensor([[[0, 0, 255]]])
    assert math.isclose(nll(probabilities, target), -math.log(0.9 * 0.4) / 2, rel_tol=1e-6)
    assert error_detection_auroc(probabilities, target) == 1.0
    assert 0 <= aurc(probabilities, target) <= 1
