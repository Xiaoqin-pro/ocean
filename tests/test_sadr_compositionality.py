import pytest
import torch

from scripts.analyze_sadr_composition_controls import pixel_order_metrics, relation_metrics
from scripts.evaluate_sadr_self_gate import selector_mask
from reliability.sadr_seg import CompositionalBasisSADRFrontEnd
from reliability.sadr_seg import (
    semantic_compositionality_loss,
    semantic_order_consistency_loss,
    semantic_probability_order_loss,
)


def test_semantic_compositionality_loss_is_zero_for_matching_target():
    first = torch.randn(2, 8, 4, 4)
    second = torch.randn(2, 8, 4, 4)
    composite = first + second
    value = semantic_compositionality_loss(composite, first, second)
    assert value.item() < 1e-6


def test_semantic_order_consistency_loss_rejects_shape_mismatch():
    with pytest.raises(ValueError):
        semantic_order_consistency_loss(torch.zeros(2, 4), torch.zeros(3, 4))


def test_semantic_probability_order_loss_is_symmetric_and_finite():
    first = torch.randn(2, 8, 4, 4)
    second = torch.randn(2, 8, 4, 4)
    forward = semantic_probability_order_loss(first, second)
    reverse = semantic_probability_order_loss(second, first)
    assert torch.isfinite(forward)
    assert torch.allclose(forward, reverse, atol=1e-7)


def test_relation_metrics_detects_same_direction_and_opposite_direction():
    first = torch.tensor([[[[1.0, 0.0]]]])
    same = torch.tensor([[[[2.0, 0.0]]]])
    opposite = torch.tensor([[[[-1.0, 0.0]]]])
    assert relation_metrics(first, same)["cosine"] > 0.99
    assert relation_metrics(first, opposite)["cosine"] < -0.99


def test_pixel_order_metrics_is_zero_for_identical_images():
    image = torch.rand(2, 3, 4, 4)
    metrics = pixel_order_metrics(image, image.clone())
    assert metrics["cosine"] > 0.999
    assert metrics["mean_abs_difference"] == pytest.approx(0.0)
    assert metrics["relative_l1"] == pytest.approx(0.0)


def test_self_gate_selector_is_prediction_only_and_shape_preserving():
    raw = torch.zeros(2, 3, 2, 2)
    adapted = raw.clone()
    adapted[:, 1] = 2.0
    mask = selector_mask(raw, adapted, "confidence")
    assert mask.shape == (2,)
    assert mask.dtype == torch.bool
    assert mask.all()


def test_compositional_basis_is_near_identity_without_coefficient_symmetry():
    front = CompositionalBasisSADRFrontEnd(bases=4)
    image = torch.rand(2, 3, 16, 16)
    restored, residual, coefficients = front(image)
    assert restored.shape == image.shape
    assert residual.shape == image.shape
    assert coefficients.shape == (2, 4)
    assert residual.abs().max().item() < 0.01
    assert not torch.allclose(coefficients[:, 0], coefficients[:, -1])
