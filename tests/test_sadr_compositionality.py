import pytest
import torch

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
