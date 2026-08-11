import torch

from reliability.dapt_seg import DegradationAdaptivePrototypeRectifier, degradation_rank_loss


def test_dapt_rectifier_is_identity_initially() -> None:
    torch.manual_seed(4)
    module = DegradationAdaptivePrototypeRectifier(8, num_classes=3, reduction=2)
    feature = torch.randn(2, 8, 4, 4)
    corrected, residual, router, prototypes, score = module(feature)
    assert torch.allclose(corrected, feature)
    assert residual.shape == feature.shape
    assert router.shape == (2, 3, 4, 4)
    assert prototypes.shape == (3, 8)
    assert score.shape == (2, 1)


def test_dapt_rank_loss_is_finite_and_differentiable() -> None:
    scores = tuple(torch.randn(2, 1, requires_grad=True) for _ in range(4))
    loss = degradation_rank_loss(scores)
    assert torch.isfinite(loss)
    loss.backward()
    assert all(score.grad is not None for score in scores)
