import torch

from reliability.spt_seg import SemanticPrototypeTransportRectifier, semantic_prototype_transport_loss


def test_spt_rectifier_preserves_identity_at_initialization() -> None:
    torch.manual_seed(1)
    module = SemanticPrototypeTransportRectifier(8, num_classes=3, reduction=2)
    feature = torch.randn(2, 8, 4, 4)
    corrected, residual, router, prototypes = module(feature)
    assert corrected.shape == feature.shape
    assert residual.shape == feature.shape
    assert router.shape == (2, 3, 4, 4)
    assert prototypes.shape == (3, 8)
    assert torch.allclose(corrected, feature)


def test_spt_prototype_loss_has_gradient_and_handles_ignore() -> None:
    torch.manual_seed(2)
    clean = (torch.randn(2, 8, 4, 4),)
    corrected = (clean[0].clone().requires_grad_(),)
    labels = torch.randint(0, 3, (2, 8, 8))
    labels[:, 0, 0] = 255
    loss, pairs = semantic_prototype_transport_loss(clean, corrected, labels, num_classes=3, minimum_pixels=1)
    assert pairs > 0
    assert torch.isfinite(loss)
    loss.backward()
    assert corrected[0].grad is not None
