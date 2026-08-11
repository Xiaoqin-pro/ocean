import torch

from reliability.bpt_seg import BPTSegformer, BoundaryPrototypeTransportRectifier, boundary_supervision_loss


def test_boundary_rectifier_is_identity_initially() -> None:
    torch.manual_seed(3)
    module = BoundaryPrototypeTransportRectifier(8, num_classes=3, reduction=2)
    feature = torch.randn(2, 8, 4, 4)
    corrected, residual, router, prototypes, boundary = module(feature)
    assert torch.allclose(corrected, feature)
    assert residual.shape == feature.shape
    assert router.shape == (2, 3, 4, 4)
    assert prototypes.shape == (3, 8)
    assert boundary.shape == (2, 1, 4, 4)


def test_boundary_loss_is_finite() -> None:
    labels = torch.zeros(2, 8, 8, dtype=torch.long)
    labels[:, 2:6, 3:5] = 1
    logits = (torch.randn(2, 1, 4, 4, requires_grad=True),)
    loss = boundary_supervision_loss(logits, labels)
    assert torch.isfinite(loss)
    loss.backward()
    assert logits[0].grad is not None
