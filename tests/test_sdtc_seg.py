import torch
from torch import nn

from reliability.sdtc_seg import (
    SDTCSegformer,
    SpatialTangentRectifier,
    clean_identity_loss,
    semantic_tangent_loss,
)


def test_rectifier_is_exact_identity_at_initialization():
    module = SpatialTangentRectifier(16)
    feature = torch.randn(2, 16, 8, 8)
    corrected, residual = module(feature)
    assert torch.equal(corrected, feature)
    assert torch.count_nonzero(residual) == 0


def test_tangent_loss_prefers_inverse_classwise_shift():
    labels = torch.tensor([[[0, 0], [1, 1]]])
    clean = (torch.zeros(1, 2, 2, 2),)
    degraded = (torch.tensor([[[[1.0, 1.0], [2.0, 2.0]], [[-1.0, -1.0], [0.5, 0.5]]]]),)
    perfect = (clean[0] - degraded[0],)
    zero = (torch.zeros_like(degraded[0]),)
    perfect_loss, count = semantic_tangent_loss(clean, degraded, perfect, labels, num_classes=2, minimum_pixels=1)
    zero_loss, _ = semantic_tangent_loss(clean, degraded, zero, labels, num_classes=2, minimum_pixels=1)
    assert count == 2
    assert perfect_loss < 1e-7
    assert zero_loss > perfect_loss


def test_tangent_target_is_stop_gradient_but_rectifier_receives_gradient():
    labels = torch.zeros(1, 2, 2, dtype=torch.long)
    clean = (torch.randn(1, 3, 2, 2, requires_grad=True),)
    degraded = (torch.randn(1, 3, 2, 2, requires_grad=True),)
    residual = (torch.randn(1, 3, 2, 2, requires_grad=True),)
    loss, _ = semantic_tangent_loss(clean, degraded, residual, labels, num_classes=1, minimum_pixels=1)
    loss.backward()
    assert clean[0].grad is None
    assert degraded[0].grad is None
    assert residual[0].grad is not None


def test_clean_identity_loss_is_zero_for_zero_residual():
    raw = (torch.randn(1, 4, 3, 3), torch.randn(1, 8, 2, 2))
    residual = tuple(torch.zeros_like(item) for item in raw)
    assert clean_identity_loss(raw, residual) == 0


class FakeEncoder(nn.Module):
    def forward(self, pixels, **kwargs):
        class Output:
            hidden_states = (pixels, pixels[:, :2])
        return Output()


class FakeBase(nn.Module):
    def __init__(self):
        super().__init__()
        self.segformer = FakeEncoder()
        self.decode_head = FakeDecodeHead()


class FakeDecodeHead(nn.Module):
    def __init__(self):
        super().__init__()
        self.classifier = nn.Conv2d(2, 2, 1)

    def forward(self, features):
        return self.classifier(features[-1])


def test_wrapper_exposes_raw_corrected_and_residual_features():
    model = SDTCSegformer(FakeBase(), hidden_sizes=(3, 2))
    output = model(torch.randn(1, 3, 4, 4))
    assert output.logits.shape == (1, 2, 4, 4)
    assert len(output.raw_features) == len(output.corrected_features) == len(output.residuals) == 2
