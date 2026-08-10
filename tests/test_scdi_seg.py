import torch
from torch import nn

from reliability.scdi_seg import (
    SCDISegformer,
    SemanticInverseExpertRectifier,
    semantic_router_loss,
)


def test_expert_rectifier_is_identity_at_initialization():
    module = SemanticInverseExpertRectifier(16, num_classes=3)
    feature = torch.randn(2, 16, 8, 8)
    corrected, residual, router = module(feature)
    assert torch.equal(corrected, feature)
    assert torch.count_nonzero(residual) == 0
    assert router.shape == (2, 3, 8, 8)


def test_semantic_router_loss_rewards_correct_class_routing():
    labels = torch.tensor([[[0, 0], [1, 1]]])
    neutral = (torch.zeros(1, 2, 2, 2),)
    correct = torch.full((1, 2, 2, 2), -5.0)
    correct[:, 0, 0] = 5.0
    correct[:, 1, 1] = 5.0
    assert semantic_router_loss((correct,), labels) < semantic_router_loss(neutral, labels)


def test_experts_receive_distinct_gradients_after_semantic_routing():
    module = SemanticInverseExpertRectifier(8, num_classes=2)
    with torch.no_grad():
        module.router.weight.zero_()
        module.router.bias.copy_(torch.tensor([5.0, -5.0]))
    corrected, _, _ = module(torch.randn(1, 8, 4, 4))
    corrected.sum().backward()
    gradient = module.expert_up.weight.grad.reshape(2, 8, *module.expert_up.weight.shape[1:])
    assert gradient[0].abs().sum() > gradient[1].abs().sum()


class FakeEncoder(nn.Module):
    def forward(self, pixels, **kwargs):
        class Output:
            hidden_states = (pixels, pixels[:, :2])
        return Output()


class FakeDecodeHead(nn.Module):
    def __init__(self):
        super().__init__()
        self.classifier = nn.Conv2d(2, 2, 1)

    def forward(self, features):
        return self.classifier(features[-1])


class FakeBase(nn.Module):
    def __init__(self):
        super().__init__()
        self.segformer = FakeEncoder()
        self.decode_head = FakeDecodeHead()


def test_wrapper_returns_semantic_router_pyramid():
    model = SCDISegformer(FakeBase(), hidden_sizes=(3, 2), num_classes=2)
    output = model(torch.randn(1, 3, 4, 4))
    assert output.logits.shape == (1, 2, 4, 4)
    assert len(output.router_logits) == len(output.residuals) == 2
