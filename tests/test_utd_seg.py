import torch

from reliability.utd_seg import semantic_trajectory_distillation_loss


def test_trajectory_distillation_is_zero_for_identical_logits() -> None:
    torch.manual_seed(5)
    logits = torch.randn(2, 3, 4, 4)
    labels = torch.zeros(2, 8, 8, dtype=torch.long)
    loss = semantic_trajectory_distillation_loss(logits, logits.clone(), labels)
    assert loss < 1e-6


def test_trajectory_distillation_has_gradient() -> None:
    clean = torch.randn(2, 3, 4, 4)
    degraded = torch.randn(2, 3, 4, 4, requires_grad=True)
    labels = torch.zeros(2, 8, 8, dtype=torch.long)
    loss = semantic_trajectory_distillation_loss(clean, degraded, labels)
    assert torch.isfinite(loss)
    loss.backward()
    assert degraded.grad is not None


def test_class_balanced_distillation_is_finite() -> None:
    clean = torch.randn(1, 3, 4, 4)
    degraded = torch.randn(1, 3, 4, 4, requires_grad=True)
    labels = torch.zeros(1, 8, 8, dtype=torch.long)
    labels[:, 2:4, 2:4] = 1
    loss = semantic_trajectory_distillation_loss(clean, degraded, labels, class_balance=True)
    assert torch.isfinite(loss)
    loss.backward()
    assert degraded.grad is not None
