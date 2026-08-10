import torch

from reliability.dts_seg import degradation_acceleration_loss, trajectory_names, true_class_margin


def test_trajectory_names_are_deterministic_and_ordered():
    first = trajectory_names("scene-7", 3)
    assert first == trajectory_names("scene-7", 3)
    family = first[1].removesuffix("_s1")
    assert first == ("clean", f"{family}_s1", f"{family}_s2", f"{family}_s3")


def test_true_class_margin_and_ignore_mask():
    logits = torch.tensor([[[[3.0, 0.0]], [[1.0, 2.0]], [[0.0, 1.0]]]])
    labels = torch.tensor([[[0, 255]]])
    margin, valid = true_class_margin(logits, labels)
    assert valid.tolist() == [[[True, False]]]
    assert torch.isfinite(margin).all()
    assert margin[0, 0, 0] > 1.0


def test_loss_ignores_constant_or_improving_slope():
    mask = torch.ones(1, 2, dtype=torch.bool)
    old = torch.tensor([[2.0, 2.0]])
    previous = torch.tensor([[1.5, 1.5]])
    current = torch.tensor([[1.0, 1.2]], requires_grad=True)
    loss = degradation_acceleration_loss(old, previous, current, mask, tolerance=0.0)
    assert torch.allclose(loss, torch.tensor(0.0))


def test_loss_penalizes_acceleration_and_only_current_gets_gradient():
    old = torch.tensor([[2.0]])
    previous = torch.tensor([[1.8]], requires_grad=True)
    current = torch.tensor([[1.0]], requires_grad=True)
    loss = degradation_acceleration_loss(old, previous, current, torch.ones(1, 1, dtype=torch.bool), tolerance=0.1)
    assert torch.allclose(loss, torch.tensor(0.5))
    loss.backward()
    assert previous.grad is None
    assert torch.allclose(current.grad, torch.tensor([[-1.0]]))


def test_empty_eligibility_returns_differentiable_zero():
    current = torch.tensor([[1.0]], requires_grad=True)
    loss = degradation_acceleration_loss(torch.tensor([[2.0]]), torch.tensor([[1.5]]), current, torch.zeros(1, 1, dtype=torch.bool))
    loss.backward()
    assert loss.item() == 0.0
    assert current.grad.item() == 0.0

