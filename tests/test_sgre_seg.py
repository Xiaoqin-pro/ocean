import torch

from reliability.sgre_seg import SemanticGroupState, classwise_cross_entropy


def test_classwise_loss_equalizes_classes_not_pixel_counts():
    labels = torch.tensor([[[0, 0, 0], [0, 0, 1]]])
    logits = torch.zeros(1, 2, 2, 3)
    logits[:, 0] = 2.0
    losses, present = classwise_cross_entropy(logits, labels, num_classes=2, minimum_pixels=1)
    assert present.tolist() == [True, True]
    assert losses[1] > losses[0]


def test_group_state_upweights_persistently_hard_group():
    state = SemanticGroupState.create(views=2, classes=2, momentum=0.5, temperature=0.5)
    present = torch.tensor([True, True])
    for _ in range(3):
        state.update(1, torch.tensor([0.2, 2.0]), present)
    weights = state.weights(torch.device("cpu"))
    assert weights[1, 1] == weights.max()
    assert torch.isclose(weights.sum(), torch.tensor(1.0))


def test_first_observation_is_not_diluted_by_default_value():
    state = SemanticGroupState.create(views=1, classes=2, momentum=0.9)
    state.update(0, torch.tensor([3.0, 0.0]), torch.tensor([True, False]))
    assert state.ema_losses[0, 0] == 3.0
    assert state.ema_losses[0, 1] == 1.0
    assert state.observed.tolist() == [[True, False]]
