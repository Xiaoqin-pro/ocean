import torch

from calibration.temperature_scaling import fit_temperature, scale_logits


def test_temperature_scaling_preserves_argmax() -> None:
    logits = torch.tensor([[[[2.0, -1.0]], [[0.5, 3.0]]]])
    assert torch.equal(logits.argmax(1), scale_logits(logits, 2.7).argmax(1))


def test_fit_temperature_is_positive_and_does_not_worsen_nll() -> None:
    logits = torch.tensor([[[[4.0, 3.0]], [[0.0, 1.0]]]], dtype=torch.float32)
    labels = torch.tensor([[[0, 1]]])
    result = fit_temperature(logits, labels)
    assert 0.05 <= result.temperature <= 20.0
    assert result.final_nll <= result.initial_nll + 1e-7
