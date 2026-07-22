import pytest
import torch

from calibration.temperature_scaling import fit_temperature, fit_temperature_from_batches, scale_logits
from scripts.evaluate_temperature_scaling import validate_cache_payload


def test_temperature_scaling_preserves_argmax() -> None:
    logits = torch.tensor([[[[2.0, -1.0]], [[0.5, 3.0]]]])
    assert torch.equal(logits.argmax(1), scale_logits(logits, 2.7).argmax(1))


def test_fit_temperature_is_positive_and_does_not_worsen_nll() -> None:
    logits = torch.tensor([[[[4.0, 3.0]], [[0.0, 1.0]]]], dtype=torch.float32)
    labels = torch.tensor([[[0, 1]]])
    result = fit_temperature(logits, labels)
    assert 0.05 <= result.temperature <= 20.0
    assert result.final_nll <= result.initial_nll + 1e-7


@pytest.mark.parametrize("temperature", [0.0, -1.0, float("nan"), float("inf")])
def test_invalid_temperature_is_rejected(temperature: float) -> None:
    with pytest.raises(ValueError, match="finite and positive"):
        scale_logits(torch.ones((1, 2, 1, 1)), temperature)


def test_temperature_one_is_an_exact_identity() -> None:
    logits = torch.randn((2, 3, 4, 4))
    assert torch.equal(scale_logits(logits, 1.0), logits)
    assert torch.equal(torch.softmax(scale_logits(logits, 1.0), 1), torch.softmax(logits, 1))


def test_streaming_fit_rejects_empty_valid_pixels() -> None:
    def batches():
        yield torch.ones((1, 2, 2, 2)), torch.full((1, 2, 2), 255)

    with pytest.raises(ValueError, match="No valid labels"):
        fit_temperature_from_batches(batches)


def test_cache_integrity_rejects_mismatched_hash_or_duplicate_samples() -> None:
    payload = {
        "split": "val", "condition": "clean", "checkpoint_sha256": "checkpoint",
        "degradation_config_sha256": "degradation", "sample_id": ["a", "b"],
        "logits": torch.zeros((2, 8, 2, 2), dtype=torch.float16), "labels": torch.zeros((2, 4, 4), dtype=torch.long),
    }
    validate_cache_payload(payload, split="val", condition="clean", checkpoint_sha256="checkpoint", degradation_config_sha256="degradation")
    with pytest.raises(ValueError, match="checkpoint hash mismatch"):
        validate_cache_payload(payload, split="val", condition="clean", checkpoint_sha256="other", degradation_config_sha256="degradation")
    payload["sample_id"] = ["a", "a"]
    with pytest.raises(ValueError, match="duplicate sample IDs"):
        validate_cache_payload(payload, split="val", condition="clean", checkpoint_sha256="checkpoint", degradation_config_sha256="degradation")
