import pytest
import torch

from scripts.cache_uiis_deeplab_logits import build_model, low_resolution_logits, validate_checkpoint, validate_protocol


def test_deeplab_cache_protocol_rejects_test_split():
    config = {"experiment": {"fit_split": "calibration", "evaluation_split": "confirmation", "official_suim_test_locked": True}, "protocol": {"confirmation_opened": True, "confirmation_used_for_fitting": False, "official_suim_test_evaluated": False, "model_retrained_after_protocol_freeze": False}}
    validate_protocol(config, ["calibration", "confirmation"])
    with pytest.raises(ValueError, match="Only calibration"):
        validate_protocol(config, ["test"])


def test_deeplab_cache_requires_final_fixed_epoch():
    checkpoint = {"checkpoint_format": "uiis_deeplabv3_fixed_protocol_v1", "epoch": 60, "checkpoint_selection": "final_epoch", "official_suim_test_evaluated": False, "calibration_evaluated": False, "confirmation_evaluated": False}
    validate_checkpoint(checkpoint, 60)
    checkpoint["epoch"] = 59
    with pytest.raises(ValueError, match="fixed final epoch"):
        validate_checkpoint(checkpoint, 60)


def test_deeplab_cache_uses_pre_upsampling_logits():
    model = build_model(8, None).eval()
    with torch.no_grad():
        logits = low_resolution_logits(model, torch.zeros((1, 3, 64, 64)))
    assert logits.shape == (1, 8, 4, 4)
