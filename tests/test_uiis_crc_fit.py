import numpy as np
import pandas as pd
import pytest

from scripts.fit_uiis_alpha010_crc import quality_group_losses, validate_protocol


def _config() -> dict:
    return {
        "experiment": {"train_split": "train", "fit_split": "calibration", "evaluation_split": "confirmation"},
        "protocol": {"confirmation_opened": False, "confirmation_used_for_fitting": False, "official_suim_test_evaluated": False},
        "risk": {"target_alpha": 0.10},
        "quality": {"groups": 3},
    }


def test_quality_losses_keep_each_image_once_per_group():
    envelope = np.array([[[0.02, 0.04], [0.03, 0.05]], [[0.04, 0.06], [0.01, 0.03]]])
    assignments = pd.DataFrame(
        [
            {"sample_id": "a", "condition": "clean", "quality_group": 0},
            {"sample_id": "a", "condition": "blur_s1", "quality_group": 0},
            {"sample_id": "b", "condition": "clean", "quality_group": 1},
            {"sample_id": "b", "condition": "blur_s1", "quality_group": 0},
        ]
    )
    assert np.allclose(quality_group_losses(envelope, assignments, ["clean", "blur_s1"], ["a", "b"], 0), [[0.03, 0.05], [0.01, 0.03]])


def test_confirmation_is_rejected_while_fitting():
    config = _config()
    validate_protocol(config)
    config["protocol"]["confirmation_opened"] = True
    with pytest.raises(ValueError, match="Confirmation"):
        validate_protocol(config)
