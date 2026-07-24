import pytest

from scripts.cache_uiis_crc_logits import validate_protocol


def _config() -> dict:
    return {
        "experiment": {
            "fit_split": "calibration",
            "evaluation_split": "confirmation",
            "official_suim_test_locked": True,
        },
        "protocol": {
            "confirmation_used_for_fitting": False,
            "confirmation_opened": False,
            "official_suim_test_evaluated": False,
            "model_retrained_after_protocol_freeze": False,
        },
    }


def test_calibration_cache_is_allowed_before_confirmation_opening():
    validate_protocol(_config(), ["calibration"])


def test_confirmation_cache_is_rejected_before_parameters_are_frozen():
    with pytest.raises(ValueError, match="Confirmation is locked"):
        validate_protocol(_config(), ["confirmation"])
