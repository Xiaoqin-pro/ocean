import pytest

from scripts.evaluate_uiis_alpha010_confirmation import validate_protocol


def _config() -> dict:
    return {
        "experiment": {"fit_split": "calibration", "evaluation_split": "confirmation", "official_suim_test_locked": True},
        "protocol": {"confirmation_opened": True, "confirmation_used_for_fitting": False, "official_suim_test_evaluated": False},
        "risk": {"target_alpha": 0.10},
    }


def _parameters() -> dict:
    return {"target_alpha": 0.10, "fit_split": "calibration", "confirmation_opened": False}


def test_frozen_calibration_parameters_can_open_confirmation_once():
    validate_protocol(_config(), _parameters())


def test_confirmation_parameters_cannot_be_refit_after_opening():
    parameters = _parameters()
    parameters["confirmation_opened"] = True
    with pytest.raises(ValueError, match="frozen"):
        validate_protocol(_config(), parameters)
