import numpy as np
import pytest

from scripts.evaluate_uiis_crc_sensitivity import bootstrap_interval, validate_protocol


def test_crc_sensitivity_protocol_is_fixed_to_three_targets_and_locked_test():
    config = {
        "experiment": {"official_suim_test_locked": True},
        "risk": {"targets": [0.05, 0.10, 0.15]},
        "quality": {"groups": 3, "seeds": [20260722, 20260723, 20260724]},
        "protocol": {"external_benchmark_extension": True, "confirmation_previously_opened_for_darc_negative_control": True, "confirmation_used_for_fitting": False, "official_suim_test_evaluated": False, "model_retrained": False},
    }
    validate_protocol(config)
    config["risk"]["targets"] = [0.10]
    with pytest.raises(ValueError, match="three-target"):
        validate_protocol(config)


def test_bootstrap_interval_is_deterministic_and_contains_mean():
    values = np.asarray([0.01, 0.03, 0.05, 0.07])
    first = bootstrap_interval(values, 200, 9)
    second = bootstrap_interval(values, 200, 9)
    assert first == second
    assert first[1] <= first[0] <= first[2]
