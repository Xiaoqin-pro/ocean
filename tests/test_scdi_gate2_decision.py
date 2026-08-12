import pandas as pd

from scripts.evaluate_scdi_seg_gate2 import decision_from


CONDITIONS = [
    "clean", "color_s1", "color_s2", "color_s3", "turbidity_s1", "turbidity_s2", "turbidity_s3",
    "lowlight_s1", "lowlight_s2", "lowlight_s3", "blur_s1", "blur_s2", "blur_s3",
]


def config():
    return {"gate": {"primary_min_miou_pp": 0.5, "alternative_severe_mean_miou_pp_min": 1.0, "clean_miou_decrease_pp_max": 0.3, "severe_families_nonnegative_min": 3}}


def table(gain=0.006, clean_gain=0.0):
    rows = [{"variant": "F4", "condition": name, "miou": 0.6} for name in CONDITIONS]
    rows += [{"variant": "SDTC", "condition": name, "miou": 0.6 + (clean_gain if name == "clean" else gain)} for name in CONDITIONS]
    return pd.DataFrame(rows)


def scdi_table(**kwargs):
    frame = table(**kwargs)
    frame.loc[frame.variant == "SDTC", "variant"] = "SCDI"
    return frame


def test_scdi_gate_passes_and_uses_scdi_metric_names():
    result = decision_from(scdi_table(), config(), CONDITIONS)
    assert result["decision"] == "PASS"
    assert "scdi_minus_f4_13_condition_mean_miou_pp" in result


def test_scdi_gate_fails_clean_safety():
    assert decision_from(scdi_table(gain=0.02, clean_gain=-0.004), config(), CONDITIONS)["decision"] == "FAIL"
