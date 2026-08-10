import json

import pandas as pd

from scripts.evaluate_dts_seg_gate0 import decision_from


def config():
    return {"gate": {"primary_min": 0.5, "alternative_severe_mean_miou_pp_min": 1.0, "clean_miou_decrease_pp_max": 0.3, "severe_families_nonnegative_min": 3}}


def table(gain=0.006, clean_gain=0.0, negative_severe=0):
    conditions = ["clean", "color_s1", "color_s2", "color_s3", "turbidity_s1", "turbidity_s2", "turbidity_s3", "lowlight_s1", "lowlight_s2", "lowlight_s3", "blur_s1", "blur_s2", "blur_s3"]
    rows = [{"variant": "F4", "condition": name, "miou": 0.6} for name in conditions]
    for index, name in enumerate(conditions):
        delta = clean_gain if name == "clean" else gain
        if name.endswith("_s3") and negative_severe:
            negative_severe -= 1
            delta = -0.001
        rows.append({"variant": "DTS", "condition": name, "miou": 0.6 + delta})
    return pd.DataFrame(rows)


def test_gate_passes_fixed_primary_and_family_rules():
    assert decision_from(table(), config())["decision"] == "PASS"


def test_gate_fails_when_too_many_severe_families_are_negative():
    assert decision_from(table(negative_severe=2), config())["decision"] == "FAIL"


def test_gate_fails_clean_safety_even_with_large_mean_gain():
    assert decision_from(table(gain=0.02, clean_gain=-0.004), config())["decision"] == "FAIL"


def test_gate_decision_is_json_serializable():
    json.dumps(decision_from(table(), config()))
