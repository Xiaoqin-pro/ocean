import pandas as pd

from scripts.evaluate_sgre_seg_gate3 import decision_from


CONDITIONS = [
    "clean", "color_s1", "color_s2", "color_s3", "turbidity_s1", "turbidity_s2", "turbidity_s3",
    "lowlight_s1", "lowlight_s2", "lowlight_s3", "blur_s1", "blur_s2", "blur_s3",
]


def test_sgre_decision_maps_candidate_and_metric_name():
    rows = [{"variant": "F4", "condition": name, "miou": 0.6} for name in CONDITIONS]
    rows += [{"variant": "SGRE", "condition": name, "miou": 0.606 if name != "clean" else 0.6} for name in CONDITIONS]
    config = {"gate": {"primary_min_miou_pp": 0.5, "alternative_severe_mean_miou_pp_min": 1.0, "clean_miou_decrease_pp_max": 0.3, "severe_families_nonnegative_min": 3}}
    result = decision_from(pd.DataFrame(rows), config, CONDITIONS)
    assert result["decision"] == "PASS"
    assert "sgre_minus_f4_13_condition_mean_miou_pp" in result
