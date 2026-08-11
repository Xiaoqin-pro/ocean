from __future__ import annotations

import pandas as pd

import numpy as np
import pytest
import torch

from scripts.evaluate_ft_reliability_pilot import (
    _ece_from_sums, _tensor_sha256, _validate_cache_payload, diagnostic_value, error_auroc_diagnostic,
    paired_bootstrap, validate_finite_table,
)


def test_primary_endpoint_is_unique_and_uses_complete_sample_clusters():
    rows = []
    for variant, shift in (("B", 0.0), ("C", 0.0), ("E", -0.02)):
        for sample in range(231):
            for condition in ("clean", "color_s1"):
                for region in ("full", "boundary", "interior"):
                    rows.append({"variant": variant, "sample_id": f"s{sample}", "condition": condition, "region": region,
                                 "eaurc": 0.2 + shift, "error_auprc": 0.4 - shift, "miou": 0.5})
    bootstrap, primary = paired_bootstrap(pd.DataFrame(rows))
    assert len(bootstrap) == 18
    endpoint = bootstrap[(bootstrap.comparison == "E_vs_C") & (bootstrap.region == "full") & (bootstrap.metric == "eaurc")].iloc[0]
    assert primary["primary_endpoint"] == "full_eaurc_C_minus_E"
    assert endpoint.ci95_low > 0


def test_single_class_error_target_makes_only_error_auroc_undefined():
    value, defined, reason = error_auroc_diagnostic(float("nan"), np.zeros(12, dtype=bool))
    assert value is None and not defined and reason == "single_class_error_target"


def test_other_nonfinite_metric_and_two_class_auroc_still_fail():
    with pytest.raises(AssertionError):
        error_auroc_diagnostic(float("nan"), np.array([False, True]))
    with pytest.raises(AssertionError):
        validate_finite_table(pd.DataFrame({"miou": [float("nan")], "error_auroc": [float("nan")]}))


def test_no_positive_error_diagnostics_are_null_not_imputed():
    errors = np.zeros(12, dtype=bool)
    for name in ("error_auprc", "top_10_uncertainty_recall"):
        value, defined, reason = diagnostic_value(name, float("nan"), errors)
        assert value is None
        assert not defined
        assert reason == "no_positive_errors"
    with pytest.raises(AssertionError):
        diagnostic_value("error_auprc", float("nan"), np.array([False, True]))


def test_declared_diagnostic_nulls_and_empty_regions_are_the_only_permitted_gaps():
    row = {"miou": 0.5, "region_defined": True}
    for name, reason in (("error_auroc", "single_class_error_target"),
                         ("error_auprc", "no_positive_errors"),
                         ("top_10_uncertainty_recall", "no_positive_errors")):
        row[name] = None
        row[f"{name}_defined"] = False
        row[f"{name}_undefined_reason"] = reason
    validate_finite_table(pd.DataFrame([row]))
    validate_finite_table(pd.DataFrame([{"region_defined": False, "region_undefined_reason": "empty_region"}]))
    with pytest.raises(AssertionError):
        validate_finite_table(pd.DataFrame([{"region_defined": False, "region_undefined_reason": "other"}]))


def test_prediction_cache_rejects_wrong_shape_or_identity():
    sample_ids = ["a", "b"]
    labels = torch.zeros((2, 384, 384), dtype=torch.uint8)
    expected = {"schema": "test", "variant": "A", "condition": "clean", "checkpoint_sha256": "x", "development_csv_sha256": "y"}
    payload = {**expected, "sample_ids": sample_ids, "logits": torch.zeros((2, 8, 2, 2), dtype=torch.float16),
               "labels": labels, "labels_sha256": _tensor_sha256(labels)}
    _validate_cache_payload(payload, expected=expected, sample_ids=sample_ids)
    payload["sample_ids"] = ["a", "different"]
    with pytest.raises(ValueError):
        _validate_cache_payload(payload, expected=expected, sample_ids=sample_ids)


def test_bootstrap_preserves_primary_but_marks_undefined_secondary_metric():
    rows = []
    for variant in ("B", "C", "E"):
        for sample in range(231):
            for region in ("full", "boundary", "interior"):
                rows.append({"variant": variant, "sample_id": f"s{sample}", "condition": "clean", "region": region,
                             "eaurc": 0.2, "miou": 0.5,
                             "error_auprc": float("nan") if sample == 0 else 0.4})
    bootstrap, primary = paired_bootstrap(pd.DataFrame(rows))
    assert primary["primary_endpoint"] == "full_eaurc_C_minus_E"
    undefined = bootstrap[bootstrap.metric.eq("error_auprc")]
    assert not undefined.defined.any()
    assert set(undefined.undefined_reason) == {"one_or_more_clusters_metric_undefined"}


def test_direct_ece_sufficient_statistics_do_not_average_per_image_ece():
    counts = np.array([2, 2], dtype=np.int64)
    confidence = np.array([1.6, 0.4], dtype=np.float64)
    events = np.array([2.0, 0.0], dtype=np.float64)
    assert _ece_from_sums(counts, confidence, events) == pytest.approx(0.2)
    with pytest.raises(ValueError):
        _ece_from_sums(np.zeros(2, dtype=np.int64), np.zeros(2), np.zeros(2))
