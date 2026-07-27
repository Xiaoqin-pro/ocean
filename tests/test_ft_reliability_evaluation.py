from __future__ import annotations

import pandas as pd

from scripts.evaluate_ft_reliability_pilot import paired_bootstrap


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
