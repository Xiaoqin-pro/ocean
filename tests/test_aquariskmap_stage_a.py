import pandas as pd
import pytest

from reliability.aquariskmap import CONDITIONS
from scripts.evaluate_aquariskmap_stage_a import REGIONS, SCORES, aggregate_and_bootstrap


def _table() -> pd.DataFrame:
    rows = []
    for sample_id in ("a", "b"):
        for condition in CONDITIONS:
            for region in REGIONS:
                for score in SCORES:
                    if score == "raw_msp":
                        values = (0.20, 0.30, 0.40)
                    elif score == "aquariskmap":
                        values = (0.15, 0.35, 0.46)
                    else:
                        values = (0.19, 0.31, 0.41)
                    rows.append({"sample_id": sample_id, "condition": condition, "score": score, "region": region, "eaurc": values[0], "error_auprc": values[1], "top_10_uncertainty_recall": values[2]})
    return pd.DataFrame(rows)


def test_stage_a_clustered_aggregate_and_gate_are_deterministic():
    first = aggregate_and_bootstrap(_table())
    second = aggregate_and_bootstrap(_table())
    assert len(first[0]) == len(SCORES) * len(REGIONS)
    assert len(first[1]) == len(REGIONS) * 3
    assert first[1].equals(second[1])
    assert first[2]["ranking_gate_pass"]


def test_stage_a_rejects_incomplete_condition_cluster():
    table = _table().iloc[1:].copy()
    with pytest.raises(ValueError, match="all 13 conditions"):
        aggregate_and_bootstrap(table)
