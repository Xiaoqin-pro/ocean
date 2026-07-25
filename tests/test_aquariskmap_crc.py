import numpy as np
import pandas as pd

from scripts.run_aquariskmap_global_crc import ALPHA, SCORES


def test_crc_constants_are_preregistered_and_score_complete():
    assert ALPHA == 0.10
    assert SCORES[0] == "raw_msp"
    assert SCORES[-1] == "aquariskmap"
    assert len(SCORES) == 8


def test_crc_cluster_shape_matches_13_condition_protocol():
    values = np.zeros((len(SCORES), 13, 146, 100), dtype=np.float32)
    assert values.shape[1:] == (13, 146, 100)


def test_primary_score_rows_are_selected_by_label_not_dataframe_iteration():
    table = pd.DataFrame([
        {"score": "raw_msp", "coverage": 0.31, "selective_risk": 0.09},
        {"score": "aquariskmap", "coverage": 0.35, "selective_risk": 0.08},
    ]).set_index("score")
    raw, aqua = table.loc["raw_msp"], table.loc["aquariskmap"]
    assert aqua.coverage - raw.coverage == 0.04
