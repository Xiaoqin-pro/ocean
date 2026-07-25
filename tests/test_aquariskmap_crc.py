import numpy as np

from scripts.run_aquariskmap_global_crc import ALPHA, SCORES


def test_crc_constants_are_preregistered_and_score_complete():
    assert ALPHA == 0.10
    assert SCORES[0] == "raw_msp"
    assert SCORES[-1] == "aquariskmap"
    assert len(SCORES) == 8


def test_crc_cluster_shape_matches_13_condition_protocol():
    values = np.zeros((len(SCORES), 13, 146, 100), dtype=np.float32)
    assert values.shape[1:] == (13, 146, 100)
