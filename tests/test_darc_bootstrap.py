import numpy as np

from scripts.bootstrap_darc_crc import _cluster_statistics


def test_cluster_statistics_retains_all_condition_versions_per_sample():
    actual = np.array([
        [[0.1, 0.2], [0.3, 0.4]],
        [[0.2, 0.3], [0.4, 0.5]],
    ])
    indices = np.array([[0, 1], [1, 0]])
    result = _cluster_statistics(actual, np.array([0.5, 1.0]), indices, lowlight_index=0, blur_index=1, alpha=0.15)
    assert result["coverage"].shape == (2,)
    assert np.allclose(result["coverage"], [0.75, 0.75])
    assert np.allclose(result["risk_excess"], [0.05, 0.25])
