import torch

from scripts.analyze_boundary_residual import RegionStats, boundary_mask


def test_boundary_mask_and_excess_aurc_are_well_formed() -> None:
    labels = torch.tensor([[[0, 0, 1], [0, 1, 1], [2, 2, 1]]])
    mask = boundary_mask(labels, radius=1)
    assert mask.shape == labels.shape
    assert mask.dtype is torch.bool
    probabilities = torch.tensor(
        [[[[0.90, 0.90, 0.20], [0.90, 0.20, 0.20], [0.20, 0.20, 0.20]],
          [[0.05, 0.05, 0.70], [0.05, 0.70, 0.70], [0.20, 0.20, 0.70]],
          [[0.05, 0.05, 0.10], [0.05, 0.10, 0.10], [0.60, 0.60, 0.10]]]]
    )
    stats = RegionStats(bins=5)
    stats.update(probabilities, labels, torch.ones_like(labels, dtype=torch.bool))
    result = stats.result()
    assert 0.0 <= result["error_rate"] <= 1.0
    assert result["aurc"] >= result["oracle_aurc"]
    assert result["eaurc"] >= 0.0
