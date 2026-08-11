import pandas as pd
import torch

import pytest

from scripts.analyze_boundary_residual import RegionStats, bootstrap_boundary_error_gap, boundary_mask, resolve_boundary_cache_context


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
    assert 0.0 <= result["mean_confidence"] <= 1.0


def test_boundary_cache_context_supports_confirmation_but_rejects_test() -> None:
    split, cache_root = resolve_boundary_cache_context(
        {
            "splits": ["calibration", "confirmation"],
            "evaluation_split": "confirmation",
            "cache_dir": "outputs/uiis_alpha010_crc/cache",
        }
    )
    assert split == "confirmation"
    assert str(cache_root).endswith("outputs\\uiis_alpha010_crc\\cache\\confirmation")
    with pytest.raises(ValueError, match="Official TEST is locked"):
        resolve_boundary_cache_context(
            {"splits": ["calibration", "test"], "evaluation_split": "test", "cache_dir": "ignored"}
        )


def test_boundary_gap_bootstrap_uses_original_image_clusters() -> None:
    table = pd.DataFrame(
        [
            {"condition": "clean", "method": "raw", "sample_id": "a", "region": "boundary", "pixels": 10, "errors": 5},
            {"condition": "clean", "method": "raw", "sample_id": "a", "region": "interior", "pixels": 10, "errors": 1},
            {"condition": "clean", "method": "raw", "sample_id": "b", "region": "boundary", "pixels": 10, "errors": 6},
            {"condition": "clean", "method": "raw", "sample_id": "b", "region": "interior", "pixels": 10, "errors": 2},
        ]
    )
    result = bootstrap_boundary_error_gap(table, iterations=100, seed=7)
    assert result.loc[0, "cluster_unit"] == "original_sample_id"
    assert result.loc[0, "samples"] == 2
    assert result.loc[0, "boundary_minus_interior_error_rate"] == pytest.approx(0.4)
    assert result.loc[0, "ci95_low"] > 0.0
