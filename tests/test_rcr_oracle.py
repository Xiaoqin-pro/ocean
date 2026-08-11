from __future__ import annotations

import pytest
import torch

from reliability.rcr_oracle import equal_probability_ensemble, pixelwise_oracle_union, repair_statistics


def test_repair_damage_net_mass_and_oracle_have_fixed_gt_semantics():
    labels = torch.tensor([[0, 1, 2, 3, 255]])
    local = torch.tensor([[0, 0, 2, 2, 0]])
    remote = torch.tensor([[1, 1, 1, 3, 0]])
    region = labels.ne(255)
    values = repair_statistics(local, remote, labels, region)
    # Local errs at indices 1 and 3; remote repairs index 1 and damages index 0/2.
    assert values["local_wrong_pixels"] == 2 and values["repaired_pixels"] == 2
    assert values["damaged_pixels"] == 2 and values["repair_rate"] == 1.0
    assert values["damage_rate"] == 2 / 2 and values["net_correction_mass"] == 0.0
    assert pixelwise_oracle_union(local, remote, labels).tolist() == [[0, 1, 2, 3, 0]]


def test_equal_probability_ensemble_is_not_raw_logit_average():
    seg = torch.tensor([[[[2.0]], [[1.0]], [[0.0]]]])
    deep = torch.tensor([[[[0.0]], [[1.2]], [[1.1]]]])
    # Raw-logit average would pick class 1, while the fixed probability mean picks class 0.
    assert ((seg + deep) / 2).argmax(1).item() == 1
    assert equal_probability_ensemble(seg, deep).item() == 0


def test_repair_statistics_rejects_misaligned_region():
    with pytest.raises(ValueError):
        repair_statistics(torch.zeros((2, 2), dtype=torch.long), torch.zeros((2, 2), dtype=torch.long), torch.zeros((2, 2), dtype=torch.long), torch.ones((1, 2), dtype=torch.bool))
