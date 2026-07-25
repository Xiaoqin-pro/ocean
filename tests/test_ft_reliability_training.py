from __future__ import annotations

import random

import numpy as np
import pytest
import torch

from reliability.ft_reliability import atomic_torch_save
from scripts.train_ft_reliability_pilot import (
    PROTOCOL_COMMIT, checkpoint_payload, load_completed_epoch_checkpoint, variant_terms,
)


def test_variant_schedules_are_fixed_and_non_overlapping():
    assert variant_terms("A", 1) == {"clean_only": True, "three_view_ce": False, "generic_ranking": False, "failure_transition": False, "retention": False}
    assert not variant_terms("C", 5)["generic_ranking"] and variant_terms("C", 6)["generic_ranking"]
    assert not variant_terms("D", 5)["failure_transition"] and variant_terms("D", 6)["failure_transition"]
    assert variant_terms("E", 1)["retention"] and variant_terms("E", 6)["failure_transition"]
    with pytest.raises(ValueError):
        variant_terms("F", 1)


def test_completed_epoch_checkpoint_restores_model_optimizer_scaler_and_rng(tmp_path):
    torch.manual_seed(42); random.seed(42); np.random.seed(42)
    model = torch.nn.Linear(3, 2)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    scaler = torch.amp.GradScaler("cuda", enabled=False)
    initial = "initialization"; csv_hash = "csv"; degradation = "degradation"
    path = tmp_path / "last.pt"
    atomic_torch_save(checkpoint_payload(4, 19, model, optimizer, scaler, variant="D", initialization_sha256=initial, method_train_csv_sha256=csv_hash, degradation_config_sha256=degradation), path)
    expected_python, expected_numpy, expected_torch = random.random(), np.random.rand(), torch.rand(1)
    restored = torch.nn.Linear(3, 2)
    restored_optimizer = torch.optim.AdamW(restored.parameters(), lr=1e-3)
    restored_scaler = torch.amp.GradScaler("cuda", enabled=False)
    assert load_completed_epoch_checkpoint(path, restored, restored_optimizer, restored_scaler, torch.device("cpu"), variant="D") == (5, 19)
    assert random.random() == expected_python
    assert np.random.rand() == expected_numpy
    assert torch.equal(torch.rand(1), expected_torch)
    for before, after in zip(model.parameters(), restored.parameters()):
        assert torch.equal(before, after)


def test_checkpoint_rejects_wrong_variant_or_any_evaluation_access(tmp_path):
    model = torch.nn.Linear(2, 2)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    scaler = torch.amp.GradScaler("cuda", enabled=False)
    path = tmp_path / "bad.pt"
    payload = checkpoint_payload(1, 1, model, optimizer, scaler, variant="B", initialization_sha256="i", method_train_csv_sha256="c", degradation_config_sha256="d")
    payload["validation_evaluated"] = True
    atomic_torch_save(payload, path)
    with pytest.raises(ValueError):
        load_completed_epoch_checkpoint(path, model, optimizer, scaler, torch.device("cpu"), variant="B")
    assert PROTOCOL_COMMIT == "9f54a1c"
