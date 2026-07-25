from __future__ import annotations

import random
import copy
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from reliability.ft_reliability import atomic_torch_save
from scripts.train_ft_reliability_pilot import (
    PROTOCOL_COMMIT, checkpoint_payload, load_completed_epoch_checkpoint, run_one_step, validate_teacher_checkpoint_metadata, variant_terms,
)


def _provenance() -> dict[str, str | None]:
    return {"initialization_sha256": "initial", "method_train_csv_sha256": "train", "method_development_csv_sha256": "development", "split_audit_sha256": "audit", "degradation_config_sha256": "degradation", "ft_config_sha256": "config", "teacher_checkpoint_sha256": None}


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
    provenance = _provenance(); provenance.update(initialization_sha256=initial, method_train_csv_sha256=csv_hash, degradation_config_sha256=degradation)
    atomic_torch_save(checkpoint_payload(4, 19, model, optimizer, scaler, variant="D", **provenance, run_kind="formal", epoch_completed=True, batch_index=None, smoke_target=None), path)
    expected_python, expected_numpy, expected_torch = random.random(), np.random.rand(), torch.rand(1)
    restored = torch.nn.Linear(3, 2)
    restored_optimizer = torch.optim.AdamW(restored.parameters(), lr=1e-3)
    restored_scaler = torch.amp.GradScaler("cuda", enabled=False)
    assert load_completed_epoch_checkpoint(path, restored, restored_optimizer, restored_scaler, torch.device("cpu"), variant="D", expected_provenance=provenance, requested_run_kind="formal") == (5, 0, 19)
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
    provenance = _provenance()
    payload = checkpoint_payload(1, 1, model, optimizer, scaler, variant="B", **provenance, run_kind="formal", epoch_completed=True, batch_index=None, smoke_target=None)
    payload["validation_evaluated"] = True
    atomic_torch_save(payload, path)
    with pytest.raises(ValueError):
        load_completed_epoch_checkpoint(path, model, optimizer, scaler, torch.device("cpu"), variant="B", expected_provenance=provenance, requested_run_kind="formal")
    assert PROTOCOL_COMMIT == "9f54a1c"


def test_completed_smoke_exits_and_formal_rejects_smoke_checkpoint(tmp_path):
    model = torch.nn.Linear(2, 2); optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3); scaler = torch.amp.GradScaler("cuda", enabled=False)
    provenance = _provenance(); path = tmp_path / "smoke.pt"
    atomic_torch_save(checkpoint_payload(1, 5, model, optimizer, scaler, variant="C", **provenance, run_kind="smoke", epoch_completed=False, batch_index=4, smoke_target=5), path)
    assert load_completed_epoch_checkpoint(path, model, optimizer, scaler, torch.device("cpu"), variant="C", expected_provenance=provenance, requested_run_kind="smoke", smoke_target=5) == (1, 5, 5)
    with pytest.raises(ValueError):
        load_completed_epoch_checkpoint(path, model, optimizer, scaler, torch.device("cpu"), variant="C", expected_provenance=provenance, requested_run_kind="formal")


def test_resume_rejects_any_provenance_change_and_teacher_metadata_is_strict(tmp_path):
    model = torch.nn.Linear(2, 2); optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3); scaler = torch.amp.GradScaler("cuda", enabled=False)
    provenance = _provenance(); path = tmp_path / "formal.pt"
    payload = checkpoint_payload(100, 11, model, optimizer, scaler, variant="A", **provenance, run_kind="formal", epoch_completed=True, batch_index=None, smoke_target=None)
    atomic_torch_save(payload, path)
    altered = {**provenance, "ft_config_sha256": "other"}
    with pytest.raises(ValueError):
        load_completed_epoch_checkpoint(path, model, optimizer, scaler, torch.device("cpu"), variant="A", expected_provenance=altered, requested_run_kind="formal")
    validate_teacher_checkpoint_metadata(payload, provenance)
    payload["epoch"] = 99
    with pytest.raises(ValueError):
        validate_teacher_checkpoint_metadata(payload, provenance)


def test_gradient_accumulation_divides_each_micro_loss_and_matches_effective_batch():
    torch.manual_seed(12)
    whole = torch.nn.Linear(2, 1, bias=False)
    accumulated = torch.nn.Linear(2, 1, bias=False)
    accumulated.load_state_dict(whole.state_dict())
    x = torch.randn(4, 2); y = torch.randn(4, 1)
    whole_loss = ((whole(x) - y) ** 2).mean(); whole_loss.backward()
    for micro_x, micro_y in ((x[:2], y[:2]), (x[2:], y[2:])):
        (((accumulated(micro_x) - micro_y) ** 2).mean() / 2.0).backward()
    assert torch.allclose(whole.weight.grad, accumulated.weight.grad, atol=1e-7, rtol=1e-6)


class _TinySegmentationModel(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.projection = torch.nn.Conv2d(3, 2, 1)

    def forward(self, *, pixel_values: torch.Tensor) -> SimpleNamespace:
        return SimpleNamespace(logits=self.projection(pixel_values))


def test_actual_driver_one_synthetic_step_for_all_five_variants():
    torch.manual_seed(5)
    batch = {
        "clean": torch.randn(2, 3, 4, 4), "s1": torch.randn(2, 3, 4, 4),
        "s3": torch.randn(2, 3, 4, 4), "labels": torch.randint(0, 2, (2, 4, 4)),
    }
    for variant in ("A", "B", "C", "D", "E"):
        model = _TinySegmentationModel()
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
        scaler = torch.amp.GradScaler("cuda", enabled=False)
        teacher = copy.deepcopy(model).eval() if variant == "E" else None
        if teacher is not None:
            for parameter in teacher.parameters():
                parameter.requires_grad_(False)
        values = run_one_step(model, optimizer, scaler, batch, variant=variant, epoch=6, batch_index=0, amp=False, teacher=teacher)
        assert all(np.isfinite(value) for value in values.values())
