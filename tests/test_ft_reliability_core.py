from __future__ import annotations

import copy
from pathlib import Path

import pytest
import torch
import torch.nn.functional as functional

from reliability.ft_reliability import (
    IGNORE_INDEX, assert_method_train_access, clean_retention_kl,
    deterministic_correctness_pair_sampler, deterministic_transition_sampler,
    failure_transition_loss, failure_transition_mask, generic_correctness_ranking_loss,
    top1_top2_logit_gap, trajectory_family, validate_method_train_membership, validate_split_manifest,
)


def _transition_logits(q_s3: float = 1.0) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    # s1 predicts class 0 correctly; s3 predicts class 1 incorrectly.
    s1 = torch.tensor([[[[2.0]], [[0.0]]]], requires_grad=True)
    s3 = torch.tensor([[[[0.0]], [[q_s3]]]], requires_grad=True)
    labels = torch.tensor([[[0]]])
    boundary = torch.tensor([[[True]]])
    return s1, s3, labels, boundary


def test_top_gap_and_failure_transition_mask_are_exact_and_detached():
    s1, s3, labels, _ = _transition_logits()
    assert torch.equal(top1_top2_logit_gap(s1), torch.tensor([[[2.0]]]))
    assert torch.equal(top1_top2_logit_gap(s3), torch.tensor([[[1.0]]]))
    transition = failure_transition_mask(s1, s3, labels)
    assert transition.dtype is torch.bool and transition.item()
    assert not transition.requires_grad
    assert not failure_transition_mask(s1, s1, labels).item()


def test_ft_direction_detaches_s1_and_reduces_severe_wrong_gap():
    s1_low, s3_low, labels, boundary = _transition_logits(0.4)
    low, _ = failure_transition_loss(s1_low, s3_low, labels, boundary, epoch=6, batch_index=0, max_pixels=1)
    s1_high, s3_high, _, _ = _transition_logits(1.2)
    high, _ = failure_transition_loss(s1_high, s3_high, labels, boundary, epoch=6, batch_index=0, max_pixels=1)
    assert low < high
    high.backward()
    assert s1_high.grad is None
    assert s3_high.grad is not None
    # Gradient descent decreases the current wrong top logit, hence its top gap.
    assert s3_high.grad[0, 1, 0, 0] > 0


def test_ft_empty_transition_is_finite_graph_connected_zero():
    s1, s3, labels, boundary = _transition_logits()
    s3 = s3.detach().clone().requires_grad_()  # s3 now predicts class 0 correctly.
    s3.data[0, 0, 0, 0] = 2.0
    loss, counts = failure_transition_loss(s1, s3, labels, boundary, epoch=6, batch_index=0, max_pixels=1)
    assert loss.item() == 0.0 and counts == {"boundary": 0, "interior": 0}
    loss.backward()
    assert s3.grad is not None


def test_generic_correctness_ranking_direction_empty_pool_and_determinism():
    # Each region has one correct q=3 and one wrong q=1 pixel.
    logits = torch.tensor([[[[3.0, 0.0, 3.0, 0.0]], [[0.0, 1.0, 0.0, 1.0]]]], requires_grad=True)
    labels = torch.tensor([[[0, 0, 0, 0]]])
    boundary = torch.tensor([[[True, True, False, False]]])
    good, counts = generic_correctness_ranking_loss(logits, labels, boundary, epoch=6, batch_index=2, pairs=8)
    worse = logits.detach().clone(); worse[0, 0, 0, 0] = 1.0; worse[0, 1, 0, 1] = 3.0
    bad, _ = generic_correctness_ranking_loss(worse.requires_grad_(), labels, boundary, epoch=6, batch_index=2, pairs=8)
    assert good < bad and sum(counts.values()) == 8
    correct = torch.tensor([[[True, True]]]); labels_ignore = torch.full((1, 1, 2), IGNORE_INDEX)
    c, w, counts = deterministic_correctness_pair_sampler(correct, torch.tensor([[[True, False]]]), labels_ignore, epoch=1, batch_index=1)
    assert len(c) == len(w) == 0 and counts == {"boundary": 0, "interior": 0}


def test_transition_sampler_uses_valid_pixels_and_fixed_fallback():
    transition = torch.tensor([[[True, True, True, False, True, True]]])
    boundary = torch.tensor([[[True, True, False, False, False, False]]])
    labels = torch.tensor([[[0, 0, 0, IGNORE_INDEX, 0, 0]]])
    indices_a, counts_a = deterministic_transition_sampler(transition, boundary, labels, epoch=3, batch_index=4, max_pixels=6)
    indices_b, counts_b = deterministic_transition_sampler(transition, boundary, labels, epoch=3, batch_index=4, max_pixels=6)
    assert torch.equal(indices_a, indices_b) and counts_a == counts_b
    assert all(int(index) != 3 for index in indices_a)
    assert counts_a["boundary"] > 0 and counts_a["interior"] > 0


def test_clean_retention_masks_ignore_and_keeps_teacher_gradient_free():
    teacher = torch.tensor([[[[2.0, 1.0]], [[0.0, 0.0]]]], requires_grad=True)
    student = teacher.detach().clone().requires_grad_()
    labels = torch.tensor([[[0, IGNORE_INDEX]]])
    same = clean_retention_kl(teacher, student, labels)
    assert same.item() == pytest.approx(0.0, abs=1e-7)
    changed = student.detach().clone(); changed[0, 0, 0, 0] = -2.0
    value = clean_retention_kl(teacher, changed.requires_grad_(), labels)
    assert value > 0
    value.backward()
    assert teacher.grad is None


def test_trajectory_cycles_four_families_without_s2_or_cross_family():
    views = [trajectory_family("scene-7", epoch) for epoch in range(1, 5)]
    assert {value[1].split("_")[0] for value in views} == {"color", "turbidity", "lowlight", "blur"}
    for clean, s1, s3 in views:
        assert clean == "clean" and s1.endswith("_s1") and s3.endswith("_s3")
        assert "s2" not in s1 + s3 and s1.split("_")[0] == s3.split("_")[0]


def test_sequential_three_view_backward_equals_joint_mean_ce_gradient():
    torch.manual_seed(7)
    joint = torch.nn.Conv2d(2, 2, 1, bias=False)
    sequential = copy.deepcopy(joint)
    inputs = [torch.randn(1, 2, 2, 2) for _ in range(3)]
    labels = torch.tensor([[[0, 1], [1, 0]]])
    joint_loss = sum(functional.cross_entropy(joint(value), labels) for value in inputs) / 3.0
    joint_loss.backward()
    for value in inputs:
        (functional.cross_entropy(sequential(value), labels) / 3.0).backward()
    assert torch.allclose(joint.weight.grad, sequential.weight.grad, atol=1e-7, rtol=1e-6)


def test_access_guard_and_metadata_only_split_audit():
    allowed = assert_method_train_access(Path("data/suim_processed/splits/aquariskmap_risk_head_v1/risk_head_train.csv"))
    assert allowed.name == "risk_head_train.csv"
    for bad in ("risk_head_development.csv", "val.csv", "calibration.csv", "test.csv", "UIIS_train.csv"):
        with pytest.raises(PermissionError):
            assert_method_train_access(Path(bad))
    with pytest.raises(PermissionError):
        assert_method_train_access(Path("untrusted/risk_head_train.csv"), allowed_directory=Path("trusted"))
    validate_method_train_membership(["a", "b"], ["a", "b"])
    with pytest.raises(PermissionError):
        validate_method_train_membership(["a", "forbidden_test_id"], ["a", "b"])
    validate_split_manifest({"method_train_count": 936, "method_development_count": 231, "sample_id_overlap": 0, "scene_group_overlap": 0, "exact_duplicate_overlap": 0})
    with pytest.raises(ValueError):
        validate_split_manifest({"method_train_count": 936, "method_development_count": 231, "sample_id_overlap": 1, "scene_group_overlap": 0, "exact_duplicate_overlap": 0})
