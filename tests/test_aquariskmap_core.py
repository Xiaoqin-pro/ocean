import numpy as np
import pytest
import torch

from reliability.aquariskmap import (
    CONDITIONS,
    FEATURE_SCHEMA_VERSION,
    AquaRiskMap,
    atomic_torch_save,
    build_features,
    full_resolution_risk,
    paired_batch_plan,
    ranking_loss,
    trajectory_loss,
    validate_cache_payload,
    weighted_error_bce,
)


def test_features_are_14_channel_finite_and_class_count_agnostic():
    rgb = torch.randn(2, 3, 384, 384)
    for classes in (4, 8):
        features = build_features(torch.randn(2, classes, 24, 24), rgb)
        assert features.shape == (2, 14, 96, 96)
        assert torch.isfinite(features).all()


def test_head_shape_budget_and_no_attention():
    head = AquaRiskMap()
    assert head(torch.randn(3, 14, 96, 96)).shape == (3, 1, 96, 96)
    assert head.parameter_count() < 200_000
    assert not any("attention" in name.lower() for name, _ in head.named_modules())


def test_cache_schema_rejects_test_and_bad_shape():
    payload = {
        "features": np.zeros((13, 14, 96, 96), dtype=np.float16),
        "predicted_class": np.zeros((13, 384, 384), dtype=np.uint8),
        "sample_id": "sample", "scene_group_id": "scene", "conditions": CONDITIONS,
        "split": "risk_head_train", "model_name": "segformer", "feature_schema_version": FEATURE_SCHEMA_VERSION,
        "checkpoint_sha256": "checkpoint", "degradation_config_sha256": "degradation", "source_image_sha256": "image", "source_mask_sha256": "mask",
        "official_suim_test_evaluated": False,
    }
    validate_cache_payload(payload, split="risk_head_train", model_name="segformer")
    with pytest.raises(ValueError, match="only risk_head"):
        validate_cache_payload(payload, split="test", model_name="segformer")
    payload["features"] = np.zeros((12, 14, 96, 96), dtype=np.float16)
    with pytest.raises(ValueError, match="features"):
        validate_cache_payload(payload, split="risk_head_train", model_name="segformer")


def test_weighted_bce_ignores_ignore_pixels_and_uses_full_resolution():
    scores = full_resolution_risk(torch.zeros(1, 1, 96, 96))
    errors = torch.zeros(1, 384, 384, dtype=torch.bool)
    labels = torch.zeros(1, 384, 384, dtype=torch.long)
    labels[:, :8, :8] = 255
    errors[:, 100, 100] = True
    value = weighted_error_bce(scores, errors, labels, positive_weight=2.0)
    assert value.isfinite() and value > 0
    with pytest.raises(ValueError, match=r"\[1,10\]"):
        weighted_error_bce(scores, errors, labels, positive_weight=11.0)


def test_boundary_bce_weight_is_numerically_stronger_than_interior_weight():
    labels = torch.zeros(1, 32, 32, dtype=torch.long); labels[:, :, 16:] = 1
    scores = torch.zeros(1, 1, 32, 32)
    boundary_error = torch.zeros(1, 32, 32, dtype=torch.bool); boundary_error[:, 15, 15] = True
    interior_error = torch.zeros(1, 32, 32, dtype=torch.bool); interior_error[:, 15, 4] = True
    scores[:, 0, 15, 15] = -5.0; scores[:, 0, 15, 4] = -5.0
    assert weighted_error_bce(scores, boundary_error, labels, positive_weight=2.0) > weighted_error_bce(scores, interior_error, labels, positive_weight=2.0)


def test_ranking_loss_direction_empty_case_and_determinism():
    labels = torch.zeros(1, 4, 4, dtype=torch.long)
    errors = torch.zeros(1, 4, 4, dtype=torch.bool)
    errors[:, :2] = True
    good = torch.where(errors, torch.full_like(errors, True, dtype=torch.float32) * 3.0, torch.zeros_like(errors, dtype=torch.float32)).unsqueeze(1)
    bad = -good
    assert ranking_loss(good, errors, labels, epoch=1, batch_index=2, pairs=64) < ranking_loss(bad, errors, labels, epoch=1, batch_index=2, pairs=64)
    assert ranking_loss(good, errors, labels, epoch=1, batch_index=2, pairs=64) == ranking_loss(good, errors, labels, epoch=1, batch_index=2, pairs=64)
    assert ranking_loss(good, torch.zeros_like(errors), labels, epoch=0, batch_index=0).item() == 0.0


def test_trajectory_loss_requires_matching_pairs_and_new_errors():
    labels = torch.zeros(1, 4, 4, dtype=torch.long)
    clean_errors = torch.zeros(1, 4, 4, dtype=torch.bool)
    degraded_errors = clean_errors.clone(); degraded_errors[:, 0, 0] = True
    clean = torch.zeros(1, 1, 4, 4)
    degraded = torch.full((1, 1, 4, 4), 8.0)
    assert trajectory_loss(clean, degraded, clean_errors, degraded_errors, labels, ["a"], ["a"]).item() == 0.0
    slightly_degraded = torch.full((1, 1, 4, 4), 0.1)
    assert trajectory_loss(clean, slightly_degraded, clean_errors, degraded_errors, labels, ["a"], ["a"]).item() > 0.0
    assert trajectory_loss(clean, clean, clean_errors, clean_errors, labels, ["a"], ["a"]).item() == 0.0
    with pytest.raises(ValueError, match="matching sample_id"):
        trajectory_loss(clean, degraded, clean_errors, degraded_errors, labels, ["a"], ["b"])


def test_paired_sampler_and_atomic_checkpoint(tmp_path):
    plan_a = paired_batch_plan([f"scene_{index}" for index in range(8)], epoch=3)
    plan_b = paired_batch_plan([f"scene_{index}" for index in range(8)], epoch=3)
    assert plan_a == plan_b
    assert all(len(batch) == 8 for batch in plan_a)
    for batch in plan_a:
        assert [sample for sample, _ in batch[::2]] == [sample for sample, _ in batch[1::2]]
        assert all(condition == "clean" for _, condition in batch[::2])
        assert all(condition != "clean" for _, condition in batch[1::2])
    observed = {
        condition
        for epoch in range(12)
        for batch in paired_batch_plan(["scene_0", "scene_1", "scene_2", "scene_3"], epoch=epoch)
        for sample_id, condition in batch
        if sample_id == "scene_0" and condition != "clean"
    }
    assert len(observed) == 12
    path = tmp_path / "last.pt"
    atomic_torch_save({"epoch": 1}, path)
    assert torch.load(path, map_location="cpu", weights_only=False)["epoch"] == 1
