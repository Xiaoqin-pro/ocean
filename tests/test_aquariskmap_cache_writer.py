import hashlib

import numpy as np
import pandas as pd
import pytest
import torch
import torch.nn.functional as functional

from reliability.aquariskmap import CONDITIONS, FEATURE_SCHEMA_VERSION, validate_cache_payload
from scripts.cache_aquariskmap_features import (
    atomic_npz,
    cache_is_complete,
    deterministic_npz_bytes,
    load_npz_payload,
    prediction_from_logits,
    resolve_context,
    validate_membership,
)


def payload():
    return {
        "features": np.zeros((13, 14, 96, 96), dtype=np.float16),
        "predicted_class": np.zeros((13, 384, 384), dtype=np.uint8),
        "conditions": np.asarray(CONDITIONS),
        "sample_id": "a", "scene_group_id": "scene_a", "split": "risk_head_train", "model_name": "segformer",
        "feature_schema_version": FEATURE_SCHEMA_VERSION, "checkpoint_sha256": "checkpoint", "degradation_config_sha256": "degradation",
        "source_image_sha256": "image", "source_mask_sha256": "mask", "official_suim_test_evaluated": False,
    }


def fake_root(tmp_path):
    inner = tmp_path / "data" / "suim_processed" / "splits" / "aquariskmap_risk_head_v1"; inner.mkdir(parents=True)
    formal = tmp_path / "data" / "suim_processed" / "splits" / "v2_scene_grouped_deduplicated"; formal.mkdir(parents=True)
    train = pd.DataFrame({"sample_id": ["a", "b"], "scene_group_id": ["scene_a", "scene_b"]})
    development = pd.DataFrame({"sample_id": ["c"], "scene_group_id": ["scene_c"]})
    train.to_csv(inner / "risk_head_train.csv", index=False); development.to_csv(inner / "risk_head_development.csv", index=False)
    pd.DataFrame({"sample_id": ["v"]}).to_csv(formal / "val.csv", index=False)
    pd.DataFrame({"sample_id": ["k"]}).to_csv(formal / "calibration.csv", index=False)
    pd.DataFrame({"sample_id": ["t"]}).to_csv(formal / "test.csv", index=False)
    for name in ("baseline.yaml", "checkpoint.pt", "degradation.yaml"):
        (tmp_path / name).write_bytes(b"x")
    config = {
        "protocol": {"risk_head_split": "data/suim_processed/splits/aquariskmap_risk_head_v1"},
        "base_models": {"segformer": {"baseline_config": "baseline.yaml", "checkpoint": "checkpoint.pt"}, "deeplab": {"baseline_config": "baseline.yaml", "checkpoint": "checkpoint.pt"}},
        "degradations": {"config": "degradation.yaml"},
    }
    return config, train, tmp_path


def test_deterministic_atomic_cache_and_hash_guard(tmp_path):
    item = payload()
    validate_cache_payload(item, split="risk_head_train", model_name="segformer")
    assert deterministic_npz_bytes(item) == deterministic_npz_bytes(item)
    path = tmp_path / "a.npz"
    atomic_npz(path, item)
    first_hash = hashlib.sha256(path.read_bytes()).hexdigest()
    atomic_npz(path, item)
    assert hashlib.sha256(path.read_bytes()).hexdigest() == first_hash
    assert not list(tmp_path.glob("*.tmp"))
    loaded = load_npz_payload(path)
    validate_cache_payload(loaded, split="risk_head_train", model_name="segformer")
    assert cache_is_complete(path, split="risk_head_train", model_name="segformer", checkpoint_sha256="checkpoint", degradation_config_sha256="degradation", source_image_sha256="image", source_mask_sha256="mask")
    assert not cache_is_complete(path, split="risk_head_train", model_name="segformer", checkpoint_sha256="changed", degradation_config_sha256="degradation", source_image_sha256="image", source_mask_sha256="mask")


def test_context_and_membership_reject_barred_or_wrong_inputs(tmp_path):
    config, train, root = fake_root(tmp_path)
    split, _, _, _ = resolve_context(root, config, model_name="segformer", split="risk_head_train")
    assert split.name == "risk_head_train.csv"
    validate_membership(root, config, split="risk_head_train", frame=train)
    with pytest.raises(ValueError, match="Only preregistered"):
        resolve_context(root, config, model_name="segformer", split="test")
    test_disguised = train.copy(); test_disguised.loc[0, "sample_id"] = "t"
    with pytest.raises(ValueError, match="exactly match"):
        validate_membership(root, config, split="risk_head_train", frame=test_disguised)


def test_predicted_class_is_directly_from_full_resolution_logits():
    logits = torch.tensor([[[[2.0, -2.0], [-2.0, 2.0]], [[-2.0, 2.0], [2.0, -2.0]]]])
    expected = functional.interpolate(logits, size=(384, 384), mode="bilinear", align_corners=False).argmax(dim=1).to(torch.uint8)
    assert torch.equal(prediction_from_logits(logits), expected)


def test_cache_rejects_incomplete_condition_registry():
    item = payload(); item["conditions"] = np.asarray(CONDITIONS[:-1])
    with pytest.raises(ValueError, match="condition order"):
        validate_cache_payload(item, split="risk_head_train", model_name="segformer")
    item = payload(); item["predicted_class"][0, 0, 0] = 8
    with pytest.raises(ValueError, match="Predicted classes"):
        validate_cache_payload(item, split="risk_head_train", model_name="segformer")
