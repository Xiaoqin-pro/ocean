"""Preregistered AquaRiskMap feature, loss, sampler, and cache primitives.

This module deliberately contains no dataset split other than risk-head
train/development and no evaluation path for the SUIM official TEST split.
"""
from __future__ import annotations

import hashlib
import math
import os
import tempfile
from pathlib import Path
from typing import Iterable, Mapping, Sequence

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as functional


FEATURE_SCHEMA_VERSION = "aquariskmap_features_v1"
CONDITIONS = ("clean", "color_s1", "color_s2", "color_s3", "turbidity_s1", "turbidity_s2", "turbidity_s3", "lowlight_s1", "lowlight_s2", "lowlight_s3", "blur_s1", "blur_s2", "blur_s3")
DEGRADED_CONDITIONS = CONDITIONS[1:]
GRID_SIZE = 96
IMAGE_SIZE = 384
INPUT_CHANNELS = 14
IGNORE_INDEX = 255


def _neighbor_features(prediction: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Return 8-neighbor disagreement fraction and 4-neighbor boundary flag."""
    if prediction.ndim != 3:
        raise ValueError("Prediction must be [B,H,W].")
    batch, height, width = prediction.shape
    disagreement = torch.zeros((batch, height, width), device=prediction.device, dtype=torch.float32)
    neighbor_count = torch.zeros_like(disagreement)
    boundary = torch.zeros((batch, height, width), device=prediction.device, dtype=torch.bool)
    for dy, dx in ((-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1)):
        src_y0, src_y1 = max(0, -dy), height - max(0, dy)
        src_x0, src_x1 = max(0, -dx), width - max(0, dx)
        dst_y0, dst_y1 = max(0, dy), height - max(0, -dy)
        dst_x0, dst_x1 = max(0, dx), width - max(0, -dx)
        different = prediction[:, dst_y0:dst_y1, dst_x0:dst_x1].ne(prediction[:, src_y0:src_y1, src_x0:src_x1])
        disagreement[:, dst_y0:dst_y1, dst_x0:dst_x1] += different.float()
        neighbor_count[:, dst_y0:dst_y1, dst_x0:dst_x1] += 1.0
        if abs(dy) + abs(dx) == 1:
            boundary[:, dst_y0:dst_y1, dst_x0:dst_x1] |= different
    return disagreement / neighbor_count.clamp_min(1.0), boundary.float()


def build_features(logits: torch.Tensor, normalized_rgb: torch.Tensor, *, grid_size: int = GRID_SIZE) -> torch.Tensor:
    """Build the frozen 14-channel class-agnostic AquaRiskMap representation.

    Logits are bilinearly resized before softmax.  RGB is resized to the same
    grid.  The representation never exposes class-indexed channels.
    """
    if logits.ndim != 4 or normalized_rgb.ndim != 4 or logits.shape[0] != normalized_rgb.shape[0]:
        raise ValueError("Logits and RGB must be [B,C,H,W] and [B,3,H,W] with equal batch size.")
    if normalized_rgb.shape[1] != 3 or logits.shape[1] < 2:
        raise ValueError("AquaRiskMap requires RGB and at least two segmentation classes.")
    logits = functional.interpolate(logits.float(), size=(grid_size, grid_size), mode="bilinear", align_corners=False)
    rgb = functional.interpolate(normalized_rgb.float(), size=(grid_size, grid_size), mode="bilinear", align_corners=False)
    probabilities = logits.softmax(dim=1)
    classes = logits.shape[1]
    top_count = min(4, classes)
    top = probabilities.topk(top_count, dim=1).values
    if top_count < 4:
        top = functional.pad(top, (0, 0, 0, 0, 0, 4 - top_count))
    gaps = torch.stack((top[:, 0] - top[:, 1], top[:, 1] - top[:, 2], top[:, 2] - top[:, 3]), dim=1)
    entropy = -(probabilities * probabilities.clamp_min(torch.finfo(probabilities.dtype).eps).log()).sum(dim=1, keepdim=True) / math.log(classes)
    normalized_logsumexp = torch.logsumexp(logits, dim=1, keepdim=True) - math.log(classes)
    prediction = probabilities.argmax(dim=1)
    disagreement, boundary = _neighbor_features(prediction)
    result = torch.cat((rgb, top, gaps, entropy, normalized_logsumexp, disagreement.unsqueeze(1), boundary.unsqueeze(1)), dim=1)
    if result.shape[1] != INPUT_CHANNELS or not torch.isfinite(result).all():
        raise AssertionError("Invalid AquaRiskMap feature tensor.")
    return result


class _DepthwiseSeparable(nn.Module):
    def __init__(self, channels_in: int, channels_out: int, *, dilation: int = 1) -> None:
        super().__init__()
        self.depthwise = nn.Conv2d(channels_in, channels_in, 3, padding=dilation, dilation=dilation, groups=channels_in, bias=False)
        self.pointwise = nn.Conv2d(channels_in, channels_out, 1, bias=False)
        self.norm = nn.GroupNorm(8, channels_out)
        self.activation = nn.GELU()

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        return self.activation(self.norm(self.pointwise(self.depthwise(value))))


class AquaRiskMap(nn.Module):
    """The preregistered <0.20M-parameter error-risk head."""
    def __init__(self, channels: int = INPUT_CHANNELS, width: int = 64) -> None:
        super().__init__()
        if channels != INPUT_CHANNELS or width != 64:
            raise ValueError("AquaRiskMap architecture is preregistered as 14 -> 64 only.")
        self.stem = _DepthwiseSeparable(channels, width)
        self.blocks = nn.ModuleList([_DepthwiseSeparable(width, width, dilation=dilation) for dilation in (1, 2, 3)])
        self.head = nn.Conv2d(width, 1, 1)
        if self.parameter_count() >= 200_000:
            raise AssertionError("AquaRiskMap exceeds its preregistered parameter budget.")

    def parameter_count(self) -> int:
        return sum(parameter.numel() for parameter in self.parameters() if parameter.requires_grad)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        if features.ndim != 4 or features.shape[1] != INPUT_CHANNELS or features.shape[-2:] != (GRID_SIZE, GRID_SIZE):
            raise ValueError("AquaRiskMap expects [B,14,96,96] features.")
        value = self.stem(features.float())
        for block in self.blocks:
            value = value + block(value)
        return self.head(value)


def full_resolution_risk(logits: torch.Tensor, *, size: tuple[int, int] = (IMAGE_SIZE, IMAGE_SIZE)) -> torch.Tensor:
    if logits.ndim != 4 or logits.shape[1] != 1:
        raise ValueError("Risk logits must be [B,1,H,W].")
    return functional.interpolate(logits.float(), size=size, mode="bilinear", align_corners=False)


def gt_boundary(labels: torch.Tensor, *, radius: int = 3, ignore_index: int = IGNORE_INDEX) -> torch.Tensor:
    if labels.ndim != 3 or radius < 1:
        raise ValueError("Labels must be [B,H,W] and radius must be positive.")
    # This deliberately matches the radius-r definition already used by the
    # frozen UWR-Bench boundary analysis.  Ignore pixels are removed later by
    # the valid-mask in the loss, exactly as in that analysis.
    values = labels.float().unsqueeze(1)
    maximum = functional.max_pool2d(values, kernel_size=2 * radius + 1, stride=1, padding=radius)
    minimum = -functional.max_pool2d(-values, kernel_size=2 * radius + 1, stride=1, padding=radius)
    return maximum[:, 0].ne(minimum[:, 0])


def weighted_error_bce(risk_logits: torch.Tensor, errors: torch.Tensor, labels: torch.Tensor, *, positive_weight: float, ignore_index: int = IGNORE_INDEX) -> torch.Tensor:
    if not (1.0 <= positive_weight <= 10.0):
        raise ValueError("Positive error weight must be in [1,10].")
    if risk_logits.shape[1] != 1 or errors.shape != labels.shape or risk_logits.shape[0] != labels.shape[0] or risk_logits.shape[-2:] != labels.shape[-2:]:
        raise ValueError("Risk logits, errors, and labels have incompatible shapes.")
    valid = labels.ne(ignore_index)
    if not valid.any():
        return risk_logits.float().sum() * 0.0
    boundary = gt_boundary(labels, ignore_index=ignore_index)
    target = errors.float()
    weights = valid.float() * (1.0 + boundary.float()) * torch.where(errors, torch.full_like(target, positive_weight), torch.ones_like(target))
    value = functional.binary_cross_entropy_with_logits(risk_logits[:, 0].float(), target, reduction="none")
    return (value * weights).sum() / weights.sum().clamp_min(1.0)


def _draw(indices: torch.Tensor, count: int, generator: torch.Generator) -> torch.Tensor:
    if len(indices) >= count:
        return indices[torch.randperm(len(indices), generator=generator)[:count]]
    return indices[torch.randint(len(indices), (count,), generator=generator)]


def ranking_loss(risk_logits: torch.Tensor, errors: torch.Tensor, labels: torch.Tensor, *, epoch: int, batch_index: int, pairs: int = 2048, ignore_index: int = IGNORE_INDEX) -> torch.Tensor:
    if pairs < 1:
        raise ValueError("Ranking pair count must be positive.")
    scores = risk_logits[:, 0].float().reshape(-1)
    flat_errors = errors.reshape(-1).bool()
    valid = labels.reshape(-1).ne(ignore_index)
    error_indices = torch.nonzero(valid & flat_errors, as_tuple=False).flatten().cpu()
    correct_indices = torch.nonzero(valid & ~flat_errors, as_tuple=False).flatten().cpu()
    if len(error_indices) == 0 or len(correct_indices) == 0:
        return scores.sum() * 0.0
    generator = torch.Generator(device="cpu").manual_seed(20260725 + 100003 * epoch + batch_index)
    selected_errors = _draw(error_indices, pairs, generator).to(scores.device)
    selected_correct = _draw(correct_indices, pairs, generator).to(scores.device)
    return functional.softplus(scores[selected_correct] - scores[selected_errors]).mean()


def trajectory_loss(clean_logits: torch.Tensor, degraded_logits: torch.Tensor, clean_errors: torch.Tensor, degraded_errors: torch.Tensor, labels: torch.Tensor, clean_ids: Sequence[str], degraded_ids: Sequence[str], *, margin: float = 0.10, ignore_index: int = IGNORE_INDEX) -> torch.Tensor:
    if tuple(clean_ids) != tuple(degraded_ids):
        raise ValueError("Trajectory pairs must have matching sample_id order.")
    if clean_logits.shape != degraded_logits.shape or clean_errors.shape != degraded_errors.shape or clean_errors.shape != labels.shape:
        raise ValueError("Trajectory inputs have incompatible shapes.")
    new_error = ~clean_errors.bool() & degraded_errors.bool() & labels.ne(ignore_index)
    if not new_error.any():
        return (clean_logits.float().sum() + degraded_logits.float().sum()) * 0.0
    clean_risk = clean_logits[:, 0].float().sigmoid()
    degraded_risk = degraded_logits[:, 0].float().sigmoid()
    return functional.relu(margin - (degraded_risk - clean_risk))[new_error].mean()


def stable_hash(sample_id: str) -> int:
    return int.from_bytes(hashlib.sha256(sample_id.encode("utf-8")).digest()[:8], "big")


def paired_batch_plan(sample_ids: Iterable[str], *, epoch: int, base_scenes_per_batch: int = 4) -> list[list[tuple[str, str]]]:
    ids = sorted(set(str(value) for value in sample_ids))
    if len(ids) % base_scenes_per_batch:
        raise ValueError("Risk-head train sample count must divide into complete paired batches.")
    generator = torch.Generator(device="cpu").manual_seed(20260725 + epoch)
    order = torch.randperm(len(ids), generator=generator).tolist()
    shuffled = [ids[index] for index in order]
    batches: list[list[tuple[str, str]]] = []
    for start in range(0, len(shuffled), base_scenes_per_batch):
        scenes = shuffled[start:start + base_scenes_per_batch]
        pairs: list[tuple[str, str]] = []
        for sample_id in scenes:
            condition = DEGRADED_CONDITIONS[(stable_hash(sample_id) + epoch) % len(DEGRADED_CONDITIONS)]
            pairs.extend(((sample_id, "clean"), (sample_id, condition)))
        batches.append(pairs)
    return batches


def validate_cache_payload(payload: Mapping[str, object], *, split: str, model_name: str, expected_conditions: Sequence[str] = CONDITIONS) -> None:
    if split not in {"risk_head_train", "risk_head_development"}:
        raise ValueError("AquaRiskMap cache permits only risk_head_train/development.")
    required = {"features", "predicted_class", "sample_id", "scene_group_id", "conditions", "split", "model_name", "feature_schema_version", "checkpoint_sha256", "degradation_config_sha256", "source_image_sha256", "source_mask_sha256", "official_suim_test_evaluated"}
    missing = required.difference(payload)
    if missing:
        raise ValueError(f"Cache payload is missing keys: {sorted(missing)}")
    scalar = lambda key: payload[key].item() if isinstance(payload[key], np.ndarray) and payload[key].shape == () else payload[key]
    if scalar("split") != split or scalar("model_name") != model_name or scalar("feature_schema_version") != FEATURE_SCHEMA_VERSION:
        raise ValueError("Cache metadata does not match the requested context.")
    if bool(scalar("official_suim_test_evaluated")):
        raise ValueError("Official SUIM TEST must remain locked.")
    if not all(isinstance(scalar(key), str) and scalar(key) for key in ("checkpoint_sha256", "degradation_config_sha256", "source_image_sha256", "source_mask_sha256")):
        raise ValueError("Cache provenance hashes must be non-empty strings.")
    features, prediction = payload["features"], payload["predicted_class"]
    if not isinstance(features, np.ndarray) or features.shape != (len(expected_conditions), INPUT_CHANNELS, GRID_SIZE, GRID_SIZE) or features.dtype != np.float16 or not np.isfinite(features).all():
        raise ValueError("Cache features must be finite float16 [13,14,96,96].")
    if not isinstance(prediction, np.ndarray) or prediction.shape != (len(expected_conditions), IMAGE_SIZE, IMAGE_SIZE) or prediction.dtype != np.uint8 or np.any(prediction > 7):
        raise ValueError("Predicted classes must be uint8 [13,384,384].")
    if tuple(str(value) for value in payload["conditions"]) != tuple(expected_conditions):
        raise ValueError("Cache condition order differs from the preregistered registry.")


def atomic_torch_save(payload: Mapping[str, object], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile("wb", suffix=".tmp", dir=path.parent, delete=False) as handle:
        temporary = Path(handle.name)
    try:
        torch.save(dict(payload), temporary)
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()
