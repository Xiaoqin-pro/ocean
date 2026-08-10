"""Semantic degradation-tangent cancellation modules and losses."""
from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as functional
from torch import nn


@dataclass
class SDTCOutput:
    logits: torch.Tensor
    raw_features: tuple[torch.Tensor, ...]
    corrected_features: tuple[torch.Tensor, ...]
    residuals: tuple[torch.Tensor, ...]


class SpatialTangentRectifier(nn.Module):
    """Small identity-initialized spatial residual adapter."""

    def __init__(self, channels: int, reduction: int = 4) -> None:
        super().__init__()
        hidden = max(8, channels // reduction)
        self.norm = nn.GroupNorm(1, channels, affine=False)
        self.down = nn.Conv2d(channels, hidden, 1)
        self.depthwise = nn.Conv2d(hidden, hidden, 3, padding=1, groups=hidden)
        self.up = nn.Conv2d(hidden, channels, 1)
        nn.init.zeros_(self.up.weight)
        nn.init.zeros_(self.up.bias)

    def forward(self, feature: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        value = functional.gelu(self.down(self.norm(feature)))
        value = functional.gelu(self.depthwise(value))
        residual = self.up(value)
        return feature + residual, residual


class SDTCSegformer(nn.Module):
    """SegFormer with multi-scale spatial tangent rectifiers before decoding."""

    def __init__(
        self,
        base_model: nn.Module,
        hidden_sizes: tuple[int, ...] = (32, 64, 160, 256),
        reduction: int = 4,
    ) -> None:
        super().__init__()
        self.base_model = base_model
        self.rectifiers = nn.ModuleList(SpatialTangentRectifier(channels, reduction) for channels in hidden_sizes)

    def forward(self, pixel_values: torch.Tensor) -> SDTCOutput:
        encoded = self.base_model.segformer(
            pixel_values,
            output_hidden_states=True,
            return_dict=True,
        )
        raw = tuple(encoded.hidden_states)
        pairs = tuple(rectifier(feature) for rectifier, feature in zip(self.rectifiers, raw, strict=True))
        corrected = tuple(item[0] for item in pairs)
        residuals = tuple(item[1] for item in pairs)
        logits = self.base_model.decode_head(corrected)
        return SDTCOutput(logits=logits, raw_features=raw, corrected_features=corrected, residuals=residuals)


def resized_labels(labels: torch.Tensor, feature: torch.Tensor) -> torch.Tensor:
    return functional.interpolate(
        labels.float().unsqueeze(1), size=feature.shape[-2:], mode="nearest"
    )[:, 0].long()


def semantic_tangent_loss(
    clean_features: tuple[torch.Tensor, ...],
    degraded_features: tuple[torch.Tensor, ...],
    residuals: tuple[torch.Tensor, ...],
    labels: torch.Tensor,
    *,
    num_classes: int = 8,
    ignore_index: int = 255,
    minimum_pixels: int = 4,
) -> tuple[torch.Tensor, int]:
    """Regress class-wise residual centroids toward inverse degradation tangents.

    The target tangent is detached so the encoder cannot reduce this loss by
    moving both endpoints. Each scene/class/scale contributes equally.
    """
    if not (len(clean_features) == len(degraded_features) == len(residuals)):
        raise ValueError("SDTC feature tuples must have equal lengths.")
    terms: list[torch.Tensor] = []
    total_pairs = 0
    for clean, degraded, residual in zip(clean_features, degraded_features, residuals, strict=True):
        if clean.shape != degraded.shape or degraded.shape != residual.shape:
            raise ValueError("SDTC paired features must share shape.")
        native_labels = resized_labels(labels, degraded)
        batch_size, channels = degraded.shape[:2]
        flat_labels = native_labels.reshape(batch_size, -1)
        valid = flat_labels.ne(ignore_index) & flat_labels.ge(0) & flat_labels.lt(num_classes)
        offsets = torch.arange(batch_size, device=degraded.device).unsqueeze(1) * num_classes
        indices = (flat_labels + offsets)[valid]
        counts = torch.bincount(indices, minlength=batch_size * num_classes)

        def centers(feature: torch.Tensor) -> torch.Tensor:
            values = feature.float().flatten(2).transpose(1, 2)[valid]
            sums = torch.zeros((batch_size * num_classes, channels), device=feature.device, dtype=torch.float32)
            sums.index_add_(0, indices, values)
            return sums / counts.clamp_min(1).unsqueeze(1)

        clean_centers = centers(clean)
        degraded_centers = centers(degraded)
        predicted = centers(residual)
        eligible = counts.ge(minimum_pixels)
        target = (clean_centers - degraded_centers).detach()
        scale = target.square().mean(1).sqrt().clamp_min(0.05).unsqueeze(1)
        per_pair = functional.smooth_l1_loss(predicted / scale, target / scale, reduction="none").mean(1)
        if bool(eligible.any()):
            terms.append(per_pair[eligible].mean())
            total_pairs += int(eligible.sum())
    if not terms:
        return sum(item.sum() for item in residuals) * 0.0, 0
    return torch.stack(terms).mean(), total_pairs


def clean_identity_loss(
    raw_features: tuple[torch.Tensor, ...], residuals: tuple[torch.Tensor, ...]
) -> torch.Tensor:
    """Keep rectifiers near identity for already clean training views."""
    if len(raw_features) != len(residuals):
        raise ValueError("SDTC identity tuples must have equal lengths.")
    terms = []
    for raw, residual in zip(raw_features, residuals, strict=True):
        scale = raw.detach().float().square().mean().clamp_min(1e-4)
        terms.append(residual.float().square().mean() / scale)
    return torch.stack(terms).mean()


def rectifier_parameter_count(model: SDTCSegformer) -> int:
    return sum(parameter.numel() for parameter in model.rectifiers.parameters())
