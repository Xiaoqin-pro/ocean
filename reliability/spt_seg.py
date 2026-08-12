"""Semantic Prototype Transport (SPT) adapters for underwater segmentation."""
from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as functional
from torch import nn

from reliability.sdtc_seg import resized_labels


@dataclass
class SPTOutput:
    logits: torch.Tensor
    raw_features: tuple[torch.Tensor, ...]
    corrected_features: tuple[torch.Tensor, ...]
    residuals: tuple[torch.Tensor, ...]
    router_logits: tuple[torch.Tensor, ...]
    prototypes: tuple[torch.Tensor, ...]


class SemanticPrototypeTransportRectifier(nn.Module):
    """Identity-initialized class-routed rectifier with learned semantic anchors.

    The router combines a local spatial classifier with cosine similarity to
    scale-specific class prototypes.  Experts remain low-rank and spatial,
    so the forward path is lightweight; the prototype loss supplies the
    cross-view semantic transport signal during training.
    """

    def __init__(self, channels: int, num_classes: int = 8, reduction: int = 4) -> None:
        super().__init__()
        self.channels = channels
        self.num_classes = num_classes
        hidden = max(8, channels // reduction)
        self.norm = nn.GroupNorm(1, channels, affine=False)
        self.down = nn.Conv2d(channels, hidden, 1)
        self.depthwise = nn.Conv2d(hidden, hidden, 3, padding=1, groups=hidden)
        self.expert_up = nn.Conv2d(hidden, num_classes * channels, 1)
        self.router = nn.Conv2d(channels, num_classes, 1)
        self.prototype_query = nn.Conv2d(channels, channels, 1, bias=False)
        self.prototypes = nn.Parameter(torch.randn(num_classes, channels) * 0.02)
        nn.init.zeros_(self.expert_up.weight)
        nn.init.zeros_(self.expert_up.bias)

    def forward(self, feature: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        normalized = self.norm(feature)
        latent = functional.gelu(self.down(normalized))
        latent = functional.gelu(self.depthwise(latent))
        batch, _, height, width = feature.shape
        experts = self.expert_up(latent).reshape(batch, self.num_classes, self.channels, height, width)
        local_logits = self.router(normalized)
        query = functional.normalize(self.prototype_query(normalized), dim=1)
        anchors = functional.normalize(self.prototypes, dim=1)
        similarity = torch.einsum("bchw,kc->bkhw", query, anchors)
        router_logits = local_logits + 0.25 * similarity
        probabilities = router_logits.softmax(1)
        residual = (experts * probabilities.unsqueeze(2)).sum(1)
        return feature + residual, residual, router_logits, self.prototypes


class SPTSegformer(nn.Module):
    """SegFormer with multi-scale semantic prototype transport adapters."""

    def __init__(
        self,
        base_model: nn.Module,
        hidden_sizes: tuple[int, ...] = (32, 64, 160, 256),
        num_classes: int = 8,
        reduction: int = 4,
    ) -> None:
        super().__init__()
        self.base_model = base_model
        self.rectifiers = nn.ModuleList(
            SemanticPrototypeTransportRectifier(channels, num_classes, reduction) for channels in hidden_sizes
        )

    def forward(self, pixel_values: torch.Tensor) -> SPTOutput:
        encoded = self.base_model.segformer(pixel_values, output_hidden_states=True, return_dict=True)
        raw = tuple(encoded.hidden_states)
        values = tuple(rectifier(feature) for rectifier, feature in zip(self.rectifiers, raw, strict=True))
        corrected = tuple(item[0] for item in values)
        residuals = tuple(item[1] for item in values)
        routers = tuple(item[2] for item in values)
        prototypes = tuple(item[3] for item in values)
        logits = self.base_model.decode_head(corrected)
        return SPTOutput(logits, raw, corrected, residuals, routers, prototypes)


def semantic_prototype_transport_loss(
    clean_features: tuple[torch.Tensor, ...],
    corrected_features: tuple[torch.Tensor, ...],
    labels: torch.Tensor,
    *,
    num_classes: int = 8,
    ignore_index: int = 255,
    minimum_pixels: int = 4,
    margin: float = 0.15,
) -> tuple[torch.Tensor, int]:
    """Align degraded class centroids to clean anchors and repel other classes."""
    if len(clean_features) != len(corrected_features):
        raise ValueError("SPT feature tuples must have equal lengths.")
    terms: list[torch.Tensor] = []
    total_pairs = 0
    for clean, corrected in zip(clean_features, corrected_features, strict=True):
        if clean.shape != corrected.shape:
            raise ValueError("SPT paired features must share shape.")
        native = resized_labels(labels, corrected)
        batch, channels = corrected.shape[:2]
        flat = native.reshape(batch, -1)
        valid = flat.ne(ignore_index) & flat.ge(0) & flat.lt(num_classes)
        offsets = torch.arange(batch, device=corrected.device).unsqueeze(1) * num_classes
        indices = (flat + offsets)[valid]
        counts = torch.bincount(indices, minlength=batch * num_classes)

        def centers(feature: torch.Tensor) -> torch.Tensor:
            values = feature.float().flatten(2).transpose(1, 2)[valid]
            sums = torch.zeros((batch * num_classes, channels), device=feature.device, dtype=torch.float32)
            sums.index_add_(0, indices, values)
            return sums / counts.clamp_min(1).unsqueeze(1)

        clean_centers = functional.normalize(centers(clean).detach(), dim=1)
        corrected_centers = functional.normalize(centers(corrected), dim=1)
        eligible = counts.ge(minimum_pixels)
        positive = 1.0 - (corrected_centers * clean_centers).sum(1)
        similarity = corrected_centers @ clean_centers.transpose(0, 1)
        identity = torch.arange(batch * num_classes, device=corrected.device)
        negative = (similarity - similarity.diagonal().unsqueeze(1) + margin).clamp_min(0.0)
        negative[torch.arange(batch * num_classes, device=corrected.device), torch.arange(batch * num_classes, device=corrected.device)] = 0.0
        negative = negative.sum(1) / max(num_classes - 1, 1)
        value = positive + negative
        if bool(eligible.any()):
            terms.append(value[eligible].mean())
            total_pairs += int(eligible.sum())
    if not terms:
        return sum(item.sum() for item in corrected_features) * 0.0, 0
    return torch.stack(terms).mean(), total_pairs


def rectifier_parameter_count(model: SPTSegformer) -> int:
    return sum(parameter.numel() for parameter in model.rectifiers.parameters())
