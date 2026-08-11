"""Boundary- and frequency-gated Semantic Prototype Transport (BPT)."""
from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as functional
from torch import nn

from reliability.sdtc_seg import resized_labels
from reliability.spt_seg import semantic_prototype_transport_loss


@dataclass
class BPTOutput:
    logits: torch.Tensor
    raw_features: tuple[torch.Tensor, ...]
    corrected_features: tuple[torch.Tensor, ...]
    residuals: tuple[torch.Tensor, ...]
    router_logits: tuple[torch.Tensor, ...]
    prototypes: tuple[torch.Tensor, ...]
    boundary_logits: tuple[torch.Tensor, ...]


class BoundaryPrototypeTransportRectifier(nn.Module):
    """SPT rectifier whose residual is modulated by a high-frequency boundary gate."""

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
        self.boundary_head = nn.Conv2d(channels, 1, 1)
        self.boundary_scale = nn.Parameter(torch.zeros(()))
        nn.init.zeros_(self.expert_up.weight)
        nn.init.zeros_(self.expert_up.bias)
        nn.init.zeros_(self.boundary_head.weight)
        nn.init.zeros_(self.boundary_head.bias)

    def forward(self, feature: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
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
        high_frequency = normalized - functional.avg_pool2d(normalized, kernel_size=3, stride=1, padding=1)
        boundary_logits = self.boundary_head(high_frequency)
        boundary_gate = 1.0 + torch.tanh(self.boundary_scale) * boundary_logits.sigmoid()
        residual = residual * boundary_gate
        return feature + residual, residual, router_logits, self.prototypes, boundary_logits


class BPTSegformer(nn.Module):
    def __init__(self, base_model: nn.Module, hidden_sizes: tuple[int, ...] = (32, 64, 160, 256), num_classes: int = 8, reduction: int = 4) -> None:
        super().__init__()
        self.base_model = base_model
        self.rectifiers = nn.ModuleList(BoundaryPrototypeTransportRectifier(channels, num_classes, reduction) for channels in hidden_sizes)

    def forward(self, pixel_values: torch.Tensor) -> BPTOutput:
        encoded = self.base_model.segformer(pixel_values, output_hidden_states=True, return_dict=True)
        raw = tuple(encoded.hidden_states)
        values = tuple(rectifier(feature) for rectifier, feature in zip(self.rectifiers, raw, strict=True))
        corrected = tuple(item[0] for item in values); residuals = tuple(item[1] for item in values); routers = tuple(item[2] for item in values); prototypes = tuple(item[3] for item in values); boundaries = tuple(item[4] for item in values)
        logits = self.base_model.decode_head(corrected)
        return BPTOutput(logits, raw, corrected, residuals, routers, prototypes, boundaries)


def boundary_supervision_loss(boundary_logits: tuple[torch.Tensor, ...], labels: torch.Tensor, *, ignore_index: int = 255, positive_weight: float = 4.0) -> torch.Tensor:
    """Supervise boundaries from local class discontinuities at every feature scale."""
    terms = []
    for logits in boundary_logits:
        target = resized_labels(labels, logits)
        valid = target.ne(ignore_index)
        edge = torch.zeros_like(valid)
        for dy, dx in ((1, 0), (-1, 0), (0, 1), (0, -1)):
            shifted = torch.roll(target, shifts=(dy, dx), dims=(-2, -1))
            shifted_valid = torch.roll(valid, shifts=(dy, dx), dims=(-2, -1))
            edge |= valid & shifted_valid & target.ne(shifted)
        if bool(valid.any()):
            terms.append(functional.binary_cross_entropy_with_logits(logits[:, 0][valid], edge.float()[valid], pos_weight=torch.tensor(positive_weight, device=logits.device)))
    if not terms:
        return sum(item.sum() for item in boundary_logits) * 0.0
    return torch.stack(terms).mean()


def rectifier_parameter_count(model: BPTSegformer) -> int:
    return sum(parameter.numel() for parameter in model.rectifiers.parameters())


__all__ = ["BPTOutput", "BPTSegformer", "BoundaryPrototypeTransportRectifier", "boundary_supervision_loss", "rectifier_parameter_count", "semantic_prototype_transport_loss"]
