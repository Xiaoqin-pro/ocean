"""Source-anchored domain-conditioned prototype transport for two public domains."""
from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as functional
from torch import nn

from reliability.spt_seg import SPTOutput, SPTSegformer


@dataclass
class DCPTOutput:
    logits: torch.Tensor
    gate_logits: torch.Tensor
    gate_probabilities: torch.Tensor
    experts: tuple[SPTOutput, SPTOutput]


class DomainConditionedSPT(nn.Module):
    """Two frozen source backbones with trainable low-rank domain adapters.

    The gate sees pooled encoder features from both source models and routes each
    image without an inference-time domain label.  A shared prototype bank is
    exposed for cross-domain alignment during training.
    """

    def __init__(
        self,
        source_models: tuple[nn.Module, nn.Module],
        hidden_sizes: tuple[int, ...] = (32, 64, 160, 256),
        num_classes: int = 8,
        rectifier_reduction: int = 4,
    ) -> None:
        super().__init__()
        self.experts = nn.ModuleList(
            SPTSegformer(model, hidden_sizes, num_classes, rectifier_reduction)
            for model in source_models
        )
        for expert in self.experts:
            for parameter in expert.base_model.parameters():
                parameter.requires_grad_(False)
        gate_hidden = max(32, hidden_sizes[-1] // 2)
        self.gate = nn.Sequential(
            nn.LayerNorm(hidden_sizes[-1] * 2),
            nn.Linear(hidden_sizes[-1] * 2, gate_hidden),
            nn.GELU(),
            nn.Linear(gate_hidden, 2),
        )
        self.shared_prototypes = nn.ParameterList(
            [nn.Parameter(torch.randn(num_classes, channels) * 0.02) for channels in hidden_sizes]
        )
        self.hard_routing = False

    def forward_expert(self, pixel_values: torch.Tensor, domain: int) -> SPTOutput:
        if domain not in (0, 1):
            raise ValueError("domain must be 0 (SUIM) or 1 (UIIS)")
        return self.experts[domain](pixel_values)

    def gate_from_outputs(self, first: SPTOutput, second: SPTOutput) -> torch.Tensor:
        first_pool = first.raw_features[-1].mean(dim=(2, 3))
        second_pool = second.raw_features[-1].mean(dim=(2, 3))
        return self.gate(torch.cat((first_pool, second_pool), dim=1))

    def forward(self, pixel_values: torch.Tensor) -> DCPTOutput:
        first = self.experts[0](pixel_values)
        second = self.experts[1](pixel_values)
        gate_logits = self.gate_from_outputs(first, second)
        probabilities = gate_logits.softmax(dim=1)
        if self.hard_routing:
            hard = torch.nn.functional.one_hot(probabilities.argmax(dim=1), num_classes=2).to(probabilities.dtype)
            probabilities = hard + probabilities - probabilities.detach()
        logits = probabilities[:, 0, None, None, None] * first.logits + probabilities[:, 1, None, None, None] * second.logits
        return DCPTOutput(logits, gate_logits, probabilities, (first, second))


def shared_prototype_alignment_loss(
    features: tuple[torch.Tensor, ...],
    labels: torch.Tensor,
    shared_prototypes: nn.ParameterList,
    *,
    num_classes: int = 8,
    ignore_index: int = 255,
    minimum_pixels: int = 4,
) -> tuple[torch.Tensor, int]:
    """Align each domain expert's class centroids to a shared semantic bank."""
    terms: list[torch.Tensor] = []
    pairs = 0
    for feature, prototype in zip(features, shared_prototypes, strict=True):
        native = functional.interpolate(labels[:, None].float(), size=feature.shape[-2:], mode="nearest")[:, 0].long()
        flat = native.reshape(native.shape[0], -1)
        valid = flat.ne(ignore_index) & flat.ge(0) & flat.lt(num_classes)
        offsets = torch.arange(flat.shape[0], device=feature.device)[:, None] * num_classes
        indices = (flat + offsets)[valid]
        counts = torch.bincount(indices, minlength=flat.shape[0] * num_classes)
        values = feature.float().flatten(2).transpose(1, 2)[valid]
        sums = torch.zeros((flat.shape[0] * num_classes, feature.shape[1]), device=feature.device)
        sums.index_add_(0, indices, values)
        centers = sums / counts.clamp_min(1).unsqueeze(1)
        eligible = counts.ge(minimum_pixels)
        if bool(eligible.any()):
            target = functional.normalize(prototype, dim=1)
            normalized = functional.normalize(centers, dim=1)
            terms.append((1.0 - (normalized * target.repeat(flat.shape[0], 1))[eligible].sum(dim=1)).mean())
            pairs += int(eligible.sum())
    if not terms:
        return sum(item.sum() for item in features) * 0.0, 0
    return torch.stack(terms).mean(), pairs


def gate_accuracy(gate_logits: torch.Tensor, domain: torch.Tensor) -> torch.Tensor:
    return (gate_logits.argmax(1) == domain).float().mean()
