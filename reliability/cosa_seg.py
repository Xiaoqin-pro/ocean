"""Cross-ontology semantic anchoring losses for COSA-Seg."""
from __future__ import annotations

import torch
import torch.nn.functional as functional


def mapped_cross_entropy(logits: torch.Tensor, labels: torch.Tensor, ignore_index: int = 255) -> torch.Tensor:
    """Cross entropy for a mask whose labels have already been mapped to SUIM ids."""
    resized = functional.interpolate(logits.float(), size=labels.shape[-2:], mode="bilinear", align_corners=False)
    return functional.cross_entropy(resized, labels, ignore_index=ignore_index)


def symmetric_kl(logits_a: torch.Tensor, logits_b: torch.Tensor) -> torch.Tensor:
    """Bounded view-consistency loss between two semantic predictions."""
    a = functional.log_softmax(logits_a.float(), dim=1)
    b = functional.log_softmax(logits_b.float(), dim=1)
    forward = functional.kl_div(a, b.exp().detach(), reduction="none").sum(1).mean()
    backward = functional.kl_div(b, a.exp().detach(), reduction="none").sum(1).mean()
    return 0.5 * (forward + backward)


def class_anchor_loss(
    feature_a: torch.Tensor,
    labels_a: torch.Tensor,
    feature_b: torch.Tensor,
    labels_b: torch.Tensor,
    class_ids: tuple[int, ...] = (0, 3, 5, 6),
    ignore_index: int = 255,
) -> torch.Tensor:
    """Align mapped class centroids across the two label ontologies.

    The first view is treated as the replay anchor; gradients flow only through
    the second view. Missing classes in a mini-batch are skipped.
    """
    if feature_a.shape[1] != feature_b.shape[1]:
        raise ValueError("Anchor features must have equal channel counts.")
    labels_a = functional.interpolate(labels_a.unsqueeze(1).float(), size=feature_a.shape[-2:], mode="nearest").squeeze(1).long()
    labels_b = functional.interpolate(labels_b.unsqueeze(1).float(), size=feature_b.shape[-2:], mode="nearest").squeeze(1).long()
    terms: list[torch.Tensor] = []
    for class_id in class_ids:
        masks_a = labels_a.eq(class_id)
        masks_b = labels_b.eq(class_id)
        if not bool(masks_a.any()) or not bool(masks_b.any()):
            continue
        center_a = feature_a.float().permute(0, 2, 3, 1)[masks_a].mean(0).detach()
        center_b = feature_b.float().permute(0, 2, 3, 1)[masks_b].mean(0)
        terms.append(1.0 - functional.cosine_similarity(center_a.unsqueeze(0), center_b.unsqueeze(0)).mean())
    if not terms:
        return feature_b.sum() * 0.0
    return torch.stack(terms).mean()


def residual_penalty(residuals: tuple[torch.Tensor, ...]) -> torch.Tensor:
    return sum(item.float().pow(2).mean() for item in residuals)
