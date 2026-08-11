"""Primitives for Degradation-Trajectory Straightening (DTS-Seg).

The loss is causal and memory-safe: earlier trajectory margins are detached,
so each progressively degraded view can be forwarded and back-propagated
sequentially on an 8 GB GPU.
"""
from __future__ import annotations

import hashlib

import torch
import torch.nn.functional as functional


IGNORE_INDEX = 255
FAMILIES = ("color", "turbidity", "lowlight", "blur")


def stable_family(sample_id: str, epoch: int) -> str:
    """Choose one degradation family deterministically per scene and epoch."""
    if not sample_id or epoch < 1:
        raise ValueError("sample_id must be non-empty and epoch starts at one.")
    digest = hashlib.sha256(sample_id.encode("utf-8")).digest()
    offset = int.from_bytes(digest[:8], "big")
    return FAMILIES[(offset + epoch) % len(FAMILIES)]


def trajectory_names(sample_id: str, epoch: int) -> tuple[str, str, str, str]:
    family = stable_family(sample_id, epoch)
    return "clean", f"{family}_s1", f"{family}_s2", f"{family}_s3"


def resize_labels(labels: torch.Tensor, size: tuple[int, int]) -> torch.Tensor:
    if labels.ndim != 3:
        raise ValueError("labels must have shape [N,H,W].")
    return functional.interpolate(labels.float().unsqueeze(1), size=size, mode="nearest")[:, 0].long()


def true_class_margin(logits: torch.Tensor, labels: torch.Tensor, *, ignore_index: int = IGNORE_INDEX) -> tuple[torch.Tensor, torch.Tensor]:
    """Return true-class logit margin and its valid-pixel mask.

    The margin is z_y - logsumexp(z_not_y), a smooth multiclass analogue of
    the binary decision margin. Invalid labels are replaced only for gather;
    callers must reduce using the returned mask.
    """
    if logits.ndim != 4 or labels.ndim != 3 or logits.shape[0] != labels.shape[0]:
        raise ValueError("Expected logits [N,C,H,W] and labels [N,H,W].")
    if labels.shape[-2:] != logits.shape[-2:]:
        labels = resize_labels(labels, tuple(logits.shape[-2:]))
    valid = labels.ne(ignore_index) & labels.ge(0) & labels.lt(logits.shape[1])
    safe = labels.masked_fill(~valid, 0)
    truth = logits.gather(1, safe.unsqueeze(1)).squeeze(1)
    competing = logits.masked_fill(functional.one_hot(safe, logits.shape[1]).permute(0, 3, 1, 2).bool(), -torch.inf)
    margin = truth - torch.logsumexp(competing, dim=1)
    return margin, valid


def degradation_acceleration_loss(
    previous_previous_margin: torch.Tensor,
    previous_margin: torch.Tensor,
    current_margin: torch.Tensor,
    eligible: torch.Tensor,
    *,
    tolerance: float = 0.05,
) -> torch.Tensor:
    """Penalize only an accelerating loss of true-class semantic margin.

    Let d_prev=m_{k-2}-m_{k-1} and d_now=m_{k-1}-m_k. The loss is
    relu(d_now-d_prev-tolerance). Earlier margins and eligibility are detached
    by construction, so gradients flow only through the current view.
    """
    if tolerance < 0:
        raise ValueError("tolerance must be non-negative.")
    if previous_previous_margin.shape != previous_margin.shape or previous_margin.shape != current_margin.shape:
        raise ValueError("All trajectory margins must have the same shape.")
    if eligible.shape != current_margin.shape or eligible.dtype != torch.bool:
        raise ValueError("eligible must be a boolean mask matching the margins.")
    if not bool(eligible.any()):
        return current_margin.sum() * 0.0
    prior_drop = previous_previous_margin.detach() - previous_margin.detach()
    current_drop = previous_margin.detach() - current_margin
    acceleration = functional.relu(current_drop - prior_drop - tolerance)
    return acceleration[eligible.detach()].mean()

