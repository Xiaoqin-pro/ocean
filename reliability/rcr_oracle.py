"""Pure, GT-oracle repairability primitives for RCR Gate R0."""
from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass(frozen=True)
class RepairMasks:
    valid: torch.Tensor
    local_wrong: torch.Tensor
    remote_wrong: torch.Tensor
    repaired: torch.Tensor
    damaged: torch.Tensor


def repair_masks(local: torch.Tensor, remote: torch.Tensor, labels: torch.Tensor, *, ignore_index: int = 255) -> RepairMasks:
    if local.shape != remote.shape or local.shape != labels.shape:
        raise ValueError("Local, remote, and labels must share exactly the same spatial shape.")
    valid = labels.ne(ignore_index)
    local_wrong, remote_wrong = local.ne(labels) & valid, remote.ne(labels) & valid
    return RepairMasks(valid, local_wrong, remote_wrong, local_wrong & ~remote_wrong, ~local_wrong & remote_wrong)


def pixelwise_oracle_union(local: torch.Tensor, remote: torch.Tensor, labels: torch.Tensor, *, ignore_index: int = 255) -> torch.Tensor:
    """Use remote prediction only where GT proves that it repairs a local error."""
    masks = repair_masks(local, remote, labels, ignore_index=ignore_index)
    return torch.where(masks.repaired, remote, local)


def equal_probability_ensemble(segformer_logits: torch.Tensor, deeplab_logits: torch.Tensor) -> torch.Tensor:
    """Fixed equal-probability ensemble; never averages uncalibrated logits."""
    if segformer_logits.shape != deeplab_logits.shape or segformer_logits.ndim != 4:
        raise ValueError("Both models must provide matching [B,C,H,W] logits.")
    return (segformer_logits.float().softmax(1) * 0.5 + deeplab_logits.float().softmax(1) * 0.5).argmax(1)


def repair_statistics(local: torch.Tensor, remote: torch.Tensor, labels: torch.Tensor, region: torch.Tensor, *, ignore_index: int = 255) -> dict[str, float | int]:
    if region.shape != labels.shape:
        raise ValueError("Region mask must align with labels.")
    masks = repair_masks(local, remote, labels, ignore_index=ignore_index)
    selected = masks.valid & region.bool()
    local_wrong, local_correct = masks.local_wrong & selected, ~masks.local_wrong & selected
    repaired, damaged = masks.repaired & selected, masks.damaged & selected
    valid_count, wrong_count, correct_count = int(selected.sum()), int(local_wrong.sum()), int(local_correct.sum())
    repair_count, damage_count = int(repaired.sum()), int(damaged.sum())
    return {
        "valid_pixels": valid_count, "local_wrong_pixels": wrong_count, "local_correct_pixels": correct_count,
        "repaired_pixels": repair_count, "damaged_pixels": damage_count,
        "repair_rate": repair_count / wrong_count if wrong_count else None,
        "damage_rate": damage_count / correct_count if correct_count else None,
        "net_correction_mass": (repair_count - damage_count) / valid_count if valid_count else None,
    }
