"""Frozen FT-Reliability v1.1 training primitives.

This module contains no evaluation entry point and no permission to load a
formal SUIM split.  It deliberately operates on tensors and an explicitly
allowed ``method_train`` path only.
"""
from __future__ import annotations

import hashlib
import os
import tempfile
from pathlib import Path
from typing import Iterable, Mapping, Sequence

import torch
import torch.nn.functional as functional


IGNORE_INDEX = 255
FAMILIES = ("color", "turbidity", "lowlight", "blur")
PROHIBITED_PATH_TOKENS = (
    "risk_head_development", "method_development", "val.csv", "calibration.csv",
    "test.csv", "uiis", "dut-useg", "usis10k",
)


def top1_top2_logit_gap(logits: torch.Tensor) -> torch.Tensor:
    """Return the class-agnostic top-1 minus top-2 logit gap, ``[B,H,W]``."""
    if logits.ndim != 4 or logits.shape[1] < 2:
        raise ValueError("Expected logits with shape [B,C,H,W] and C >= 2.")
    top2 = logits.topk(2, dim=1).values
    return top2[:, 0] - top2[:, 1]


def failure_transition_mask(
    logits_s1: torch.Tensor, logits_s3: torch.Tensor, labels: torch.Tensor, *, ignore_index: int = IGNORE_INDEX
) -> torch.Tensor:
    """Select only detached same-pixel s1-correct -> s3-wrong transitions."""
    if logits_s1.shape != logits_s3.shape or logits_s1.ndim != 4 or labels.shape != logits_s1.shape[:1] + logits_s1.shape[-2:]:
        raise ValueError("Expected matching [B,C,H,W] logits and [B,H,W] labels.")
    prediction_s1 = logits_s1.detach().argmax(dim=1)
    prediction_s3 = logits_s3.detach().argmax(dim=1)
    return prediction_s1.eq(labels) & prediction_s3.ne(labels) & labels.ne(ignore_index)


def stable_hash(sample_id: str) -> int:
    return int.from_bytes(hashlib.sha256(str(sample_id).encode("utf-8")).digest()[:8], "big")


def trajectory_family(sample_id: str, epoch: int) -> tuple[str, str, str]:
    """Return the fixed clean/s1/s3 trajectory for one scene and epoch."""
    if epoch < 1:
        raise ValueError("Epoch numbering starts at one.")
    family = FAMILIES[(stable_hash(sample_id) + epoch) % len(FAMILIES)]
    return "clean", f"{family}_s1", f"{family}_s3"


def _generator(seed: int) -> torch.Generator:
    return torch.Generator(device="cpu").manual_seed(int(seed))


def _draw(indices: torch.Tensor, count: int, generator: torch.Generator, *, replacement: bool) -> torch.Tensor:
    if count == 0:
        return indices.new_empty((0,), dtype=torch.long)
    if len(indices) == 0:
        raise ValueError("Cannot sample an empty pool.")
    if len(indices) >= count and not replacement:
        return indices[torch.randperm(len(indices), generator=generator)[:count]]
    return indices[torch.randint(len(indices), (count,), generator=generator)]


def _balanced_indices(
    eligible: torch.Tensor, boundary: torch.Tensor, *, count: int, boundary_fraction: float, seed: int
) -> tuple[torch.Tensor, dict[str, int]]:
    """Sample valid flattened indices with deterministic boundary fallback.

    The target starts at the requested boundary/interior ratio.  A deficient
    region contributes all of its unique pixels first; the other region fills
    the shortfall.  If both pools together are smaller than ``count``, every
    eligible pixel is retained and the remainder is sampled with replacement.
    """
    if count < 1 or not 0.0 <= boundary_fraction <= 1.0:
        raise ValueError("Invalid sampling count or boundary fraction.")
    if eligible.dtype != torch.bool or boundary.dtype != torch.bool or eligible.shape != boundary.shape:
        raise ValueError("Eligible and boundary masks must be matching bool tensors.")
    flat_eligible, flat_boundary = eligible.reshape(-1).cpu(), boundary.reshape(-1).cpu()
    boundary_pool = torch.nonzero(flat_eligible & flat_boundary, as_tuple=False).flatten()
    interior_pool = torch.nonzero(flat_eligible & ~flat_boundary, as_tuple=False).flatten()
    unique_eligible = int(len(boundary_pool) + len(interior_pool))
    if unique_eligible == 0:
        return boundary_pool, {"boundary": 0, "interior": 0, "unique_eligible": 0, "sampled_terms": 0, "duplication_factor": 0.0, "zero_loss": 1}
    target_boundary = int(round(count * boundary_fraction))
    target_interior = count - target_boundary
    generator = _generator(seed)
    chosen_boundary = _draw(boundary_pool, min(len(boundary_pool), target_boundary), generator, replacement=False) if len(boundary_pool) else boundary_pool
    chosen_interior = _draw(interior_pool, min(len(interior_pool), target_interior), generator, replacement=False) if len(interior_pool) else interior_pool
    remaining = count - len(chosen_boundary) - len(chosen_interior)
    if remaining:
        preferred = interior_pool if len(chosen_boundary) < target_boundary else boundary_pool
        alternate = boundary_pool if preferred.data_ptr() == interior_pool.data_ptr() else interior_pool
        if len(preferred):
            available = preferred[~torch.isin(preferred, chosen_interior if preferred.data_ptr() == interior_pool.data_ptr() else chosen_boundary)]
            take = min(remaining, len(available))
            extra = _draw(available, take, generator, replacement=False) if take else available
            if preferred.data_ptr() == interior_pool.data_ptr():
                chosen_interior = torch.cat((chosen_interior, extra))
            else:
                chosen_boundary = torch.cat((chosen_boundary, extra))
            remaining -= len(extra)
        if remaining and len(alternate):
            chosen = chosen_boundary if alternate.data_ptr() == boundary_pool.data_ptr() else chosen_interior
            available = alternate[~torch.isin(alternate, chosen)]
            take = min(remaining, len(available))
            extra = _draw(available, take, generator, replacement=False) if take else available
            if alternate.data_ptr() == boundary_pool.data_ptr():
                chosen_boundary = torch.cat((chosen_boundary, extra))
            else:
                chosen_interior = torch.cat((chosen_interior, extra))
            remaining -= len(extra)
    selected = torch.cat((chosen_boundary, chosen_interior))
    if remaining:
        pool = torch.cat((boundary_pool, interior_pool))
        selected = torch.cat((selected, _draw(pool, remaining, generator, replacement=True)))
    order = torch.randperm(len(selected), generator=generator)
    selected = selected[order]
    sampled_terms = int(len(selected))
    return selected, {
        "boundary": int(flat_boundary[selected].sum()), "interior": int((~flat_boundary[selected]).sum()),
        "unique_eligible": unique_eligible, "sampled_terms": sampled_terms,
        "duplication_factor": float(sampled_terms / unique_eligible), "zero_loss": 0,
    }


def deterministic_transition_sampler(
    transition: torch.Tensor, boundary: torch.Tensor, labels: torch.Tensor, *, epoch: int, batch_index: int,
    max_pixels: int = 4096, boundary_fraction: float = 0.50, ignore_index: int = IGNORE_INDEX,
) -> tuple[torch.Tensor, dict[str, int]]:
    """Sample transition pixels; ignore labels can never enter the sample."""
    eligible = transition.bool() & labels.ne(ignore_index)
    return _balanced_indices(eligible, boundary.bool() & labels.ne(ignore_index), count=max_pixels, boundary_fraction=boundary_fraction, seed=20260725 + 100003 * epoch + batch_index)


def failure_transition_loss(
    logits_s1: torch.Tensor, logits_s3: torch.Tensor, labels: torch.Tensor, boundary: torch.Tensor, *, epoch: int,
    batch_index: int, margin: float = 0.20, max_pixels: int = 4096, boundary_fraction: float = 0.50,
    ignore_index: int = IGNORE_INDEX,
) -> tuple[torch.Tensor, dict[str, int]]:
    """FT v1.1 loss: lower severe wrong-prediction top-gap confidence only."""
    if margin != 0.20:
        raise ValueError("FT-Reliability v1.1 freezes the margin at 0.20.")
    return failure_transition_loss_from_s1(
        top1_top2_logit_gap(logits_s1).detach(), logits_s1.detach().argmax(dim=1), logits_s3, labels, boundary,
        epoch=epoch, batch_index=batch_index, margin=margin, max_pixels=max_pixels,
        boundary_fraction=boundary_fraction, ignore_index=ignore_index,
    )


def failure_transition_loss_from_s1(
    q_s1: torch.Tensor, prediction_s1: torch.Tensor, logits_s3: torch.Tensor, labels: torch.Tensor, boundary: torch.Tensor, *,
    epoch: int, batch_index: int, margin: float = 0.20, max_pixels: int = 4096,
    boundary_fraction: float = 0.50, ignore_index: int = IGNORE_INDEX,
) -> tuple[torch.Tensor, dict[str, int]]:
    """FT loss using saved detached s1 gap/prediction after the s1 graph is freed."""
    if q_s1.shape != labels.shape or prediction_s1.shape != labels.shape:
        raise ValueError("Saved s1 prediction/gap must match labels.")
    prediction_s3 = logits_s3.detach().argmax(dim=1)
    transition = prediction_s1.detach().eq(labels) & prediction_s3.ne(labels) & labels.ne(ignore_index)
    indices, counts = deterministic_transition_sampler(transition, boundary, labels, epoch=epoch, batch_index=batch_index, max_pixels=max_pixels, boundary_fraction=boundary_fraction, ignore_index=ignore_index)
    q_s1, q_s3 = q_s1.detach().reshape(-1), top1_top2_logit_gap(logits_s3).reshape(-1)
    if len(indices) == 0:
        return q_s3.sum() * 0.0, counts
    chosen = indices.to(q_s3.device)
    return functional.softplus(margin - q_s1[chosen] + q_s3[chosen]).mean(), counts


def deterministic_correctness_pair_sampler(
    correct: torch.Tensor, boundary: torch.Tensor, labels: torch.Tensor, *, epoch: int, batch_index: int,
    pairs: int = 4096, boundary_fraction: float = 0.50, ignore_index: int = IGNORE_INDEX,
) -> tuple[torch.Tensor, torch.Tensor, dict[str, int]]:
    """Pair correct/wrong s3 pixels with fixed region-aware fallback rules."""
    valid = labels.ne(ignore_index)
    correct, boundary = correct.bool() & valid, boundary.bool() & valid
    requested_boundary = int(round(pairs * boundary_fraction))
    requested_interior = pairs - requested_boundary
    generator = _generator(20260725 + 100003 * epoch + batch_index)
    pools = {}
    for name, region in (("boundary", boundary), ("interior", ~boundary & valid)):
        pools[name] = (torch.nonzero((correct & region).reshape(-1), as_tuple=False).flatten().cpu(), torch.nonzero((~correct & region & valid).reshape(-1), as_tuple=False).flatten().cpu())
    usable = {name: min(len(pair[0]), len(pair[1])) for name, pair in pools.items()}
    selected: list[tuple[torch.Tensor, torch.Tensor, str]] = []
    for name, amount in (("boundary", requested_boundary), ("interior", requested_interior)):
        if usable[name]:
            c, w = pools[name]
            selected.append((_draw(c, amount, generator, replacement=len(c) < amount), _draw(w, amount, generator, replacement=len(w) < amount), name))
    have = sum(len(value[0]) for value in selected)
    if have < pairs:
        fallback = "interior" if usable["interior"] else "boundary"
        if usable[fallback]:
            c, w = pools[fallback]
            amount = pairs - have
            selected.append((_draw(c, amount, generator, replacement=True), _draw(w, amount, generator, replacement=True), fallback))
    if not selected:
        empty = torch.empty((0,), dtype=torch.long)
        return empty, empty, {"boundary": 0, "interior": 0, "unique_eligible": 0, "sampled_terms": 0, "duplication_factor": 0.0, "zero_loss": 1}
    correct_indices, wrong_indices = torch.cat([value[0] for value in selected]), torch.cat([value[1] for value in selected])
    order = torch.randperm(len(correct_indices), generator=generator)
    counts = {name: sum(len(value[0]) for value in selected if value[2] == name) for name in ("boundary", "interior")}
    unique_eligible = int(sum(usable.values()))
    counts.update(unique_eligible=unique_eligible, sampled_terms=int(len(correct_indices)), duplication_factor=float(len(correct_indices) / max(unique_eligible, 1)), zero_loss=0)
    return correct_indices[order], wrong_indices[order], counts


def generic_correctness_ranking_loss(
    logits_s3: torch.Tensor, labels: torch.Tensor, boundary: torch.Tensor, *, epoch: int, batch_index: int,
    margin: float = 0.20, pairs: int = 4096, boundary_fraction: float = 0.50, ignore_index: int = IGNORE_INDEX,
) -> tuple[torch.Tensor, dict[str, int]]:
    """Fixed C comparator: generic s3 correct-versus-wrong top-gap ranking."""
    if margin != 0.20:
        raise ValueError("Generic ranking comparator freezes the margin at 0.20.")
    q = top1_top2_logit_gap(logits_s3).reshape(-1)
    correct = logits_s3.detach().argmax(dim=1).eq(labels)
    correct_indices, wrong_indices, counts = deterministic_correctness_pair_sampler(correct, boundary, labels, epoch=epoch, batch_index=batch_index, pairs=pairs, boundary_fraction=boundary_fraction, ignore_index=ignore_index)
    if len(correct_indices) == 0:
        return q.sum() * 0.0, counts
    return functional.softplus(margin - q[correct_indices.to(q.device)] + q[wrong_indices.to(q.device)]).mean(), counts


def clean_retention_kl(
    teacher_logits: torch.Tensor, student_logits: torch.Tensor, labels: torch.Tensor, *, ignore_index: int = IGNORE_INDEX
) -> torch.Tensor:
    """Teacher-to-student KL on valid clean pixels the teacher predicts correctly."""
    if teacher_logits.shape != student_logits.shape or teacher_logits.ndim != 4 or labels.shape != teacher_logits.shape[:1] + teacher_logits.shape[-2:]:
        raise ValueError("Teacher/student logits and labels have incompatible shapes.")
    teacher_probability = teacher_logits.detach().float().softmax(dim=1)
    student_log_probability = student_logits.float().log_softmax(dim=1)
    per_pixel = (teacher_probability * (teacher_probability.clamp_min(torch.finfo(torch.float32).eps).log() - student_log_probability)).sum(dim=1)
    retained = labels.ne(ignore_index) & teacher_logits.detach().argmax(dim=1).eq(labels)
    if not retained.any():
        return student_logits.float().sum() * 0.0
    return per_pixel[retained].mean()


def assert_method_train_access(
    path: Path, *, expected_filename: str = "risk_head_train.csv", allowed_directory: Path | None = None,
) -> Path:
    """Reject arbitrary data sources; the driver has one admissible CSV name."""
    resolved = path.resolve()
    lower = str(resolved).replace("\\", "/").lower()
    if resolved.name != expected_filename or any(token in lower for token in PROHIBITED_PATH_TOKENS):
        raise PermissionError("FT-Reliability training only permits the frozen method_train CSV.")
    if allowed_directory is not None and resolved.parent != allowed_directory.resolve():
        raise PermissionError("method_train CSV is outside the frozen split directory.")
    return resolved


def validate_method_train_membership(sample_ids: Iterable[str], allowed_sample_ids: Iterable[str]) -> None:
    """Reject duplicate or disguised sample IDs before a training loader exists."""
    observed, allowed = tuple(map(str, sample_ids)), set(map(str, allowed_sample_ids))
    if len(observed) != len(set(observed)) or set(observed) != allowed:
        raise PermissionError("method_train sample IDs do not exactly match the frozen allowlist.")


def validate_split_manifest(
    manifest: Mapping[str, object], *, method_train_sha256: str, method_development_sha256: str,
) -> None:
    """Validate the committed metadata audit without reading development rows."""
    if manifest.get("split_version") != "aquariskmap_risk_head_v1":
        raise ValueError("Unexpected frozen split version.")
    counts = manifest.get("counts", {})
    hashes = manifest.get("csv_sha256", {})
    if not isinstance(counts, Mapping) or not isinstance(hashes, Mapping):
        raise ValueError("Frozen split audit lacks counts or CSV hashes.")
    if counts.get("risk_head_train") != 936 or counts.get("risk_head_development") != 231:
        raise ValueError("Frozen split audit has incorrect sample counts.")
    if str(hashes.get("risk_head_train", "")).upper() != method_train_sha256.upper() or str(hashes.get("risk_head_development", "")).upper() != method_development_sha256.upper():
        raise ValueError("Frozen split audit CSV hashes differ from the protocol.")
    if bool(manifest.get("sample_leakage", True)) or bool(manifest.get("scene_group_leakage", True)):
        raise ValueError("Frozen split audit reports cross-role leakage.")
    if any(bool(manifest.get(key, True)) for key in ("formal_validation_read", "formal_calibration_read", "official_suim_test_evaluated")):
        raise ValueError("Frozen split audit records impermissible formal-split access.")


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
