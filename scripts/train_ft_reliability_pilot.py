"""Fixed FT-Reliability v1.2 SegFormer pilot driver.

The command interface deliberately exposes no data-path or evaluation-split
argument.  Running this module is a future authorized action; this commit only
provides the implementation and synthetic-tensor-tested checkpoint machinery.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
import subprocess
import sys
from pathlib import Path
from typing import Any, Mapping

import albumentations as A
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as functional
import yaml
from albumentations.pytorch import ToTensorV2
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from transformers import SegformerForSemanticSegmentation

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from datasets.label_mapping import ID2LABEL, LABEL2ID  # noqa: E402
from datasets.suim_dataset import IMAGENET_MEAN, IMAGENET_STD  # noqa: E402
from degradations.registry import build_image_degradation, load_conditions  # noqa: E402
from reliability.ft_reliability import (  # noqa: E402
    IGNORE_INDEX, assert_method_train_access, atomic_torch_save, clean_retention_kl,
    failure_transition_loss_from_s1, generic_correctness_ranking_loss, top1_top2_logit_gap, trajectory_family,
    stable_hash, validate_split_manifest,
)


PROTOCOL_COMMIT = "ft_reliability_v1_2"
CHECKPOINT_FORMAT = "ft_reliability_pilot_v1_2"
ALLOWED_VARIANTS = ("A", "B", "C", "D", "E")


def load_config(root: Path = ROOT) -> dict[str, Any]:
    return yaml.safe_load((root / "configs" / "ft_reliability_pilot.yaml").read_text(encoding="utf-8"))


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def implementation_commit(root: Path = ROOT) -> str:
    """Record the exact source revision without making it a user-supplied value."""
    try:
        return subprocess.check_output(
            ["git", "-C", str(root), "rev-parse", "HEAD"], text=True, stderr=subprocess.DEVNULL,
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return "unavailable"


def state_dict_sha256(state_dict: Mapping[str, torch.Tensor]) -> str:
    """Hash tensor names, dtypes, shapes, and CPU bytes deterministically."""
    digest = hashlib.sha256()
    for name in sorted(state_dict):
        value = state_dict[name].detach().cpu().contiguous()
        digest.update(name.encode("utf-8")); digest.update(str(value.dtype).encode("ascii"))
        digest.update(json.dumps(list(value.shape)).encode("ascii")); digest.update(value.numpy().tobytes())
    return digest.hexdigest()


def variant_terms(variant: str, epoch: int) -> dict[str, bool]:
    if variant not in ALLOWED_VARIANTS or epoch < 1:
        raise ValueError("Unknown variant or invalid epoch.")
    warm = epoch > 5
    return {
        "clean_only": variant == "A",
        "three_view_ce": variant != "A",
        "generic_ranking": variant == "C" and warm,
        "failure_transition": variant in {"D", "E"} and warm,
        # E is identical to the shared Degradation-CE warm-up through epoch 5.
        "retention": variant == "E" and warm,
    }


def resize_logits(logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    return functional.interpolate(logits, size=labels.shape[-2:], mode="bilinear", align_corners=False)


def capture_rng_state() -> dict[str, object]:
    return {
        "python": random.getstate(), "numpy": np.random.get_state(), "torch": torch.get_rng_state(),
        "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
    }


def restore_rng_state(state: Mapping[str, object]) -> None:
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(torch.as_tensor(state["torch"], dtype=torch.uint8, device="cpu"))
    if state.get("cuda") is not None and torch.cuda.is_available():
        torch.cuda.set_rng_state_all([torch.as_tensor(value, dtype=torch.uint8, device="cpu") for value in state["cuda"]])


def checkpoint_payload(
    epoch: int, global_step: int, model: torch.nn.Module, optimizer: torch.optim.Optimizer,
    scaler: torch.amp.GradScaler, *, variant: str, initialization_sha256: str,
    method_train_csv_sha256: str, method_development_csv_sha256: str, split_audit_sha256: str,
    degradation_config_sha256: str, ft_config_sha256: str, protocol_document_sha256: str,
    implementation_git_commit: str, teacher_checkpoint_sha256: str | None = None,
    run_kind: str, epoch_completed: bool, batch_index: int | None, smoke_target: int | None,
) -> dict[str, object]:
    if run_kind not in {"smoke", "formal"}:
        raise ValueError("run_kind must be smoke or formal.")
    return {
        "checkpoint_format": CHECKPOINT_FORMAT, "protocol_commit": PROTOCOL_COMMIT,
        "protocol_document_sha256": protocol_document_sha256,
        "implementation_git_commit": implementation_git_commit,
        "variant": variant, "model_name": "segformer", "epoch": epoch, "global_step": global_step,
        "model_state_dict": model.state_dict(), "optimizer_state_dict": optimizer.state_dict(),
        "scaler_state_dict": scaler.state_dict(), "rng_state": capture_rng_state(),
        "initialization_sha256": initialization_sha256, "method_train_csv_sha256": method_train_csv_sha256,
        "method_development_csv_sha256": method_development_csv_sha256, "split_audit_sha256": split_audit_sha256,
        "degradation_config_sha256": degradation_config_sha256, "ft_config_sha256": ft_config_sha256,
        "teacher_checkpoint_sha256": teacher_checkpoint_sha256, "run_kind": run_kind,
        "epoch_completed": epoch_completed, "batch_index": batch_index, "smoke_target": smoke_target,
        "checkpoint_selection": "final_epoch",
        "method_development_evaluated": False, "validation_evaluated": False,
        "calibration_evaluated": False, "official_suim_test_evaluated": False,
    }


def load_completed_epoch_checkpoint(
    path: Path, model: torch.nn.Module, optimizer: torch.optim.Optimizer, scaler: torch.amp.GradScaler,
    device: torch.device, *, variant: str, expected_provenance: Mapping[str, str | None],
    requested_run_kind: str, smoke_target: int | None = None,
) -> tuple[int, int, int]:
    checkpoint = torch.load(path, map_location=device, weights_only=False)
    if checkpoint.get("checkpoint_format") != CHECKPOINT_FORMAT or checkpoint.get("protocol_commit") != PROTOCOL_COMMIT:
        raise ValueError("Unsupported FT-Reliability checkpoint.")
    if checkpoint.get("variant") != variant or any(bool(checkpoint.get(key, True)) for key in ("method_development_evaluated", "validation_evaluated", "calibration_evaluated", "official_suim_test_evaluated")):
        raise ValueError("Checkpoint violates the frozen FT-Reliability access protocol.")
    if requested_run_kind not in {"smoke", "formal"} or checkpoint.get("run_kind") != requested_run_kind:
        raise ValueError("Smoke and formal checkpoints are not interchangeable.")
    if requested_run_kind == "formal" and not bool(checkpoint.get("epoch_completed")):
        raise ValueError("Formal training may resume only a completed epoch checkpoint.")
    if requested_run_kind == "smoke" and checkpoint.get("smoke_target") != smoke_target:
        raise ValueError("Smoke checkpoint target differs from the requested bounded run.")
    for key, expected in expected_provenance.items():
        if checkpoint.get(key) != expected:
            raise ValueError(f"Checkpoint provenance mismatch for {key}.")
    model.load_state_dict(checkpoint["model_state_dict"])
    optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    scaler.load_state_dict(checkpoint["scaler_state_dict"])
    restore_rng_state(checkpoint["rng_state"])
    next_epoch = int(checkpoint["epoch"]) + 1 if bool(checkpoint["epoch_completed"]) else int(checkpoint["epoch"])
    next_batch = 0 if bool(checkpoint["epoch_completed"]) else int(checkpoint["batch_index"]) + 1
    return next_epoch, next_batch, int(checkpoint["global_step"])


class TrajectoryDataset(Dataset[dict[str, Any]]):
    """Read only method_train and apply one shared spatial transform per triplet."""
    def __init__(self, root: Path, split_csv: Path, degradation_config: Path) -> None:
        self.root, self.split_csv = root, assert_method_train_access(split_csv, allowed_directory=split_csv.parent)
        self.frame = pd.read_csv(self.split_csv).sort_values("sample_id").reset_index(drop=True)
        if len(self.frame) != 936 or self.frame["sample_id"].duplicated().any():
            raise ValueError("method_train must contain exactly 936 unique samples.")
        required = {"sample_id", "image_path", "mask_path"}
        if required.difference(self.frame.columns):
            raise ValueError("method_train CSV is missing required columns.")
        self.conditions = {condition.name: build_image_degradation(condition) for condition in load_conditions(degradation_config)}
        self.epoch = 1
        self.transform = A.ReplayCompose([
            A.Resize(384, 384), A.HorizontalFlip(p=0.5),
            A.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD, max_pixel_value=255.0), ToTensorV2(),
        ])

    def __len__(self) -> int:
        return len(self.frame)

    def set_epoch(self, epoch: int) -> None:
        if epoch < 1:
            raise ValueError("Epoch numbering starts at one.")
        self.epoch = epoch

    def __getitem__(self, index: int) -> dict[str, Any]:
        row = self.frame.iloc[index]
        sample_id = str(row.sample_id)
        with Image.open(self.root / str(row.image_path)) as image:
            image = np.asarray(image.convert("RGB"), dtype=np.uint8)
        with Image.open(self.root / str(row.mask_path)) as mask:
            label = np.asarray(mask, dtype=np.uint8)
        _, s1, s3 = trajectory_family(sample_id, self.epoch)
        # Per-scene/epoch replay seed makes a resumed skipped batch identical.
        self.transform.set_random_seed((stable_hash(sample_id) + 100003 * self.epoch) % (2**32))
        result = self.transform(image=image, mask=label)
        clean = result["image"]
        s1_result = A.ReplayCompose.replay(result["replay"], image=self.conditions[s1](image, sample_id), mask=label)
        s3_result = A.ReplayCompose.replay(result["replay"], image=self.conditions[s3](image, sample_id), mask=label)
        return {"clean": clean.float(), "s1": s1_result["image"].float(), "s3": s3_result["image"].float(), "labels": result["mask"].long(), "sample_id": sample_id}


def _build_segformer(config: Mapping[str, Any], device: torch.device) -> torch.nn.Module:
    return SegformerForSemanticSegmentation.from_pretrained(
        config["models"]["segformer_b0"]["pretrained_model"], num_labels=8,
        id2label=ID2LABEL, label2id=LABEL2ID, ignore_mismatched_sizes=True,
    ).to(device)


def validate_teacher_checkpoint_metadata(checkpoint: Mapping[str, object], expected: Mapping[str, str | None]) -> None:
    """Accept only the frozen 100-epoch formal Baseline-936 teacher."""
    if checkpoint.get("checkpoint_format") != CHECKPOINT_FORMAT or checkpoint.get("protocol_commit") != PROTOCOL_COMMIT:
        raise ValueError("Teacher checkpoint has an incompatible FT protocol.")
    required = {"variant": "A", "model_name": "segformer", "run_kind": "formal", "checkpoint_selection": "final_epoch", "epoch": 100, "epoch_completed": True}
    for key, value in required.items():
        if checkpoint.get(key) != value:
            raise ValueError(f"Teacher checkpoint has invalid {key}.")
    if any(bool(checkpoint.get(key, True)) for key in ("method_development_evaluated", "validation_evaluated", "calibration_evaluated", "official_suim_test_evaluated")):
        raise ValueError("Teacher checkpoint violates the access protocol.")
    for key, value in expected.items():
        if checkpoint.get(key) != value:
            raise ValueError(f"Teacher provenance mismatch for {key}.")


def load_frozen_teacher(path: Path, config: Mapping[str, Any], device: torch.device, expected: Mapping[str, str | None]) -> tuple[torch.nn.Module, str]:
    if not path.is_file():
        raise FileNotFoundError("Frozen Baseline-936 teacher checkpoint is missing.")
    checkpoint = torch.load(path, map_location=device, weights_only=False)
    validate_teacher_checkpoint_metadata(checkpoint, expected)
    teacher = _build_segformer(config, device)
    teacher.load_state_dict(checkpoint["model_state_dict"])
    teacher.eval()
    for parameter in teacher.parameters():
        parameter.requires_grad_(False)
    return teacher, sha256(path)


def _student_logits(model: torch.nn.Module, pixels: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    return resize_logits(model(pixel_values=pixels).logits, labels)


def _boundary(labels: torch.Tensor) -> torch.Tensor:
    valid = labels.ne(IGNORE_INDEX)
    value = labels.float().masked_fill(~valid, 0.0).unsqueeze(1)
    maximum = functional.max_pool2d(value, 7, stride=1, padding=3)
    minimum = -functional.max_pool2d(-value, 7, stride=1, padding=3)
    return maximum[:, 0].ne(minimum[:, 0]) & valid


def run_one_step(
    model: torch.nn.Module, optimizer: torch.optim.Optimizer, scaler: torch.amp.GradScaler, batch: Mapping[str, Any], *,
    variant: str, epoch: int, batch_index: int, amp: bool, teacher: torch.nn.Module | None = None,
    accumulation_steps: int = 1, accumulation_index: int = 0,
) -> dict[str, float]:
    """Execute one mathematically averaged three-view update and one optimizer step."""
    terms = variant_terms(variant, epoch)
    device = next(model.parameters()).device
    labels = batch["labels"].to(device)
    boundary = _boundary(labels)
    if accumulation_steps < 1 or not 0 <= accumulation_index < accumulation_steps:
        raise ValueError("Invalid gradient accumulation state.")
    if accumulation_index == 0:
        optimizer.zero_grad(set_to_none=True)
    scale = 1.0 / accumulation_steps
    values = {
        "clean_ce": 0.0, "s1_ce": 0.0, "s3_ce": 0.0, "ft": 0.0, "retain": 0.0, "ranking": 0.0,
        # Fixed C/E fairness audit: both objectives request 4,096 softplus terms.
        "unique_eligible": 0.0, "sampled_terms": 0.0, "duplication_factor": 0.0,
        "sampled_boundary": 0.0, "sampled_interior": 0.0, "zero_loss_batch": 0.0,
        "transition_valid_pixel_rate": 0.0, "c_to_w": 0.0, "w_to_c": 0.0,
        "c_to_c": 0.0, "w_to_w": 0.0, "retention_valid_pixel_rate": 0.0,
    }
    for class_id in range(8):
        values[f"transition_class_{class_id}"] = 0.0
        values[f"retention_class_{class_id}"] = 0.0
    with torch.amp.autocast("cuda", enabled=amp):
        clean_logits = _student_logits(model, batch["clean"].to(device), labels)
        clean_ce = functional.cross_entropy(clean_logits, labels, ignore_index=IGNORE_INDEX)
        retain = clean_logits.sum() * 0.0
        if terms["retention"]:
            if teacher is None:
                raise ValueError("Variant E requires a frozen Baseline-936 teacher.")
            with torch.no_grad():
                teacher_logits = _student_logits(teacher, batch["clean"].to(device), labels)
            retain = clean_retention_kl(teacher_logits, clean_logits, labels)
            retained = labels.ne(IGNORE_INDEX) & teacher_logits.detach().argmax(dim=1).eq(labels)
            values["retention_valid_pixel_rate"] = float(retained.float().mean().detach())
            for class_id in range(8):
                values[f"retention_class_{class_id}"] = float((retained & labels.eq(class_id)).sum().detach())
        clean_loss = clean_ce if terms["clean_only"] else clean_ce / 3.0
        clean_loss = clean_loss + (0.10 * retain if terms["retention"] else 0.0)
    scaler.scale(clean_loss * scale).backward()
    values["clean_ce"], values["retain"] = float(clean_ce.detach()), float(retain.detach())
    if not terms["clean_only"]:
        with torch.amp.autocast("cuda", enabled=amp):
            s1_logits = _student_logits(model, batch["s1"].to(device), labels)
            s1_ce = functional.cross_entropy(s1_logits, labels, ignore_index=IGNORE_INDEX)
            q_s1 = top1_top2_logit_gap(s1_logits).detach()
            prediction_s1 = s1_logits.detach().argmax(dim=1)
        scaler.scale((s1_ce / 3.0) * scale).backward()
        del s1_logits
        with torch.amp.autocast("cuda", enabled=amp):
            s3_logits = _student_logits(model, batch["s3"].to(device), labels)
            s3_ce = functional.cross_entropy(s3_logits, labels, ignore_index=IGNORE_INDEX)
            ft = s3_logits.sum() * 0.0
            ranking = s3_logits.sum() * 0.0
            if terms["failure_transition"]:
                ft, transition_counts = failure_transition_loss_from_s1(q_s1, prediction_s1, s3_logits, labels, boundary, epoch=epoch, batch_index=batch_index)
            else:
                transition_counts = {"boundary": 0, "interior": 0, "unique_eligible": 0, "sampled_terms": 0, "duplication_factor": 0.0, "zero_loss": 1}
            ranking_counts = {"boundary": 0, "interior": 0, "unique_eligible": 0, "sampled_terms": 0, "duplication_factor": 0.0, "zero_loss": 1}
            if terms["generic_ranking"]:
                ranking, ranking_counts = generic_correctness_ranking_loss(s3_logits, labels, boundary, epoch=epoch, batch_index=batch_index)
            audit_counts = transition_counts if terms["failure_transition"] else ranking_counts
            pred_s3 = s3_logits.detach().argmax(dim=1)
            valid = labels.ne(IGNORE_INDEX)
            c1, c3 = prediction_s1.eq(labels), pred_s3.eq(labels)
            values.update(
                unique_eligible=float(audit_counts["unique_eligible"]),
                sampled_terms=float(audit_counts["sampled_terms"]),
                duplication_factor=float(audit_counts["duplication_factor"]),
                sampled_boundary=float(audit_counts["boundary"]),
                sampled_interior=float(audit_counts["interior"]),
                zero_loss_batch=float(audit_counts["zero_loss"]),
                transition_valid_pixel_rate=float((c1 & ~c3 & valid).float().mean().detach()),
                c_to_w=float((c1 & ~c3 & valid).sum().detach()),
                w_to_c=float((~c1 & c3 & valid).sum().detach()),
                c_to_c=float((c1 & c3 & valid).sum().detach()),
                w_to_w=float((~c1 & ~c3 & valid).sum().detach()),
            )
            transition_mask = c1 & ~c3 & valid
            for class_id in range(8):
                values[f"transition_class_{class_id}"] = float((transition_mask & labels.eq(class_id)).sum().detach())
            s3_loss = s3_ce / 3.0 + 0.10 * ft + 0.10 * ranking
        scaler.scale(s3_loss * scale).backward()
        values.update(s1_ce=float(s1_ce.detach()), s3_ce=float(s3_ce.detach()), ft=float(ft.detach()), ranking=float(ranking.detach()), transition_boundary=float(transition_counts["boundary"]), transition_interior=float(transition_counts["interior"]))
    if accumulation_index == accumulation_steps - 1:
        scaler.step(optimizer)
        scaler.update()
    return values


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--variant", required=True, choices=ALLOWED_VARIANTS)
    parser.add_argument("--model", required=True, choices=("segformer",))
    parser.add_argument("--smoke-steps", type=int, default=0)
    parser.add_argument("--resume", type=Path)
    args = parser.parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("FT-Reliability training is configured for CUDA.")
    config_path = ROOT / "configs" / "ft_reliability_pilot.yaml"
    config = load_config(ROOT)
    protocol_path = ROOT / str(config["provenance"]["protocol_document"])
    if any(bool(config["access_control"][key]) for key in ("method_development_evaluated", "validation_evaluated", "calibration_evaluated", "official_suim_test_evaluated")):
        raise ValueError("FT-Reliability protocol forbids evaluation during pilot training.")
    access = config["access_control"]
    split = ROOT / access["method_train_csv"]
    development_csv = ROOT / access["method_development_csv"]
    audit_path = ROOT / access["frozen_split_audit_path"]
    if sha256(split).upper() != str(access["method_train_csv_sha256"]).upper():
        raise ValueError("method_train CSV hash differs from the frozen protocol.")
    # Hash-only development verification: no development rows are loaded.
    if sha256(development_csv).upper() != str(access["method_development_csv_sha256"]).upper():
        raise ValueError("method_development CSV hash differs from the frozen protocol.")
    if sha256(audit_path).upper() != str(access["frozen_split_audit_sha256"]).upper():
        raise ValueError("Frozen split audit hash differs from the protocol.")
    validate_split_manifest(json.loads(audit_path.read_text(encoding="utf-8")), method_train_sha256=str(access["method_train_csv_sha256"]), method_development_sha256=str(access["method_development_csv_sha256"]))
    seed = int(config["experiment"]["seed"])
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    device = torch.device("cuda")
    dataset = TrajectoryDataset(ROOT, split, ROOT / config["degradations"]["registry_config"])
    model = _build_segformer(config, device)
    initialization_sha = state_dict_sha256(model.state_dict())
    optimizer = torch.optim.AdamW(model.parameters(), lr=float(config["models"]["segformer_b0"]["learning_rate"]), weight_decay=float(config["models"]["segformer_b0"]["weight_decay"]))
    scaler = torch.amp.GradScaler("cuda", enabled=True)
    run_kind = "smoke" if args.smoke_steps else "formal"
    output = ROOT / config["experiment"]["output_dir"] / "segformer" / run_kind / args.variant
    checkpoint = output / "checkpoints" / "last.pt"
    provenance = {
        "initialization_sha256": initialization_sha,
        "method_train_csv_sha256": sha256(split),
        "method_development_csv_sha256": sha256(development_csv),
        "split_audit_sha256": sha256(audit_path),
        "degradation_config_sha256": sha256(ROOT / config["degradations"]["registry_config"]),
        "ft_config_sha256": sha256(config_path),
        "protocol_document_sha256": sha256(protocol_path),
        "implementation_git_commit": implementation_commit(ROOT),
        "teacher_checkpoint_sha256": None,
    }
    teacher = None
    if args.variant == "E":
        teacher, teacher_hash = load_frozen_teacher(ROOT / config["clean_retention"]["teacher_checkpoint"], config, device, provenance)
        provenance["teacher_checkpoint_sha256"] = teacher_hash
    start_epoch, start_batch, global_step = 1, 0, 0
    if args.resume:
        start_epoch, start_batch, global_step = load_completed_epoch_checkpoint(args.resume, model, optimizer, scaler, device, variant=args.variant, expected_provenance=provenance, requested_run_kind=run_kind, smoke_target=args.smoke_steps or None)
        if run_kind == "smoke" and global_step >= args.smoke_steps:
            print(json.dumps({"smoke_steps": global_step, "already_complete": True, "official_suim_test_evaluated": False}), flush=True)
            return
    elif args.variant in {"C", "D", "E"}:
        warmup = ROOT / config["experiment"]["output_dir"] / "segformer" / "formal" / "shared_warmup" / "checkpoints" / "final.pt"
        warmup_provenance = dict(provenance); warmup_provenance["teacher_checkpoint_sha256"] = None
        start_epoch, start_batch, global_step = load_completed_epoch_checkpoint(warmup, model, optimizer, scaler, device, variant="shared_warmup", expected_provenance=warmup_provenance, requested_run_kind="formal")
        if start_epoch != 6 or start_batch != 0:
            raise ValueError("Shared warm-up must end at completed epoch five.")
    if args.variant == "E" and run_kind == "smoke" and start_epoch != 6:
        raise ValueError("Variant E smoke may begin only from the completed shared warm-up at epoch six.")
    history: list[dict[str, object]] = []
    for epoch in range(start_epoch, int(config["models"]["segformer_b0"]["epochs"]) + 1):
        dataset.set_epoch(epoch)
        generator = torch.Generator().manual_seed(seed + epoch)
        model_config = config["models"]["segformer_b0"]
        accumulation_steps = int(model_config["gradient_accumulation_steps"])
        effective_batch = int(model_config["effective_base_scene_batch_size"])
        if effective_batch % accumulation_steps:
            raise ValueError("Effective base-scene batch must divide by gradient accumulation steps.")
        loader = DataLoader(dataset, batch_size=effective_batch // accumulation_steps, shuffle=True, generator=generator, num_workers=0, pin_memory=True)
        totals: dict[str, float] = {}
        batches = 0
        for batch_index, batch in enumerate(loader):
            if epoch == start_epoch and batch_index < start_batch:
                continue
            values = run_one_step(model, optimizer, scaler, batch, variant=args.variant, epoch=epoch, batch_index=batch_index, amp=True, teacher=teacher, accumulation_steps=accumulation_steps, accumulation_index=batch_index % accumulation_steps)
            if not all(np.isfinite(value) for value in values.values()):
                raise FloatingPointError("FT-Reliability loss became non-finite.")
            global_step += 1
            batches += 1
            for key, value in values.items():
                totals[key] = totals.get(key, 0.0) + value
            if args.smoke_steps and global_step >= args.smoke_steps:
                atomic_torch_save(checkpoint_payload(epoch, global_step, model, optimizer, scaler, variant=args.variant, **provenance, run_kind="smoke", epoch_completed=False, batch_index=batch_index, smoke_target=args.smoke_steps), checkpoint)
                print(json.dumps({"smoke_steps": global_step, "losses": values, "method_development_evaluated": False, "official_suim_test_evaluated": False}), flush=True)
                return
        row = {"epoch": epoch, "global_step": global_step, **{key: value / max(batches, 1) for key, value in totals.items()}, "method_development_evaluated": False, "validation_evaluated": False, "calibration_evaluated": False, "official_suim_test_evaluated": False}
        history.append(row)
        output.mkdir(parents=True, exist_ok=True)
        temporary = output / "train_history.json.tmp"; temporary.write_text(json.dumps(history, indent=2) + "\n", encoding="utf-8"); os.replace(temporary, output / "train_history.json")
        atomic_torch_save(checkpoint_payload(epoch, global_step, model, optimizer, scaler, variant=args.variant, **provenance, run_kind=run_kind, epoch_completed=True, batch_index=None, smoke_target=args.smoke_steps or None), checkpoint)
        if run_kind == "formal" and args.variant == "B" and epoch == 5:
            warmup = ROOT / config["experiment"]["output_dir"] / "segformer" / "formal" / "shared_warmup" / "checkpoints" / "final.pt"
            atomic_torch_save(checkpoint_payload(epoch, global_step, model, optimizer, scaler, variant="shared_warmup", **provenance, run_kind="formal", epoch_completed=True, batch_index=None, smoke_target=None), warmup)


if __name__ == "__main__":
    main()
