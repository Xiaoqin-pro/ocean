"""Fixed FT-Reliability v1.1 SegFormer pilot driver.

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
    failure_transition_loss, generic_correctness_ranking_loss, trajectory_family,
    validate_split_manifest,
)


PROTOCOL_COMMIT = "9f54a1c"
ALLOWED_VARIANTS = ("A", "B", "C", "D", "E")


def load_config(root: Path = ROOT) -> dict[str, Any]:
    return yaml.safe_load((root / "configs" / "ft_reliability_pilot.yaml").read_text(encoding="utf-8"))


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
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
        "retention": variant == "E",
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
    method_train_csv_sha256: str, degradation_config_sha256: str, teacher_checkpoint_sha256: str | None = None,
) -> dict[str, object]:
    return {
        "checkpoint_format": "ft_reliability_pilot_v1_1", "protocol_commit": PROTOCOL_COMMIT,
        "variant": variant, "model_name": "segformer", "epoch": epoch, "global_step": global_step,
        "model_state_dict": model.state_dict(), "optimizer_state_dict": optimizer.state_dict(),
        "scaler_state_dict": scaler.state_dict(), "rng_state": capture_rng_state(),
        "initialization_sha256": initialization_sha256, "method_train_csv_sha256": method_train_csv_sha256,
        "degradation_config_sha256": degradation_config_sha256, "teacher_checkpoint_sha256": teacher_checkpoint_sha256,
        "checkpoint_selection": "final_epoch",
        "method_development_evaluated": False, "validation_evaluated": False,
        "calibration_evaluated": False, "official_suim_test_evaluated": False,
    }


def load_completed_epoch_checkpoint(
    path: Path, model: torch.nn.Module, optimizer: torch.optim.Optimizer, scaler: torch.amp.GradScaler,
    device: torch.device, *, variant: str,
) -> tuple[int, int]:
    checkpoint = torch.load(path, map_location=device, weights_only=False)
    if checkpoint.get("checkpoint_format") != "ft_reliability_pilot_v1_1" or checkpoint.get("protocol_commit") != PROTOCOL_COMMIT:
        raise ValueError("Unsupported FT-Reliability checkpoint.")
    if checkpoint.get("variant") != variant or any(bool(checkpoint.get(key, True)) for key in ("method_development_evaluated", "validation_evaluated", "calibration_evaluated", "official_suim_test_evaluated")):
        raise ValueError("Checkpoint violates the frozen FT-Reliability access protocol.")
    model.load_state_dict(checkpoint["model_state_dict"])
    optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    scaler.load_state_dict(checkpoint["scaler_state_dict"])
    restore_rng_state(checkpoint["rng_state"])
    return int(checkpoint["epoch"]) + 1, int(checkpoint["global_step"])


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
) -> dict[str, float]:
    """Execute one mathematically averaged three-view update and one optimizer step."""
    terms = variant_terms(variant, epoch)
    device = next(model.parameters()).device
    labels = batch["labels"].to(device)
    boundary = _boundary(labels)
    optimizer.zero_grad(set_to_none=True)
    values = {"clean_ce": 0.0, "s1_ce": 0.0, "s3_ce": 0.0, "ft": 0.0, "retain": 0.0, "ranking": 0.0}
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
        clean_loss = clean_ce if terms["clean_only"] else clean_ce / 3.0
        clean_loss = clean_loss + (0.10 * retain if terms["retention"] else 0.0)
    scaler.scale(clean_loss).backward()
    values["clean_ce"], values["retain"] = float(clean_ce.detach()), float(retain.detach())
    if not terms["clean_only"]:
        with torch.amp.autocast("cuda", enabled=amp):
            s1_logits = _student_logits(model, batch["s1"].to(device), labels)
            s1_ce = functional.cross_entropy(s1_logits, labels, ignore_index=IGNORE_INDEX)
        scaler.scale(s1_ce / 3.0).backward()
        with torch.amp.autocast("cuda", enabled=amp):
            s3_logits = _student_logits(model, batch["s3"].to(device), labels)
            s3_ce = functional.cross_entropy(s3_logits, labels, ignore_index=IGNORE_INDEX)
            ft = s3_logits.sum() * 0.0
            ranking = s3_logits.sum() * 0.0
            if terms["failure_transition"]:
                ft, _ = failure_transition_loss(s1_logits, s3_logits, labels, boundary, epoch=epoch, batch_index=batch_index)
            if terms["generic_ranking"]:
                ranking, _ = generic_correctness_ranking_loss(s3_logits, labels, boundary, epoch=epoch, batch_index=batch_index)
            s3_loss = s3_ce / 3.0 + 0.10 * ft + 0.10 * ranking
        scaler.scale(s3_loss).backward()
        values.update(s1_ce=float(s1_ce.detach()), s3_ce=float(s3_ce.detach()), ft=float(ft.detach()), ranking=float(ranking.detach()))
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
    config = load_config(ROOT)
    if any(bool(config["access_control"][key]) for key in ("method_development_evaluated", "validation_evaluated", "calibration_evaluated", "official_suim_test_evaluated")):
        raise ValueError("FT-Reliability protocol forbids evaluation during pilot training.")
    # Metadata-only audit: this deliberately does not read development contents.
    validate_split_manifest({
        "method_train_count": int(config["access_control"]["method_train_count"]),
        "method_development_count": int(config["access_control"]["method_development_count"]),
        **dict(config["access_control"]["frozen_split_audit"]),
    })
    seed = int(config["experiment"]["seed"])
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    device = torch.device("cuda")
    split = ROOT / config["access_control"]["method_train_csv"]
    dataset = TrajectoryDataset(ROOT, split, ROOT / config["degradations"]["registry_config"])
    loader = DataLoader(dataset, batch_size=int(config["models"]["segformer_b0"]["base_scene_batch_size"]), shuffle=True, num_workers=0, pin_memory=True)
    model = _build_segformer(config, device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=float(config["models"]["segformer_b0"]["learning_rate"]), weight_decay=float(config["models"]["segformer_b0"]["weight_decay"]))
    scaler = torch.amp.GradScaler("cuda", enabled=True)
    teacher = None
    if args.variant == "E":
        raise RuntimeError("Baseline-936 teacher production is intentionally deferred until the integration authorization.")
    output = ROOT / config["experiment"]["output_dir"] / "segformer" / args.variant
    checkpoint = output / "checkpoints" / "last.pt"
    start_epoch, global_step = 1, 0
    if args.resume:
        start_epoch, global_step = load_completed_epoch_checkpoint(args.resume, model, optimizer, scaler, device, variant=args.variant)
    initialization_sha = "recorded_at_first_authorized_training_run"
    for epoch in range(start_epoch, int(config["models"]["segformer_b0"]["epochs"]) + 1):
        dataset.set_epoch(epoch)
        for batch_index, batch in enumerate(loader):
            values = run_one_step(model, optimizer, scaler, batch, variant=args.variant, epoch=epoch, batch_index=batch_index, amp=True, teacher=teacher)
            global_step += 1
            if args.smoke_steps and global_step >= args.smoke_steps:
                atomic_torch_save(checkpoint_payload(epoch, global_step, model, optimizer, scaler, variant=args.variant, initialization_sha256=initialization_sha, method_train_csv_sha256=sha256(split), degradation_config_sha256=sha256(ROOT / config["degradations"]["registry_config"])), checkpoint)
                print(json.dumps({"smoke_steps": global_step, "losses": values, "method_development_evaluated": False, "official_suim_test_evaluated": False}), flush=True)
                return
        atomic_torch_save(checkpoint_payload(epoch, global_step, model, optimizer, scaler, variant=args.variant, initialization_sha256=initialization_sha, method_train_csv_sha256=sha256(split), degradation_config_sha256=sha256(ROOT / config["degradations"]["registry_config"])), checkpoint)


if __name__ == "__main__":
    main()
