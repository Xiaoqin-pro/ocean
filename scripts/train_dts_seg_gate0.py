"""Train the preregistered F4 and DTS Gate-0 variants on method_train only."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
import sys
import tempfile
from pathlib import Path
from typing import Any

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
from reliability.dts_seg import degradation_acceleration_loss, trajectory_names, true_class_margin  # noqa: E402
from reliability.ft_reliability import stable_hash  # noqa: E402


FORMAT = "dts_seg_gate0_v1"
VARIANTS = ("F4", "DTS")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest().upper()


def atomic_torch_save(value: dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, raw = tempfile.mkstemp(suffix=".pt", dir=path.parent)
    os.close(descriptor)
    temporary = Path(raw)
    try:
        torch.save(value, temporary)
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


class FourViewTrajectoryDataset(Dataset[dict[str, Any]]):
    def __init__(self, split_csv: Path, registry: Path, image_size: int) -> None:
        if split_csv.name != "risk_head_train.csv":
            raise PermissionError("DTS Gate-0 training accepts method_train only.")
        self.frame = pd.read_csv(split_csv).sort_values("sample_id").reset_index(drop=True)
        if len(self.frame) != 936 or self.frame.sample_id.duplicated().any():
            raise ValueError("method_train must contain 936 unique scenes.")
        self.conditions = {item.name: build_image_degradation(item) for item in load_conditions(registry)}
        self.epoch = 1
        self.transform = A.ReplayCompose(
            [
                A.Resize(image_size, image_size),
                A.HorizontalFlip(p=0.5),
                A.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD, max_pixel_value=255.0),
                ToTensorV2(),
            ]
        )

    def set_epoch(self, epoch: int) -> None:
        self.epoch = epoch

    def __len__(self) -> int:
        return len(self.frame)

    def __getitem__(self, index: int) -> dict[str, Any]:
        row = self.frame.iloc[index]
        sample_id = str(row.sample_id)
        with Image.open(ROOT / str(row.image_path)) as handle:
            image = np.asarray(handle.convert("RGB"), dtype=np.uint8)
        with Image.open(ROOT / str(row.mask_path)) as handle:
            label = np.asarray(handle, dtype=np.uint8)
        names = trajectory_names(sample_id, self.epoch)
        self.transform.set_random_seed((stable_hash(sample_id) + 100003 * self.epoch) % (2**32))
        base = self.transform(image=image, mask=label)
        views = {"clean": base["image"].float()}
        for key, name in zip(("s1", "s2", "s3"), names[1:], strict=True):
            replay = A.ReplayCompose.replay(base["replay"], image=self.conditions[name](image, sample_id), mask=label)
            views[key] = replay["image"].float()
        return {**views, "labels": base["mask"].long(), "sample_id": sample_id}


def build_model(config: dict[str, Any], device: torch.device) -> torch.nn.Module:
    model = SegformerForSemanticSegmentation.from_pretrained(
        config["model"]["pretrained_model"],
        num_labels=int(config["model"]["num_classes"]),
        id2label=ID2LABEL,
        label2id=LABEL2ID,
        ignore_mismatched_sizes=True,
    ).to(device)
    source_path = ROOT / str(config["model"]["source_checkpoint"])
    source = torch.load(source_path, map_location=device, weights_only=False)
    required = {"variant": "B", "model_name": "segformer", "epoch": 100, "run_kind": "formal", "epoch_completed": True}
    for key, expected in required.items():
        if source.get(key) != expected:
            raise ValueError(f"Strong source checkpoint has invalid {key}.")
    if any(bool(source.get(key, True)) for key in ("validation_evaluated", "calibration_evaluated", "official_suim_test_evaluated")):
        raise ValueError("Source checkpoint records prohibited formal-split access.")
    model.load_state_dict(source["model_state_dict"])
    return model


def checkpoint_payload(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    scaler: torch.amp.GradScaler,
    *,
    variant: str,
    epoch: int,
    config_path: Path,
    source_path: Path,
    history: list[dict[str, float]],
    smoke: bool,
) -> dict[str, Any]:
    return {
        "checkpoint_format": FORMAT,
        "variant": variant,
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "scaler_state_dict": scaler.state_dict(),
        "history": history,
        "config_sha256": sha256(config_path),
        "source_checkpoint_sha256": sha256(source_path),
        "smoke": smoke,
        "method_development_evaluated": False,
        "validation_evaluated": False,
        "calibration_evaluated": False,
        "official_suim_test_evaluated": False,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--variant", required=True, choices=VARIANTS)
    parser.add_argument("--config", type=Path, default=ROOT / "configs/dts_seg_gate0.yaml")
    parser.add_argument("--smoke-batches", type=int, default=0)
    args = parser.parse_args()
    config_path = args.config.resolve()
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    if args.smoke_batches < 0:
        raise ValueError("smoke-batches must be non-negative.")
    seed = int(config["experiment"]["seed"])
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)
    if not torch.cuda.is_available():
        raise RuntimeError("DTS Gate-0 requires CUDA.")
    device = torch.device("cuda")
    training = config["training"]
    dataset = FourViewTrajectoryDataset(
        ROOT / str(config["data"]["train_csv"]),
        ROOT / str(config["data"]["degradation_registry"]),
        int(config["data"]["image_size"]),
    )
    generator = torch.Generator().manual_seed(seed)
    loader = DataLoader(dataset, batch_size=int(training["batch_size"]), shuffle=True, num_workers=0, pin_memory=True, generator=generator)
    model = build_model(config, device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=float(training["learning_rate"]), weight_decay=float(training["weight_decay"]))
    scaler = torch.amp.GradScaler("cuda", enabled=bool(training["amp"]))
    epochs = 1 if args.smoke_batches else int(training["epochs"])
    output = ROOT / str(config["experiment"]["output_dir"]) / ("smoke" if args.smoke_batches else "formal") / args.variant
    history: list[dict[str, float]] = []
    source_path = ROOT / str(config["model"]["source_checkpoint"])
    model.train()
    for epoch in range(1, epochs + 1):
        dataset.set_epoch(epoch)
        totals = {"ce": 0.0, "dts_s2": 0.0, "dts_s3": 0.0, "eligible_s2": 0.0, "eligible_s3": 0.0}
        batches = 0
        for batch_index, batch in enumerate(loader):
            if args.smoke_batches and batch_index >= args.smoke_batches:
                break
            labels = batch["labels"].to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            detached_margins: list[torch.Tensor] = []
            detached_correct: list[torch.Tensor] = []
            for view_index, key in enumerate(("clean", "s1", "s2", "s3")):
                pixels = batch[key].to(device, non_blocking=True)
                with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=bool(training["amp"])):
                    logits = model(pixel_values=pixels).logits
                    full_logits = functional.interpolate(logits.float(), size=labels.shape[-2:], mode="bilinear", align_corners=False)
                    ce = functional.cross_entropy(full_logits, labels, ignore_index=255) / 4.0
                    margin, valid = true_class_margin(logits.float(), labels)
                    native_labels = functional.interpolate(labels.float().unsqueeze(1), size=logits.shape[-2:], mode="nearest")[:, 0].long()
                    correct = logits.detach().argmax(1).eq(native_labels) & valid
                    dts = logits.sum() * 0.0
                    eligible_count = 0
                    if args.variant == "DTS" and view_index >= 2:
                        eligible = detached_correct[-1] & valid
                        dts = degradation_acceleration_loss(
                            detached_margins[-2], detached_margins[-1], margin, eligible,
                            tolerance=float(training["acceleration_tolerance"]),
                        )
                        eligible_count = int(eligible.sum())
                    loss = ce + float(training["dts_weight"]) * dts
                scaler.scale(loss).backward()
                detached_margins.append(margin.detach())
                detached_correct.append(correct.detach())
                totals["ce"] += float(ce.detach())
                if key in {"s2", "s3"}:
                    totals[f"dts_{key}"] += float(dts.detach())
                    totals[f"eligible_{key}"] += eligible_count
                del pixels, logits, full_logits, margin, loss
            scaler.step(optimizer)
            scaler.update()
            batches += 1
            if batch_index % 25 == 0:
                print(f"{args.variant} epoch={epoch} batch={batch_index}/{len(loader)} ce={totals['ce']/batches:.4f} dts={(totals['dts_s2']+totals['dts_s3'])/batches:.4f}", flush=True)
        if batches == 0:
            raise RuntimeError("No training batch completed.")
        row = {"epoch": float(epoch), "batches": float(batches), **{key: value / batches for key, value in totals.items()}}
        history.append(row)
        payload = checkpoint_payload(model, optimizer, scaler, variant=args.variant, epoch=epoch, config_path=config_path, source_path=source_path, history=history, smoke=bool(args.smoke_batches))
        atomic_torch_save(payload, output / "checkpoints" / "last.pt")
        print(json.dumps(row, sort_keys=True), flush=True)
    if not args.smoke_batches:
        atomic_torch_save(payload, output / "checkpoints" / "final.pt")


if __name__ == "__main__":
    main()
