"""Train a DeepLabV3-MobileNetV3 CNN replication on the formal SUIM protocol."""
from __future__ import annotations

import argparse
import csv
import logging
import os
import random
import shutil
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as functional
import yaml
from torch.utils.data import DataLoader
from torchvision.models import MobileNet_V3_Large_Weights
from torchvision.models.segmentation import deeplabv3_mobilenet_v3_large

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from datasets.suim_dataset import SUIMDataset, build_eval_transform, build_train_transform  # noqa: E402
from metrics.segmentation import confusion_matrix, metrics_from_confusion_matrix  # noqa: E402
from utils.reproducibility import set_seed  # noqa: E402


def load_config(path: Path) -> dict[str, Any]:
    return yaml.safe_load(path.read_text(encoding="utf-8"))


def build_model(num_classes: int, backbone_weights: str | None) -> torch.nn.Module:
    weights = None if backbone_weights is None else MobileNet_V3_Large_Weights[backbone_weights]
    return deeplabv3_mobilenet_v3_large(weights=None, weights_backbone=weights, num_classes=num_classes, aux_loss=False)


def resize_logits(logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    return functional.interpolate(logits, size=labels.shape[-2:], mode="bilinear", align_corners=False)


def capture_rng_state() -> dict[str, Any]:
    return {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch": torch.get_rng_state(),
        "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
    }


def restore_rng_state(state: dict[str, Any]) -> None:
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch"].detach().to(device="cpu", dtype=torch.uint8))
    if state.get("cuda") is not None and torch.cuda.is_available():
        torch.cuda.set_rng_state_all([value.detach().to(device="cpu", dtype=torch.uint8) for value in state["cuda"]])


def atomic_torch_save(payload: dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, temporary)
    os.replace(temporary, path)


def checkpoint_payload(epoch: int, model: torch.nn.Module, optimizer: torch.optim.Optimizer, scheduler: torch.optim.lr_scheduler.LRScheduler, scaler: torch.amp.GradScaler, best_miou: float, config: dict[str, Any]) -> dict[str, Any]:
    return {
        "checkpoint_format": "deeplabv3_mobilenetv3_suim_v1",
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": scheduler.state_dict(),
        "scaler_state_dict": scaler.state_dict(),
        "rng_state": capture_rng_state(),
        "best_miou": best_miou,
        "config": config,
        "official_test_evaluated": False,
    }


@torch.no_grad()
def evaluate(model: torch.nn.Module, loader: DataLoader, device: torch.device, amp: bool, classes: int, ignore_index: int) -> dict[str, Any]:
    model.eval()
    matrix = torch.zeros((classes, classes), dtype=torch.long, device=device)
    loss_sum, pixels = 0.0, 0
    for batch in loader:
        labels = batch["labels"].to(device)
        with torch.amp.autocast("cuda", enabled=amp):
            logits = resize_logits(model(batch["pixel_values"].to(device))["out"], labels)
            loss = functional.cross_entropy(logits, labels, ignore_index=ignore_index)
        valid = labels.ne(ignore_index)
        count = int(valid.sum())
        loss_sum += float(loss) * count
        pixels += count
        matrix += confusion_matrix(logits.argmax(dim=1), labels, num_classes=classes, ignore_index=ignore_index).to(device)
    metrics = metrics_from_confusion_matrix(matrix.cpu())
    metrics["loss"] = loss_sum / pixels
    return metrics


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=ROOT / "configs" / "baseline" / "deeplabv3_mobilenetv3_suim_v2_scene.yaml")
    parser.add_argument("--resume", type=Path)
    parser.add_argument("--smoke-steps", type=int, default=0)
    args = parser.parse_args()
    config = load_config(args.config.resolve())
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for this formal CNN replication.")
    data, training, loss = config["data"], config["training"], config["loss"]
    set_seed(int(config["experiment"]["seed"]))
    device = torch.device("cuda")
    output = ROOT / config["experiment"]["output_dir"]
    if args.smoke_steps:
        output = ROOT / "outputs" / "deeplabv3_mobilenetv3_suim_smoke"
    checkpoints, logs = output / "checkpoints", output / "logs"
    checkpoints.mkdir(parents=True, exist_ok=True)
    logs.mkdir(parents=True, exist_ok=True)
    shutil.copy2(args.config, output / "config.yaml")
    logger = logging.getLogger("deeplabv3_suim")
    logger.handlers.clear()
    logger.setLevel(logging.INFO)
    formatter = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s")
    for handler in (logging.StreamHandler(), logging.FileHandler(logs / "console.log", encoding="utf-8")):
        handler.setFormatter(formatter)
        logger.addHandler(handler)

    split_dir = ROOT / data["split_dir"]
    train = SUIMDataset(split_dir / "train.csv", transform=build_train_transform(int(data["image_size"])))
    validation = SUIMDataset(split_dir / "val.csv", transform=build_eval_transform(int(data["image_size"])))
    loader_args = {"batch_size": int(data["batch_size"]), "num_workers": int(data["num_workers"]), "pin_memory": True}
    train_loader = DataLoader(train, shuffle=True, **loader_args)
    validation_loader = DataLoader(validation, shuffle=False, **loader_args)
    model = build_model(int(data["num_classes"]), config["model"].get("backbone_weights")).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=float(training["learning_rate"]), weight_decay=float(training["weight_decay"]))
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=int(training["epochs"]))
    scaler = torch.amp.GradScaler("cuda", enabled=bool(training["amp"]))
    start_epoch, best_miou = 1, float("-inf")
    if args.resume:
        saved = torch.load(args.resume, map_location=device, weights_only=False)
        if saved.get("checkpoint_format") != "deeplabv3_mobilenetv3_suim_v1":
            raise ValueError("Unsupported DeepLab checkpoint.")
        model.load_state_dict(saved["model_state_dict"])
        optimizer.load_state_dict(saved["optimizer_state_dict"])
        scheduler.load_state_dict(saved["scheduler_state_dict"])
        scaler.load_state_dict(saved["scaler_state_dict"])
        restore_rng_state(saved["rng_state"])
        start_epoch, best_miou = int(saved["epoch"]) + 1, float(saved["best_miou"])
        logger.info("resumed completed epoch=%s", start_epoch - 1)

    if args.smoke_steps:
        model.train()
        for step, batch in enumerate(train_loader, start=1):
            labels = batch["labels"].to(device)
            optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast("cuda", enabled=bool(training["amp"])):
                logits = resize_logits(model(batch["pixel_values"].to(device))["out"], labels)
                value = functional.cross_entropy(logits, labels, ignore_index=int(loss["ignore_index"]))
            scaler.scale(value).backward()
            scaler.step(optimizer)
            scaler.update()
            if step >= args.smoke_steps:
                break
        logger.info("smoke steps=%s loss=%.4f", step, float(value))
        return

    history_path = logs / "train_history.csv"
    write_header = not history_path.exists() or start_epoch == 1
    without_improvement = 0
    with history_path.open("a", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["epoch", "train_loss", "val_loss", "val_miou", "learning_rate"])
        if write_header:
            writer.writeheader()
        for epoch in range(start_epoch, int(training["epochs"]) + 1):
            model.train()
            loss_sum, pixels = 0.0, 0
            for batch in train_loader:
                labels = batch["labels"].to(device)
                optimizer.zero_grad(set_to_none=True)
                with torch.amp.autocast("cuda", enabled=bool(training["amp"])):
                    logits = resize_logits(model(batch["pixel_values"].to(device))["out"], labels)
                    value = functional.cross_entropy(logits, labels, ignore_index=int(loss["ignore_index"]))
                scaler.scale(value).backward()
                scaler.step(optimizer)
                scaler.update()
                count = int(labels.ne(int(loss["ignore_index"])).sum())
                loss_sum += float(value) * count
                pixels += count
            scheduler.step()
            row: dict[str, Any] = {"epoch": epoch, "train_loss": loss_sum / pixels, "val_loss": "", "val_miou": "", "learning_rate": optimizer.param_groups[0]["lr"]}
            if epoch % int(training["validate_every"]) == 0 or epoch == int(training["epochs"]):
                validation_metrics = evaluate(model, validation_loader, device, bool(training["amp"]), int(data["num_classes"]), int(loss["ignore_index"]))
                row.update(val_loss=validation_metrics["loss"], val_miou=validation_metrics["miou"])
                if validation_metrics["miou"] > best_miou:
                    best_miou, without_improvement = validation_metrics["miou"], 0
                    atomic_torch_save(checkpoint_payload(epoch, model, optimizer, scheduler, scaler, best_miou, config), checkpoints / "best.pt")
                else:
                    without_improvement += 1
                logger.info("epoch=%s train_loss=%.4f val_miou=%.4f", epoch, row["train_loss"], validation_metrics["miou"])
            else:
                logger.info("epoch=%s train_loss=%.4f", epoch, row["train_loss"])
            writer.writerow(row)
            handle.flush()
            atomic_torch_save(checkpoint_payload(epoch, model, optimizer, scheduler, scaler, best_miou, config), checkpoints / "last.pt")
            if without_improvement >= int(training["early_stopping_patience"]):
                logger.info("Early stopping triggered.")
                break


if __name__ == "__main__":
    main()
