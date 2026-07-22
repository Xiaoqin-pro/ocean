from __future__ import annotations

import argparse
import csv
import logging
import shutil
import sys
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as functional
import yaml
from torch.utils.data import DataLoader
from transformers import SegformerForSemanticSegmentation

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from datasets.suim_dataset import SUIMDataset, build_eval_transform, build_train_transform  # noqa: E402
from datasets.label_mapping import ID2LABEL, LABEL2ID  # noqa: E402
from metrics.segmentation import confusion_matrix, metrics_from_confusion_matrix  # noqa: E402
from utils.experiment_logger import append_experiment  # noqa: E402
from utils.reproducibility import set_seed  # noqa: E402


def load_config(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def resize_logits(logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    return functional.interpolate(logits, size=labels.shape[-2:], mode="bilinear", align_corners=False)


@torch.no_grad()
def evaluate(model: torch.nn.Module, loader: DataLoader, device: torch.device, amp: bool, classes: int, ignore_index: int) -> dict[str, Any]:
    model.eval()
    matrix = torch.zeros((classes, classes), dtype=torch.long, device=device)
    loss_sum, valid_pixel_count = 0.0, 0
    for batch in loader:
        pixels, labels = batch["pixel_values"].to(device), batch["labels"].to(device)
        with torch.amp.autocast("cuda", enabled=amp):
            logits = resize_logits(model(pixel_values=pixels).logits, labels)
            loss = functional.cross_entropy(logits, labels, ignore_index=ignore_index)
        valid = labels.ne(ignore_index)
        pixels_in_batch = int(valid.sum().item())
        loss_sum += float(loss.item()) * pixels_in_batch
        valid_pixel_count += pixels_in_batch
        matrix += confusion_matrix(logits.argmax(dim=1), labels, num_classes=classes, ignore_index=ignore_index).to(device)
    result = metrics_from_confusion_matrix(matrix.cpu())
    result["loss"] = loss_sum / valid_pixel_count if valid_pixel_count else float("nan")
    return result


def save_checkpoint(path: Path, epoch: int, model: torch.nn.Module, optimizer: torch.optim.Optimizer, scaler: torch.amp.GradScaler, best_miou: float, config: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "scaler_state_dict": scaler.state_dict(),
        "best_miou": best_miou,
        "config": config,
    }, path)


def main() -> None:
    parser = argparse.ArgumentParser(description="Train the SUIM SegFormer-B0 cross-entropy baseline.")
    parser.add_argument("--config", type=Path, default=PROJECT_ROOT / "configs" / "segformer_b0_suim_baseline.yaml")
    parser.add_argument("--resume", type=Path)
    args = parser.parse_args()
    config = load_config(args.config)
    data, model_config, training, loss_config = config["data"], config["model"], config["training"], config["loss"]
    if not torch.cuda.is_available():
        raise RuntimeError("Baseline training is configured for CUDA.")
    set_seed(config["experiment"]["seed"])
    device = torch.device("cuda")
    output_dir = PROJECT_ROOT / config["experiment"]["output_dir"]
    checkpoints, logs = output_dir / "checkpoints", output_dir / "logs"
    checkpoints.mkdir(parents=True, exist_ok=True)
    logs.mkdir(parents=True, exist_ok=True)
    (output_dir / "visualizations").mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger("train_baseline")
    logger.handlers.clear()
    logger.setLevel(logging.INFO)
    formatter = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s")
    for handler in (logging.StreamHandler(), logging.FileHandler(logs / "console.log", encoding="utf-8")):
        handler.setFormatter(formatter)
        logger.addHandler(handler)
    shutil.copy2(args.config, output_dir / "config.yaml")

    split_dir = PROJECT_ROOT / data["split_dir"]
    train_dataset = SUIMDataset(split_dir / "train.csv", transform=build_train_transform(data["image_size"]))
    val_dataset = SUIMDataset(split_dir / "val.csv", transform=build_eval_transform(data["image_size"]))
    loader_args = dict(batch_size=data["batch_size"], num_workers=data["num_workers"], pin_memory=True)
    train_loader = DataLoader(train_dataset, shuffle=True, **loader_args)
    val_loader = DataLoader(val_dataset, shuffle=False, **loader_args)
    model = SegformerForSemanticSegmentation.from_pretrained(
        model_config["pretrained_model"], num_labels=data["num_classes"], id2label=ID2LABEL,
        label2id=LABEL2ID, ignore_mismatched_sizes=True,
    ).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=training["learning_rate"], weight_decay=training["weight_decay"])
    amp = bool(training["amp"])
    scaler = torch.amp.GradScaler("cuda", enabled=amp)
    start_epoch, best_miou, best_epoch, without_improvement = 1, float("-inf"), 0, 0
    if args.resume:
        checkpoint = torch.load(args.resume, map_location=device, weights_only=False)
        model.load_state_dict(checkpoint["model_state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        scaler.load_state_dict(checkpoint["scaler_state_dict"])
        start_epoch, best_miou = checkpoint["epoch"] + 1, checkpoint["best_miou"]

    history_path = logs / "train_history.csv"
    write_header = not history_path.exists() or start_epoch == 1
    with history_path.open("a", newline="", encoding="utf-8") as history:
        writer = csv.DictWriter(history, fieldnames=["epoch", "train_loss", "val_loss", "val_miou"])
        if write_header:
            writer.writeheader()
        for epoch in range(start_epoch, training["epochs"] + 1):
            model.train()
            loss_sum, valid_pixel_count = 0.0, 0
            for batch in train_loader:
                pixels, labels = batch["pixel_values"].to(device), batch["labels"].to(device)
                optimizer.zero_grad(set_to_none=True)
                with torch.amp.autocast("cuda", enabled=amp):
                    logits = resize_logits(model(pixel_values=pixels).logits, labels)
                    loss = functional.cross_entropy(logits, labels, ignore_index=loss_config["ignore_index"])
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
                valid = labels.ne(loss_config["ignore_index"])
                pixels_in_batch = int(valid.sum().item())
                loss_sum += float(loss.item()) * pixels_in_batch
                valid_pixel_count += pixels_in_batch
            row: dict[str, Any] = {"epoch": epoch, "train_loss": loss_sum / valid_pixel_count if valid_pixel_count else float("nan"), "val_loss": "", "val_miou": ""}
            if epoch % training["validate_every"] == 0 or epoch == training["epochs"]:
                validation = evaluate(model, val_loader, device, amp, data["num_classes"], loss_config["ignore_index"])
                row.update(val_loss=validation["loss"], val_miou=validation["miou"])
                if validation["miou"] > best_miou:
                    best_miou, best_epoch, without_improvement = validation["miou"], epoch, 0
                    save_checkpoint(checkpoints / "best.pt", epoch, model, optimizer, scaler, best_miou, config)
                else:
                    without_improvement += 1
                logger.info("epoch=%s train_loss=%.4f val_miou=%.4f", epoch, row["train_loss"], validation["miou"])
            else:
                logger.info("epoch=%s train_loss=%.4f", epoch, row["train_loss"])
            writer.writerow(row)
            history.flush()
            save_checkpoint(checkpoints / "last.pt", epoch, model, optimizer, scaler, best_miou, config)
            if without_improvement >= training["early_stopping_patience"]:
                logger.info("Early stopping triggered.")
                break
    append_experiment({
        "experiment_id": config["experiment"]["name"], "model": model_config["pretrained_model"],
        "dataset_split": data["split_dir"], "seed": config["experiment"]["seed"], "image_size": data["image_size"],
        "batch_size": data["batch_size"], "epochs": epoch, "learning_rate": training["learning_rate"],
        "best_epoch": best_epoch, "val_miou": best_miou, "checkpoint_path": str(checkpoints / "best.pt"), "notes": "CE baseline",
    })


if __name__ == "__main__":
    main()
