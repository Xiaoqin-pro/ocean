"""Train the preregistered UIIS SegFormer baseline without touching confirmation."""
from __future__ import annotations

import argparse
import csv
import json
import logging
import os
import random
import shutil
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as functional
import yaml
from torch.utils.data import DataLoader
from transformers import SegformerForSemanticSegmentation

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from datasets.label_mapping import ID2LABEL, LABEL2ID  # noqa: E402
from datasets.suim_dataset import SUIMDataset, build_train_transform  # noqa: E402
from utils.reproducibility import set_seed  # noqa: E402


def load_config(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def _rng_state() -> dict[str, Any]:
    return {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch": torch.get_rng_state(),
        "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
    }


def _restore_rng_state(state: dict[str, Any]) -> None:
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch"])
    if state.get("cuda") is not None and torch.cuda.is_available():
        torch.cuda.set_rng_state_all(state["cuda"])


def atomic_torch_save(payload: dict[str, Any], path: Path) -> None:
    """Write a complete checkpoint before replacing the prior last.pt."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, temporary)
    os.replace(temporary, path)


def checkpoint_payload(
    epoch: int,
    global_step: int,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LRScheduler,
    scaler: torch.amp.GradScaler,
    config: dict[str, Any],
) -> dict[str, Any]:
    return {
        "checkpoint_format": "uiis_fixed_protocol_v1",
        "epoch": epoch,
        "global_step": global_step,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": scheduler.state_dict(),
        "scaler_state_dict": scaler.state_dict(),
        "rng_state": _rng_state(),
        "config": config,
        "official_suim_test_evaluated": False,
        "confirmation_evaluated": False,
    }


def load_completed_epoch_checkpoint(
    path: Path,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LRScheduler,
    scaler: torch.amp.GradScaler,
    device: torch.device,
) -> tuple[int, int]:
    checkpoint = torch.load(path, map_location=device, weights_only=False)
    if checkpoint.get("checkpoint_format") != "uiis_fixed_protocol_v1":
        raise ValueError("Unsupported checkpoint format.")
    model.load_state_dict(checkpoint["model_state_dict"])
    optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
    scaler.load_state_dict(checkpoint["scaler_state_dict"])
    _restore_rng_state(checkpoint["rng_state"])
    return int(checkpoint["epoch"]) + 1, int(checkpoint["global_step"])


def resize_logits(logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    return functional.interpolate(logits, size=labels.shape[-2:], mode="bilinear", align_corners=False)


def build_epoch_loader(dataset: SUIMDataset, batch_size: int, workers: int, seed: int) -> DataLoader:
    generator = torch.Generator()
    generator.manual_seed(seed)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        generator=generator,
        num_workers=workers,
        pin_memory=True,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=ROOT / "configs" / "uiis" / "segformer_b0_alpha010_confirmation.yaml")
    parser.add_argument("--resume", type=Path, help="Resume only from a completed epoch last.pt checkpoint.")
    parser.add_argument("--benchmark-steps", type=int, default=0, help="Run a non-formal timing benchmark and exit.")
    parser.add_argument("--benchmark-output-dir", type=Path, default=ROOT / "outputs" / "uiis_speed_benchmark")
    args = parser.parse_args()
    if args.benchmark_steps < 0:
        raise ValueError("benchmark-steps must be non-negative.")
    config = load_config(args.config)
    data, model_config, training, loss_config = (config[key] for key in ("data", "model", "training", "loss"))
    if not torch.cuda.is_available():
        raise RuntimeError("UIIS training is configured for CUDA.")
    set_seed(int(config["experiment"]["seed"]))
    device = torch.device("cuda")
    output_dir = ROOT / config["experiment"]["output_dir"]
    if args.benchmark_steps:
        output_dir = args.benchmark_output_dir
    checkpoints, logs = output_dir / "checkpoints", output_dir / "logs"
    checkpoints.mkdir(parents=True, exist_ok=True)
    logs.mkdir(parents=True, exist_ok=True)
    shutil.copy2(args.config, output_dir / "config.yaml")
    logger = logging.getLogger("uiis_fixed_protocol")
    logger.handlers.clear()
    logger.setLevel(logging.INFO)
    formatter = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s")
    for handler in (logging.StreamHandler(), logging.FileHandler(logs / "console.log", encoding="utf-8")):
        handler.setFormatter(formatter)
        logger.addHandler(handler)

    split_dir = ROOT / data["split_dir"]
    train_dataset = SUIMDataset(split_dir / "train.csv", transform=build_train_transform(int(data["image_size"])))
    model = SegformerForSemanticSegmentation.from_pretrained(
        model_config["pretrained_model"],
        num_labels=int(data["num_classes"]),
        id2label=ID2LABEL,
        label2id=LABEL2ID,
        ignore_mismatched_sizes=True,
    ).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=float(training["learning_rate"]), weight_decay=float(training["weight_decay"]))
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=int(training["epochs"]))
    amp = bool(training["amp"])
    scaler = torch.amp.GradScaler("cuda", enabled=amp)
    start_epoch, global_step = 1, 0
    if args.resume:
        start_epoch, global_step = load_completed_epoch_checkpoint(args.resume, model, optimizer, scheduler, scaler, device)
        logger.info("resumed completed epoch=%s global_step=%s", start_epoch - 1, global_step)

    if args.benchmark_steps:
        loader = build_epoch_loader(train_dataset, int(data["batch_size"]), int(data["num_workers"]), int(config["experiment"]["seed"]))
        model.train()
        elapsed, data_elapsed, loss_sum = 0.0, 0.0, 0.0
        data_start = time.perf_counter()
        for step, batch in enumerate(loader, start=1):
            data_elapsed += time.perf_counter() - data_start
            started = time.perf_counter()
            pixels, labels = batch["pixel_values"].to(device), batch["labels"].to(device)
            optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast("cuda", enabled=amp):
                loss = functional.cross_entropy(resize_logits(model(pixel_values=pixels).logits, labels), labels, ignore_index=int(loss_config["ignore_index"]))
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            torch.cuda.synchronize()
            elapsed += time.perf_counter() - started
            loss_sum += float(loss.item())
            if step >= args.benchmark_steps:
                break
            data_start = time.perf_counter()
        payload = checkpoint_payload(0, step, model, optimizer, scheduler, scaler, config)
        payload["benchmark_only"] = True
        atomic_torch_save(payload, checkpoints / "benchmark_last.pt")
        result = {
            "benchmark_only": True,
            "steps": step,
            "mean_step_seconds": elapsed / step,
            "mean_data_seconds": data_elapsed / step,
            "mean_loss": loss_sum / step,
            "peak_memory_mib": round(torch.cuda.max_memory_allocated(device) / 1024**2, 2),
            "estimated_epoch_seconds": (len(train_dataset) / int(data["batch_size"])) * elapsed / step,
            "official_suim_test_evaluated": False,
            "confirmation_evaluated": False,
        }
        (output_dir / "benchmark.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
        logger.info("benchmark=%s", result)
        return

    history_path = logs / "train_history.csv"
    write_header = not history_path.exists() or start_epoch == 1
    with history_path.open("a", newline="", encoding="utf-8") as history:
        writer = csv.DictWriter(history, fieldnames=["epoch", "train_loss", "learning_rate", "global_step"])
        if write_header:
            writer.writeheader()
        for epoch in range(start_epoch, int(training["epochs"]) + 1):
            loader = build_epoch_loader(train_dataset, int(data["batch_size"]), int(data["num_workers"]), int(config["experiment"]["seed"]) + epoch)
            model.train()
            loss_sum, valid_count = 0.0, 0
            for batch in loader:
                pixels, labels = batch["pixel_values"].to(device), batch["labels"].to(device)
                optimizer.zero_grad(set_to_none=True)
                with torch.amp.autocast("cuda", enabled=amp):
                    logits = resize_logits(model(pixel_values=pixels).logits, labels)
                    loss = functional.cross_entropy(logits, labels, ignore_index=int(loss_config["ignore_index"]))
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
                count = int(labels.ne(int(loss_config["ignore_index"])).sum().item())
                loss_sum += float(loss.item()) * count
                valid_count += count
                global_step += 1
            scheduler.step()
            row = {
                "epoch": epoch,
                "train_loss": loss_sum / valid_count,
                "learning_rate": optimizer.param_groups[0]["lr"],
                "global_step": global_step,
            }
            writer.writerow(row)
            history.flush()
            atomic_torch_save(checkpoint_payload(epoch, global_step, model, optimizer, scheduler, scaler, config), checkpoints / "last.pt")
            logger.info("epoch=%s train_loss=%.4f", epoch, row["train_loss"])


if __name__ == "__main__":
    main()
