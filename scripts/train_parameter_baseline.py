"""Train a frozen-protocol partial/full fine-tuning baseline on the UIIS train split."""
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

import numpy as np
import torch
import torch.nn.functional as F
import yaml
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.train_uiis_scdi_replication import UIISTrajectoryDataset, build_models  # noqa: E402
from scripts.lora_utils import lora_trainable_parameters, replace_lora_modules  # noqa: E402

FORMAT = "parameter_efficiency_v1"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest().upper()


def atomic_save(value: dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, raw = tempfile.mkstemp(suffix=".pt", dir=path.parent)
    os.close(descriptor)
    temporary = Path(raw)
    try:
        torch.save(value, temporary)
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def configure_trainable(model: torch.nn.Module, scope: str) -> tuple[list[torch.nn.Parameter], list[str]]:
    if scope not in {"head", "last_block", "full"}:
        raise ValueError("scope must be head, last_block, or full")
    for parameter in model.parameters():
        parameter.requires_grad_(scope == "full")
    if scope == "full":
        names = [name for name, parameter in model.named_parameters() if parameter.requires_grad]
    else:
        prefix = "decode_head." if scope == "head" else "segformer.stages.3.blocks.1."
        names = []
        for name, parameter in model.named_parameters():
            parameter.requires_grad_(name.startswith(prefix))
            if parameter.requires_grad:
                names.append(name)
    parameters = [parameter for parameter in model.parameters() if parameter.requires_grad]
    if not parameters:
        raise RuntimeError(f"No trainable parameters selected for scope={scope}")
    return parameters, names


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scope", choices=("head", "last_block", "full", "lora"), required=True)
    parser.add_argument("--config", type=Path, default=ROOT / "configs/parameter_efficiency.yaml")
    parser.add_argument("--smoke-batches", type=int, default=0)
    parser.add_argument("--lora-rank", type=int, default=2)
    args = parser.parse_args()
    if args.smoke_batches < 0:
        raise ValueError("smoke-batches must be non-negative")
    config_path = args.config.resolve()
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    seed = int(config["experiment"]["seed"])
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)
    if not torch.cuda.is_available():
        raise RuntimeError("Parameter-efficiency training requires CUDA.")
    device = torch.device("cuda")
    data = config["data"]; training = config["training"]
    uiis_config = yaml.safe_load((ROOT / str(data["uiis_config"])).read_text(encoding="utf-8"))
    dataset = UIISTrajectoryDataset(ROOT / str(data["train_csv"]), ROOT / str(data["degradation_registry"]), int(data["image_size"]))
    loader = DataLoader(dataset, batch_size=int(training["batch_size"]), shuffle=True, num_workers=0, pin_memory=True, generator=torch.Generator().manual_seed(seed))
    model = build_models(uiis_config, "F4", device)
    if args.scope == "lora":
        for parameter in model.parameters():
            parameter.requires_grad_(False)
        replaced = replace_lora_modules(model, rank=args.lora_rank, alpha=float(args.lora_rank))
        parameters = lora_trainable_parameters(model)
        trainable_names = [name for name, parameter in model.named_parameters() if parameter.requires_grad]
        if len(parameters) == 0:
            raise RuntimeError("No LoRA parameters selected")
        print(json.dumps({"lora_modules": replaced, "trainable_parameters": sum(item.numel() for item in parameters)}), flush=True)
    else:
        parameters, trainable_names = configure_trainable(model, args.scope)
    total_parameters = sum(parameter.numel() for parameter in model.parameters())
    trainable_parameters = sum(parameter.numel() for parameter in parameters)
    optimizer = torch.optim.AdamW(parameters, lr=float(training["learning_rate"]), weight_decay=float(training["weight_decay"]))
    scaler = torch.amp.GradScaler("cuda", enabled=bool(training["amp"]))
    epochs = 1 if args.smoke_batches else int(training["epochs"])
    output = ROOT / str(config["experiment"]["output_dir"]) / ("smoke" if args.smoke_batches else "formal") / args.scope
    history: list[dict[str, float]] = []
    source_path = ROOT / str(uiis_config["model"]["source_checkpoint"])
    model.train()
    for epoch in range(1, epochs + 1):
        dataset.set_epoch(epoch)
        total_loss = 0.0; batches = 0
        for batch_index, batch in enumerate(loader):
            if args.smoke_batches and batch_index >= args.smoke_batches:
                break
            labels = batch["labels"].to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            loss = labels.float().sum() * 0.0
            for key in ("clean", "s1", "s2", "s3"):
                pixels = batch[key].to(device, non_blocking=True)
                with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=bool(training["amp"])):
                    logits = model(pixel_values=pixels).logits
                    full_logits = F.interpolate(logits.float(), size=labels.shape[-2:], mode="bilinear", align_corners=False)
                    loss = loss + F.cross_entropy(full_logits, labels, ignore_index=255) / 4.0
                del pixels, logits, full_logits
            scaler.scale(loss).backward(); scaler.step(optimizer); scaler.update()
            total_loss += float(loss.detach()); batches += 1
            if batch_index % 50 == 0:
                print(f"{args.scope} epoch={epoch} batch={batch_index}/{len(loader)} loss={total_loss / batches:.4f}", flush=True)
        row = {"epoch": float(epoch), "batches": float(batches), "loss": total_loss / batches}
        history.append(row); print(json.dumps(row, sort_keys=True), flush=True)
        payload = {
            "checkpoint_format": FORMAT, "variant": args.scope, "epoch": epoch,
            "model_state_dict": model.state_dict(), "optimizer_state_dict": optimizer.state_dict(),
            "scaler_state_dict": scaler.state_dict(), "history": history,
            "config_sha256": sha256(config_path), "source_checkpoint_sha256": sha256(source_path),
            "smoke": bool(args.smoke_batches), "confirmation_evaluated": False,
            "official_suim_test_evaluated": False, "total_parameters": total_parameters,
            "trainable_parameters": trainable_parameters, "trainable_parameter_names": trainable_names,
            "lora_rank": int(args.lora_rank) if args.scope == "lora" else None,
        }
        atomic_save(payload, output / "checkpoints/last.pt")
    if not args.smoke_batches:
        atomic_save(payload, output / "checkpoints/final.pt")


if __name__ == "__main__":
    main()
