"""Train SGRE-Seg with online class-by-severity group robustness."""
from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as functional
import yaml
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from reliability.scdi_seg import rectifier_parameter_count, semantic_router_loss  # noqa: E402
from reliability.sgre_seg import SemanticGroupState, classwise_cross_entropy  # noqa: E402
from scripts.train_dts_seg_gate0 import FourViewTrajectoryDataset, atomic_torch_save, sha256  # noqa: E402
from scripts.train_scdi_seg_gate2 import build_model  # noqa: E402


FORMAT = "sgre_seg_gate3_v1"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=ROOT / "configs/sgre_seg_gate3.yaml")
    parser.add_argument("--smoke-batches", type=int, default=0)
    args = parser.parse_args()
    if args.smoke_batches < 0:
        raise ValueError("smoke-batches must be non-negative.")
    config_path = args.config.resolve()
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    seed = int(config["experiment"]["seed"])
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)
    if not torch.cuda.is_available():
        raise RuntimeError("SGRE Gate-3 requires CUDA.")
    device = torch.device("cuda")
    training = config["training"]
    dataset = FourViewTrajectoryDataset(
        ROOT / str(config["data"]["train_csv"]), ROOT / str(config["data"]["degradation_registry"]), int(config["data"]["image_size"])
    )
    loader = DataLoader(
        dataset, batch_size=int(training["batch_size"]), shuffle=True, num_workers=0, pin_memory=True,
        generator=torch.Generator().manual_seed(seed),
    )
    model = build_model(config, device)
    expert_ids = {id(parameter) for parameter in model.rectifiers.parameters()}
    optimizer = torch.optim.AdamW(
        [
            {"params": [parameter for parameter in model.parameters() if id(parameter) not in expert_ids], "lr": float(training["backbone_learning_rate"])},
            {"params": list(model.rectifiers.parameters()), "lr": float(training["expert_learning_rate"])},
        ], weight_decay=float(training["weight_decay"]),
    )
    scaler = torch.amp.GradScaler("cuda", enabled=bool(training["amp"]))
    group_state = SemanticGroupState.create(
        views=4, classes=int(config["model"]["num_classes"]),
        momentum=float(training["group_ema_momentum"]), temperature=float(training["group_temperature"]),
    )
    epochs = 1 if args.smoke_batches else int(training["epochs"])
    output = ROOT / str(config["experiment"]["output_dir"]) / ("smoke" if args.smoke_batches else "formal") / "SGRE"
    history: list[dict[str, float]] = []
    source_path = ROOT / str(config["model"]["source_checkpoint"])
    model.train()
    view_keys = ("clean", "s1", "s2", "s3")
    for epoch in range(1, epochs + 1):
        dataset.set_epoch(epoch)
        totals = {"pixel_ce": 0.0, "group_loss": 0.0, "router": 0.0, "present_groups": 0.0}
        batches = 0
        for batch_index, batch in enumerate(loader):
            if args.smoke_batches and batch_index >= args.smoke_batches:
                break
            labels = batch["labels"].to(device, non_blocking=True)
            weights = group_state.weights(device)
            updates = []
            optimizer.zero_grad(set_to_none=True)
            for view_index, key in enumerate(view_keys):
                pixels = batch[key].to(device, non_blocking=True)
                with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=bool(training["amp"])):
                    result = model(pixels)
                    logits = functional.interpolate(result.logits.float(), size=labels.shape[-2:], mode="bilinear", align_corners=False)
                    pixel_ce = functional.cross_entropy(logits, labels, ignore_index=255) / 4.0
                    class_losses, present = classwise_cross_entropy(
                        logits, labels, num_classes=int(config["model"]["num_classes"]),
                        minimum_pixels=int(training["group_minimum_pixels"]),
                    )
                    group_loss = (weights[view_index, present] * class_losses[present]).sum()
                    router = semantic_router_loss(result.router_logits, labels)
                    loss = (
                        float(training["normal_ce_mix"]) * pixel_ce
                        + float(training["group_robust_mix"]) * group_loss
                        + float(training["router_weight"]) * router
                    )
                scaler.scale(loss).backward()
                updates.append((view_index, class_losses.detach(), present.detach()))
                totals["pixel_ce"] += float(pixel_ce.detach())
                totals["group_loss"] += float(group_loss.detach())
                totals["router"] += float(router.detach())
                totals["present_groups"] += int(present.sum())
                del pixels, result, logits, loss
            scaler.step(optimizer); scaler.update()
            for view_index, losses, present in updates:
                group_state.update(view_index, losses, present)
            batches += 1
            if batch_index % 25 == 0:
                flat_index = int(group_state.weights(torch.device("cpu")).argmax())
                print(
                    f"SGRE epoch={epoch} batch={batch_index}/{len(loader)} ce={totals['pixel_ce']/batches:.4f} "
                    f"group={totals['group_loss']/batches:.4f} router={totals['router']/batches:.4f} "
                    f"worst=v{flat_index // 8}c{flat_index % 8}", flush=True,
                )
        if batches == 0:
            raise RuntimeError("No training batch completed.")
        row = {"epoch": float(epoch), "batches": float(batches), **{key: value / batches for key, value in totals.items()}}
        history.append(row)
        payload = {
            "checkpoint_format": FORMAT, "variant": "SGRE", "epoch": epoch,
            "model_state_dict": model.state_dict(), "optimizer_state_dict": optimizer.state_dict(),
            "scaler_state_dict": scaler.state_dict(), "group_state": group_state.state_dict(), "history": history,
            "config_sha256": sha256(config_path), "source_checkpoint_sha256": sha256(source_path),
            "rectifier_parameters": rectifier_parameter_count(model), "smoke": bool(args.smoke_batches),
            "method_development_evaluated": False, "validation_evaluated": False,
            "calibration_evaluated": False, "official_suim_test_evaluated": False,
        }
        atomic_torch_save(payload, output / "checkpoints/last.pt")
        print(json.dumps(row, sort_keys=True), flush=True)
    if not args.smoke_batches:
        atomic_torch_save(payload, output / "checkpoints/final.pt")


if __name__ == "__main__":
    main()
