"""Train SADR on UIIS clean/degraded pairs with a frozen segmentation expert."""
from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

import numpy as np
import torch
import yaml
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from reliability.sadr_seg import SADRFrontEnd, confidence_distillation, segmentation_loss  # noqa: E402
from scripts.train_dts_seg_gate0 import atomic_torch_save, sha256  # noqa: E402
from scripts.train_uiis_scdi_replication import UIISTrajectoryDataset, build_models as build_uiis_f4  # noqa: E402

FORMAT = "sadr_v1"


def build_model(config: dict, device: torch.device) -> tuple[torch.nn.Module, SADRFrontEnd]:
    uiis_config = yaml.safe_load((ROOT / str(config["data"]["uiis_config"])).read_text(encoding="utf-8"))
    base = build_uiis_f4(uiis_config, "F4", device)
    front = SADRFrontEnd(gated=bool(config["training"].get("gated", False))).to(device)
    for parameter in base.parameters():
        parameter.requires_grad_(False)
    return base.eval(), front


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__); parser.add_argument("--config", type=Path, default=ROOT / "configs/sadr_fast.yaml"); parser.add_argument("--smoke-batches", type=int, default=0); args = parser.parse_args()
    config_path = args.config.resolve(); config = yaml.safe_load(config_path.read_text(encoding="utf-8")); seed = int(config["experiment"]["seed"]); random.seed(seed); np.random.seed(seed); torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)
    if not torch.cuda.is_available(): raise RuntimeError("SADR requires CUDA.")
    device = torch.device("cuda"); data = config["data"]; training = config["training"]
    dataset = UIISTrajectoryDataset(ROOT / str(data["train_csv"]), ROOT / str(data["degradation_registry"]), int(data["image_size"]))
    loader = DataLoader(dataset, batch_size=int(training["batch_size"]), shuffle=True, num_workers=0, pin_memory=True, generator=torch.Generator().manual_seed(seed))
    base, front = build_model(config, device); optimizer = torch.optim.AdamW(front.parameters(), lr=float(training["learning_rate"]), weight_decay=float(training["weight_decay"])); scaler = torch.amp.GradScaler("cuda", enabled=bool(training["amp"])); epochs = 1 if args.smoke_batches else int(training["epochs"]); output = ROOT / str(config["experiment"]["output_dir"]) / ("smoke" if args.smoke_batches else "formal"); history = []
    for epoch in range(1, epochs + 1):
        dataset.set_epoch(epoch); front.train(); totals = {key: 0.0 for key in ("segmentation", "reconstruction", "identity", "distillation", "residual")}; batches = 0
        for batch_index, batch in enumerate(loader):
            if args.smoke_batches and batch_index >= args.smoke_batches: break
            clean = batch["clean"].to(device, non_blocking=True); labels = batch["labels"].to(device, non_blocking=True); degraded = torch.cat([batch["s1"], batch["s2"], batch["s3"]], dim=0).to(device, non_blocking=True); repeated_labels = labels.repeat(3, 1, 1); clean_repeated = clean.repeat(3, 1, 1, 1); optimizer.zero_grad(set_to_none=True)
            with torch.no_grad(): teacher = base(pixel_values=clean).logits.detach()
            with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=bool(training["amp"])):
                restored, residual = front(degraded); restored_clean, clean_residual = front(clean); student = base(pixel_values=restored).logits; teacher_rep = teacher.repeat(3, 1, 1, 1); segmentation = segmentation_loss(student, repeated_labels); reconstruction = (restored - clean_repeated).abs().mean(); identity = clean_residual.abs().mean(); distillation = confidence_distillation(student, teacher_rep); residual_value = residual.float().pow(2).mean() + clean_residual.float().pow(2).mean(); loss = float(training["segmentation_weight"]) * segmentation + float(training["reconstruction_weight"]) * reconstruction + float(training["identity_weight"]) * identity + float(training["distillation_weight"]) * distillation + float(training["residual_weight"]) * residual_value
            scaler.scale(loss).backward(); scaler.step(optimizer); scaler.update();
            for key, value in (("segmentation", segmentation), ("reconstruction", reconstruction), ("identity", identity), ("distillation", distillation), ("residual", residual_value)): totals[key] += float(value.detach())
            batches += 1
            if batch_index % 50 == 0: print(f"SADR epoch={epoch} batch={batch_index}/{len(loader)} seg={totals['segmentation']/batches:.4f} recon={totals['reconstruction']/batches:.4f} distill={totals['distillation']/batches:.4f}", flush=True)
            del clean, labels, degraded, repeated_labels, clean_repeated, teacher, restored, residual, restored_clean, clean_residual, student, teacher_rep, loss
        if batches == 0: raise RuntimeError("No batch completed.")
        row = {"epoch": float(epoch), "batches": float(batches), **{key: value / batches for key, value in totals.items()}}; history.append(row); print(json.dumps(row, sort_keys=True), flush=True); payload = {"checkpoint_format": FORMAT, "variant": "SADR", "epoch": epoch, "model_state_dict": front.state_dict(), "optimizer_state_dict": optimizer.state_dict(), "scaler_state_dict": scaler.state_dict(), "history": history, "config_sha256": sha256(config_path), "smoke": bool(args.smoke_batches), "calibration_evaluated": False, "confirmation_evaluated": False, "official_suim_test_evaluated": False}; atomic_torch_save(payload, output / "checkpoints" / "last.pt")
    if not args.smoke_batches: atomic_torch_save(payload, output / "checkpoints" / "final.pt")


if __name__ == "__main__": main()
