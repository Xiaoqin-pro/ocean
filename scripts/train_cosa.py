"""Train COSA-Seg: cross-ontology semantic anchoring with UVMulti pairs."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
import sys
import tempfile
from pathlib import Path

import albumentations as A
import numpy as np
import pandas as pd
import torch
import yaml
from albumentations.pytorch import ToTensorV2
from PIL import Image
from torch.utils.data import DataLoader, Dataset

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from datasets.suim_dataset import IMAGENET_MEAN, IMAGENET_STD  # noqa: E402
from reliability.cosa_seg import class_anchor_loss, mapped_cross_entropy, residual_penalty, symmetric_kl  # noqa: E402
from reliability.spt_seg import SPTSegformer  # noqa: E402
from scripts.train_dts_seg_gate0 import atomic_torch_save, sha256  # noqa: E402
from scripts.train_uiis_scdi_replication import UIISTrajectoryDataset, build_models as build_uiis_f4  # noqa: E402

FORMAT = "cosa_v1"


def build_model(config: dict, device: torch.device) -> SPTSegformer:
    uiis_config = yaml.safe_load((ROOT / str(config["data"]["uiis_config"])).read_text(encoding="utf-8"))
    base = build_uiis_f4(uiis_config, "F4", device)
    model = SPTSegformer(base, tuple(config["model"]["hidden_sizes"]), int(config["model"]["num_classes"]), int(config["model"]["rectifier_reduction"])).to(device)
    for parameter in model.base_model.parameters():
        parameter.requires_grad_(False)
    return model


UVMULTI_TO_SUIM = {0: 0, 1: 3, 2: 5, 3: 5, 4: 5, 5: 6, 6: 5}


class UVMultiLabeledPairs(Dataset[dict[str, torch.Tensor]]):
    def __init__(self, root: Path, image_size: int) -> None:
        self.items: list[tuple[Path, Path, Path]] = []
        for video in sorted(root.iterdir()):
            if not video.is_dir():
                continue
            images = video / "images"; enhanced = video / "sequence_enh"; masks = video / "mask"
            for mask_path in sorted(masks.glob("*.png")):
                image_path = images / f"{mask_path.stem}.jpg"
                enhanced_path = enhanced / f"{mask_path.stem}.jpg"
                if image_path.is_file() and enhanced_path.is_file():
                    self.items.append((image_path, enhanced_path, mask_path))
        if len(self.items) < 100:
            raise ValueError(f"Expected at least 100 UVMulti labeled pairs, found {len(self.items)}")
        self.transform = A.ReplayCompose([A.Resize(image_size, image_size), A.HorizontalFlip(p=0.5), A.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD, max_pixel_value=255.0), ToTensorV2()])
        self.epoch = 1

    def set_epoch(self, epoch: int) -> None:
        self.epoch = epoch

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        image_path, enhanced_path, mask_path = self.items[index]
        with Image.open(image_path) as image, Image.open(enhanced_path) as enhanced, Image.open(mask_path) as mask:
            raw = np.asarray(image.convert("RGB"), dtype=np.uint8)
            enh = np.asarray(enhanced.convert("RGB"), dtype=np.uint8)
            labels = np.asarray(mask, dtype=np.uint8)
        mapped = np.full_like(labels, 255, dtype=np.uint8)
        for source, target in UVMULTI_TO_SUIM.items():
            mapped[labels == source] = target
        seed = (int(hashlib.sha256(str(image_path).encode()).hexdigest()[:16], 16) + 100003 * self.epoch) % (2**32)
        self.transform.set_random_seed(seed)
        base = self.transform(image=raw, mask=mapped)
        paired = A.ReplayCompose.replay(base["replay"], image=enh, mask=mapped)
        return {"raw": base["image"].float(), "enhanced": paired["image"].float(), "labels": base["mask"].long(), "sample_id": image_path.stem}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__); parser.add_argument("--config", type=Path, default=ROOT / "configs/cosa_fast.yaml"); parser.add_argument("--smoke-batches", type=int, default=0); args = parser.parse_args()
    config_path = args.config.resolve(); config = yaml.safe_load(config_path.read_text(encoding="utf-8")); seed = int(config["experiment"]["seed"]); random.seed(seed); np.random.seed(seed); torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)
    if not torch.cuda.is_available(): raise RuntimeError("COSA requires CUDA.")
    device = torch.device("cuda"); data = config["data"]; training = config["training"]
    uiis = UIISTrajectoryDataset(ROOT / str(data["uiis_train_csv"]), ROOT / str(data["degradation_registry"]), int(data["image_size"]))
    uvmulti = UVMultiLabeledPairs(ROOT / str(data["uvmulti_root"]), int(data["image_size"]))
    uiis_loader = DataLoader(uiis, batch_size=int(training["batch_size"]), shuffle=True, num_workers=0, pin_memory=True, generator=torch.Generator().manual_seed(seed))
    uv_loader = DataLoader(uvmulti, batch_size=int(training["uv_batch_size"]), shuffle=True, num_workers=0, pin_memory=True, generator=torch.Generator().manual_seed(seed + 17))
    model = build_model(config, device); model.train(); model.base_model.eval(); optimizer = torch.optim.AdamW(list(model.rectifiers.parameters()), lr=float(training["learning_rate"]), weight_decay=float(training["weight_decay"])); scaler = torch.amp.GradScaler("cuda", enabled=bool(training["amp"])); epochs = 1 if args.smoke_batches else int(training["epochs"]); output = ROOT / str(config["experiment"]["output_dir"]) / ("smoke" if args.smoke_batches else "formal"); history = []
    for epoch in range(1, epochs + 1):
        uiis.set_epoch(epoch); uvmulti.set_epoch(epoch); uv_iter = iter(uv_loader); totals = {key: 0.0 for key in ("uiis", "uiis_s3", "uvmulti", "consistency", "anchor", "residual")}; batches = 0
        for batch_index, uiis_batch in enumerate(uiis_loader):
            if args.smoke_batches and batch_index >= args.smoke_batches: break
            try: uv_batch = next(uv_iter)
            except StopIteration: uv_iter = iter(uv_loader); uv_batch = next(uv_iter)
            optimizer.zero_grad(set_to_none=True); clean = uiis_batch["clean"].to(device, non_blocking=True); s3 = uiis_batch["s3"].to(device, non_blocking=True); uiis_labels = uiis_batch["labels"].to(device, non_blocking=True); raw = uv_batch["raw"].to(device, non_blocking=True); enhanced = uv_batch["enhanced"].to(device, non_blocking=True); uv_labels = uv_batch["labels"].to(device, non_blocking=True)
            with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=bool(training["amp"])):
                uiis_out = model(clean); uiis_s3_out = model(s3); raw_out = model(raw); enh_out = model(enhanced)
                loss_uiis = mapped_cross_entropy(uiis_out.logits, uiis_labels); loss_uiis_s3 = mapped_cross_entropy(uiis_s3_out.logits, uiis_labels); loss_uv = 0.5 * (mapped_cross_entropy(raw_out.logits, uv_labels) + mapped_cross_entropy(enh_out.logits, uv_labels)); consistency = symmetric_kl(raw_out.logits, enh_out.logits); anchor = class_anchor_loss(uiis_out.corrected_features[-1], uiis_labels, raw_out.corrected_features[-1], uv_labels); residual = residual_penalty(raw_out.residuals) + residual_penalty(enh_out.residuals)
                loss = float(training["uiis_weight"]) * loss_uiis + float(training["uiis_s3_weight"]) * loss_uiis_s3 + float(training["uvmulti_weight"]) * loss_uv + float(training["view_consistency_weight"]) * consistency + float(training["cross_ontology_weight"]) * anchor + float(training["residual_weight"]) * residual
            scaler.scale(loss).backward(); scaler.step(optimizer); scaler.update();
            for key, value in (("uiis", loss_uiis), ("uiis_s3", loss_uiis_s3), ("uvmulti", loss_uv), ("consistency", consistency), ("anchor", anchor), ("residual", residual)): totals[key] += float(value.detach())
            batches += 1
            if batch_index % 50 == 0: print(f"COSA epoch={epoch} batch={batch_index}/{len(uiis_loader)} uiis={totals['uiis']/batches:.4f} uvmulti={totals['uvmulti']/batches:.4f} anchor={totals['anchor']/batches:.4f}", flush=True)
            del clean, s3, uiis_labels, raw, enhanced, uv_labels, uiis_out, uiis_s3_out, raw_out, enh_out, loss
        if batches == 0: raise RuntimeError("No batch completed.")
        row = {"epoch": float(epoch), "batches": float(batches), **{key: value / batches for key, value in totals.items()}}; history.append(row); print(json.dumps(row, sort_keys=True), flush=True); payload = {"checkpoint_format": FORMAT, "variant": "COSA-Seg", "epoch": epoch, "model_state_dict": model.state_dict(), "optimizer_state_dict": optimizer.state_dict(), "scaler_state_dict": scaler.state_dict(), "history": history, "config_sha256": sha256(config_path), "smoke": bool(args.smoke_batches), "calibration_evaluated": False, "confirmation_evaluated": False, "official_suim_test_evaluated": False, "uvmulti_train_pairs": len(uvmulti)}; atomic_torch_save(payload, output / "checkpoints" / "last.pt")
    if not args.smoke_batches: atomic_torch_save(payload, output / "checkpoints" / "final.pt")


if __name__ == "__main__": main()
