"""Train Paired Enhancement Semantic Consistency (PESC) on public UIIS/UVMulti."""
from __future__ import annotations

import argparse
import itertools
import json
import random
import sys
from pathlib import Path

import albumentations as A
import numpy as np
import torch
import torch.nn.functional as functional
import yaml
from albumentations.pytorch import ToTensorV2
from PIL import Image
from torch.utils.data import DataLoader, Dataset

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from datasets.suim_dataset import IMAGENET_MEAN, IMAGENET_STD  # noqa: E402
from reliability.anchor_replay import source_logit_anchor_loss  # noqa: E402
from reliability.pesc_seg import paired_enhancement_consistency_loss  # noqa: E402
from reliability.sdtc_seg import clean_identity_loss  # noqa: E402
from reliability.spt_seg import SPTSegformer, semantic_prototype_transport_loss  # noqa: E402
from reliability.utd_seg import semantic_trajectory_distillation_loss  # noqa: E402
from scripts.train_dts_seg_gate0 import atomic_torch_save, sha256  # noqa: E402
from scripts.train_uiis_scdi_replication import UIISTrajectoryDataset, build_models  # noqa: E402

FORMAT = "pesc_seg_v1"


class UVMultiPairDataset(Dataset[dict[str, torch.Tensor]]):
    def __init__(self, root: Path, image_size: int) -> None:
        pairs: list[tuple[Path, Path]] = []
        for video in sorted((root / "train").glob("video*")):
            raw_dir = video / "sequence"; enhanced_dir = video / "sequence_enh"
            if not raw_dir.is_dir() or not enhanced_dir.is_dir(): continue
            for raw in sorted(raw_dir.glob("*.jpg")):
                enhanced = enhanced_dir / raw.name
                if enhanced.exists(): pairs.append((raw, enhanced))
        if len(pairs) < 1000: raise PermissionError(f"Expected public UVMulti paired frames, found only {len(pairs)}.")
        self.pairs = pairs; self.epoch = 1
        self.transform = A.ReplayCompose([A.Resize(image_size, image_size), A.HorizontalFlip(p=0.5), A.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD, max_pixel_value=255.0), ToTensorV2()])

    def set_epoch(self, epoch: int) -> None: self.epoch = epoch
    def __len__(self) -> int: return len(self.pairs)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        raw_path, enhanced_path = self.pairs[index]
        with Image.open(raw_path) as raw_image, Image.open(enhanced_path) as enhanced_image:
            raw = np.asarray(raw_image.convert("RGB"), dtype=np.uint8); enhanced = np.asarray(enhanced_image.convert("RGB"), dtype=np.uint8)
        self.transform.set_random_seed((index * 100003 + self.epoch * 9176) % (2**32)); first = self.transform(image=raw); second = A.ReplayCompose.replay(first["replay"], image=enhanced)
        return {"raw": first["image"].float(), "enhanced": second["image"].float()}


def build_pesc_model(config: dict, device: torch.device) -> SPTSegformer:
    uiis_config = yaml.safe_load((ROOT / "configs/uiis_scdi_replication.yaml").read_text(encoding="utf-8")); base = build_models(uiis_config, "F4", device)
    source = torch.load(ROOT / "outputs/uiis_scdi_replication/formal/F4/checkpoints/final.pt", map_location=device, weights_only=False); base.load_state_dict(source["model_state_dict"])
    model = SPTSegformer(base, tuple(config["model"]["hidden_sizes"]), int(config["model"]["num_classes"]), int(config["model"]["rectifier_reduction"])).to(device)
    for parameter in model.base_model.parameters(): parameter.requires_grad_(False)
    return model


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__); parser.add_argument("--config", type=Path, default=ROOT / "configs/pesc_fast.yaml"); parser.add_argument("--smoke-batches", type=int, default=0); args = parser.parse_args(); config_path = args.config.resolve(); config = yaml.safe_load(config_path.read_text(encoding="utf-8")); seed = int(config["experiment"]["seed"]); random.seed(seed); torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)
    if not torch.cuda.is_available(): raise RuntimeError("PESC requires CUDA.")
    device = torch.device("cuda"); data = config["data"]; training = config["training"]; labeled = UIISTrajectoryDataset(ROOT / str(data["uiis_train_csv"]), ROOT / str(data["degradation_registry"]), int(data["image_size"])); pairs = UVMultiPairDataset(ROOT / str(data["uvmulti_root"]), int(data["image_size"])); labeled_loader = DataLoader(labeled, batch_size=int(training["batch_size"]), shuffle=True, num_workers=0, pin_memory=True, generator=torch.Generator().manual_seed(seed)); pair_loader = DataLoader(pairs, batch_size=int(training["pair_batch_size"]), shuffle=True, num_workers=0, pin_memory=True, generator=torch.Generator().manual_seed(seed + 1)); model = build_pesc_model(config, device); model.train(); model.base_model.eval(); optimizer = torch.optim.AdamW(list(model.rectifiers.parameters()), lr=float(training["learning_rate"]), weight_decay=float(training["weight_decay"])); scaler = torch.amp.GradScaler("cuda", enabled=bool(training["amp"])); epochs = 1 if args.smoke_batches else int(training["epochs"]); output = ROOT / str(config["experiment"]["output_dir"]) / ("smoke" if args.smoke_batches else "formal"); history = []
    for epoch in range(1, epochs + 1):
        labeled.set_epoch(epoch); pairs.set_epoch(epoch); label_iterator = iter(labeled_loader); pair_iterator = iter(pair_loader); steps = min(len(labeled_loader), len(pair_loader)) if args.smoke_batches == 0 else args.smoke_batches; totals = {key: 0.0 for key in ("supervised", "prototype", "distillation", "source_anchor", "pair_consistency", "pair_coverage", "identity")}
        for step in range(steps):
            try: batch = next(label_iterator)
            except StopIteration: label_iterator = iter(labeled_loader); batch = next(label_iterator)
            try: pair_batch = next(pair_iterator)
            except StopIteration: pair_iterator = iter(pair_loader); pair_batch = next(pair_iterator)
            labels = batch["labels"].to(device, non_blocking=True); optimizer.zero_grad(set_to_none=True); clean = batch["clean"].to(device, non_blocking=True); clean_result = model(clean); clean_reference = tuple(item.detach() for item in clean_result.raw_features); clean_teacher = model.base_model(pixel_values=clean).logits.detach(); clean_identity = clean_identity_loss(clean_result.raw_features, clean_result.residuals); total_loss = functional.cross_entropy(functional.interpolate(clean_result.logits.float(), size=labels.shape[-2:], mode="bilinear", align_corners=False), labels, ignore_index=255) / 4.0; supervised_total = float(total_loss.detach()); prototype_total = 0.0; distill_total = 0.0; anchor_total = 0.0
            for key in ("s1", "s2", "s3"):
                pixels = batch[key].to(device, non_blocking=True)
                with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=bool(training["amp"])):
                    result = model(pixels); logits = result.logits; full = functional.interpolate(logits.float(), size=labels.shape[-2:], mode="bilinear", align_corners=False); supervised = functional.cross_entropy(full, labels, ignore_index=255) / 4.0; prototype, _ = semantic_prototype_transport_loss(clean_reference, result.corrected_features, labels, num_classes=8, minimum_pixels=int(training["prototype_minimum_pixels"]), margin=float(training["prototype_margin"])); distillation = semantic_trajectory_distillation_loss(clean_result.logits.detach(), logits, labels, temperature=float(training["distillation_temperature"]), confidence_power=float(training["confidence_power"]), class_balance=True); source_anchor = source_logit_anchor_loss(logits, model.base_model(pixel_values=pixels).logits.detach(), temperature=float(training["source_temperature"])); residual_anchor = sum(item.float().pow(2).mean() for item in result.residuals); loss = supervised + float(training["prototype_weight"]) * prototype + float(training["distillation_weight"]) * (int(key[-1]) / 3.0) * distillation + float(training["source_anchor_weight"]) * source_anchor + float(training["anchor_weight"]) * residual_anchor
                total_loss = total_loss + loss; supervised_total += float(supervised.detach()); prototype_total += float(prototype.detach()); distill_total += float(distillation.detach()); anchor_total += float(source_anchor.detach()); del pixels, result, logits, full, loss
            with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=bool(training["amp"])):
                raw_result = model(pair_batch["raw"].to(device, non_blocking=True)); enhanced_result = model(pair_batch["enhanced"].to(device, non_blocking=True)); pair_loss, coverage = paired_enhancement_consistency_loss(raw_result.logits, enhanced_result.logits, temperature=float(training["pair_temperature"]), confidence_threshold=float(training["pair_confidence_threshold"])); pair_residual = sum(item.float().pow(2).mean() for item in raw_result.residuals) + sum(item.float().pow(2).mean() for item in enhanced_result.residuals); total_loss = total_loss + float(training["pair_weight"]) * pair_loss + float(training["anchor_weight"]) * pair_residual
            total_loss = total_loss + float(training["clean_identity_weight"]) * clean_identity + float(training["source_anchor_weight"]) * source_logit_anchor_loss(clean_result.logits, clean_teacher, temperature=float(training["source_temperature"])); scaler.scale(total_loss).backward(); scaler.step(optimizer); scaler.update(); totals["supervised"] += supervised_total; totals["prototype"] += prototype_total; totals["distillation"] += distill_total; totals["source_anchor"] += anchor_total; totals["pair_consistency"] += float(pair_loss.detach()); totals["pair_coverage"] += coverage; totals["identity"] += float(clean_identity.detach());
            if step % 50 == 0: print(f"PESC epoch={epoch} step={step}/{steps} sup={totals['supervised']/(step+1):.4f} pair={totals['pair_consistency']/(step+1):.4f} coverage={totals['pair_coverage']/(step+1):.3f}", flush=True)
            del clean, clean_result, clean_teacher, total_loss, clean_identity, raw_result, enhanced_result
        row = {"epoch": float(epoch), "steps": float(steps), **{key: value / steps for key, value in totals.items()}}; history.append(row); print(json.dumps(row, sort_keys=True), flush=True); payload = {"checkpoint_format": FORMAT, "variant": "PESC", "epoch": epoch, "model_state_dict": model.state_dict(), "optimizer_state_dict": optimizer.state_dict(), "scaler_state_dict": scaler.state_dict(), "history": history, "config_sha256": sha256(config_path), "smoke": bool(args.smoke_batches), "calibration_evaluated": False, "confirmation_evaluated": False, "official_suim_test_evaluated": False}; atomic_torch_save(payload, output / "checkpoints" / "last.pt")
    if not args.smoke_batches: atomic_torch_save(payload, output / "checkpoints" / "final.pt")


if __name__ == "__main__": main()
