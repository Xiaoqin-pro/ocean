"""Train joint SUIM+UIIS F4 or class-balanced UTD on public train roles only."""
from __future__ import annotations

import argparse
import hashlib
import json
import random
import sys
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
from reliability.scdi_seg import semantic_router_loss  # noqa: E402
from reliability.sdtc_seg import clean_identity_loss  # noqa: E402
from reliability.spt_seg import SPTSegformer, rectifier_parameter_count, semantic_prototype_transport_loss  # noqa: E402
from reliability.utd_seg import semantic_trajectory_distillation_loss  # noqa: E402
from scripts.train_dts_seg_gate0 import atomic_torch_save, sha256  # noqa: E402
from scripts.train_uiis_spt import atomic_save  # noqa: E402
from degradations.registry import build_image_degradation, load_conditions  # noqa: E402
from reliability.dts_seg import trajectory_names  # noqa: E402

FORMAT = "joint_cb_utd_v1"


class CombinedTrajectoryDataset(Dataset[dict[str, Any]]):
    def __init__(self, suim_csv: Path, uiis_csv: Path, registry: Path, image_size: int) -> None:
        suim = pd.read_csv(suim_csv); uiis = pd.read_csv(uiis_csv)
        if suim_csv.name != "risk_head_train.csv" or len(suim) != 936: raise PermissionError("Joint training requires the frozen 936-image SUIM train role.")
        if uiis_csv.name != "train.csv" or len(uiis) != 2371: raise PermissionError("Joint training requires the frozen 2,371-image UIIS train role.")
        suim["domain"] = "suim"; uiis["domain"] = "uiis"; self.frame = pd.concat([suim, uiis], ignore_index=True).sort_values(["domain", "sample_id"]).reset_index(drop=True)
        self.conditions = {item.name: build_image_degradation(item) for item in load_conditions(registry)}; self.epoch = 1
        self.transform = A.ReplayCompose([A.Resize(image_size, image_size), A.HorizontalFlip(p=0.5), A.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD, max_pixel_value=255.0), ToTensorV2()])

    def set_epoch(self, epoch: int) -> None: self.epoch = epoch
    def __len__(self) -> int: return len(self.frame)

    def __getitem__(self, index: int) -> dict[str, Any]:
        row = self.frame.iloc[index]; domain = str(row.domain); sample_id = f"{domain}_{row.sample_id}"; image_path = ROOT / str(row.image_path); mask_path = ROOT / str(row.mask_path)
        with Image.open(image_path) as image: image_array = np.asarray(image.convert("RGB"), dtype=np.uint8)
        with Image.open(mask_path) as mask: mask_array = np.asarray(mask, dtype=np.uint8)
        seed = int(hashlib.sha256(sample_id.encode()).hexdigest()[:16], 16) + 100003 * self.epoch; self.transform.set_random_seed(seed % (2**32)); base = self.transform(image=image_array, mask=mask_array); names = trajectory_names(sample_id, self.epoch); views = {"clean": base["image"].float()}
        for key, name in zip(("s1", "s2", "s3"), names[1:], strict=True):
            replay = A.ReplayCompose.replay(base["replay"], image=self.conditions[name](image_array, sample_id), mask=mask_array); views[key] = replay["image"].float()
        return {**views, "labels": base["mask"].long(), "sample_id": sample_id, "domain": domain}


def build_model(config: dict[str, Any], variant: str, device: torch.device) -> torch.nn.Module:
    base = SegformerForSemanticSegmentation.from_pretrained(config["model"]["pretrained_model"], num_labels=8, id2label=ID2LABEL, label2id=LABEL2ID, ignore_mismatched_sizes=True).to(device); source_path = ROOT / str(config["model"]["source_checkpoint"]); source = torch.load(source_path, map_location=device, weights_only=False)
    required = {"variant": "B", "model_name": "segformer", "epoch": 100, "run_kind": "formal", "epoch_completed": True}
    for key, expected in required.items():
        if source.get(key) != expected: raise ValueError(f"Joint source has invalid {key}.")
    if any(bool(source.get(key, True)) for key in ("validation_evaluated", "calibration_evaluated", "official_suim_test_evaluated")): raise ValueError("Joint source records prohibited access.")
    base.load_state_dict(source["model_state_dict"])
    if variant == "F4": return base
    if variant == "CB-UTD": return SPTSegformer(base, tuple(config["model"]["hidden_sizes"]), int(config["model"]["num_classes"]), int(config["model"]["rectifier_reduction"])).to(device)
    raise ValueError(f"Unknown joint variant {variant}.")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__); parser.add_argument("--variant", choices=("F4", "CB-UTD"), required=True); parser.add_argument("--config", type=Path, default=ROOT / "configs/joint_cb_utd.yaml"); parser.add_argument("--smoke-batches", type=int, default=0); args = parser.parse_args(); config_path = args.config.resolve(); config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    seed = int(config["experiment"]["seed"]); random.seed(seed); torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)
    if not torch.cuda.is_available(): raise RuntimeError("Joint training requires CUDA.")
    device = torch.device("cuda"); training = config["training"]; dataset = CombinedTrajectoryDataset(ROOT / str(config["data"]["suim_train_csv"]), ROOT / str(config["data"]["uiis_train_csv"]), ROOT / str(config["data"]["degradation_registry"]), int(config["data"]["image_size"])); loader = DataLoader(dataset, batch_size=int(training["batch_size"]), shuffle=True, num_workers=0, pin_memory=True, generator=torch.Generator().manual_seed(seed)); model = build_model(config, args.variant, device)
    if args.variant == "CB-UTD":
        expert_ids = {id(parameter) for parameter in model.rectifiers.parameters()}; parameters = [{"params": [parameter for parameter in model.parameters() if id(parameter) not in expert_ids], "lr": float(training["backbone_learning_rate"])}, {"params": list(model.rectifiers.parameters()), "lr": float(training["expert_learning_rate"])}]
    else: parameters = [{"params": list(model.parameters()), "lr": float(training["backbone_learning_rate"])}]
    optimizer = torch.optim.AdamW(parameters, weight_decay=float(training["weight_decay"])); scaler = torch.amp.GradScaler("cuda", enabled=bool(training["amp"])); epochs = 1 if args.smoke_batches else int(training["epochs"]); output = ROOT / str(config["experiment"]["output_dir"]) / ("smoke" if args.smoke_batches else "formal") / args.variant; source_path = ROOT / str(config["model"]["source_checkpoint"]); history = []; model.train()
    for epoch in range(1, epochs + 1):
        dataset.set_epoch(epoch); totals = {"ce": 0.0, "prototype": 0.0, "distillation": 0.0, "identity": 0.0, "router": 0.0, "pairs": 0.0}; batches = 0
        for batch_index, batch in enumerate(loader):
            if args.smoke_batches and batch_index >= args.smoke_batches: break
            labels = batch["labels"].to(device, non_blocking=True); optimizer.zero_grad(set_to_none=True); clean_reference = None; teacher_logits = None
            for view_index, key in enumerate(("clean", "s1", "s2", "s3")):
                pixels = batch[key].to(device, non_blocking=True)
                with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=bool(training["amp"])):
                    if args.variant == "F4":
                        result = model(pixel_values=pixels); logits = result.logits; raw = residuals = corrected = routers = None
                    else:
                        result = model(pixels); logits = result.logits; raw = result.raw_features; residuals = result.residuals; corrected = result.corrected_features; routers = result.router_logits
                    full_logits = functional.interpolate(logits.float(), size=labels.shape[-2:], mode="bilinear", align_corners=False); ce = functional.cross_entropy(full_logits, labels, ignore_index=255) / 4.0; prototype = logits.sum() * 0.0; identity = logits.sum() * 0.0; distillation = logits.sum() * 0.0; router = logits.sum() * 0.0; pairs = 0
                    if args.variant == "CB-UTD":
                        router = semantic_router_loss(routers, labels)
                        if key == "clean": identity = clean_identity_loss(raw, residuals); clean_reference = tuple(item.detach() for item in raw); teacher_logits = logits.detach()
                        else:
                            prototype, pairs = semantic_prototype_transport_loss(clean_reference, corrected, labels, num_classes=int(config["model"]["num_classes"]), minimum_pixels=int(training["prototype_minimum_pixels"]), margin=float(training["prototype_margin"])); distillation = semantic_trajectory_distillation_loss(teacher_logits, logits, labels, temperature=float(training["distillation_temperature"]), confidence_power=float(training["confidence_power"]), class_balance=bool(training.get("class_balance", False)))
                    loss = ce + float(training["prototype_weight"]) * prototype + float(training["clean_identity_weight"]) * identity + float(training["router_weight"]) * router + float(training["distillation_weight"]) * (float(view_index) / 3.0) * distillation
                scaler.scale(loss).backward(); totals["ce"] += float(ce.detach()); totals["prototype"] += float(prototype.detach()); totals["distillation"] += float(distillation.detach()); totals["identity"] += float(identity.detach()); totals["router"] += float(router.detach()); totals["pairs"] += pairs; del pixels, result, logits, full_logits, loss
            scaler.step(optimizer); scaler.update(); batches += 1
            if batch_index % 50 == 0: print(f"{args.variant} epoch={epoch} batch={batch_index}/{len(loader)} ce={totals['ce']/batches:.4f} proto={totals['prototype']/batches:.4f} distill={totals['distillation']/batches:.4f}", flush=True)
        if batches == 0: raise RuntimeError("No training batch completed.")
        row = {"epoch": float(epoch), "batches": float(batches), **{key: value / batches for key, value in totals.items()}}; history.append(row); print(json.dumps(row, sort_keys=True), flush=True)
        payload = {"checkpoint_format": FORMAT, "variant": args.variant, "epoch": epoch, "model_state_dict": model.state_dict(), "optimizer_state_dict": optimizer.state_dict(), "scaler_state_dict": scaler.state_dict(), "history": history, "config_sha256": sha256(config_path), "source_checkpoint_sha256": sha256(source_path), "smoke": bool(args.smoke_batches), "official_suim_test_evaluated": False, "uiis_confirmation_evaluated": False}; atomic_torch_save(payload, output / "checkpoints" / "last.pt")
    if not args.smoke_batches: atomic_torch_save(payload, output / "checkpoints" / "final.pt")


if __name__ == "__main__": main()
