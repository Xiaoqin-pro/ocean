"""Train paired F4/SCDI replication runs on UIIS train only."""
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
from reliability.dts_seg import trajectory_names  # noqa: E402
from reliability.sdtc_seg import clean_identity_loss, semantic_tangent_loss  # noqa: E402
from reliability.sdtc_seg import SDTCSegformer  # noqa: E402


FORMAT = "uiis_scdi_replication_v1"


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


class UIISTrajectoryDataset(Dataset[dict[str, Any]]):
    def __init__(self, split_csv: Path, registry: Path, image_size: int) -> None:
        self.frame = pd.read_csv(split_csv).sort_values("sample_id").reset_index(drop=True)
        if split_csv.name != "train.csv" or len(self.frame) != 2371:
            raise PermissionError("UIIS replication training accepts only the frozen 2,371-image train split.")
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
        with Image.open(ROOT / str(row.image_path)) as image:
            image_array = np.asarray(image.convert("RGB"), dtype=np.uint8)
        with Image.open(ROOT / str(row.mask_path)) as mask:
            mask_array = np.asarray(mask, dtype=np.uint8)
        names = trajectory_names(sample_id, self.epoch)
        self.transform.set_random_seed((int(hashlib.sha256(sample_id.encode()).hexdigest()[:16], 16) + 100003 * self.epoch) % (2**32))
        base = self.transform(image=image_array, mask=mask_array)
        views = {"clean": base["image"].float()}
        for key, name in zip(("s1", "s2", "s3"), names[1:], strict=True):
            replay = A.ReplayCompose.replay(base["replay"], image=self.conditions[name](image_array, sample_id), mask=mask_array)
            views[key] = replay["image"].float()
        return {**views, "labels": base["mask"].long(), "sample_id": sample_id}


def build_models(config: dict[str, Any], variant: str, device: torch.device) -> torch.nn.Module:
    base = SegformerForSemanticSegmentation.from_pretrained(
        config["model"]["pretrained_model"], num_labels=8, id2label=ID2LABEL, label2id=LABEL2ID, ignore_mismatched_sizes=True,
    ).to(device)
    source_path = ROOT / str(config["model"]["source_checkpoint"])
    source = torch.load(source_path, map_location=device, weights_only=False)
    if source.get("checkpoint_format") != "uiis_fixed_protocol_v1" or int(source.get("epoch", 0)) != 60:
        raise ValueError("The UIIS source must be the completed 60-epoch train-only checkpoint.")
    if source.get("confirmation_evaluated") or source.get("official_suim_test_evaluated"):
        raise ValueError("UIIS source records prohibited confirmation/test access.")
    base.load_state_dict(source["model_state_dict"])
    if variant == "F4":
        return base
    if variant == "SCDI":
        return SDTCSegformer(base, tuple(config["model"]["hidden_sizes"]), int(config["model"]["rectifier_reduction"])).to(device)
    raise ValueError(f"Unknown replication variant: {variant}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--variant", choices=("F4", "SCDI"), required=True)
    parser.add_argument("--config", type=Path, default=ROOT / "configs/uiis_scdi_replication.yaml")
    parser.add_argument("--smoke-batches", type=int, default=0)
    args = parser.parse_args()
    config_path = args.config.resolve(); config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    if args.smoke_batches < 0:
        raise ValueError("smoke-batches must be non-negative.")
    seed = int(config["experiment"]["seed"])
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)
    if not torch.cuda.is_available():
        raise RuntimeError("UIIS replication requires CUDA.")
    device = torch.device("cuda"); training = config["training"]
    dataset = UIISTrajectoryDataset(ROOT / str(config["data"]["train_csv"]), ROOT / str(config["data"]["degradation_registry"]), int(config["data"]["image_size"]))
    loader = DataLoader(dataset, batch_size=int(training["batch_size"]), shuffle=True, num_workers=0, pin_memory=True, generator=torch.Generator().manual_seed(seed))
    model = build_models(config, args.variant, device)
    if args.variant == "SCDI":
        expert_ids = {id(parameter) for parameter in model.rectifiers.parameters()}
        parameters = [
            {"params": [parameter for parameter in model.parameters() if id(parameter) not in expert_ids], "lr": float(training["backbone_learning_rate"])},
            {"params": list(model.rectifiers.parameters()), "lr": float(training["expert_learning_rate"])},
        ]
    else:
        parameters = [{"params": list(model.parameters()), "lr": float(training["backbone_learning_rate"])}]
    optimizer = torch.optim.AdamW(parameters, weight_decay=float(training["weight_decay"]))
    scaler = torch.amp.GradScaler("cuda", enabled=bool(training["amp"]))
    epochs = 1 if args.smoke_batches else int(training["epochs"])
    output = ROOT / str(config["experiment"]["output_dir"]) / ("smoke" if args.smoke_batches else "formal") / args.variant
    history = []; source_path = ROOT / str(config["model"]["source_checkpoint"]); model.train()
    for epoch in range(1, epochs + 1):
        dataset.set_epoch(epoch); totals = {"ce": 0.0, "tangent": 0.0, "identity": 0.0}; batches = 0
        for batch_index, batch in enumerate(loader):
            if args.smoke_batches and batch_index >= args.smoke_batches: break
            labels = batch["labels"].to(device, non_blocking=True); optimizer.zero_grad(set_to_none=True); clean_reference = None
            for key in ("clean", "s1", "s2", "s3"):
                pixels = batch[key].to(device, non_blocking=True)
                with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=bool(training["amp"])):
                    if args.variant == "F4":
                        result = model(pixel_values=pixels); logits = result.logits; raw = residuals = None
                    else:
                        result = model(pixels); logits = result.logits; raw = result.raw_features; residuals = result.residuals
                    full_logits = functional.interpolate(logits.float(), size=labels.shape[-2:], mode="bilinear", align_corners=False)
                    ce = functional.cross_entropy(full_logits, labels, ignore_index=255) / 4.0
                    tangent = logits.sum() * 0.0; identity = logits.sum() * 0.0
                    if args.variant == "SCDI":
                        if key == "clean":
                            identity = clean_identity_loss(raw, residuals); clean_reference = tuple(item.detach() for item in raw)
                        else:
                            tangent, _ = semantic_tangent_loss(clean_reference, raw, residuals, labels, minimum_pixels=int(training["tangent_minimum_pixels"]))
                    loss = ce + float(training["tangent_weight"]) * tangent + float(training["clean_identity_weight"]) * identity
                scaler.scale(loss).backward(); totals["ce"] += float(ce.detach()); totals["tangent"] += float(tangent.detach()); totals["identity"] += float(identity.detach())
                del pixels, result, logits, full_logits, loss
            scaler.step(optimizer); scaler.update(); batches += 1
            if batch_index % 50 == 0: print(f"{args.variant} epoch={epoch} batch={batch_index}/{len(loader)} ce={totals['ce']/batches:.4f} tangent={totals['tangent']/batches:.4f}", flush=True)
        row = {"epoch": float(epoch), "batches": float(batches), **{key: value / batches for key, value in totals.items()}}; history.append(row); print(json.dumps(row, sort_keys=True), flush=True)
        payload = {"checkpoint_format": FORMAT, "variant": args.variant, "epoch": epoch, "model_state_dict": model.state_dict(), "optimizer_state_dict": optimizer.state_dict(), "scaler_state_dict": scaler.state_dict(), "history": history, "config_sha256": sha256(config_path), "source_checkpoint_sha256": sha256(source_path), "smoke": bool(args.smoke_batches), "confirmation_evaluated": False, "official_suim_test_evaluated": False}
        atomic_save(payload, output / "checkpoints/last.pt")
    if not args.smoke_batches: atomic_save(payload, output / "checkpoints/final.pt")


if __name__ == "__main__": main()
