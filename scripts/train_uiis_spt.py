"""Train the UIIS Semantic Prototype Transport (SPT) candidate on train only."""
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

import torch
import torch.nn.functional as functional
import yaml
from torch.utils.data import DataLoader
from transformers import SegformerForSemanticSegmentation

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from datasets.label_mapping import ID2LABEL, LABEL2ID  # noqa: E402
from reliability.sdtc_seg import clean_identity_loss  # noqa: E402
from reliability.scdi_seg import semantic_router_loss  # noqa: E402
from reliability.spt_seg import SPTSegformer, rectifier_parameter_count, semantic_prototype_transport_loss  # noqa: E402
from scripts.train_uiis_scdi_replication import UIISTrajectoryDataset  # noqa: E402


FORMAT = "uiis_spt_v1"


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


def build_model(config: dict[str, Any], device: torch.device) -> SPTSegformer:
    base = SegformerForSemanticSegmentation.from_pretrained(
        config["model"]["pretrained_model"], num_labels=8, id2label=ID2LABEL, label2id=LABEL2ID, ignore_mismatched_sizes=True,
    ).to(device)
    source_path = ROOT / str(config["model"]["source_checkpoint"])
    source = torch.load(source_path, map_location=device, weights_only=False)
    if source.get("checkpoint_format") != "uiis_fixed_protocol_v1" or int(source.get("epoch", 0)) != 60:
        raise ValueError("SPT requires the completed 60-epoch UIIS train-only checkpoint.")
    if source.get("confirmation_evaluated") or source.get("official_suim_test_evaluated"):
        raise ValueError("SPT source records prohibited evaluation access.")
    base.load_state_dict(source["model_state_dict"])
    return SPTSegformer(
        base, tuple(config["model"]["hidden_sizes"]), int(config["model"]["num_classes"]), int(config["model"]["rectifier_reduction"]),
    ).to(device)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=ROOT / "configs/uiis_spt.yaml")
    parser.add_argument("--smoke-batches", type=int, default=0)
    args = parser.parse_args()
    if args.smoke_batches < 0:
        raise ValueError("smoke-batches must be non-negative.")
    config_path = args.config.resolve(); config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    seed = int(config["experiment"]["seed"])
    random.seed(seed); torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)
    if not torch.cuda.is_available():
        raise RuntimeError("UIIS SPT requires CUDA.")
    device = torch.device("cuda"); training = config["training"]
    dataset = UIISTrajectoryDataset(ROOT / str(config["data"]["train_csv"]), ROOT / str(config["data"]["degradation_registry"]), int(config["data"]["image_size"]))
    loader = DataLoader(dataset, batch_size=int(training["batch_size"]), shuffle=True, num_workers=0, pin_memory=True, generator=torch.Generator().manual_seed(seed))
    model = build_model(config, device)
    expert_ids = {id(parameter) for parameter in model.rectifiers.parameters()}
    optimizer = torch.optim.AdamW(
        [
            {"params": [parameter for parameter in model.parameters() if id(parameter) not in expert_ids], "lr": float(training["backbone_learning_rate"])},
            {"params": list(model.rectifiers.parameters()), "lr": float(training["expert_learning_rate"])},
        ], weight_decay=float(training["weight_decay"]),
    )
    scaler = torch.amp.GradScaler("cuda", enabled=bool(training["amp"]))
    epochs = 1 if args.smoke_batches else int(training["epochs"])
    output = ROOT / str(config["experiment"]["output_dir"]) / ("smoke" if args.smoke_batches else "formal")
    source_path = ROOT / str(config["model"]["source_checkpoint"]); history: list[dict[str, float]] = []
    model.train()
    for epoch in range(1, epochs + 1):
        dataset.set_epoch(epoch); totals = {"ce": 0.0, "prototype": 0.0, "identity": 0.0, "router": 0.0, "pairs": 0.0}; batches = 0
        for batch_index, batch in enumerate(loader):
            if args.smoke_batches and batch_index >= args.smoke_batches: break
            labels = batch["labels"].to(device, non_blocking=True); optimizer.zero_grad(set_to_none=True); clean_reference = None
            for key in ("clean", "s1", "s2", "s3"):
                pixels = batch[key].to(device, non_blocking=True)
                with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=bool(training["amp"])):
                    result = model(pixels)
                    full_logits = functional.interpolate(result.logits.float(), size=labels.shape[-2:], mode="bilinear", align_corners=False)
                    ce = functional.cross_entropy(full_logits, labels, ignore_index=255) / 4.0
                    router = semantic_router_loss(result.router_logits, labels)
                    prototype = result.logits.sum() * 0.0; identity = result.logits.sum() * 0.0; pairs = 0
                    if key == "clean":
                        identity = clean_identity_loss(result.raw_features, result.residuals); clean_reference = tuple(item.detach() for item in result.raw_features)
                    else:
                        if clean_reference is None: raise AssertionError("Clean reference must be computed first.")
                        prototype, pairs = semantic_prototype_transport_loss(clean_reference, result.corrected_features, labels, num_classes=int(config["model"]["num_classes"]), minimum_pixels=int(training["prototype_minimum_pixels"]), margin=float(training["prototype_margin"]))
                    loss = ce + float(training["prototype_weight"]) * prototype + float(training["clean_identity_weight"]) * identity + float(training["router_weight"]) * router
                scaler.scale(loss).backward(); totals["ce"] += float(ce.detach()); totals["prototype"] += float(prototype.detach()); totals["identity"] += float(identity.detach()); totals["router"] += float(router.detach()); totals["pairs"] += pairs
                del pixels, result, full_logits, loss
            scaler.step(optimizer); scaler.update(); batches += 1
            if batch_index % 50 == 0: print(f"SPT epoch={epoch} batch={batch_index}/{len(loader)} ce={totals['ce']/batches:.4f} proto={totals['prototype']/batches:.4f} router={totals['router']/batches:.4f}", flush=True)
        if batches == 0: raise RuntimeError("No training batch completed.")
        row = {"epoch": float(epoch), "batches": float(batches), **{key: value / batches for key, value in totals.items()}}; history.append(row); print(json.dumps(row, sort_keys=True), flush=True)
        payload = {"checkpoint_format": FORMAT, "variant": "SPT", "epoch": epoch, "model_state_dict": model.state_dict(), "optimizer_state_dict": optimizer.state_dict(), "scaler_state_dict": scaler.state_dict(), "history": history, "config_sha256": sha256(config_path), "source_checkpoint_sha256": sha256(source_path), "rectifier_parameters": rectifier_parameter_count(model), "smoke": bool(args.smoke_batches), "calibration_evaluated": False, "confirmation_evaluated": False, "official_suim_test_evaluated": False}
        atomic_save(payload, output / "checkpoints/last.pt")
    if not args.smoke_batches: atomic_save(payload, output / "checkpoints/final.pt")


if __name__ == "__main__": main()
