"""Unsupervised target-domain adaptation of a frozen SUIM source model."""
from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

import torch
import torch.nn.functional as functional
import yaml
from torch.utils.data import DataLoader
from transformers import SegformerForSemanticSegmentation

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from datasets.label_mapping import ID2LABEL, LABEL2ID  # noqa: E402
from reliability.sdtc_seg import clean_identity_loss  # noqa: E402
from reliability.spt_seg import SPTSegformer  # noqa: E402
from reliability.tta import confident_pseudo_label_loss, prediction_entropy  # noqa: E402
from scripts.train_dts_seg_gate0 import atomic_torch_save, sha256  # noqa: E402
from scripts.train_uiis_scdi_replication import UIISTrajectoryDataset  # noqa: E402

FORMAT = "tta_spt_v1"


def build_model(config: dict, device: torch.device) -> SPTSegformer:
    base = SegformerForSemanticSegmentation.from_pretrained(config["model"]["pretrained_model"], num_labels=8, id2label=ID2LABEL, label2id=LABEL2ID, ignore_mismatched_sizes=True).to(device); source = torch.load(ROOT / str(config["model"]["source_checkpoint"]), map_location=device, weights_only=False); required = {"variant": "B", "model_name": "segformer", "epoch": 100, "run_kind": "formal", "epoch_completed": True}
    for key, expected in required.items():
        if source.get(key) != expected: raise ValueError(f"Invalid source field {key}.")
    if any(bool(source.get(key, True)) for key in ("validation_evaluated", "calibration_evaluated", "official_suim_test_evaluated")): raise ValueError("Source records prohibited access.")
    base.load_state_dict(source["model_state_dict"]); model = SPTSegformer(base, tuple(config["model"]["hidden_sizes"]), int(config["model"]["num_classes"]), int(config["model"]["rectifier_reduction"])).to(device)
    for parameter in model.base_model.parameters(): parameter.requires_grad_(False)
    return model


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__); parser.add_argument("--config", type=Path, default=ROOT / "configs/tta_spt_fast.yaml"); parser.add_argument("--smoke-batches", type=int, default=0); args = parser.parse_args(); config_path = args.config.resolve(); config = yaml.safe_load(config_path.read_text(encoding="utf-8")); seed = int(config["experiment"]["seed"]); random.seed(seed); torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)
    if not torch.cuda.is_available(): raise RuntimeError("TTA requires CUDA.")
    device = torch.device("cuda"); data = config["data"]; training = config["training"]; dataset = UIISTrajectoryDataset(ROOT / str(data["uiis_train_csv"]), ROOT / str(data["degradation_registry"]), int(data["image_size"])); loader = DataLoader(dataset, batch_size=int(training["batch_size"]), shuffle=True, num_workers=0, pin_memory=True, generator=torch.Generator().manual_seed(seed)); model = build_model(config, device); model.train(); model.base_model.eval(); optimizer = torch.optim.AdamW(list(model.rectifiers.parameters()), lr=float(training["learning_rate"]), weight_decay=float(training["weight_decay"])); scaler = torch.amp.GradScaler("cuda", enabled=bool(training["amp"])); epochs = 1 if args.smoke_batches else int(training["epochs"]); output = ROOT / str(config["experiment"]["output_dir"]) / ("smoke" if args.smoke_batches else "formal"); history = []
    for epoch in range(1, epochs + 1):
        dataset.set_epoch(epoch); totals = {key: 0.0 for key in ("pseudo", "entropy", "identity", "coverage")}; batches = 0
        for batch_index, batch in enumerate(loader):
            if args.smoke_batches and batch_index >= args.smoke_batches: break
            optimizer.zero_grad(set_to_none=True); clean = batch["clean"].to(device, non_blocking=True); degraded = batch["s3"].to(device, non_blocking=True)
            with torch.no_grad(): teacher = model(clean); teacher_logits = teacher.logits.detach()
            with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=bool(training["amp"])):
                student = model(degraded); pseudo, coverage = confident_pseudo_label_loss(student.logits, teacher_logits, threshold=float(training["pseudo_threshold"])); entropy = prediction_entropy(student.logits); identity = clean_identity_loss(student.raw_features, student.residuals); residual = sum(item.float().pow(2).mean() for item in student.residuals); loss = float(training["pseudo_weight"]) * pseudo + float(training["entropy_weight"]) * entropy + float(training["identity_weight"]) * identity + float(training["residual_weight"]) * residual
            scaler.scale(loss).backward(); scaler.step(optimizer); scaler.update(); totals["pseudo"] += float(pseudo.detach()); totals["entropy"] += float(entropy.detach()); totals["identity"] += float(identity.detach()); totals["coverage"] += coverage; batches += 1
            if batch_index % 50 == 0: print(f"TTA epoch={epoch} batch={batch_index}/{len(loader)} pseudo={totals['pseudo']/batches:.4f} entropy={totals['entropy']/batches:.4f} coverage={totals['coverage']/batches:.3f}", flush=True)
            del clean, degraded, teacher, student, teacher_logits, loss
        if batches == 0: raise RuntimeError("No batch completed.")
        row = {"epoch": float(epoch), "batches": float(batches), **{key: value / batches for key, value in totals.items()}}; history.append(row); print(json.dumps(row, sort_keys=True), flush=True); payload = {"checkpoint_format": FORMAT, "variant": "TTA-SPT", "epoch": epoch, "model_state_dict": model.state_dict(), "optimizer_state_dict": optimizer.state_dict(), "scaler_state_dict": scaler.state_dict(), "history": history, "config_sha256": sha256(config_path), "smoke": bool(args.smoke_batches), "calibration_evaluated": False, "confirmation_evaluated": False, "official_suim_test_evaluated": False}; atomic_torch_save(payload, output / "checkpoints" / "last.pt")
    if not args.smoke_batches: atomic_torch_save(payload, output / "checkpoints" / "final.pt")


if __name__ == "__main__": main()
