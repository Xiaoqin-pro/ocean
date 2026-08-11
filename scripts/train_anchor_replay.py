"""Train source-anchor replay Semantic Prototype Transport on SUIM + UIIS."""
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
from reliability.anchor_replay import source_logit_anchor_loss  # noqa: E402
from reliability.sdtc_seg import clean_identity_loss  # noqa: E402
from reliability.spt_seg import SPTSegformer, semantic_prototype_transport_loss  # noqa: E402
from reliability.utd_seg import semantic_trajectory_distillation_loss  # noqa: E402
from scripts.train_dts_seg_gate0 import atomic_torch_save, sha256  # noqa: E402
from scripts.train_dcpt import EpochSubset  # noqa: E402
from scripts.train_joint_cb_utd import CombinedTrajectoryDataset  # noqa: E402
from reliability.dcpt_seg import shared_prototype_alignment_loss  # noqa: E402

FORMAT = "ar_spt_seg_v1"


def build_model(config: dict, device: torch.device) -> SPTSegformer:
    base = SegformerForSemanticSegmentation.from_pretrained(config["model"]["pretrained_model"], num_labels=8, id2label=ID2LABEL, label2id=LABEL2ID, ignore_mismatched_sizes=True).to(device)
    source_path = ROOT / str(config["model"]["source_checkpoint"]); source = torch.load(source_path, map_location=device, weights_only=False); required = {"variant": "B", "model_name": "segformer", "epoch": 100, "run_kind": "formal", "epoch_completed": True}
    for key, expected in required.items():
        if source.get(key) != expected: raise ValueError(f"Invalid source field {key}.")
    if any(bool(source.get(key, True)) for key in ("validation_evaluated", "calibration_evaluated", "official_suim_test_evaluated")): raise ValueError("Source records prohibited access.")
    base.load_state_dict(source["model_state_dict"]); model = SPTSegformer(base, tuple(config["model"]["hidden_sizes"]), int(config["model"]["num_classes"]), int(config["model"]["rectifier_reduction"])).to(device)
    for parameter in model.base_model.parameters(): parameter.requires_grad_(False)
    return model


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__); parser.add_argument("--config", type=Path, default=ROOT / "configs/anchor_replay_fast.yaml"); parser.add_argument("--smoke-batches", type=int, default=0); args = parser.parse_args(); config_path = args.config.resolve(); config = yaml.safe_load(config_path.read_text(encoding="utf-8")); seed = int(config["experiment"]["seed"]); random.seed(seed); torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)
    if not torch.cuda.is_available(): raise RuntimeError("Anchor replay requires CUDA.")
    device = torch.device("cuda"); data_config = config["data"]; training = config["training"]; combined = CombinedTrajectoryDataset(ROOT / str(data_config["suim_train_csv"]), ROOT / str(data_config["uiis_train_csv"]), ROOT / str(data_config["degradation_registry"]), int(data_config["image_size"])); indices = {domain: [index for index, value in enumerate(combined.frame.domain.tolist()) if value == name] for domain, name in ((0, "suim"), (1, "uiis"))}; datasets = {domain: EpochSubset(combined, values) for domain, values in indices.items()}; loaders = {domain: DataLoader(dataset, batch_size=int(training["batch_size"]), shuffle=True, num_workers=0, pin_memory=True, generator=torch.Generator().manual_seed(seed + domain)) for domain, dataset in datasets.items()}; model = build_model(config, device); model.train(); model.base_model.eval(); parameters = list(model.rectifiers.parameters()); optimizer = torch.optim.AdamW(parameters, lr=float(training["learning_rate"]), weight_decay=float(training["weight_decay"])); scaler = torch.amp.GradScaler("cuda", enabled=bool(training["amp"])); epochs = 1 if args.smoke_batches else int(training["epochs"]); output = ROOT / str(config["experiment"]["output_dir"]) / ("smoke" if args.smoke_batches else "formal"); history = []
    for epoch in range(1, epochs + 1):
        totals = {key: 0.0 for key in ("ce", "prototype", "shared", "distillation", "source_anchor", "identity", "anchor", "pairs")}; batches = 0
        for domain in (0, 1):
            datasets[domain].set_epoch(epoch)
            for batch_index, batch in enumerate(loaders[domain]):
                if args.smoke_batches and batch_index >= args.smoke_batches: break
                labels = batch["labels"].to(device, non_blocking=True); optimizer.zero_grad(set_to_none=True); clean_pixels = batch["clean"].to(device, non_blocking=True); clean_result = model(clean_pixels); clean_reference = tuple(item.detach() for item in clean_result.raw_features); teacher_clean = model.base_model(pixel_values=clean_pixels).logits.detach(); teacher_logits = clean_result.logits.detach(); clean_identity = clean_identity_loss(clean_result.raw_features, clean_result.residuals); clean_full = functional.interpolate(clean_result.logits.float(), size=labels.shape[-2:], mode="bilinear", align_corners=False); clean_ce = functional.cross_entropy(clean_full, labels, ignore_index=255) / 4.0
                total_loss = clean_ce; totals["ce"] += float(clean_ce.detach()); totals["identity"] += float(clean_identity.detach())
                for key in ("s1", "s2", "s3"):
                    pixels = batch[key].to(device, non_blocking=True)
                    with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=bool(training["amp"])):
                        result = model(pixels); logits = result.logits; full_logits = functional.interpolate(logits.float(), size=labels.shape[-2:], mode="bilinear", align_corners=False); ce = functional.cross_entropy(full_logits, labels, ignore_index=255) / 4.0; prototype, pairs = semantic_prototype_transport_loss(clean_reference, result.corrected_features, labels, num_classes=8, minimum_pixels=int(training["prototype_minimum_pixels"]), margin=float(training["prototype_margin"])); distillation = semantic_trajectory_distillation_loss(teacher_logits, logits, labels, temperature=float(training["distillation_temperature"]), confidence_power=float(training["confidence_power"]), class_balance=True); residual_anchor = sum(item.float().pow(2).mean() for item in result.residuals); source_anchor = source_logit_anchor_loss(logits, model.base_model(pixel_values=pixels).logits.detach(), temperature=float(training["source_temperature"])) if domain == 0 else logits.sum() * 0.0; loss = ce + float(training["prototype_weight"]) * prototype + float(training["distillation_weight"]) * (int(key[-1]) / 3.0) * distillation + float(training["anchor_weight"]) * residual_anchor + float(training["source_anchor_weight"]) * source_anchor
                    total_loss = total_loss + loss; totals["ce"] += float(ce.detach()); totals["prototype"] += float(prototype.detach()); totals["distillation"] += float(distillation.detach()); totals["source_anchor"] += float(source_anchor.detach()); totals["anchor"] += float(residual_anchor.detach()); totals["pairs"] += float(pairs); del pixels, result, logits, full_logits, loss
                total_loss = total_loss + float(training["clean_identity_weight"]) * clean_identity; scaler.scale(total_loss).backward(); scaler.step(optimizer); scaler.update(); batches += 1
                if batch_index % 50 == 0: print(f"AR-SPT epoch={epoch} domain={domain} batch={batch_index}/{len(loaders[domain])} ce={totals['ce']/max(batches,1):.4f} source_anchor={totals['source_anchor']/max(batches,1):.4f}", flush=True)
                del clean_pixels, clean_result, teacher_clean, teacher_logits, total_loss, clean_identity, clean_full
        if batches == 0: raise RuntimeError("No training batch completed.")
        row = {"epoch": float(epoch), "batches": float(batches), **{key: value / batches for key, value in totals.items()}}; history.append(row); print(json.dumps(row, sort_keys=True), flush=True); payload = {"checkpoint_format": FORMAT, "variant": "AR-SPT", "epoch": epoch, "model_state_dict": model.state_dict(), "optimizer_state_dict": optimizer.state_dict(), "scaler_state_dict": scaler.state_dict(), "history": history, "config_sha256": sha256(config_path), "smoke": bool(args.smoke_batches), "calibration_evaluated": False, "confirmation_evaluated": False, "official_suim_test_evaluated": False}; atomic_torch_save(payload, output / "checkpoints" / "last.pt")
    if not args.smoke_batches: atomic_torch_save(payload, output / "checkpoints" / "final.pt")


if __name__ == "__main__": main()
