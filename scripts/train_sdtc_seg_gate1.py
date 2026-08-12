"""Train the preregistered SDTC-Seg Gate-1 candidate on method_train only."""
from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as functional
import yaml
from torch.utils.data import DataLoader
from transformers import SegformerForSemanticSegmentation

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from datasets.label_mapping import ID2LABEL, LABEL2ID  # noqa: E402
from reliability.sdtc_seg import (  # noqa: E402
    SDTCSegformer,
    clean_identity_loss,
    rectifier_parameter_count,
    semantic_tangent_loss,
)
from scripts.train_dts_seg_gate0 import (  # noqa: E402
    FourViewTrajectoryDataset,
    atomic_torch_save,
    sha256,
)


FORMAT = "sdtc_seg_gate1_v1"


def build_model(config: dict[str, Any], device: torch.device) -> SDTCSegformer:
    base = SegformerForSemanticSegmentation.from_pretrained(
        config["model"]["pretrained_model"],
        num_labels=int(config["model"]["num_classes"]),
        id2label=ID2LABEL,
        label2id=LABEL2ID,
        ignore_mismatched_sizes=True,
    ).to(device)
    source_path = ROOT / str(config["model"]["source_checkpoint"])
    source = torch.load(source_path, map_location=device, weights_only=False)
    required = {"variant": "B", "model_name": "segformer", "epoch": 100, "run_kind": "formal", "epoch_completed": True}
    for key, expected in required.items():
        if source.get(key) != expected:
            raise ValueError(f"Strong source checkpoint has invalid {key}.")
    if any(bool(source.get(key, True)) for key in ("validation_evaluated", "calibration_evaluated", "official_suim_test_evaluated")):
        raise ValueError("Source checkpoint records prohibited formal-split access.")
    base.load_state_dict(source["model_state_dict"])
    return SDTCSegformer(
        base,
        tuple(int(value) for value in config["model"]["hidden_sizes"]),
        int(config["model"]["rectifier_reduction"]),
    ).to(device)


def checkpoint_payload(
    model: SDTCSegformer,
    optimizer: torch.optim.Optimizer,
    scaler: torch.amp.GradScaler,
    *,
    epoch: int,
    config_path: Path,
    source_path: Path,
    history: list[dict[str, float]],
    smoke: bool,
) -> dict[str, Any]:
    return {
        "checkpoint_format": FORMAT,
        "variant": "SDTC",
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "scaler_state_dict": scaler.state_dict(),
        "history": history,
        "config_sha256": sha256(config_path),
        "source_checkpoint_sha256": sha256(source_path),
        "rectifier_parameters": rectifier_parameter_count(model),
        "smoke": smoke,
        "method_development_evaluated": False,
        "validation_evaluated": False,
        "calibration_evaluated": False,
        "official_suim_test_evaluated": False,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=ROOT / "configs/sdtc_seg_gate1.yaml")
    parser.add_argument("--smoke-batches", type=int, default=0)
    args = parser.parse_args()
    if args.smoke_batches < 0:
        raise ValueError("smoke-batches must be non-negative.")
    config_path = args.config.resolve()
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    seed = int(config["experiment"]["seed"])
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)
    if not torch.cuda.is_available():
        raise RuntimeError("SDTC Gate-1 requires CUDA.")
    device = torch.device("cuda")
    training = config["training"]
    dataset = FourViewTrajectoryDataset(
        ROOT / str(config["data"]["train_csv"]),
        ROOT / str(config["data"]["degradation_registry"]),
        int(config["data"]["image_size"]),
    )
    generator = torch.Generator().manual_seed(seed)
    loader = DataLoader(dataset, batch_size=int(training["batch_size"]), shuffle=True, num_workers=0, pin_memory=True, generator=generator)
    model = build_model(config, device)
    rectifier_ids = {id(parameter) for parameter in model.rectifiers.parameters()}
    optimizer = torch.optim.AdamW(
        [
            {"params": [parameter for parameter in model.parameters() if id(parameter) not in rectifier_ids], "lr": float(training["backbone_learning_rate"])},
            {"params": list(model.rectifiers.parameters()), "lr": float(training["rectifier_learning_rate"])},
        ],
        weight_decay=float(training["weight_decay"]),
    )
    scaler = torch.amp.GradScaler("cuda", enabled=bool(training["amp"]))
    epochs = 1 if args.smoke_batches else int(training["epochs"])
    output = ROOT / str(config["experiment"]["output_dir"]) / ("smoke" if args.smoke_batches else "formal") / "SDTC"
    history: list[dict[str, float]] = []
    source_path = ROOT / str(config["model"]["source_checkpoint"])
    model.train()
    for epoch in range(1, epochs + 1):
        dataset.set_epoch(epoch)
        totals = {"ce": 0.0, "tangent": 0.0, "identity": 0.0, "class_scale_pairs": 0.0}
        batches = 0
        for batch_index, batch in enumerate(loader):
            if args.smoke_batches and batch_index >= args.smoke_batches:
                break
            labels = batch["labels"].to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            clean_reference: tuple[torch.Tensor, ...] | None = None
            for key in ("clean", "s1", "s2", "s3"):
                pixels = batch[key].to(device, non_blocking=True)
                with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=bool(training["amp"])):
                    result = model(pixels)
                    full_logits = functional.interpolate(result.logits.float(), size=labels.shape[-2:], mode="bilinear", align_corners=False)
                    ce = functional.cross_entropy(full_logits, labels, ignore_index=255) / 4.0
                    tangent = result.logits.sum() * 0.0
                    identity = result.logits.sum() * 0.0
                    pairs = 0
                    if key == "clean":
                        identity = clean_identity_loss(result.raw_features, result.residuals)
                        clean_reference = tuple(item.detach() for item in result.raw_features)
                    else:
                        if clean_reference is None:
                            raise AssertionError("Clean reference must be computed first.")
                        tangent, pairs = semantic_tangent_loss(
                            clean_reference,
                            result.raw_features,
                            result.residuals,
                            labels,
                            num_classes=int(config["model"]["num_classes"]),
                            minimum_pixels=int(training["tangent_minimum_pixels"]),
                        )
                    loss = ce + float(training["tangent_weight"]) * tangent + float(training["clean_identity_weight"]) * identity
                scaler.scale(loss).backward()
                totals["ce"] += float(ce.detach())
                totals["tangent"] += float(tangent.detach())
                totals["identity"] += float(identity.detach())
                totals["class_scale_pairs"] += pairs
                del pixels, result, full_logits, loss
            scaler.step(optimizer)
            scaler.update()
            batches += 1
            if batch_index % 25 == 0:
                print(
                    f"SDTC epoch={epoch} batch={batch_index}/{len(loader)} ce={totals['ce']/batches:.4f} "
                    f"tangent={totals['tangent']/batches:.4f} pairs={totals['class_scale_pairs']/batches:.1f}",
                    flush=True,
                )
        if batches == 0:
            raise RuntimeError("No training batch completed.")
        row = {"epoch": float(epoch), "batches": float(batches), **{key: value / batches for key, value in totals.items()}}
        history.append(row)
        payload = checkpoint_payload(model, optimizer, scaler, epoch=epoch, config_path=config_path, source_path=source_path, history=history, smoke=bool(args.smoke_batches))
        atomic_torch_save(payload, output / "checkpoints" / "last.pt")
        print(json.dumps(row, sort_keys=True), flush=True)
    if not args.smoke_batches:
        atomic_torch_save(payload, output / "checkpoints" / "final.pt")


if __name__ == "__main__":
    main()
