"""Train the preregistered AquaRiskMap head from complete inner-split caches.

The frozen segmentation backbones are not loaded or changed here.  This script
never accepts validation, calibration, or official TEST as a training source.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
import sys
import tempfile
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import pandas as pd
import torch
import yaml
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from reliability.aquariskmap import (  # noqa: E402
    CONDITIONS, IGNORE_INDEX, AquaRiskMap, atomic_torch_save, paired_batch_plan,
    ranking_loss, trajectory_loss, validate_cache_payload, weighted_error_bce,
)
from scripts.cache_aquariskmap_features import load_npz_payload, sha256  # noqa: E402


ALLOWED_MODELS = ("segformer", "deeplab")


def load_config(root: Path = ROOT) -> dict[str, Any]:
    return yaml.safe_load((root / "configs" / "aquariskmap_pilot.yaml").read_text(encoding="utf-8"))


def load_labels(root: Path, frame: pd.DataFrame) -> dict[str, torch.Tensor]:
    result: dict[str, torch.Tensor] = {}
    for row in frame.itertuples(index=False):
        with Image.open(root / str(row.mask_path)) as image:
            resized = image.resize((384, 384), resample=Image.Resampling.NEAREST)
            values = np.asarray(resized, dtype=np.uint8).copy()
        result[str(row.sample_id)] = torch.from_numpy(values).long()
    return result


def cache_root(root: Path, config: Mapping[str, Any], model_name: str, split: str) -> Path:
    if model_name not in ALLOWED_MODELS or split not in {"risk_head_train", "risk_head_development"}:
        raise ValueError("Only the preregistered model and inner split names are allowed.")
    return root / config["experiment"]["output_dir"] / "cache" / model_name / split


def split_frame(root: Path, config: Mapping[str, Any], split: str) -> pd.DataFrame:
    path = root / config["protocol"]["risk_head_split"] / f"{split}.csv"
    frame = pd.read_csv(path).sort_values("sample_id").reset_index(drop=True)
    if frame.empty or frame["sample_id"].duplicated().any() or frame["scene_group_id"].isna().any():
        raise ValueError("Inner split is not a valid unique scene-group split.")
    return frame


def validate_complete_cache(root: Path, config: Mapping[str, Any], *, model_name: str, split: str, frame: pd.DataFrame) -> Path:
    directory = cache_root(root, config, model_name, split)
    manifest_path = directory / "cache_manifest.json"
    if not manifest_path.is_file():
        raise FileNotFoundError(f"Missing cache manifest: {manifest_path}")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if int(manifest.get("samples", -1)) != len(frame) or manifest.get("split") != split or manifest.get("model_name") != model_name:
        raise ValueError("Cache manifest does not represent the complete requested inner split.")
    if any(bool(manifest.get(key, True)) for key in ("validation_evaluated", "calibration_evaluated", "official_suim_test_evaluated")):
        raise ValueError("Cache provenance records an impermissible evaluation.")
    expected_checkpoint = sha256(root / config["base_models"][model_name]["checkpoint"])
    expected_degradation = sha256(root / config["degradations"]["config"])
    if manifest.get("checkpoint_sha256") != expected_checkpoint or manifest.get("degradation_config_sha256") != expected_degradation:
        raise ValueError("Cache manifest hash does not match the frozen base assets.")
    cached = {path.stem for path in directory.glob("*.npz")}
    expected = set(frame["sample_id"])
    if cached != expected:
        raise ValueError("Cache files do not exactly match the preregistered inner split.")
    for row in frame.itertuples(index=False):
        payload = load_npz_payload(directory / f"{row.sample_id}.npz")
        validate_cache_payload(payload, split=split, model_name=model_name)
        if payload["scene_group_id"].item() != str(row.scene_group_id):
            raise ValueError("Cached scene group differs from the frozen split.")
        if payload["source_image_sha256"].item() != sha256(root / str(row.image_path)) or payload["source_mask_sha256"].item() != sha256(root / str(row.mask_path)):
            raise ValueError("Cached source hash differs from the admitted source pair.")
    return directory


class CacheReader:
    def __init__(self, directory: Path, *, split: str, model_name: str, labels: Mapping[str, torch.Tensor]) -> None:
        self.directory, self.split, self.model_name, self.labels = directory, split, model_name, labels
        self.condition_index = {name: index for index, name in enumerate(CONDITIONS)}

    def batch(self, records: list[tuple[str, str]], device: torch.device) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, list[str], list[str]]:
        files: dict[str, dict[str, object]] = {}
        features, predictions, labels, sample_ids, conditions = [], [], [], [], []
        for sample_id, condition in records:
            if condition not in self.condition_index:
                raise ValueError("Unknown fixed degradation condition.")
            files.setdefault(sample_id, load_npz_payload(self.directory / f"{sample_id}.npz"))
            payload = files[sample_id]
            index = self.condition_index[condition]
            features.append(torch.from_numpy(np.asarray(payload["features"])[index].copy()))
            predictions.append(torch.from_numpy(np.asarray(payload["predicted_class"])[index].copy()))
            labels.append(self.labels[sample_id])
            sample_ids.append(sample_id); conditions.append(condition)
        return (torch.stack(features).to(device, dtype=torch.float32), torch.stack(predictions).to(device), torch.stack(labels).to(device), sample_ids, conditions)


def positive_error_weight(reader: CacheReader, sample_ids: list[str]) -> float:
    errors, pixels = 0, 0
    for sample_id in sample_ids:
        payload = load_npz_payload(reader.directory / f"{sample_id}.npz")
        prediction = np.asarray(payload["predicted_class"])
        label = reader.labels[sample_id].numpy()
        valid = label != IGNORE_INDEX
        errors += int((prediction[:, valid] != label[valid]).sum())
        pixels += int(valid.sum()) * len(CONDITIONS)
    if errors == 0:
        return 10.0
    return float(np.clip((pixels - errors) / errors, 1.0, 10.0))


def capture_rng_state() -> dict[str, object]:
    return {"python": random.getstate(), "numpy": np.random.get_state(), "torch": torch.get_rng_state(), "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None}


def restore_rng_state(state: Mapping[str, object]) -> None:
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(torch.as_tensor(state["torch"], dtype=torch.uint8, device="cpu"))
    if state.get("cuda") is not None and torch.cuda.is_available():
        torch.cuda.set_rng_state_all([torch.as_tensor(value, dtype=torch.uint8, device="cpu") for value in state["cuda"]])


def checkpoint_payload(epoch: int, global_step: int, model: AquaRiskMap, optimizer: torch.optim.Optimizer, scaler: torch.amp.GradScaler, *, model_name: str, positive_weight: float) -> dict[str, object]:
    return {
        "checkpoint_format": "aquariskmap_pilot_v1", "epoch": epoch, "global_step": global_step,
        "model_state_dict": model.state_dict(), "optimizer_state_dict": optimizer.state_dict(), "scaler_state_dict": scaler.state_dict(),
        "rng_state": capture_rng_state(), "model_name": model_name, "positive_weight": positive_weight,
        "checkpoint_selection": "final_epoch", "validation_evaluated": False, "calibration_evaluated": False, "official_suim_test_evaluated": False,
    }


def load_completed_epoch_checkpoint(path: Path, model: AquaRiskMap, optimizer: torch.optim.Optimizer, scaler: torch.amp.GradScaler, device: torch.device) -> tuple[int, int, float]:
    checkpoint = torch.load(path, map_location=device, weights_only=False)
    if checkpoint.get("checkpoint_format") != "aquariskmap_pilot_v1" or any(bool(checkpoint.get(key, True)) for key in ("validation_evaluated", "calibration_evaluated", "official_suim_test_evaluated")):
        raise ValueError("Checkpoint does not satisfy the locked AquaRiskMap protocol.")
    model.load_state_dict(checkpoint["model_state_dict"]); optimizer.load_state_dict(checkpoint["optimizer_state_dict"]); scaler.load_state_dict(checkpoint["scaler_state_dict"])
    restore_rng_state(checkpoint["rng_state"])
    return int(checkpoint["epoch"]) + 1, int(checkpoint["global_step"]), float(checkpoint["positive_weight"])


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True, choices=ALLOWED_MODELS)
    parser.add_argument("--resume", type=Path)
    parser.add_argument("--smoke-steps", type=int, default=0)
    args = parser.parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for AquaRiskMap training.")
    config = load_config(ROOT)
    if not config["protocol"]["official_suim_test_locked"] or config["protocol"]["official_suim_test_evaluated"]:
        raise ValueError("Official SUIM TEST must remain locked.")
    torch.manual_seed(int(config["experiment"]["seed"])); np.random.seed(int(config["experiment"]["seed"])); random.seed(int(config["experiment"]["seed"]))
    frame = split_frame(ROOT, config, "risk_head_train")
    directory = validate_complete_cache(ROOT, config, model_name=args.model, split="risk_head_train", frame=frame)
    labels = load_labels(ROOT, frame)
    reader = CacheReader(directory, split="risk_head_train", model_name=args.model, labels=labels)
    weight = positive_error_weight(reader, frame["sample_id"].tolist())
    device = torch.device("cuda"); model = AquaRiskMap().to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=float(config["training"]["learning_rate"]), weight_decay=float(config["training"]["weight_decay"]))
    scaler = torch.amp.GradScaler("cuda", enabled=bool(config["training"]["amp"]))
    output = ROOT / config["experiment"]["output_dir"] / args.model
    if args.smoke_steps:
        output = output / "smoke"
    checkpoint_path = output / "checkpoints" / "last.pt"
    start_epoch, global_step = 1, 0
    if args.resume:
        start_epoch, global_step, restored_weight = load_completed_epoch_checkpoint(args.resume, model, optimizer, scaler, device)
        if not np.isclose(restored_weight, weight):
            raise ValueError("Resumed checkpoint positive weight differs from frozen risk-head train cache.")
    history: list[dict[str, object]] = []
    epochs = int(config["training"]["epochs"])
    for epoch in range(start_epoch, epochs + 1):
        model.train(); sums = {"bce": 0.0, "rank": 0.0, "trajectory": 0.0, "total": 0.0}; batches = 0
        for batch_index, records in enumerate(paired_batch_plan(frame["sample_id"], epoch=epoch, base_scenes_per_batch=int(config["training"]["base_scenes_per_batch"]))):
            features, prediction, label, sample_ids, conditions = reader.batch(records, device)
            errors = prediction.ne(label)
            optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast("cuda", enabled=bool(config["training"]["amp"])):
                risk = model(features)
                full = torch.nn.functional.interpolate(risk.float(), size=label.shape[-2:], mode="bilinear", align_corners=False)
                bce = weighted_error_bce(full, errors, label, positive_weight=weight)
                rank = ranking_loss(full, errors, label, epoch=epoch, batch_index=batch_index, pairs=int(config["loss"]["ranking_pairs_per_batch"]))
                trajectory = trajectory_loss(full[::2], full[1::2], errors[::2], errors[1::2], label[::2], sample_ids[::2], sample_ids[1::2], margin=float(config["loss"]["trajectory_margin"]))
                total = bce + float(config["loss"]["ranking_weight"]) * rank + float(config["loss"]["trajectory_weight"]) * trajectory
            if not torch.isfinite(total):
                raise FloatingPointError("AquaRiskMap loss became non-finite.")
            scaler.scale(total).backward(); scaler.step(optimizer); scaler.update(); global_step += 1; batches += 1
            for key, value in (("bce", bce), ("rank", rank), ("trajectory", trajectory), ("total", total)):
                sums[key] += float(value.detach())
            if args.smoke_steps and global_step >= args.smoke_steps:
                atomic_torch_save(checkpoint_payload(epoch, global_step, model, optimizer, scaler, model_name=args.model, positive_weight=weight), checkpoint_path)
                print(json.dumps({"smoke_steps": global_step, "losses": {key: value / batches for key, value in sums.items()}, "official_suim_test_evaluated": False}), flush=True)
                return
        history.append({"epoch": epoch, "global_step": global_step, "positive_weight": weight, **{key: value / batches for key, value in sums.items()}, "validation_evaluated": False, "calibration_evaluated": False, "official_suim_test_evaluated": False})
        output.mkdir(parents=True, exist_ok=True)
        temporary = output / "train_history.json.tmp"; temporary.write_text(json.dumps(history, indent=2) + "\n", encoding="utf-8"); os.replace(temporary, output / "train_history.json")
        atomic_torch_save(checkpoint_payload(epoch, global_step, model, optimizer, scaler, model_name=args.model, positive_weight=weight), checkpoint_path)
        print(json.dumps(history[-1]), flush=True)


if __name__ == "__main__":
    main()
