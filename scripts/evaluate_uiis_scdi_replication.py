"""Evaluate UIIS F4/SCDI replication only on the locked confirmation split."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd
import torch
import torch.nn.functional as functional
import yaml
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from datasets.suim_dataset import SUIMDataset, build_eval_transform  # noqa: E402
from degradations.registry import build_image_degradation, load_conditions  # noqa: E402
from scripts.train_uiis_scdi_replication import FORMAT, build_models  # noqa: E402


def evaluate(model, csv_path: Path, condition, image_size: int, device: torch.device) -> dict:
    dataset = SUIMDataset(csv_path, transform=build_eval_transform(image_size), image_degradation=build_image_degradation(condition))
    loader = DataLoader(dataset, batch_size=8, shuffle=False, num_workers=0, pin_memory=True)
    confusion = torch.zeros(8, 8, dtype=torch.int64, device=device)
    with torch.no_grad():
        for batch in loader:
            pixels = batch["pixel_values"].to(device); labels = batch["labels"].to(device)
            with torch.autocast(device_type="cuda", dtype=torch.float16):
                output = model(pixel_values=pixels) if isinstance(model, torch.nn.Module) and not hasattr(model, "rectifiers") else model(pixels)
                logits = output.logits
            prediction = functional.interpolate(logits.float(), size=labels.shape[-2:], mode="bilinear", align_corners=False).argmax(1)
            valid = labels.ne(255); indices = labels[valid] * 8 + prediction[valid]
            confusion += torch.bincount(indices, minlength=64).reshape(8, 8)
    matrix = confusion.cpu().numpy(); diagonal = matrix.diagonal(); denominator = matrix.sum(0) + matrix.sum(1) - diagonal
    valid = denominator > 0; per_class = diagonal / denominator.clip(min=1)
    return {"condition": condition.name, "severity": int(condition.severity), "degradation_type": condition.degradation_type, "miou": float(per_class[valid].mean()), "per_class_iou": [float(value) if valid[index] else None for index, value in enumerate(per_class)]}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__); parser.add_argument("--config", type=Path, default=ROOT / "configs/uiis_scdi_replication.yaml"); args = parser.parse_args()
    config = yaml.safe_load(args.config.read_text(encoding="utf-8")); data = config["data"]
    confirmation = ROOT / str(data["confirmation_csv"])
    if confirmation.name != "confirmation.csv" or len(pd.read_csv(confirmation)) != 511: raise PermissionError("Only the frozen UIIS confirmation role is permitted.")
    if not torch.cuda.is_available(): raise RuntimeError("CUDA required.")
    device = torch.device("cuda"); conditions = [item for item in load_conditions(ROOT / str(data["degradation_registry"])) if item.name in set(config["evaluation"]["conditions"])]
    output = ROOT / str(config["experiment"]["output_dir"]) / "confirmation_evaluation"; output.mkdir(parents=True, exist_ok=True); rows=[]
    for variant in ("F4", "SCDI"):
        checkpoint = ROOT / str(config["experiment"]["output_dir"]) / "formal" / variant / "checkpoints" / "final.pt"
        payload = torch.load(checkpoint, map_location=device, weights_only=False)
        if payload.get("checkpoint_format") != FORMAT or payload.get("variant") != variant or int(payload.get("epoch", 0)) != int(config["training"]["epochs"]) or payload.get("smoke"): raise ValueError(f"Invalid {variant} checkpoint.")
        if payload.get("confirmation_evaluated") or payload.get("official_suim_test_evaluated"): raise ValueError("Checkpoint records prohibited evaluation.")
        model = build_models(config, variant, device); model.load_state_dict(payload["model_state_dict"]); model.eval()
        for condition in conditions:
            row = {"variant": variant, **evaluate(model, confirmation, condition, int(data["image_size"]), device)}; rows.append(row); print(f"{variant}/{condition.name}: mIoU={row['miou']:.4f}", flush=True)
        del model; torch.cuda.empty_cache()
    frame = pd.DataFrame(rows); frame.to_csv(output / "condition_metrics.csv", index=False); summary = {"confirmation_evaluated": True, "official_suim_test_evaluated": False, "rows": len(frame), "mean_miou": {variant: float(frame[frame.variant==variant].miou.mean()) for variant in frame.variant.unique()}}
    (output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8"); print(json.dumps(summary, indent=2))


if __name__ == "__main__": main()
