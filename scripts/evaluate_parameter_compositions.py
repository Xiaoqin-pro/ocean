"""Evaluate frozen/SADR/partial-FT/full-FT checkpoints on ordered compositions."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
import yaml
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from datasets.suim_dataset import SUIMDataset, build_eval_transform  # noqa: E402
from degradations.registry import build_image_degradation, load_conditions  # noqa: E402
from scripts.evaluate_parameter_efficiency import load_models  # noqa: E402


FAMILIES = ("color", "turbidity", "lowlight", "blur")
PAIRS = (("color", "turbidity"), ("lowlight", "blur"), ("color", "lowlight"), ("turbidity", "blur"))


def compose(registry: dict[str, object], names: tuple[str, ...]):
    transforms = [build_image_degradation(registry[name]) for name in names]

    def apply(image: np.ndarray, sample_id: str) -> np.ndarray:
        result = image.copy()
        for transform in transforms:
            result = transform(result, sample_id)
        return result

    return apply


def miou(confusion: torch.Tensor) -> float:
    matrix = confusion.cpu().numpy().astype(np.float64)
    diagonal = np.diag(matrix)
    denominator = matrix.sum(0) + matrix.sum(1) - diagonal
    valid = denominator > 0
    return float(np.divide(diagonal, denominator, out=np.zeros_like(diagonal), where=valid)[valid].mean())


def evaluate(model: torch.nn.Module, csv_path: Path, registry: dict[str, object], image_size: int, pair: tuple[str, str], device: torch.device) -> list[dict[str, object]]:
    rows = []
    for order, names in (("AB", (f"{pair[0]}_s2", f"{pair[1]}_s2")), ("BA", (f"{pair[1]}_s2", f"{pair[0]}_s2"))):
        dataset = SUIMDataset(csv_path, transform=build_eval_transform(image_size), image_degradation=compose(registry, names))
        loader = DataLoader(dataset, batch_size=8, shuffle=False, num_workers=0, pin_memory=True)
        raw_confusion = torch.zeros(8, 8, dtype=torch.int64, device=device)
        corrected_confusion = torch.zeros(8, 8, dtype=torch.int64, device=device)
        with torch.no_grad():
            for batch in loader:
                pixels = batch["pixel_values"].to(device, non_blocking=True)
                labels = batch["labels"].to(device, non_blocking=True)
                with torch.autocast(device_type="cuda", dtype=torch.float16):
                    output = model(pixel_values=pixels)
                    logits = output.logits
                corrected = F.interpolate(logits.float(), size=labels.shape[-2:], mode="bilinear", align_corners=False).argmax(1)
                valid = labels.ne(255)
                raw_indices = labels[valid] * 8 + corrected[valid]
                corrected_confusion += torch.bincount(raw_indices, minlength=64).reshape(8, 8)
        rows.append({"order": order, "miou": miou(corrected_confusion), "pair": "+".join(pair), "images": len(dataset)})
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--split", choices=("confirmation", "calibration"), default="confirmation")
    parser.add_argument("--config", type=Path, default=ROOT / "configs/parameter_efficiency.yaml")
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    csv_path = ROOT / f"data/uiis_processed/splits/uiis_alpha010_confirmation/{args.split}.csv"
    expected = {"confirmation": 511, "calibration": 508}[args.split]
    if len(pd.read_csv(csv_path)) != expected:
        raise PermissionError(f"Unexpected {args.split} split")
    config = yaml.safe_load(args.config.resolve().read_text(encoding="utf-8"))
    uiis_config = yaml.safe_load((ROOT / str(config["data"]["uiis_config"])).read_text(encoding="utf-8"))
    sadr_config = yaml.safe_load((ROOT / "configs/sadr_long.yaml").read_text(encoding="utf-8"))
    registry = {condition.name: condition for condition in load_conditions(ROOT / str(config["data"]["degradation_registry"]))}
    device = torch.device("cuda")
    models = load_models(uiis_config, sadr_config, device)
    rows = []
    for model_name, model in models.items():
        for pair in PAIRS:
            for row in evaluate(model, csv_path, registry, 384, pair, device):
                rows.append({"variant": model_name, **row})
                print(f"{args.split}/{model_name}/{row['pair']}/{row['order']}: {row['miou']:.4f}", flush=True)
        del model
        torch.cuda.empty_cache()
    frame = pd.DataFrame(rows)
    baseline = frame[frame.variant == "frozen"].set_index(["pair", "order"]).miou
    frame["gain_vs_frozen_pp"] = [100.0 * (value - baseline.loc[(pair, order)]) for value, pair, order in zip(frame.miou, frame.pair, frame.order, strict=True)]
    summary = frame.groupby("variant", sort=False).gain_vs_frozen_pp.mean().to_dict()
    output = args.output or ROOT / "outputs/parameter_efficiency" / f"{args.split}_composition_evaluation"
    output.mkdir(parents=True, exist_ok=True)
    frame.to_csv(output / "condition_metrics.csv", index=False)
    result = {"split": args.split, "images": expected, "pairs": ["+".join(pair) for pair in PAIRS], "mean_gain_vs_frozen_pp": {key: float(value) for key, value in summary.items()}, "note": "Post-freeze ordered-composition diagnostic; composition uses matched severity-2 registry transforms."}
    (output / "summary.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2), flush=True)


if __name__ == "__main__":
    main()
