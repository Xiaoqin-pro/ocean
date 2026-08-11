"""Measure additive and order-consistent correction structure of SADR."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import yaml
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from datasets.suim_dataset import SUIMDataset, build_eval_transform  # noqa: E402
from degradations.registry import build_image_degradation, load_conditions  # noqa: E402
from scripts.evaluate_sadr import SADRModel  # noqa: E402
from scripts.train_sadr import FORMAT, build_model  # noqa: E402


def compose_degradations(registry: dict[str, object], names: tuple[str, ...]):
    transforms = [build_image_degradation(registry[name]) for name in names]

    def apply(image: np.ndarray, sample_id: str) -> np.ndarray:
        result = image.copy()
        for transform in transforms:
            result = transform(result, sample_id)
        return result

    return apply


def correction(base: torch.nn.Module, front: torch.nn.Module, pixels: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    with torch.no_grad(), torch.autocast(device_type="cuda", dtype=torch.float16):
        raw = base(pixel_values=pixels).logits.float()
        restored, residual = front(pixels)
        corrected = base(pixel_values=restored).logits.float()
    return corrected - raw, residual.float()


def relation_metrics(left: torch.Tensor, right: torch.Tensor) -> dict[str, float]:
    left_flat = left.flatten(1); right_flat = right.flatten(1)
    cosine = torch.nn.functional.cosine_similarity(left_flat, right_flat, dim=1)
    relative_error = (left_flat - right_flat).norm(dim=1) / (left_flat.norm(dim=1) + right_flat.norm(dim=1) + 1e-6)
    return {"cosine": float(cosine.mean()), "relative_error": float(relative_error.mean())}


def build_variant(config_path: Path, device: torch.device) -> tuple[torch.nn.Module, torch.nn.Module]:
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    payload = torch.load(ROOT / str(config["experiment"]["output_dir"]) / "formal/checkpoints/final.pt", map_location=device, weights_only=False)
    if payload.get("checkpoint_format") != FORMAT or payload.get("epoch") != int(config["training"]["epochs"]):
        raise ValueError(f"Invalid checkpoint for {config_path}.")
    if any(bool(payload.get(key, True)) for key in ("calibration_evaluated", "confirmation_evaluated", "official_suim_test_evaluated")):
        raise ValueError(f"Checkpoint records prohibited evaluation access: {config_path}")
    base, front = build_model(config, device); front.load_state_dict(payload["model_state_dict"]); base.eval(); front.eval()
    return base, front


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=ROOT / "configs/sadr_long.yaml")
    parser.add_argument("--pixel-config", type=Path, default=ROOT / "configs/sadr_pixel_only.yaml")
    parser.add_argument("--sadr-only", action="store_true", help="Skip the pixel-only control variant.")
    parser.add_argument("--max-images", type=int, default=128)
    parser.add_argument("--output", type=Path, default=ROOT / "outputs/sadr_long/compositionality_analysis.json")
    args = parser.parse_args()
    config = yaml.safe_load(args.config.resolve().read_text(encoding="utf-8")); data = config["data"]
    csv_path = ROOT / str(data["confirmation_csv"]); frame = pd.read_csv(csv_path)
    if csv_path.name != "confirmation.csv" or len(frame) != 511:
        raise PermissionError("Compositionality analysis accepts only the frozen confirmation split.")
    if args.max_images <= 0 or args.max_images > len(frame):
        raise ValueError("max-images must be in [1, 511].")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required.")
    device = torch.device("cuda")
    registry = {condition.name: condition for condition in load_conditions(ROOT / str(data["degradation_registry"]))}
    conditions = {
        "clean": (),
        "A": ("color_s2",),
        "B": ("turbidity_s2",),
        "C": ("lowlight_s2",),
        "AB": ("color_s2", "turbidity_s2"),
        "BA": ("turbidity_s2", "color_s2"),
        "AA": ("color_s2", "color_s2"),
        "ABC": ("color_s2", "turbidity_s2", "lowlight_s2"),
    }
    loaders = {
        name: DataLoader(
            SUIMDataset(csv_path, transform=build_eval_transform(int(data["evaluation_image_size"])), image_degradation=compose_degradations(registry, members) if members else None),
            batch_size=8, shuffle=False, num_workers=0, pin_memory=True,
        ) for name, members in conditions.items()
    }
    variant_configs = (("SADR", args.config.resolve()),) if args.sadr_only else (("SADR", args.config.resolve()), ("pixel_only", args.pixel_config.resolve()))
    rows = []
    for variant_name, variant_config in variant_configs:
        base, front = build_variant(variant_config, device)
        accum = {key: [] for key in ("add_AB_logits", "add_AB_residual", "order_logits", "order_residual", "self_logits", "self_residual", "triple_logits", "triple_residual")}
        iterators = {name: iter(loader) for name, loader in loaders.items()}
        seen = 0
        while seen < args.max_images:
            batches = {name: next(iterator) for name, iterator in iterators.items()}
            batch_size = batches["clean"]["pixel_values"].shape[0]
            take = min(batch_size, args.max_images - seen)
            pixels = {name: batches[name]["pixel_values"][:take].to(device, non_blocking=True) for name in batches}
            corrections = {name: correction(base, front, value) for name, value in pixels.items()}
            add_delta = corrections["AB"][0]; add_residual = corrections["AB"][1]
            accum["add_AB_logits"].append(relation_metrics(add_delta, corrections["A"][0] + corrections["B"][0]))
            accum["add_AB_residual"].append(relation_metrics(add_residual, corrections["A"][1] + corrections["B"][1]))
            accum["order_logits"].append(relation_metrics(corrections["AB"][0], corrections["BA"][0]))
            accum["order_residual"].append(relation_metrics(corrections["AB"][1], corrections["BA"][1]))
            accum["self_logits"].append(relation_metrics(corrections["AA"][0], 2.0 * corrections["A"][0]))
            accum["self_residual"].append(relation_metrics(corrections["AA"][1], 2.0 * corrections["A"][1]))
            triple_delta, triple_residual = corrections["ABC"]
            accum["triple_logits"].append(relation_metrics(triple_delta, corrections["A"][0] + corrections["B"][0] + corrections["C"][0]))
            accum["triple_residual"].append(relation_metrics(triple_residual, corrections["A"][1] + corrections["B"][1] + corrections["C"][1]))
            seen += take
        summary = {key: {metric: float(np.mean([row[metric] for row in values])) for metric in ("cosine", "relative_error")} for key, values in accum.items()}
        rows.append({"variant": variant_name, "images": seen, "relations": summary}); print(json.dumps(rows[-1], sort_keys=True), flush=True)
        del base, front
        torch.cuda.empty_cache()
    payload = {"split": "confirmation_compositionality_screen", "images": args.max_images, "rows": rows, "note": "Post-freeze diagnostic on the confirmation images; not an independent generalization split."}
    args.output.resolve().parent.mkdir(parents=True, exist_ok=True); args.output.resolve().write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__": main()
