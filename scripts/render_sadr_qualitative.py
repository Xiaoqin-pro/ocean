"""Render qualitative degraded/restored pairs for the SADR paper figure."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from datasets.suim_dataset import IMAGENET_MEAN, IMAGENET_STD, SUIMDataset, build_eval_transform  # noqa: E402
from degradations.registry import build_image_degradation, load_conditions  # noqa: E402
from scripts.train_sadr import build_model  # noqa: E402


def denormalize(value: torch.Tensor) -> np.ndarray:
    array = value.detach().cpu().permute(1, 2, 0).numpy()
    array = array * np.asarray(IMAGENET_STD, dtype=np.float32) + np.asarray(IMAGENET_MEAN, dtype=np.float32)
    return np.clip(array, 0.0, 1.0)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=ROOT / "configs/sadr_long.yaml")
    parser.add_argument("--indices", type=int, nargs="+", default=[0, 7, 20])
    parser.add_argument("--conditions", nargs="+", default=["color_s3", "lowlight_s3", "blur_s3"])
    args = parser.parse_args()
    config = yaml.safe_load(args.config.resolve().read_text(encoding="utf-8"))
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required.")
    device = torch.device("cuda")
    _, front = build_model(config, device)
    payload = torch.load(ROOT / str(config["experiment"]["output_dir"]) / "formal/checkpoints/final.pt", map_location=device, weights_only=False)
    front.load_state_dict(payload["model_state_dict"]); front.eval()
    registry = {item.name: item for item in load_conditions(ROOT / str(config["data"]["degradation_registry"]))}
    csv_path = ROOT / str(config["data"]["confirmation_csv"])
    datasets = {name: SUIMDataset(csv_path, transform=build_eval_transform(int(config["data"]["evaluation_image_size"])), image_degradation=build_image_degradation(registry[name])) for name in args.conditions}
    fig, axes = plt.subplots(len(args.indices), 2 * len(args.conditions), figsize=(3.0 * len(args.conditions), 2.5 * len(args.indices)), squeeze=False)
    with torch.no_grad():
        for row, index in enumerate(args.indices):
            for col, name in enumerate(args.conditions):
                sample = datasets[name][index]
                pixels = sample["pixel_values"].unsqueeze(0).to(device)
                restored = front(pixels)[0][0]
                axes[row, 2 * col].imshow(denormalize(sample["pixel_values"])); axes[row, 2 * col].set_title(f"{name}\ndegraded")
                axes[row, 2 * col + 1].imshow(denormalize(restored)); axes[row, 2 * col + 1].set_title("SADR restored")
                axes[row, 2 * col].axis("off"); axes[row, 2 * col + 1].axis("off")
    fig.tight_layout()
    output = ROOT / str(config["experiment"]["output_dir"]) / "qualitative_grid.png"
    output.parent.mkdir(parents=True, exist_ok=True); fig.savefig(output, dpi=180, bbox_inches="tight"); print(output)


if __name__ == "__main__": main()
