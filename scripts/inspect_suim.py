from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from datasets.label_mapping import index_mask_to_rgb  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description="Save random SUIM image/mask/overlay inspection pages.")
    parser.add_argument("--split", type=Path, default=PROJECT_ROOT / "data" / "suim_processed" / "splits" / "v1_seed_20260721" / "train.csv")
    parser.add_argument("--count", type=int, default=100)
    parser.add_argument("--seed", type=int, default=20260721)
    parser.add_argument("--output-dir", type=Path, default=PROJECT_ROOT / "outputs" / "data_inspection")
    args = parser.parse_args()

    data = pd.read_csv(args.split)
    count = min(args.count, len(data))
    if count <= 0:
        raise ValueError("Inspection split is empty.")
    sampled = data.sample(n=count, random_state=args.seed).reset_index(drop=True)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    for page_number, start in enumerate(range(0, count, 10), 1):
        page = sampled.iloc[start : start + 10]
        figure, axes = plt.subplots(len(page), 3, figsize=(12, 4 * len(page)), squeeze=False)
        for row_axes, (_, sample) in zip(axes, page.iterrows()):
            with Image.open(PROJECT_ROOT / sample.image_path) as image:
                image_array = np.asarray(image.convert("RGB"))
            with Image.open(PROJECT_ROOT / sample.mask_path) as mask:
                index_mask = np.asarray(mask, dtype=np.uint8)
            rgb_mask = index_mask_to_rgb(index_mask)
            overlay = np.clip(0.6 * image_array + 0.4 * rgb_mask, 0, 255).astype(np.uint8)
            for axis, content, title in zip(row_axes, (image_array, rgb_mask, overlay), (f"{sample.sample_id}: image", "mask", "overlay")):
                axis.imshow(content)
                axis.set_title(title)
                axis.axis("off")
        figure.tight_layout()
        output = args.output_dir / f"inspection_page_{page_number:02d}.png"
        figure.savefig(output, dpi=150, bbox_inches="tight")
        plt.close(figure)
        print(output)


if __name__ == "__main__":
    main()
