"""Render deterministic UIIS image/semantic-mask examples for manual audit."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from datasets.label_mapping import index_mask_to_rgb  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description="Render a deterministic UIIS semantic-mask audit sheet.")
    parser.add_argument("--manifest", type=Path, default=ROOT / "data" / "uiis_processed" / "manifest.csv")
    parser.add_argument("--output", type=Path, default=ROOT / "data" / "uiis_processed" / "quality_reports" / "uiis_semantic_examples.png")
    parser.add_argument("--samples", type=int, default=20)
    parser.add_argument("--seed", type=int, default=20260723)
    args = parser.parse_args()
    manifest = pd.read_csv(args.manifest)
    if args.samples <= 0 or args.samples > len(manifest):
        raise ValueError("Invalid requested sample count.")
    selected = manifest.sample(n=args.samples, random_state=args.seed).sort_values("sample_id", kind="stable")
    columns = 4
    rows = int(np.ceil(len(selected) / columns))
    figure, axes = plt.subplots(rows, columns, figsize=(4 * columns, 3 * rows), squeeze=False)
    for axis, item in zip(axes.ravel(), selected.itertuples(index=False), strict=False):
        with Image.open(ROOT / item.image_path) as image:
            rgb = np.asarray(image.convert("RGB"), dtype=np.uint8)
        with Image.open(ROOT / item.mask_path) as image:
            mask = np.asarray(image, dtype=np.uint8)
        overlay = (0.55 * rgb + 0.45 * index_mask_to_rgb(mask)).astype(np.uint8)
        axis.imshow(overlay)
        axis.set_title(item.sample_id, fontsize=8)
        axis.axis("off")
    for axis in axes.ravel()[len(selected):]:
        axis.axis("off")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    figure.tight_layout()
    figure.savefig(args.output, dpi=160)
    plt.close(figure)
    print(args.output)


if __name__ == "__main__":
    main()
