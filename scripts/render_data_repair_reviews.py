"""Render deterministic human-review panels for SUIM label repairs."""
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

from datasets.label_mapping import RGB_TO_CLASS, index_mask_to_rgb, rgb_mask_to_index_bit_threshold  # noqa: E402


def raw_mask_path(partition: str, sample_id: str) -> Path:
    source = "TEST" if partition == "test" else "train_val"
    return PROJECT_ROOT / "data" / "suim_raw" / source / "masks" / f"{sample_id}.bmp"


def quantized_locations(raw_rgb: np.ndarray) -> np.ndarray:
    exact = np.zeros(raw_rgb.shape[:2], dtype=bool)
    for color in RGB_TO_CLASS:
        exact |= np.all(raw_rgb == np.asarray(color, dtype=np.uint8), axis=-1)
    return ~exact


def overlay(image: np.ndarray, mask: np.ndarray, alpha: float = 0.45) -> np.ndarray:
    return (image.astype(np.float32) * (1 - alpha) + index_mask_to_rgb(mask).astype(np.float32) * alpha).astype(np.uint8)


def save_quantization_panels(manifest: pd.DataFrame, output_dir: Path, top_k: int) -> None:
    target = output_dir / "quantization_top20"
    target.mkdir(parents=True, exist_ok=True)
    for rank, row in enumerate(manifest.sort_values("quantized_ratio", ascending=False).head(top_k).itertuples(index=False), start=1):
        with Image.open(PROJECT_ROOT / row.image_path) as image:
            image_array = np.asarray(image.convert("RGB"), dtype=np.uint8)
        with Image.open(raw_mask_path(row.partition, row.sample_id)) as mask:
            raw_rgb = np.asarray(mask.convert("RGB"), dtype=np.uint8)
        indexed, _ = rgb_mask_to_index_bit_threshold(raw_rgb)
        if indexed.shape != image_array.shape[:2]:
            indexed = np.asarray(Image.fromarray(indexed).resize((image_array.shape[1], image_array.shape[0]), Image.Resampling.NEAREST))
            raw_rgb = np.asarray(Image.fromarray(raw_rgb).resize((image_array.shape[1], image_array.shape[0]), Image.Resampling.NEAREST))
        changed = quantized_locations(raw_rgb)
        location_map = np.zeros((*changed.shape, 3), dtype=np.uint8)
        location_map[changed] = (255, 255, 255)
        figure, axes = plt.subplots(1, 4, figsize=(16, 4))
        panels = [(raw_rgb, "Raw RGB mask"), (index_mask_to_rgb(indexed), "Thresholded class mask"), (overlay(image_array, indexed), "Image + class overlay"), (location_map, "Quantized locations")]
        for axis, (panel, title) in zip(axes, panels):
            axis.imshow(panel)
            axis.set_title(title)
            axis.axis("off")
        figure.suptitle(f"#{rank:02d} {row.sample_id} | quantized ratio={row.quantized_ratio:.2%}")
        figure.tight_layout()
        figure.savefig(target / f"{rank:02d}_{row.sample_id}.png", dpi=160, bbox_inches="tight")
        plt.close(figure)


def save_size_repair_panels(manifest: pd.DataFrame, output_dir: Path) -> None:
    target = output_dir / "size_repair_37"
    target.mkdir(parents=True, exist_ok=True)
    repaired = manifest.loc[manifest["size_repaired"]].sort_values("sample_id")
    for row in repaired.itertuples(index=False):
        with Image.open(PROJECT_ROOT / row.image_path) as image:
            image_array = np.asarray(image.convert("RGB"), dtype=np.uint8)
        with Image.open(raw_mask_path(row.partition, row.sample_id)) as mask:
            raw_rgb = np.asarray(mask.convert("RGB"), dtype=np.uint8)
        raw_index, _ = rgb_mask_to_index_bit_threshold(raw_rgb)
        excess = raw_index.shape[0] - image_array.shape[0]
        if excess <= 0 or raw_index.shape[1] != image_array.shape[1]:
            raise ValueError(f"Unexpected size-repair geometry for {row.sample_id}: {raw_index.shape} vs {image_array.shape}")
        top_crop = raw_index[excess:, :]
        bottom_crop = raw_index[:-excess, :]
        resized = np.asarray(Image.fromarray(raw_index).resize((image_array.shape[1], image_array.shape[0]), Image.Resampling.NEAREST))
        figure, axes = plt.subplots(1, 4, figsize=(16, 4))
        panels = [(image_array, "Image"), (overlay(image_array, top_crop), f"Top crop ({excess}px)"), (overlay(image_array, bottom_crop), f"Bottom crop ({excess}px)"), (overlay(image_array, resized), "Nearest resize")]
        for axis, (panel, title) in zip(axes, panels):
            axis.imshow(panel)
            axis.set_title(title)
            axis.axis("off")
        figure.suptitle(f"{row.sample_id} | raw mask {raw_index.shape[1]}x{raw_index.shape[0]} -> image {image_array.shape[1]}x{image_array.shape[0]}")
        figure.tight_layout()
        figure.savefig(target / f"{row.sample_id}.png", dpi=160, bbox_inches="tight")
        plt.close(figure)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, default=PROJECT_ROOT / "data" / "suim_processed" / "manifest.csv")
    parser.add_argument("--output-dir", type=Path, default=PROJECT_ROOT / "data" / "suim_processed" / "quality_reports" / "manual_review")
    parser.add_argument("--top-k", type=int, default=20)
    args = parser.parse_args()
    manifest = pd.read_csv(args.manifest)
    if "quantized_ratio" not in manifest or "size_repaired" not in manifest:
        raise ValueError("Run report_data_quality.py before rendering review panels.")
    save_quantization_panels(manifest, args.output_dir, args.top_k)
    save_size_repair_panels(manifest, args.output_dir)
    print(f"Wrote {args.top_k} quantization panels and {int(manifest['size_repaired'].sum())} size-repair panels to {args.output_dir}")


if __name__ == "__main__":
    main()
