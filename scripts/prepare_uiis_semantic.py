"""Audit and convert UIIS COCO polygons into semantic masks; no training."""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from datasets.uiis_semantic import rasterize_semantic_mask, validate_categories  # noqa: E402


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def process_partition(*, raw_root: Path, source_partition: str, output: Path) -> tuple[list[dict[str, object]], dict[str, int]]:
    document = load_json(raw_root / "annotations" / f"{source_partition}.json")
    validate_categories(document["categories"])
    annotations: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for annotation in document["annotations"]:
        annotations[int(annotation["image_id"])].append(annotation)
    rows: list[dict[str, object]] = []
    counters = {"images": 0, "empty_masks": 0, "bad_image_dimensions": 0, "annotations": 0, "polygon_parts": 0, "cross_class_overlap_pixels": 0, "same_class_overlap_pixels": 0}
    mask_dir = output / "semantic_masks" / source_partition
    mask_dir.mkdir(parents=True, exist_ok=True)
    for image_info in document["images"]:
        image_id = int(image_info["id"])
        image_path = raw_root / source_partition / str(image_info["file_name"])
        if not image_path.is_file():
            raise FileNotFoundError(image_path)
        with Image.open(image_path) as image:
            image_array = np.asarray(image.convert("RGB"), dtype=np.uint8)
        if image_array.shape[:2] != (int(image_info["height"]), int(image_info["width"])):
            counters["bad_image_dimensions"] += 1
            raise ValueError(f"COCO/image dimension mismatch: {image_path}")
        mask, image_counts = rasterize_semantic_mask(height=image_array.shape[0], width=image_array.shape[1], annotations=annotations[image_id])
        for key in image_counts:
            counters[key] += image_counts[key]
        if not np.any(mask):
            counters["empty_masks"] += 1
        sample_id = f"uiis_{source_partition}_{image_id:06d}"
        mask_path = mask_dir / f"{sample_id}.png"
        Image.fromarray(mask, mode="L").save(mask_path)
        classes, counts = np.unique(mask, return_counts=True)
        rows.append({
            "sample_id": sample_id, "source_partition": source_partition, "source_image_id": image_id,
            "image_path": str(image_path.relative_to(ROOT)), "mask_path": str(mask_path.relative_to(ROOT)),
            "image_sha256": sha256(image_path), "mask_sha256": sha256(mask_path),
            "width": int(image_array.shape[1]), "height": int(image_array.shape[0]),
            "annotation_count": len(annotations[image_id]), "present_classes": ";".join(map(str, classes.tolist())),
            **{f"pixels_class_{class_id}": int(counts[list(classes).index(class_id)]) if class_id in classes else 0 for class_id in range(8)},
        })
        counters["images"] += 1
    return rows, counters


def main() -> None:
    parser = argparse.ArgumentParser(description="Convert downloaded UIIS COCO instances to auditable semantic masks.")
    parser.add_argument("--raw-root", type=Path, default=ROOT / "data" / "uiis_raw")
    parser.add_argument("--output", type=Path, default=ROOT / "data" / "uiis_processed")
    args = parser.parse_args()
    raw_root, output = args.raw_root.resolve(), args.output.resolve()
    if not (raw_root / "annotations" / "train.json").is_file() or not (raw_root / "annotations" / "val.json").is_file():
        raise FileNotFoundError("UIIS train.json and val.json are required.")
    all_rows: list[dict[str, object]] = []
    summary: dict[str, dict[str, int]] = {}
    for partition in ("train", "val"):
        rows, counters = process_partition(raw_root=raw_root, source_partition=partition, output=output)
        all_rows.extend(rows)
        summary[partition] = counters
    manifest = pd.DataFrame(all_rows).sort_values("sample_id", kind="stable")
    if manifest["sample_id"].duplicated().any() or len(manifest) != 4628:
        raise AssertionError("UIIS semantic manifest must contain 4,628 unique images.")
    output.mkdir(parents=True, exist_ok=True)
    manifest.to_csv(output / "manifest.csv", index=False)
    total = {key: sum(part[key] for part in summary.values()) for key in next(iter(summary.values()))}
    metadata = {"source": "LiamLian0727/UIIS", "source_partitions": ["train", "val"], "images": len(manifest), "semantic_classes": 8, "background_class": 0, "uiis_category_to_suim_class": {"1": 6, "2": 5, "3": 2, "4": 3, "5": 1, "6": 4, "7": 7}, "overlap_policy": "descending_instance_area_then_annotation_id; later_smaller_instance_overwrites", "partitions": summary, "total": total, "official_suim_test_evaluated": False, "model_retrained": False}
    (output / "conversion_metadata.json").write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(metadata, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
