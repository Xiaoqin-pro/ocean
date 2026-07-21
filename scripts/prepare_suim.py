from __future__ import annotations

import argparse
import csv
import shutil
import sys
from pathlib import Path

import numpy as np
from PIL import Image

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from datasets.label_mapping import rgb_mask_to_index, rgb_mask_to_index_bit_threshold  # noqa: E402


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp"}
EXPECTED_COUNTS = {"train_val": 1525, "test": 110}


def collect_files(directory: Path) -> dict[str, Path]:
    if not directory.is_dir():
        raise FileNotFoundError(f"Directory does not exist: {directory}")
    files: dict[str, Path] = {}
    for path in directory.iterdir():  # Intentional: do not read TEST mask subfolders.
        if not path.is_file() or path.suffix.lower() not in IMAGE_EXTENSIONS:
            continue
        if path.stem in files:
            raise RuntimeError(f"Duplicate stem {path.stem!r} in {directory}")
        files[path.stem] = path
    return files


def process_partition(
    name: str, raw_partition: Path, output_root: Path, mask_policy: str, size_policy: str
) -> list[dict[str, str | int]]:
    images = collect_files(raw_partition / "images")
    masks = collect_files(raw_partition / "masks")
    image_ids, mask_ids = set(images), set(masks)
    missing_masks, missing_images = sorted(image_ids - mask_ids), sorted(mask_ids - image_ids)
    if missing_masks or missing_images:
        raise RuntimeError(
            f"{name} pairing failed: {len(missing_masks)} images without masks, "
            f"{len(missing_images)} masks without images. Examples: "
            f"{missing_masks[:5]}, {missing_images[:5]}"
        )
    if len(image_ids) != EXPECTED_COUNTS[name]:
        raise RuntimeError(f"{name}: expected {EXPECTED_COUNTS[name]} pairs, found {len(image_ids)}")

    image_output = output_root / "images" / name
    mask_output = output_root / "masks" / name
    image_output.mkdir(parents=True, exist_ok=False)
    mask_output.mkdir(parents=True, exist_ok=False)
    records: list[dict[str, str | int]] = []
    for number, sample_id in enumerate(sorted(image_ids), 1):
        image_path, mask_path = images[sample_id], masks[sample_id]
        with Image.open(image_path) as image:
            image_rgb = image.convert("RGB")
            width, height = image_rgb.size
        with Image.open(mask_path) as mask:
            mask_rgb = mask.convert("RGB")
            original_mask_width, original_mask_height = mask_rgb.size
            if mask_rgb.size != (width, height):
                if size_policy == "strict":
                    raise ValueError(f"{sample_id}: image is {(width, height)}, mask is {mask_rgb.size}")
                mask_resized = 1
            else:
                mask_resized = 0
            if mask_policy == "exact":
                indexed = rgb_mask_to_index(mask_rgb, strict=True)
                non_exact_pixels = 0
            else:
                indexed, non_exact_pixels = rgb_mask_to_index_bit_threshold(mask_rgb)
            if mask_resized:
                indexed = np.asarray(
                    Image.fromarray(indexed).resize((width, height), resample=Image.Resampling.NEAREST),
                    dtype=np.uint8,
                )
        classes = np.unique(indexed)
        if np.any(classes > 7):
            raise ValueError(f"{sample_id}: invalid class ids {classes.tolist()}")

        destination_image = image_output / f"{sample_id}{image_path.suffix.lower()}"
        destination_mask = mask_output / f"{sample_id}.png"
        shutil.copy2(image_path, destination_image)
        Image.fromarray(indexed).save(destination_mask)
        records.append({
            "partition": name,
            "sample_id": sample_id,
            "image_path": destination_image.relative_to(PROJECT_ROOT).as_posix(),
            "mask_path": destination_mask.relative_to(PROJECT_ROOT).as_posix(),
            "width": width,
            "height": height,
            "original_mask_width": original_mask_width,
            "original_mask_height": original_mask_height,
            "mask_resized_nearest": mask_resized,
            "classes_present": "|".join(map(str, classes.tolist())),
            "mask_policy": mask_policy,
            "non_exact_rgb_pixels": non_exact_pixels,
        })
        if number % 100 == 0 or number == len(image_ids):
            print(f"{name}: processed {number}/{len(image_ids)}")
    return records


def main() -> None:
    parser = argparse.ArgumentParser(description="Strictly convert official SUIM RGB masks to class indices.")
    parser.add_argument("--raw-root", type=Path, default=PROJECT_ROOT / "data" / "suim_raw")
    parser.add_argument("--output-root", type=Path, default=PROJECT_ROOT / "data" / "suim_processed")
    parser.add_argument("--overwrite", action="store_true", help="Delete an existing processed output before writing.")
    parser.add_argument(
        "--mask-policy", choices=("exact", "bit_threshold"), default="bit_threshold",
        help="exact rejects every non-palette RGB value; bit_threshold explicitly quantizes SUIM BMP boundary colors.",
    )
    parser.add_argument(
        "--size-policy", choices=("strict", "resize_mask_nearest"), default="resize_mask_nearest",
        help="strict rejects mismatched pairs; resize_mask_nearest explicitly resizes only class-index masks to image size.",
    )
    args = parser.parse_args()
    if args.output_root.exists():
        if not args.overwrite:
            raise FileExistsError(f"Output exists: {args.output_root}. Use --overwrite only to regenerate it.")
        shutil.rmtree(args.output_root)

    records = process_partition("train_val", args.raw_root / "train_val", args.output_root, args.mask_policy, args.size_policy)
    records += process_partition("test", args.raw_root / "TEST", args.output_root, args.mask_policy, args.size_policy)
    manifest = args.output_root / "manifest.csv"
    with manifest.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(records[0]))
        writer.writeheader()
        writer.writerows(records)
    print(f"Manifest: {manifest}")
    print(f"Mask policy: {args.mask_policy}")
    print(f"Size policy: {args.size_policy}")
    print(f"Non-exact RGB pixels quantized: {sum(int(r['non_exact_rgb_pixels']) for r in records)}")
    print(f"Masks resized with nearest neighbor: {sum(int(r['mask_resized_nearest']) for r in records)}")
    print(f"train_val samples: {sum(r['partition'] == 'train_val' for r in records)}")
    print(f"test samples: {sum(r['partition'] == 'test' for r in records)}")


if __name__ == "__main__":
    main()
