from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch
from torch.utils.data import DataLoader

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from datasets.suim_dataset import SUIMDataset, build_train_transform  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate SUIM Dataset output shapes and labels.")
    parser.add_argument("--split", type=Path, default=PROJECT_ROOT / "data" / "suim_processed" / "splits" / "v1_seed_20260721" / "train.csv")
    parser.add_argument("--image-size", type=int, default=384)
    args = parser.parse_args()

    dataset = SUIMDataset(args.split, transform=build_train_transform(args.image_size))
    sample = dataset[0]
    assert tuple(sample["pixel_values"].shape) == (3, args.image_size, args.image_size)
    assert tuple(sample["labels"].shape) == (args.image_size, args.image_size)
    assert sample["pixel_values"].dtype == torch.float32
    assert sample["labels"].dtype == torch.long
    classes = torch.unique(sample["labels"])
    assert torch.all((0 <= classes) & (classes <= 7))

    loader = DataLoader(dataset, batch_size=4, shuffle=True, num_workers=0, pin_memory=torch.cuda.is_available())
    batch = next(iter(loader))
    assert tuple(batch["pixel_values"].shape) == (4, 3, args.image_size, args.image_size)
    assert tuple(batch["labels"].shape) == (4, args.image_size, args.image_size)
    print(f"Dataset size: {len(dataset)}")
    print(f"Sample ID: {sample['sample_id']}")
    print(f"Image shape: {tuple(sample['pixel_values'].shape)}; dtype: {sample['pixel_values'].dtype}")
    print(f"Label shape: {tuple(sample['labels'].shape)}; dtype: {sample['labels'].dtype}")
    print(f"Sample classes: {classes.tolist()}")
    print(f"Batch images: {tuple(batch['pixel_values'].shape)}")
    print(f"Batch labels: {tuple(batch['labels'].shape)}")


if __name__ == "__main__":
    main()
