from __future__ import annotations

from pathlib import Path
from collections.abc import Callable
from typing import Any

import albumentations as A
import cv2
import numpy as np
import pandas as pd
import torch
from albumentations.pytorch import ToTensorV2
from PIL import Image
from torch.utils.data import Dataset


PROJECT_ROOT = Path(__file__).resolve().parents[1]
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def _base_transform(image_size: int, *, training: bool) -> A.Compose:
    transforms: list[Any] = [
        A.Resize(
            height=image_size,
            width=image_size,
            interpolation=cv2.INTER_LINEAR,
            mask_interpolation=cv2.INTER_NEAREST,
        )
    ]
    if training:
        transforms.append(A.HorizontalFlip(p=0.5))
    transforms.extend(
        [
            A.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD, max_pixel_value=255.0),
            ToTensorV2(),
        ]
    )
    return A.Compose(transforms)


def build_train_transform(image_size: int = 384) -> A.Compose:
    return _base_transform(image_size, training=True)


def build_eval_transform(image_size: int = 384) -> A.Compose:
    return _base_transform(image_size, training=False)


ImageDegradation = Callable[[np.ndarray, str], np.ndarray]


class SUIMDataset(Dataset[dict[str, Any]]):
    def __init__(
        self,
        split_csv: str | Path,
        *,
        transform: A.Compose,
        image_degradation: ImageDegradation | None = None,
    ) -> None:
        self.split_csv = Path(split_csv)
        if not self.split_csv.is_file():
            raise FileNotFoundError(f"Split file does not exist: {self.split_csv}")
        self.samples = pd.read_csv(self.split_csv)
        required = {"sample_id", "image_path", "mask_path"}
        missing = required.difference(self.samples.columns)
        if missing:
            raise ValueError(f"{self.split_csv} is missing columns: {sorted(missing)}")
        if self.samples.empty:
            raise ValueError(f"Split is empty: {self.split_csv}")
        if self.samples["sample_id"].duplicated().any():
            raise ValueError(f"Split contains duplicated sample ids: {self.split_csv}")
        self.transform = transform
        self.image_degradation = image_degradation

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> dict[str, Any]:
        row = self.samples.iloc[index]
        image_path = PROJECT_ROOT / str(row["image_path"])
        mask_path = PROJECT_ROOT / str(row["mask_path"])
        if not image_path.is_file() or not mask_path.is_file():
            raise FileNotFoundError(f"Missing pair for {row['sample_id']}: {image_path}, {mask_path}")

        with Image.open(image_path) as image:
            image_array = np.asarray(image.convert("RGB"), dtype=np.uint8)
        with Image.open(mask_path) as mask:
            mask_array = np.asarray(mask, dtype=np.uint8)

        if image_array.shape[:2] != mask_array.shape:
            raise ValueError(
                f"Image/mask size mismatch for {row['sample_id']}: "
                f"{image_array.shape[:2]} vs {mask_array.shape}"
            )
        classes = np.unique(mask_array)
        if np.any(classes > 7):
            raise ValueError(f"Invalid labels for {row['sample_id']}: {classes.tolist()}")

        if self.image_degradation is not None:
            image_array = self.image_degradation(image_array, str(row["sample_id"]))
            if image_array.shape != mask_array.shape + (3,) or image_array.dtype != np.uint8:
                raise ValueError(
                    "Image degradation must return an RGB uint8 image with unchanged spatial dimensions "
                    f"for {row['sample_id']}."
                )

        transformed = self.transform(image=image_array, mask=mask_array)
        return {
            "pixel_values": transformed["image"].to(dtype=torch.float32),
            "labels": transformed["mask"].to(dtype=torch.long),
            "sample_id": str(row["sample_id"]),
        }
