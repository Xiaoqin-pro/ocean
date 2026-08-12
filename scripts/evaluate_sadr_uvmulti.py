"""Cross-dataset sanity check on the held-out UVMulti validation frames."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import albumentations as A
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as functional
import yaml
from albumentations.pytorch import ToTensorV2
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from datasets.suim_dataset import IMAGENET_MEAN, IMAGENET_STD  # noqa: E402
from reliability.sadr_seg import SADRFrontEnd  # noqa: E402
from scripts.train_uiis_scdi_replication import build_models as build_uiis_f4  # noqa: E402

UVMULTI_TO_SUIM = {0: 0, 1: 3, 2: 5, 3: 5, 4: 5, 5: 6, 6: 5}
COMMON = (0, 3, 5, 6)


class SADRModel(torch.nn.Module):
    def __init__(self, base: torch.nn.Module, front: SADRFrontEnd) -> None:
        super().__init__(); self.base = base; self.front = front

    def forward(self, pixel_values: torch.Tensor):
        restored, _ = self.front(pixel_values)
        return self.base(pixel_values=restored)


def load_sadr(config_path: str, device: torch.device) -> torch.nn.Module:
    config = yaml.safe_load((ROOT / config_path).read_text(encoding="utf-8")); uiis_config = yaml.safe_load((ROOT / str(config["data"]["uiis_config"])).read_text(encoding="utf-8")); base = build_uiis_f4(uiis_config, "F4", device); front = SADRFrontEnd().to(device); payload = torch.load(ROOT / str(config["experiment"]["output_dir"]) / "formal/checkpoints/final.pt", map_location=device, weights_only=False); front.load_state_dict(payload["model_state_dict"]); base.eval(); front.eval(); return SADRModel(base, front).eval()


def evaluate_model(model: torch.nn.Module, root: Path, enhanced: bool, device: torch.device, batch_size: int = 8) -> dict:
    transform = A.Compose([A.Resize(384, 384), A.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD, max_pixel_value=255.0), ToTensorV2()]); confusion = np.zeros((8, 8), dtype=np.int64); count = 0; image_batch = []; target_batch = []
    def flush() -> None:
        nonlocal confusion, count, image_batch, target_batch
        if not image_batch:
            return
        pixels = torch.stack(image_batch).float().to(device); target = torch.stack(target_batch).long().to(device)
        with torch.no_grad(), torch.autocast(device_type="cuda", dtype=torch.float16):
            logits = model(pixel_values=pixels).logits
        pred = functional.interpolate(logits.float(), size=target.shape[-2:], mode="bilinear", align_corners=False).argmax(1); valid = target.ne(255); indices = target[valid] * 8 + pred[valid]; confusion += torch.bincount(indices, minlength=64).reshape(8, 8).cpu().numpy(); count += len(image_batch); image_batch = []; target_batch = []
    for mask_path in sorted((root / "mask").glob("*.png")):
        image_path = root / ("sequence_enh" if enhanced else "images") / f"{mask_path.stem}.jpg"
        if not image_path.is_file(): continue
        with Image.open(image_path) as image, Image.open(mask_path) as mask:
            image_array = np.asarray(image.convert("RGB"), dtype=np.uint8); labels = np.asarray(mask, dtype=np.uint8)
        mapped = np.full_like(labels, 255, dtype=np.uint8)
        for source, target in UVMULTI_TO_SUIM.items(): mapped[labels == source] = target
        transformed = transform(image=image_array, mask=mapped); image_batch.append(transformed["image"].float()); target_batch.append(transformed["mask"].long())
        if len(image_batch) >= batch_size: flush()
    flush()
    diag = np.diag(confusion); denom = confusion.sum(0) + confusion.sum(1) - diag; valid = np.asarray([denom[c] > 0 for c in COMMON]); miou = float(np.mean(diag[list(COMMON)][valid] / denom[list(COMMON)][valid])); return {"images": count, "common_classes": list(COMMON), "miou_common": miou}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__); parser.add_argument("--root", type=Path, default=ROOT / "data/uvmulti/val/video108"); parser.add_argument("--all-videos", action="store_true", help="Evaluate every labeled video under data/uvmulti/train and val."); args = parser.parse_args();
    if not torch.cuda.is_available(): raise RuntimeError("CUDA is required.")
    device = torch.device("cuda"); uiis_config = yaml.safe_load((ROOT / "configs/uiis_scdi_replication.yaml").read_text(encoding="utf-8")); baseline = build_uiis_f4(uiis_config, "F4", device).eval(); sadr1 = load_sadr("configs/sadr_long.yaml", device); sadr2 = load_sadr("configs/sadr_seed2.yaml", device); sadr3 = load_sadr("configs/sadr_seed3.yaml", device); rows = []
    roots = [ROOT / "data/uvmulti" / split / video for split in ("train", "val") for video in sorted((ROOT / "data/uvmulti" / split).iterdir()) if video.is_dir()] if args.all_videos else [args.root]
    for root in roots:
        for view, enhanced in (("raw", False), ("enhanced", True)):
            for name, model in (("UIIS-F4", baseline), ("SADR-seed1", sadr1), ("SADR-seed2", sadr2), ("SADR-seed3", sadr3)):
                metric = evaluate_model(model, root, enhanced, device); rows.append({"split": root.parent.name, "video": root.name, "view": view, "variant": name, **metric}); print(f"{root.parent.name}/{root.name}/{view}/{name}: common_mIoU={metric['miou_common']:.4f}", flush=True)
    output = ROOT / ("outputs/sadr_uvmulti_all" if args.all_videos else "outputs/sadr_uvmulti_validation"); output.mkdir(parents=True, exist_ok=True); frame = pd.DataFrame(rows); frame.to_csv(output / "metrics.csv", index=False); (output / "summary.json").write_text(json.dumps({"videos": sorted(frame.video.unique().tolist()), "class_mapping": UVMULTI_TO_SUIM, "rows": rows}, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__": main()
