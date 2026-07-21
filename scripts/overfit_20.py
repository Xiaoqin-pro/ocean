from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as functional
import yaml
from PIL import Image
from torch.utils.data import DataLoader, Subset
from transformers import SegformerForSemanticSegmentation

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from datasets.label_mapping import index_mask_to_rgb  # noqa: E402
from datasets.label_mapping import ID2LABEL, LABEL2ID  # noqa: E402
from datasets.suim_dataset import IMAGENET_MEAN, IMAGENET_STD, SUIMDataset, build_eval_transform  # noqa: E402
from metrics.segmentation import confusion_matrix, metrics_from_confusion_matrix  # noqa: E402
from utils.reproducibility import set_seed  # noqa: E402


def save_visualization(pixel_values: torch.Tensor, prediction: torch.Tensor, target: torch.Tensor, path: Path) -> None:
    image = pixel_values.detach().cpu().permute(1, 2, 0).numpy()
    image = (image * np.asarray(IMAGENET_STD) + np.asarray(IMAGENET_MEAN)) * 255
    image = np.clip(image, 0, 255).astype(np.uint8)
    predicted_rgb = index_mask_to_rgb(prediction.detach().cpu().numpy())
    target_rgb = index_mask_to_rgb(target.detach().cpu().numpy())
    canvas = np.concatenate([image, target_rgb, predicted_rgb], axis=1)
    Image.fromarray(canvas).save(path)


def main() -> None:
    parser = argparse.ArgumentParser(description="Overfit a fixed first 20 SUIM training samples.")
    parser.add_argument("--config", type=Path, default=PROJECT_ROOT / "configs" / "segformer_b0_suim_baseline.yaml")
    parser.add_argument("--epochs", type=int, default=300)
    parser.add_argument("--batch-size", type=int, default=4)
    args = parser.parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for the overfit check.")
    config = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    set_seed(config["experiment"]["seed"])
    device = torch.device("cuda")
    data, model_config = config["data"], config["model"]
    split = PROJECT_ROOT / data["split_dir"] / "train.csv"
    dataset = SUIMDataset(split, transform=build_eval_transform(data["image_size"]))
    if len(dataset) < 20:
        raise ValueError(f"Need at least 20 training samples; received {len(dataset)}")
    fixed_dataset = Subset(dataset, list(range(20)))
    loader = DataLoader(fixed_dataset, batch_size=args.batch_size, shuffle=True, num_workers=0, pin_memory=True)
    model = SegformerForSemanticSegmentation.from_pretrained(
        model_config["pretrained_model"], num_labels=data["num_classes"], id2label=ID2LABEL,
        label2id=LABEL2ID, ignore_mismatched_sizes=True,
    ).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config["training"]["learning_rate"], weight_decay=config["training"]["weight_decay"])
    amp = bool(config["training"]["amp"])
    scaler = torch.amp.GradScaler("cuda", enabled=amp)
    output_dir = PROJECT_ROOT / "outputs" / "overfit_20"
    output_dir.mkdir(parents=True, exist_ok=True)
    history_path = output_dir / "history.csv"

    with history_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["epoch", "loss", "miou"])
        writer.writeheader()
        for epoch in range(1, args.epochs + 1):
            model.train()
            total_loss = 0.0
            matrix = torch.zeros((data["num_classes"], data["num_classes"]), dtype=torch.long, device=device)
            for batch in loader:
                pixels, labels = batch["pixel_values"].to(device), batch["labels"].to(device)
                optimizer.zero_grad(set_to_none=True)
                with torch.amp.autocast("cuda", enabled=amp):
                    logits = functional.interpolate(model(pixel_values=pixels).logits, size=labels.shape[-2:], mode="bilinear", align_corners=False)
                    loss = functional.cross_entropy(logits, labels, ignore_index=255)
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
                total_loss += loss.item()
                matrix += confusion_matrix(logits.argmax(1), labels, num_classes=data["num_classes"]).to(device)
            metric = metrics_from_confusion_matrix(matrix.cpu())
            row = {"epoch": epoch, "loss": total_loss / len(loader), "miou": metric["miou"]}
            writer.writerow(row)
            if epoch == 1 or epoch % 10 == 0 or epoch == args.epochs:
                print(f"epoch={epoch} loss={row['loss']:.4f} train_miou={row['miou']:.4f}")
        handle.flush()

    checkpoint = output_dir / "model.pt"
    torch.save({"model_state_dict": model.state_dict(), "config": config}, checkpoint)
    first = next(iter(DataLoader(fixed_dataset, batch_size=1, shuffle=False)))
    pixels, labels = first["pixel_values"].to(device), first["labels"].to(device)
    model.eval()
    with torch.no_grad(), torch.amp.autocast("cuda", enabled=amp):
        original = functional.interpolate(model(pixel_values=pixels).logits, size=labels.shape[-2:], mode="bilinear", align_corners=False)
    restored = SegformerForSemanticSegmentation.from_pretrained(
        model_config["pretrained_model"], num_labels=data["num_classes"], id2label=ID2LABEL,
        label2id=LABEL2ID, ignore_mismatched_sizes=True,
    ).to(device)
    restored.load_state_dict(torch.load(checkpoint, map_location=device, weights_only=False)["model_state_dict"])
    restored.eval()
    with torch.no_grad(), torch.amp.autocast("cuda", enabled=amp):
        reloaded = functional.interpolate(restored(pixel_values=pixels).logits, size=labels.shape[-2:], mode="bilinear", align_corners=False)
    if not torch.allclose(original, reloaded, atol=1e-5, rtol=1e-4):
        raise RuntimeError("Reloaded checkpoint predictions differ from the saved model.")
    save_visualization(pixels[0], original.argmax(1)[0], labels[0], output_dir / "sample_00_image_target_prediction.png")
    print(f"Reload check: OK; checkpoint: {checkpoint}")


if __name__ == "__main__":
    main()
