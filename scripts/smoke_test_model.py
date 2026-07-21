from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch
import torch.nn.functional as functional
from transformers import SegformerForSemanticSegmentation

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from utils.reproducibility import set_seed  # noqa: E402
from datasets.label_mapping import ID2LABEL, LABEL2ID  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description="Run one real CUDA forward/backward SegFormer-B0 step.")
    parser.add_argument("--pretrained-model", default="nvidia/mit-b0")
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--image-size", type=int, default=384)
    args = parser.parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for this smoke test.")

    set_seed(20260721)
    device = torch.device("cuda")
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats(device)
    model = SegformerForSemanticSegmentation.from_pretrained(
        args.pretrained_model, num_labels=8, id2label=ID2LABEL, label2id=LABEL2ID,
        ignore_mismatched_sizes=True,
    ).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=6e-5, weight_decay=0.01)
    scaler = torch.amp.GradScaler("cuda", enabled=True)
    pixel_values = torch.randn(args.batch_size, 3, args.image_size, args.image_size, device=device)
    labels = torch.randint(0, 8, (args.batch_size, args.image_size, args.image_size), device=device)

    model.train()
    optimizer.zero_grad(set_to_none=True)
    with torch.amp.autocast("cuda", dtype=torch.float16):
        outputs = model(pixel_values=pixel_values)
        logits = functional.interpolate(outputs.logits, size=labels.shape[-2:], mode="bilinear", align_corners=False)
        loss = functional.cross_entropy(logits, labels)
    scaler.scale(loss).backward()
    scaler.step(optimizer)
    scaler.update()
    torch.cuda.synchronize(device)

    print(f"GPU: {torch.cuda.get_device_name(device)}")
    print(f"Input: {tuple(pixel_values.shape)}")
    print(f"Raw logits: {tuple(outputs.logits.shape)}")
    print(f"Upsampled logits: {tuple(logits.shape)}")
    print(f"Loss: {loss.item():.6f}")
    print("Backward: OK")
    print("Optimizer step: OK")
    print(f"Peak VRAM: {torch.cuda.max_memory_allocated(device) / 1024**3:.2f} GB")


if __name__ == "__main__":
    main()
