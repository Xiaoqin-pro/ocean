"""Measure the degradation-family router on the frozen synthetic train views."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import torch
import yaml
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.train_sadr import build_model  # noqa: E402
from scripts.train_uiis_scdi_replication import UIISTrajectoryDataset  # noqa: E402


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    args = parser.parse_args()
    config = yaml.safe_load(args.config.resolve().read_text(encoding="utf-8"))
    if not bool(config["training"].get("routed", False)):
        raise ValueError("Router evaluation requires training.routed=true.")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required.")
    device = torch.device("cuda")
    dataset = UIISTrajectoryDataset(ROOT / str(config["data"]["train_csv"]), ROOT / str(config["data"]["degradation_registry"]), int(config["data"]["image_size"]))
    loader = DataLoader(dataset, batch_size=16, shuffle=False, num_workers=0)
    _, front = build_model(config, device)
    payload = torch.load(ROOT / str(config["experiment"]["output_dir"]) / "formal/checkpoints/final.pt", map_location=device, weights_only=False)
    front.load_state_dict(payload["model_state_dict"])
    front.eval(); correct = total = 0; entropy_sum = 0.0
    with torch.no_grad():
        for batch in loader:
            images = batch["s1"].to(device, non_blocking=True)
            labels = batch["family_id"].to(device, non_blocking=True).long()
            output = front(images); probabilities = torch.softmax(output[2], dim=1)
            correct += int((probabilities.argmax(1) == labels).sum())
            total += int(labels.numel())
            entropy_sum += float((-(probabilities * probabilities.clamp_min(1e-8).log()).sum(1)).sum())
    result = {"images": total, "family_accuracy": correct / total, "mean_entropy": entropy_sum / total}
    output = ROOT / str(config["experiment"]["output_dir"]) / "router_metrics.json"
    output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
