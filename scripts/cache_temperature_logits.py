"""Cache frozen low-resolution SegFormer logits for temperature scaling only."""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from typing import Any

import torch
import yaml
from torch.utils.data import DataLoader
from transformers import SegformerForSemanticSegmentation

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from datasets.label_mapping import ID2LABEL, LABEL2ID  # noqa: E402
from datasets.suim_dataset import SUIMDataset, build_eval_transform  # noqa: E402
from degradations.registry import build_image_degradation, load_conditions  # noqa: E402
from utils.reproducibility import set_seed  # noqa: E402


def load_yaml(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest().upper()


def main() -> None:
    parser = argparse.ArgumentParser(description="Cache frozen logits for temperature scaling; official TEST is locked.")
    parser.add_argument("--config", type=Path, default=PROJECT_ROOT / "configs" / "temperature_scaling.yaml")
    parser.add_argument("--conditions", nargs="+", help="Optional fixed subset for cache smoke validation.")
    args = parser.parse_args()
    config_path = args.config.resolve()
    config = load_yaml(config_path)
    experiment = config["experiment"]
    if list(experiment["splits"]) != ["calibration", "val"]:
        raise ValueError("Only [calibration, val] are permitted; official TEST is locked.")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for frozen-logit caching.")
    baseline_path = PROJECT_ROOT / experiment["baseline_config"]
    degradation_path = PROJECT_ROOT / experiment["degradation_config"]
    baseline = load_yaml(baseline_path)
    checkpoint_path = PROJECT_ROOT / experiment["checkpoint"]
    if not checkpoint_path.is_file():
        raise FileNotFoundError(checkpoint_path)
    output = PROJECT_ROOT / experiment["output_dir"] / "cache"
    output.mkdir(parents=True, exist_ok=True)
    checkpoint_hash, degradation_hash = sha256(checkpoint_path), sha256(degradation_path)
    data, model_config, training = baseline["data"], baseline["model"], baseline["training"]
    set_seed(int(experiment["seed"]))
    device = torch.device("cuda")
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model = SegformerForSemanticSegmentation.from_pretrained(
        model_config["pretrained_model"], num_labels=data["num_classes"], id2label=ID2LABEL,
        label2id=LABEL2ID, ignore_mismatched_sizes=True,
    ).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    split_dir = PROJECT_ROOT / data["split_dir"]
    manifest: list[dict[str, Any]] = []
    conditions = load_conditions(degradation_path)
    if args.conditions:
        requested = set(args.conditions)
        conditions = [item for item in conditions if item.name in requested]
        if not conditions or requested != {item.name for item in conditions}:
            raise ValueError("Requested cache condition is not in the frozen degradation registry.")
    for condition in conditions:
        degradation = build_image_degradation(condition)
        for split in experiment["splits"]:
            dataset = SUIMDataset(split_dir / f"{split}.csv", transform=build_eval_transform(data["image_size"]), image_degradation=degradation)
            loader = DataLoader(dataset, batch_size=data["batch_size"], shuffle=False, num_workers=data["num_workers"], pin_memory=True)
            logits, labels, sample_ids = [], [], []
            with torch.no_grad():
                for batch in loader:
                    # This must match the frozen pilot's inference precision;
                    # otherwise a seemingly harmless cache can alter boundary
                    # argmax decisions before temperature scaling even starts.
                    with torch.amp.autocast("cuda", enabled=bool(training["amp"])):
                        output_logits = model(pixel_values=batch["pixel_values"].to(device, non_blocking=True)).logits
                    logits.append(output_logits.cpu().to(torch.float16))
                    labels.append(batch["labels"].cpu())
                    sample_ids.extend(str(item) for item in batch["sample_id"])
            cache_path = output / split / f"{condition.name}.pt"
            cache_path.parent.mkdir(parents=True, exist_ok=True)
            payload = {
                "sample_id": sample_ids,
                "condition": condition.name,
                "degradation_type": condition.degradation_type,
                "severity": condition.severity,
                "split": split,
                "logits": torch.cat(logits),
                "labels": torch.cat(labels),
                "checkpoint_sha256": checkpoint_hash,
                "degradation_config_sha256": degradation_hash,
                "image_size": data["image_size"],
                "logits_shape": list(logits[0].shape[1:]),
                "dtype": "float16",
            }
            torch.save(payload, cache_path)
            manifest.append({key: payload[key] for key in ("condition", "degradation_type", "severity", "split", "checkpoint_sha256", "degradation_config_sha256", "image_size", "logits_shape", "dtype") } | {"path": str(cache_path.relative_to(PROJECT_ROOT)), "samples": len(sample_ids)})
            print(f"cached {split}/{condition.name}: {len(sample_ids)} samples")
    with (output / "manifest.json").open("w", encoding="utf-8") as handle:
        json.dump({"entries": manifest, "official_test_evaluated": False, "model_retrained": False}, handle, indent=2)


if __name__ == "__main__":
    main()
