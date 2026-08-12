"""Evaluate TTA-SPT on the frozen UIIS confirmation split."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd
import torch
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from degradations.registry import load_conditions  # noqa: E402
from scripts.evaluate_dts_seg_gate0 import evaluate_condition  # noqa: E402
from scripts.train_tta_spt import FORMAT, build_model  # noqa: E402
from scripts.train_uiis_scdi_replication import build_models as build_uiis_f4  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=ROOT / "configs/tta_spt_fast.yaml")
    parser.add_argument("--split", choices=("confirmation",), default="confirmation")
    parser.add_argument("--allow-confirmation", action="store_true")
    args = parser.parse_args()
    if not args.allow_confirmation:
        raise PermissionError("Confirmation is locked.")
    config = yaml.safe_load(args.config.resolve().read_text(encoding="utf-8"))
    data = config["data"]
    csv_path = ROOT / str(data["uiis_confirmation_csv"])
    expected = 511
    frame = pd.read_csv(csv_path)
    if len(frame) != expected or csv_path.name != "confirmation.csv":
        raise PermissionError("Unexpected frozen UIIS confirmation split.")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required.")
    device = torch.device("cuda")
    conditions = [item for item in load_conditions(ROOT / str(data["degradation_registry"])) if item.name in set(config["evaluation"]["conditions"])]
    if len(conditions) != 13:
        raise ValueError("Expected the frozen 13-condition registry.")

    tta_checkpoint = ROOT / str(config["experiment"]["output_dir"]) / "formal/checkpoints/final.pt"
    payload = torch.load(tta_checkpoint, map_location=device, weights_only=False)
    required = {"checkpoint_format": FORMAT, "variant": "TTA-SPT", "epoch": int(config["training"]["epochs"]), "smoke": False}
    for key, expected_value in required.items():
        if payload.get(key) != expected_value:
            raise ValueError(f"Invalid {key} in {tta_checkpoint}.")
    if any(bool(payload.get(key, True)) for key in ("calibration_evaluated", "confirmation_evaluated", "official_suim_test_evaluated")):
        raise ValueError("TTA checkpoint records prohibited evaluation access.")
    tta = build_model(config, device)
    tta.load_state_dict(payload["model_state_dict"])
    tta.eval()
    source = tta.base_model.eval()

    uiis_config = yaml.safe_load((ROOT / "configs/uiis_scdi_replication.yaml").read_text(encoding="utf-8"))
    uiis_checkpoint = ROOT / "outputs/uiis_scdi_replication/formal/F4/checkpoints/final.pt"
    uiis_payload = torch.load(uiis_checkpoint, map_location=device, weights_only=False)
    if int(uiis_payload.get("epoch", 0)) != 8 or bool(uiis_payload.get("confirmation_evaluated", True)):
        raise ValueError("UIIS F4 reference is not a clean train-only checkpoint.")
    uiis = build_uiis_f4(uiis_config, "F4", device)
    uiis.load_state_dict(uiis_payload["model_state_dict"])
    uiis.eval()

    image_size = int(data["evaluation_image_size"])
    rows = []
    for variant, model in (("Source-B", source), ("TTA-SPT", tta), ("UIIS-F4", uiis)):
        for condition in conditions:
            metric = evaluate_condition(model, csv_path, condition, image_size, device)
            rows.append({"variant": variant, **metric})
            print(f"{args.split}/{variant}/{condition.name}: mIoU={metric['miou']:.4f}", flush=True)
        torch.cuda.empty_cache()
    result = pd.DataFrame(rows)
    output = ROOT / str(config["experiment"]["output_dir"]) / f"{args.split}_evaluation"
    output.mkdir(parents=True, exist_ok=True)
    result.to_csv(output / "condition_metrics.csv", index=False)
    summary = {
        "split": args.split,
        "confirmation_evaluated": True,
        "official_suim_test_evaluated": False,
        "rows": len(result),
        "images": expected,
        "mean_miou": {variant: float(result[result.variant == variant].miou.mean()) for variant in result.variant.unique()},
        "gains_pp": {
            "tta_minus_source": float(100.0 * (result[result.variant == "TTA-SPT"].miou.mean() - result[result.variant == "Source-B"].miou.mean())),
            "tta_minus_uiis": float(100.0 * (result[result.variant == "TTA-SPT"].miou.mean() - result[result.variant == "UIIS-F4"].miou.mean())),
        },
    }
    (output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
