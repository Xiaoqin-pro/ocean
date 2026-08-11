"""Evaluate joint SUIM+UIIS F4 and CB-UTD checkpoints on frozen roles."""
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
from datasets.suim_dataset import SUIMDataset, build_eval_transform  # noqa: E402
from scripts.evaluate_dts_seg_gate0 import evaluate_condition  # noqa: E402
from scripts.train_joint_cb_utd import build_model  # noqa: E402


def _split_info(config: dict, split: str) -> tuple[Path, int, bool]:
    data = config["data"]
    if split == "suim_official":
        return ROOT / str(data["suim_test_csv"]), int(data["suim_test_images"]), True
    if split == "calibration":
        return ROOT / str(data["uiis_calibration_csv"]), int(data["uiis_calibration_images"]), False
    if split == "confirmation":
        return ROOT / str(data["uiis_confirmation_csv"]), int(data["uiis_confirmation_images"]), False
    raise ValueError(split)


def _evaluate(model: torch.nn.Module, csv_path: Path, conditions: list, image_size: int, device: torch.device) -> list[dict[str, object]]:
    # Reuse the locked SUIM-compatible evaluator; UIIS uses the same eight-class mask contract.
    rows: list[dict[str, object]] = []
    for condition in conditions:
        rows.append(evaluate_condition(model, csv_path, condition, image_size, device))
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=ROOT / "configs/joint_cb_utd.yaml")
    parser.add_argument("--split", choices=("calibration", "confirmation", "suim_official"), default="calibration")
    parser.add_argument("--allow-confirmation", action="store_true")
    args = parser.parse_args()
    config = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    if args.split == "confirmation" and not args.allow_confirmation:
        raise PermissionError("Confirmation is locked; pass --allow-confirmation after development freeze.")
    csv_path, expected, official = _split_info(config, args.split)
    if not csv_path.exists() or csv_path.name not in {"test.csv", "calibration.csv", "confirmation.csv"}:
        raise PermissionError("Only frozen split files are permitted.")
    if len(pd.read_csv(csv_path)) != expected:
        raise PermissionError(f"Unexpected row count for {args.split}: {len(pd.read_csv(csv_path))} != {expected}.")
    if official and not config["evaluation"].get("official_suim_test_locked"):
        raise PermissionError("Official SUIM test lock must be explicit.")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA required.")
    device = torch.device("cuda")
    conditions = [item for item in load_conditions(ROOT / str(config["data"]["degradation_registry"])) if item.name in set(config["evaluation"]["conditions"])]
    output = ROOT / str(config["experiment"]["output_dir"]) / f"{args.split}_evaluation"
    output.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, object]] = []
    for variant in ("F4", "CB-UTD"):
        checkpoint = output.parent / "formal" / variant / "checkpoints" / "final.pt"
        if not checkpoint.exists():
            raise FileNotFoundError(checkpoint)
        payload = torch.load(checkpoint, map_location=device, weights_only=False)
        if payload.get("checkpoint_format") != "joint_cb_utd_v1" or payload.get("variant") != variant or payload.get("smoke"):
            raise ValueError(f"Invalid joint {variant} checkpoint.")
        model = build_model(config, variant, device)
        model.load_state_dict(payload["model_state_dict"])
        model.eval()
        for metric in _evaluate(model, csv_path, conditions, int(config["data"]["image_size"]), device):
            row = {"variant": variant, **metric}
            rows.append(row)
            print(f"{args.split}/{variant}/{metric['condition']}: mIoU={metric['miou']:.4f}", flush=True)
    frame = pd.DataFrame(rows)
    frame.to_csv(output / "condition_metrics.csv", index=False)
    summary = {
        "split": args.split,
        "official_suim_test_evaluated": official,
        "confirmation_evaluated": args.split == "confirmation",
        "rows": len(frame),
        "images": expected,
        "mean_miou": {variant: float(frame[frame.variant == variant].miou.mean()) for variant in frame.variant.unique()},
    }
    (output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
