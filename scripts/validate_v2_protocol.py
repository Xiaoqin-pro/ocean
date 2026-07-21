"""Validate the committed v2 exact-deduplication protocol against real files."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def validate_v2_protocol(project_root: Path = PROJECT_ROOT) -> dict[str, object]:
    processed = project_root / "data" / "suim_processed"
    split_dir = processed / "splits" / "v2_grouped_deduplicated"
    report_dir = processed / "quality_reports"
    summary = json.loads((report_dir / "v2_grouped_deduplicated_summary.json").read_text(encoding="utf-8"))
    frames = {name: pd.read_csv(split_dir / f"{name}.csv") for name in ("train", "val", "calibration", "test")}
    development_ids = [set(frames[name]["sample_id"]) for name in ("train", "val", "calibration")]
    if any(left.intersection(right) for index, left in enumerate(development_ids) for right in development_ids[index + 1:]):
        raise AssertionError("Development sample IDs overlap.")
    if set().union(*development_ids).intersection(set(frames["test"]["sample_id"])):
        raise AssertionError("Official test IDs appear in development.")
    for name, frame in frames.items():
        if not all((project_root / path).is_file() for path in frame["image_path"]) or not all((project_root / path).is_file() for path in frame["mask_path"]):
            raise AssertionError(f"{name} contains a missing file path.")
        if digest(split_dir / f"{name}.csv") != summary["split_sha256"][name]:
            raise AssertionError(f"{name} CSV hash differs from the committed summary.")
        if len(frame) != summary["split_counts"][name]:
            raise AssertionError(f"{name} count differs from the committed summary.")
    dev = pd.concat([frames[name] for name in ("train", "val", "calibration")])
    owner: dict[str, str] = {}
    for name in ("train", "val", "calibration"):
        for image_hash in frames[name]["image_sha256"].unique():
            previous = owner.setdefault(str(image_hash), name)
            if previous != name:
                raise AssertionError("An exact-SHA image group crosses development splits.")
    if dev["image_sha256"].isin(set(frames["test"]["image_sha256"])).any():
        raise AssertionError("Development contains an exact official-test duplicate.")
    excluded = pd.read_csv(report_dir / "v2_excluded_samples.csv")
    if set(excluded["sample_id"]).intersection(set(dev["sample_id"])):
        raise AssertionError("An excluded sample is present in development.")
    return {"split_counts": {name: len(frame) for name, frame in frames.items()}, "exact_sha_leakage": False, "excluded_samples": len(excluded)}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--project-root", type=Path, default=PROJECT_ROOT)
    args = parser.parse_args()
    print(json.dumps(validate_v2_protocol(args.project_root), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
