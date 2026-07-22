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


def validate_v2_protocol(project_root: Path = PROJECT_ROOT, *, require_formal: bool = False) -> dict[str, object]:
    processed = project_root / "data" / "suim_processed"
    report_dir = processed / "quality_reports"
    formal_dir = processed / "splits" / "v2_scene_grouped_deduplicated"
    formal_summary = report_dir / "v2_scene_grouped_summary.json"
    is_formal = formal_dir.is_dir() and formal_summary.is_file()
    if require_formal and not is_formal:
        raise AssertionError("Formal reviewed scene-grouped v2 split does not exist.")
    split_dir = formal_dir if is_formal else processed / "splits" / "v2_grouped_deduplicated"
    summary_path = formal_summary if is_formal else report_dir / "v2_grouped_deduplicated_summary.json"
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    frames = {name: pd.read_csv(split_dir / f"{name}.csv") for name in ("train", "val", "calibration", "test")}
    members = pd.read_csv(report_dir / "scene_group_members.csv")
    hash_by_sample = members.set_index("sample_id")["image_sha256"].to_dict()
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
    group_column = "scene_group_id" if is_formal else "image_sha256"
    if group_column not in dev.columns or group_column not in frames["test"].columns:
        raise AssertionError(f"{group_column} is missing from split CSVs.")
    owner: dict[str, str] = {}
    for name in ("train", "val", "calibration"):
        for group_id in frames[name][group_column].unique():
            previous = owner.setdefault(str(group_id), name)
            if previous != name:
                raise AssertionError(f"A {group_column} group crosses development splits.")
    exact_owner: dict[str, str] = {}
    for name in ("train", "val", "calibration"):
        for sample_id in frames[name]["sample_id"]:
            image_hash = hash_by_sample[str(sample_id)]
            previous = exact_owner.setdefault(image_hash, name)
            if previous != name:
                raise AssertionError("An exact-SHA image group crosses development splits.")
    development_hashes = {hash_by_sample[str(sample_id)] for sample_id in dev["sample_id"]}
    test_hashes = {hash_by_sample[str(sample_id)] for sample_id in frames["test"]["sample_id"]}
    if development_hashes.intersection(test_hashes):
        raise AssertionError("Development contains an exact official-test duplicate.")
    excluded_path = report_dir / ("v2_scene_grouped_excluded_samples.csv" if is_formal else "v2_excluded_samples.csv")
    excluded = pd.read_csv(excluded_path)
    if set(excluded["sample_id"]).intersection(set(dev["sample_id"])):
        raise AssertionError("An excluded sample is present in development.")
    if is_formal:
        if summary.get("near_duplicate_review_status") != "review_complete":
            raise AssertionError("Formal split was created before near-duplicate review completion.")
        if set(dev["scene_group_id"]).intersection(set(frames["test"]["scene_group_id"])):
            raise AssertionError("Development and official TEST share a reviewed scene group.")
    for name in ("train", "val", "calibration"):
        present = set()
        for relative_path in frames[name]["mask_path"]:
            from PIL import Image
            import numpy as np
            with Image.open(project_root / relative_path) as image:
                labels = np.asarray(image, dtype=np.uint8)
            if labels.min() < 0 or labels.max() > 7:
                raise AssertionError(f"{name} contains an invalid label.")
            present.update(np.unique(labels).tolist())
        if present != set(range(8)):
            raise AssertionError(f"{name} does not contain all 8 classes: {sorted(present)}")
    return {"split_counts": {name: len(frame) for name, frame in frames.items()}, "exact_sha_leakage": False, "scene_group_leakage": False if is_formal else "not_checked", "excluded_samples": len(excluded), "formal_protocol": is_formal}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--project-root", type=Path, default=PROJECT_ROOT)
    parser.add_argument("--require-formal", action="store_true")
    args = parser.parse_args()
    print(json.dumps(validate_v2_protocol(args.project_root, require_formal=args.require_formal), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
