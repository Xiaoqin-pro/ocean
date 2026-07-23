"""Audit UIIS image reuse against every SUIM partition before confirmation."""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def phash(path: Path) -> int:
    with Image.open(path) as image:
        grayscale = np.asarray(image.convert("L").resize((32, 32), Image.Resampling.LANCZOS), dtype=np.float32)
    coefficients = cv2.dct(grayscale)[:8, :8]
    threshold = np.median(coefficients.ravel()[1:])
    value = 0
    for bit in (coefficients > threshold).ravel():
        value = (value << 1) | int(bit)
    return value


def _suim_manifest(split_dir: Path) -> pd.DataFrame:
    frames: list[pd.DataFrame] = []
    for partition in ("train", "val", "calibration", "test"):
        path = split_dir / f"{partition}.csv"
        if not path.is_file():
            raise FileNotFoundError(path)
        frame = pd.read_csv(path)
        if {"sample_id", "image_path"} - set(frame.columns):
            raise ValueError(f"Invalid SUIM split: {path}")
        frame = frame[["sample_id", "image_path"]].copy()
        frame["suim_partition"] = partition
        frames.append(frame)
    merged = pd.concat(frames, ignore_index=True)
    merged["absolute_path"] = [ROOT / path for path in merged["image_path"]]
    if not all(path.is_file() for path in merged["absolute_path"]):
        raise FileNotFoundError("SUIM audit image is missing.")
    merged["image_sha256"] = [sha256(path) for path in merged["absolute_path"]]
    merged["phash"] = [phash(path) for path in merged["absolute_path"]]
    return merged


def main() -> None:
    parser = argparse.ArgumentParser(description="Audit UIIS exact and strict near duplicates against SUIM without evaluation.")
    parser.add_argument("--uiis-manifest", type=Path, default=ROOT / "data" / "uiis_processed" / "manifest.csv")
    parser.add_argument("--suim-split-dir", type=Path, default=ROOT / "data" / "suim_processed" / "splits" / "v2_scene_grouped_deduplicated")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "data" / "uiis_processed" / "quality_reports")
    parser.add_argument("--phash-threshold", type=int, default=4)
    parser.add_argument("--top-nearest", type=int, default=100)
    args = parser.parse_args()
    if not 0 <= args.phash_threshold <= 64 or args.top_nearest <= 0:
        raise ValueError("Invalid duplicate-audit thresholds.")
    uiis = pd.read_csv(args.uiis_manifest)
    required = {"sample_id", "image_path", "image_sha256"}
    if required - set(uiis.columns) or uiis["sample_id"].duplicated().any() or len(uiis) != 4628:
        raise ValueError("UIIS manifest must contain 4,628 unique converted images.")
    uiis["absolute_path"] = [ROOT / path for path in uiis["image_path"]]
    if not all(path.is_file() for path in uiis["absolute_path"]):
        raise FileNotFoundError("UIIS audit image is missing.")
    uiis["phash"] = [phash(path) for path in uiis["absolute_path"]]
    suim = _suim_manifest(args.suim_split_dir)
    sha_to_suim = suim.groupby("image_sha256", sort=False)
    exact_rows: list[dict[str, object]] = []
    candidate_rows: list[dict[str, object]] = []
    nearest_rows: list[dict[str, object]] = []
    suim_hashes = suim["phash"].tolist()
    for row in uiis.itertuples(index=False):
        if row.image_sha256 in sha_to_suim.groups:
            for match in sha_to_suim.get_group(row.image_sha256).itertuples(index=False):
                exact_rows.append({"uiis_sample_id": row.sample_id, "uiis_image_path": row.image_path, "suim_sample_id": match.sample_id, "suim_partition": match.suim_partition, "suim_image_path": match.image_path, "image_sha256": row.image_sha256})
        distances = np.fromiter((int(row.phash ^ value).bit_count() for value in suim_hashes), dtype=np.int16, count=len(suim_hashes))
        closest = int(distances.min())
        for index in np.flatnonzero(distances <= args.phash_threshold):
            match = suim.iloc[int(index)]
            candidate_rows.append({"uiis_sample_id": row.sample_id, "uiis_image_path": row.image_path, "suim_sample_id": match.sample_id, "suim_partition": match.suim_partition, "suim_image_path": match.image_path, "phash_distance": int(distances[index]), "review_decision": "pending", "reviewer": "", "review_date": "", "notes": ""})
        nearest_rows.append({"uiis_sample_id": row.sample_id, "uiis_image_path": row.image_path, "nearest_suim_phash_distance": closest})
    exact = pd.DataFrame(exact_rows, columns=["uiis_sample_id", "uiis_image_path", "suim_sample_id", "suim_partition", "suim_image_path", "image_sha256"])
    candidates = pd.DataFrame(candidate_rows, columns=["uiis_sample_id", "uiis_image_path", "suim_sample_id", "suim_partition", "suim_image_path", "phash_distance", "review_decision", "reviewer", "review_date", "notes"])
    nearest = pd.DataFrame(nearest_rows).sort_values(["nearest_suim_phash_distance", "uiis_sample_id"], kind="stable").head(args.top_nearest)
    uiis_exact_duplicates = int(uiis["image_sha256"].duplicated(keep=False).sum())
    args.output_dir.mkdir(parents=True, exist_ok=True)
    exact.to_csv(args.output_dir / "uiis_vs_suim_exact_duplicates.csv", index=False)
    candidates.to_csv(args.output_dir / "uiis_vs_suim_near_duplicate_review.csv", index=False)
    nearest.to_csv(args.output_dir / "uiis_vs_suim_nearest_phash.csv", index=False)
    summary = {"uiis_images": len(uiis), "suim_images_audited": len(suim), "suim_partitions_audited": ["train", "val", "calibration", "test"], "exact_pairs": len(exact), "strict_phash_candidates": len(candidates), "phash_threshold": args.phash_threshold, "uiis_exact_duplicate_images": uiis_exact_duplicates, "official_suim_test_evaluated": False, "official_suim_test_used_for_duplicate_audit_only": True, "model_retrained": False}
    (args.output_dir / "uiis_suim_overlap_audit_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
