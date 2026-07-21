"""Search cross-split near duplicates with a deterministic 64-bit perceptual hash.

The report is a screening aid: small Hamming distance pairs should be reviewed
visually before declaring a leakage issue.
"""
from __future__ import annotations

import argparse
import hashlib
from itertools import product
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
from PIL import Image

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def perceptual_hash(path: Path) -> int:
    with Image.open(path) as image:
        gray = np.asarray(image.convert("L").resize((32, 32), Image.Resampling.LANCZOS), dtype=np.float32)
    coefficients = cv2.dct(gray)[:8, :8]
    threshold = np.median(coefficients.ravel()[1:])
    bits = coefficients > threshold
    value = 0
    for bit in bits.ravel():
        value = (value << 1) | int(bit)
    return value


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def compare(left: pd.DataFrame, right: pd.DataFrame, label: str) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for first, second in product(left.itertuples(index=False), right.itertuples(index=False)):
        distance = int(first.phash ^ second.phash).bit_count()
        rows.append({
            "comparison": label,
            "split_a": first.split,
            "sample_id_a": first.sample_id,
            "image_path_a": first.image_path,
            "split_b": second.split,
            "sample_id_b": second.sample_id,
            "image_path_b": second.image_path,
            "phash_hamming_distance": distance,
            "exact_file_duplicate": first.sha256 == second.sha256,
        })
    return pd.DataFrame(rows).sort_values(["phash_hamming_distance", "sample_id_a", "sample_id_b"], kind="stable")


def exact_matches(left: pd.DataFrame, right: pd.DataFrame, label: str) -> pd.DataFrame:
    columns = ["sample_id", "image_path", "split", "sha256"]
    merged = left[columns].merge(right[columns], on="sha256", suffixes=("_a", "_b"))
    if merged.empty:
        return pd.DataFrame(columns=["comparison", "split_a", "sample_id_a", "image_path_a", "split_b", "sample_id_b", "image_path_b", "sha256"])
    merged.insert(0, "comparison", label)
    return merged[["comparison", "split_a", "sample_id_a", "image_path_a", "split_b", "sample_id_b", "image_path_b", "sha256"]]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--processed-dir", type=Path, default=PROJECT_ROOT / "data" / "suim_processed")
    parser.add_argument("--top-k-per-comparison", type=int, default=100)
    args = parser.parse_args()
    split_dir = args.processed_dir / "splits" / "v1_seed_20260721"
    frames: dict[str, pd.DataFrame] = {}
    for split_name in ("train", "val", "calibration", "test"):
        frame = pd.read_csv(split_dir / f"{split_name}.csv")
        frame["split"] = split_name
        paths = [PROJECT_ROOT / relative for relative in frame["image_path"]]
        frame["phash"] = [perceptual_hash(path) for path in paths]
        frame["sha256"] = [file_sha256(path) for path in paths]
        frames[split_name] = frame
    comparisons = [("train", "val"), ("train", "calibration"), ("train", "test"), ("val", "test")]
    reports = [compare(frames[left], frames[right], f"{left}_vs_{right}").head(args.top_k_per_comparison) for left, right in comparisons]
    report = pd.concat(reports, ignore_index=True)
    report_dir = args.processed_dir / "quality_reports"
    report_dir.mkdir(parents=True, exist_ok=True)
    report.to_csv(report_dir / "cross_split_near_duplicate_pairs.csv", index=False)
    exact = pd.concat([exact_matches(frames[left], frames[right], f"{left}_vs_{right}") for left, right in comparisons], ignore_index=True)
    exact.to_csv(report_dir / "cross_split_exact_duplicate_pairs.csv", index=False)
    summary = report.groupby("comparison")["phash_hamming_distance"].agg(["min", "median", "max", "count"])
    summary.to_csv(report_dir / "cross_split_near_duplicate_summary.csv")
    print(summary.to_string())


if __name__ == "__main__":
    main()
