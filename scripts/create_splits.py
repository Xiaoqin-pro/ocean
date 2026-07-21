from __future__ import annotations

import argparse
import hashlib
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SEED = 20260721


def write_split(dataframe: pd.DataFrame, path: Path) -> str:
    ordered = dataframe.sort_values("sample_id").reset_index(drop=True)
    ordered.to_csv(path, index=False)
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    print(f"{path.name}: {len(ordered)} (sha256={digest})")
    return digest


def main() -> None:
    parser = argparse.ArgumentParser(description="Create the fixed SUIM v1 split.")
    parser.add_argument("--manifest", type=Path, default=PROJECT_ROOT / "data" / "suim_processed" / "manifest.csv")
    parser.add_argument("--output-dir", type=Path, default=PROJECT_ROOT / "data" / "suim_processed" / "splits" / f"v1_seed_{SEED}")
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    if not args.manifest.is_file():
        raise FileNotFoundError(f"Manifest does not exist: {args.manifest}")
    manifest = pd.read_csv(args.manifest)
    development = manifest.loc[manifest["partition"] == "train_val"].copy()
    official_test = manifest.loc[manifest["partition"] == "test"].copy()
    if len(development) != 1525 or len(official_test) != 110:
        raise ValueError(f"Expected 1525 development and 110 test samples, got {len(development)} and {len(official_test)}")
    if manifest["sample_id"].duplicated().any():
        raise ValueError("Manifest has duplicated sample_id values across partitions.")

    train, remainder = train_test_split(development, train_size=1220, random_state=args.seed, shuffle=True)
    validation, calibration = train_test_split(remainder, train_size=152, random_state=args.seed, shuffle=True)
    if len(train) != 1220 or len(validation) != 152 or len(calibration) != 153:
        raise RuntimeError("Unexpected split sizes.")
    ids = set(train.sample_id) | set(validation.sample_id) | set(calibration.sample_id)
    if len(ids) != 1525:
        raise RuntimeError("Development split overlap or omission detected.")

    if args.output_dir.exists() and any(args.output_dir.iterdir()) and not args.overwrite:
        raise FileExistsError(f"Split directory exists: {args.output_dir}. Use --overwrite to regenerate it.")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    hashes = {
        "train": write_split(train, args.output_dir / "train.csv"),
        "val": write_split(validation, args.output_dir / "val.csv"),
        "calibration": write_split(calibration, args.output_dir / "calibration.csv"),
        "test": write_split(official_test, args.output_dir / "test.csv"),
    }
    (args.output_dir / "README.txt").write_text(
        "\n".join([
            f"seed={args.seed}", "train=1220", "val=152", "calibration=153", "test=110",
            "The official TEST partition is excluded from training, tuning, and calibration.",
            *(f"{name}_sha256={digest}" for name, digest in hashes.items()),
            "",
        ]), encoding="utf-8")


if __name__ == "__main__":
    main()
