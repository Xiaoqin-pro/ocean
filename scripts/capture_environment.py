"""Capture the runtime used for a reproducible experiment snapshot."""
from __future__ import annotations

import argparse
import platform
import subprocess
from pathlib import Path

import albumentations
import torch
import transformers

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def nvidia_driver() -> str:
    try:
        return subprocess.check_output(
            ["nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader"], text=True, stderr=subprocess.DEVNULL
        ).strip().splitlines()[0]
    except (OSError, subprocess.CalledProcessError, IndexError):
        return "unavailable"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=PROJECT_ROOT / "reports" / "environment.txt")
    args = parser.parse_args()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        f"Operating system: {platform.platform()}", f"Python version: {platform.python_version()}",
        f"PyTorch version: {torch.__version__}", f"Torch CUDA version: {torch.version.cuda}",
        f"CUDA available: {torch.cuda.is_available()}",
        f"GPU name: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'unavailable'}",
        f"GPU driver: {nvidia_driver()}", f"Transformers version: {transformers.__version__}",
        f"Albumentations version: {albumentations.__version__}",
    ]
    args.output.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
