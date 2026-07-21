from __future__ import annotations

import importlib
import platform
import sys


def package_version(name: str) -> str:
    module = importlib.import_module(name)
    return getattr(module, "__version__", "installed")


def main() -> None:
    import torch
    from transformers import SegformerConfig, SegformerForSemanticSegmentation

    print(f"Python: {sys.version.split()[0]} ({platform.platform()})")
    print(f"PyTorch: {torch.__version__}")
    print(f"CUDA build: {torch.version.cuda}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")

    for package in ("transformers", "albumentations", "cv2"):
        print(f"{package}: {package_version(package)}")

    model = SegformerForSemanticSegmentation(SegformerConfig(num_labels=8))
    print(f"SegFormer labels: {model.config.num_labels}")


if __name__ == "__main__":
    main()
