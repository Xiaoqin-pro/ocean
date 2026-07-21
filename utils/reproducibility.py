from __future__ import annotations

import os
import random

import numpy as np

# Must be set before the first CUDA operation for deterministic CuBLAS paths.
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import torch


def set_seed(seed: int, *, deterministic: bool = True) -> None:
    """Set all relevant random seeds for a repeatable experiment."""
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = deterministic
    torch.backends.cudnn.benchmark = not deterministic
    if deterministic:
        torch.use_deterministic_algorithms(True, warn_only=True)
