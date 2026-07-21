# Underwater Calibration

This workspace supports the preliminary study of degradation-conditioned confidence calibration for underwater semantic segmentation.

## Current status

The current `v1_seed_20260721` random split is retained only as an engineering baseline. Cross-split exact duplicates were found, including train-to-official-test duplicates. Do not use v1 results as paper-ready results. `v2_grouped_deduplicated` removes exact-SHA leakage and known label conflicts, but remains provisional until pHash/dHash/feature near-duplicate review and scene-group isolation are completed. Do not start a formal v2 baseline until that review is recorded.

## Environment

The `venv` folder uses the system's CUDA-enabled PyTorch environment and its installed Transformers SegFormer implementation. This is the working SegFormer baseline on this computer; the currently available Python 3.13 runtime does not have a compatible full-MMCV extension for MMSegmentation. Activate it in PowerShell with:

```powershell
.\\venv\\Scripts\\Activate.ps1
```

Run `python scripts/check_environment.py` after activation to verify Python, PyTorch, CUDA, and a SegFormer model instance.

## Layout

```text
configs/        Experiment configurations
data/           Local datasets (not tracked by Git)
datasets/       Dataset conversion and loading code
models/         Model code
calibration/    Calibration methods
degradations/   Controlled degradation generation
metrics/        Reliability and selective-prediction metrics
scripts/        Setup, validation, and training utilities
outputs/        Checkpoints, logs, and visualizations (not tracked by Git)
notebooks/      Exploratory analysis
```

The SUIM and DUT-USEG data are intentionally not bundled. Put downloaded data under `data/` and do not commit it.
