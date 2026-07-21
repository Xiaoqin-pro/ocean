# Underwater Calibration

This workspace supports the preliminary study of degradation-conditioned confidence calibration for underwater semantic segmentation.

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
