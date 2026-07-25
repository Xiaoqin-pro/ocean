# UIIS DeepLabV3-MobileNetV3 Temperature-Scaling Benchmark

## Protocol

- Model: fixed final-epoch UIIS DeepLabV3-MobileNetV3 checkpoint.
- Checkpoint SHA-256: `38EB9EE1AAE4BC28B196017C17E726EE91FE63457FD845A64D6179D3F75FED67`.
- Conditions: clean plus four image-only degradations at three severities (13 conditions).
- Fitting split: calibration (508 images); evaluation split: confirmation (511 images).
- Official SUIM TEST evaluated: **false**.
- Model retrained: **false**.
- Temperatures were fit only by calibration NLL. Confirmation was never used for fitting.

## Accepted result artefacts

- 104 rows: 4 methods × 13 conditions × 2 splits.
- 832 per-class rows: 104 result rows × 8 classes.
- All numeric result fields are finite and every method/condition/split key is unique.
- The maximum per-condition, cross-method difference in mIoU, pixel accuracy,
  mean accuracy, and mean Dice is exactly `0.0`.
- 13 targeted tests covering temperature scaling and the UIIS DeepLab cache passed.

## Temperatures

| Method / scope | Temperature |
| --- | ---: |
| Raw softmax | 1.000000 |
| Clean global | 1.938024 |
| Pooled conditions | 2.062836 |
| Per-degradation clean | 1.938024 |
| Per-degradation color attenuation | 1.975146 |
| Per-degradation turbidity | 1.968173 |
| Per-degradation low light | 2.009111 |
| Per-degradation blur | 2.343192 |

## Confirmation-set summary (mean over 13 conditions)

| Method | NLL | ECE | Brier | Error AUROC | AURC |
| --- | ---: | ---: | ---: | ---: | ---: |
| Raw | 1.017409 | 0.126618 | 0.397809 | 0.751016 | 0.124524 |
| Clean global | 0.777101 | 0.027972 | 0.373552 | 0.750849 | 0.124540 |
| Pooled | 0.774448 | 0.032892 | 0.374317 | 0.750207 | 0.124728 |
| Per-degradation | 0.773086 | 0.029422 | 0.373987 | 0.750150 | 0.124779 |

Temperature scaling substantially improves probability calibration, while the
mean error-ranking metrics do not show a stable improvement. This is a fixed
external benchmark extension because UIIS confirmation was previously opened
for the DARC negative-control study; it is not a new blind confirmation.

## Execution and provenance

The evaluator reads only 26 frozen low-resolution-logit caches. It shares each
upsample and segmentation computation across the four temperature methods,
computes calibration terms on CUDA, and atomically persists one complete
condition-split partial result at a time. Thus interruption can be resumed
without recomputing completed conditions. Full provenance, cache and split
hashes are stored in `outputs/uiis_deeplab_temperature_scaling/evaluation_metadata.json`.

No independently stored pre-temperature UIIS DeepLab raw-metric table existed
before this run; therefore the raw-cache-versus-prior-pilot comparison is
explicitly marked unavailable rather than claimed. The frozen caches, checkpoint
hash, split hashes, and temperature-invariant segmentation checks provide the
recorded audit trail for this evaluation.
