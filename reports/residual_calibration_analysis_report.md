# Residual Calibration and Boundary Robustness Analysis

## Protocol

- Source: frozen `v0.4-temperature-scaling` validation caches and result tables.
- The SegFormer checkpoint was not retrained; official TEST remained locked.
- All 13 validation conditions were retained for each original image.
- Confidence intervals use 1,000 deterministic bootstrap resamples with `sample_id` as the cluster: resampling an image always retains its 13 degradation conditions.
- Ground-truth boundaries are evaluation strata only. They are never model inputs or deployment-time signals.

## Scalar-temperature residuals

Neither scalar alternative has a reliable aggregate validation NLL advantage over clean-global temperature scaling. Positive differences mean that the candidate is worse.

| Candidate minus clean-global | Metric | Mean difference | 95% CI |
| --- | --- | ---: | --- |
| Pooled | NLL | +0.000918 | [-0.000050, +0.001871] |
| Per-degradation | NLL | +0.000732 | [-0.000646, +0.002026] |
| Pooled | Brier | +0.000572 | [+0.000362, +0.000782] |
| Per-degradation | Brier | +0.000457 | [+0.000161, +0.000725] |
| Pooled | ECE | +0.005415 | [+0.005069, +0.005754] |
| Per-degradation | ECE | +0.005054 | [+0.004550, +0.005545] |

Per-degradation temperature improves NLL only in isolated conditions, most notably low-light severity 3, while degrading several low-light and blur conditions. It is therefore not a justified final calibration method.

## Boundary-radius sensitivity

The table aggregates all 13 validation conditions. `eAURC = AURC - oracle AURC`; a larger value means a larger gap from ideal error ranking.

| Radius | Region | Pixel share | Error rate | Clean-global NLL | Clean-global ECE | Clean-global eAURC | Error AUROC | Wrong-pixel confidence |
| ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | Boundary | 3.54% | 48.23% | 1.226486 | 0.097441 | 0.314003 | 0.552445 | 0.592136 |
| 1 | Interior | 96.46% | 14.56% | 0.502941 | 0.024330 | 0.048007 | 0.782502 | 0.691018 |
| 3 | Boundary | 9.88% | 38.23% | 1.067333 | 0.066519 | 0.240951 | 0.609775 | 0.593936 |
| 3 | Interior | 90.12% | 13.29% | 0.469511 | 0.021514 | 0.044662 | 0.782923 | 0.707717 |
| 5 | Boundary | 15.23% | 33.04% | 0.970402 | 0.062406 | 0.193336 | 0.646312 | 0.602383 |
| 5 | Interior | 84.77% | 12.65% | 0.449218 | 0.018202 | 0.042317 | 0.786273 | 0.717083 |

The expected drop in boundary error rate as the band widens does not explain the residual: at every radius, boundary NLL and excess selective risk remain much larger than interior values.

## Image-clustered robustness evidence

All intervals below are image-clustered across the 13 conditions. The boundary-minus-interior comparison uses clean-global temperature.

| Radius | Comparison | Metric | Difference | 95% CI |
| ---: | --- | --- | ---: | --- |
| 1 | Boundary - interior | NLL | +0.717809 | [+0.658412, +0.773163] |
| 1 | Boundary - interior | AURC | +0.352691 | [+0.334290, +0.370010] |
| 1 | Boundary - interior | eAURC | +0.236841 | [+0.225205, +0.248507] |
| 1 | Clean-global - raw boundary | NLL | -0.520463 | [-0.609931, -0.432460] |
| 1 | Clean-global - raw boundary | ECE | -0.129115 | [-0.138980, -0.119135] |
| 3 | Boundary - interior | NLL | +0.591341 | [+0.531533, +0.647743] |
| 3 | Boundary - interior | ECE | +0.018931 | [+0.003508, +0.033111] |
| 3 | Boundary - interior | AURC | +0.218846 | [+0.198384, +0.238280] |
| 3 | Boundary - interior | eAURC | +0.154530 | [+0.143027, +0.166548] |
| 3 | Clean-global - raw boundary | NLL | -0.391857 | [-0.480733, -0.305881] |
| 3 | Clean-global - raw boundary | ECE | -0.055893 | [-0.072591, -0.038609] |
| 5 | Boundary - interior | NLL | +0.509932 | [+0.445886, +0.567567] |
| 5 | Boundary - interior | ECE | +0.023732 | [+0.007782, +0.037565] |
| 5 | Boundary - interior | AURC | +0.155258 | [+0.137107, +0.173561] |
| 5 | Boundary - interior | eAURC | +0.112233 | [+0.101497, +0.124102] |
| 5 | Clean-global - raw boundary | NLL | -0.342649 | [-0.439476, -0.252914] |
| 5 | Clean-global - raw boundary | ECE | -0.026655 | [-0.045171, -0.008350] |

The boundary ECE excess is not significant at radius 1, but it is positive at radii 3 and 5. More importantly, the NLL, AURC, and eAURC gaps are positive with intervals excluding zero at every radius. Clean-global temperature reliably improves boundary probability calibration, yet it does not close the boundary error-ranking gap.

## Decision

Keep clean-global temperature scaling as the calibration baseline. Do not add a boundary head or a neural refinement module now. The justified next phase is an uncertainty-ranking benchmark: calibrated MSP versus entropy, probability margin, logit margin, energy, and local disagreement, evaluated on full, boundary, and interior strata. A ground-truth-boundary two-temperature oracle is diagnostic only and must not be treated as a deployable method.
