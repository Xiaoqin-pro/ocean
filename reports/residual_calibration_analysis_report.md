# Residual Calibration Analysis after Clean-Global Temperature Scaling

## Protocol

- Source: frozen `v0.4-temperature-scaling` validation caches and result tables.
- No image model was loaded or retrained; official TEST remained locked.
- Paired bootstrap unit: image; 1,000 deterministic resamples.
- Boundary band: label-neighbourhood radius 3 pixels.

## Condition-level and paired-bootstrap result

Neither scalar alternative has a reliable aggregate validation NLL advantage over clean-global temperature scaling:

| Comparison (candidate − clean global) | Metric | Mean difference | 95% CI |
| --- | --- | ---: | --- |
| Pooled | NLL | +0.000918 | [−0.000050, +0.001871] |
| Per-degradation | NLL | +0.000732 | [−0.000646, +0.002026] |
| Pooled | Brier | +0.000572 | [+0.000362, +0.000782] |
| Per-degradation | Brier | +0.000457 | [+0.000161, +0.000725] |
| Pooled | ECE | +0.005415 | [+0.005069, +0.005754] |
| Per-degradation | ECE | +0.005054 | [+0.004550, +0.005545] |

Positive values indicate that the candidate is worse.  Per-degradation temperature improves NLL only in isolated conditions, most notably low-light severity 3 (−0.006467), but degrades several low-light and blur conditions.  It is not a justified final method.

## Class-level residual calibration

Clean-global temperature substantially reduces classwise ECE across the evaluated conditions.  The largest absolute improvements occur for `reefs_invertebrates`, including blur severity 3: `0.077047 → 0.022361`.  This means the next gap is not simply a single globally miscalibrated class.

## Boundary versus interior

| Method | Region | NLL | ECE | Brier | Error AUROC | AURC | Wrong-pixel confidence |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Raw | Boundary | 1.482770 | 0.197162 | 0.589659 | 0.608372 | 0.327711 | 0.772276 |
| Clean global | Boundary | 1.067333 | 0.066519 | 0.541357 | 0.609775 | 0.326067 | 0.593936 |
| Raw | Interior | 0.705514 | 0.087492 | 0.224071 | 0.788302 | 0.052819 | 0.852922 |
| Clean global | Interior | 0.469511 | 0.021514 | 0.211375 | 0.782923 | 0.054197 | 0.707717 |

The boundary band represents roughly 10% of valid pixels but has markedly larger residual NLL, ECE, Brier and selective-risk error.  Global temperature improves probability calibration and lowers wrong-pixel confidence, yet does not materially improve boundary error ranking.  This supports focusing future work on spatially localized uncertainty/error localization rather than a more complex degradation-type scalar temperature.

## Decision

Keep clean-global temperature scaling as the calibration baseline.  Do not add a per-degradation scalar temperature head.  The next justified experiments compare error-ranking scores and investigate lightweight image- or boundary-aware residual uncertainty methods, using the frozen validation protocol.
