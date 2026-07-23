# Calibration-Only Logistic Ranking Fusion

## Protocol

- Features: calibrated MSP uncertainty, 3×3 local disagreement, and probability-margin uncertainty.
- Model: a regularized logistic regression only; no neural module, no SegFormer retraining.
- Fit data: calibration caches only, with a deterministic 685-pixel-per-image sample across all 13 registered conditions (1,300,130 pixels).
- Evaluation: frozen validation caches only; official TEST was not accessed.
- All ranking metrics, including per-image clustered bootstrap, use the same tie-aware definitions as the frozen uncertainty-ranking benchmark.

## Fitted model

- Feature coefficients: calibrated MSP `+0.184`, local disagreement `-0.045`, probability margin `+0.753` after feature standardization.
- Calibration-sample error rate: `0.18079`.

## Validation result versus calibrated MSP

| Stratum | Mean relative eAURC improvement | AUPRC change | Top-10% error-recall change |
| --- | ---: | ---: | ---: |
| Full | +1.22% | +0.00445 | +0.00655 |
| Boundary | -0.16% | +0.00471 | +0.00337 |
| Interior | +1.41% | +0.00616 | +0.00781 |

Image-clustered bootstrap confirms a small full eAURC improvement (`+0.000629`, 95% CI `[+0.000202, +0.001031]`) and full top-10% recall gain (`+0.003689`, CI `[+0.001241, +0.006256]`). Full AUPRC is inconclusive because its interval crosses zero. Boundary eAURC also worsens slightly on average despite small boundary AUPRC and recall gains.

No one of the 13 conditions reaches the pre-registered 5% relative eAURC improvement. Low-light severity 3 and blur severity 3 remain below threshold, and boundary top-10% recall gain is far below five percentage points.

## Decision

This calibration-only fusion is a useful negative control, not an advancement candidate. Do not add a boundary head, spatial temperature head, or neural uncertainty module on the present evidence.

The frozen conclusion is that global temperature scaling helps probability calibration, while the examined non-learned ranking scores and a simple calibration-only fusion do not provide a sufficiently large or boundary-robust error-ranking improvement. The next research decision should be a documented scope revision, rather than further tuning against this validation split.
