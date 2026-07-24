# UIIS alpha=0.10 independent confirmation

## Result

The preregistered quality-group CRC candidate did **not** pass independent
confirmation. This is a negative result and closes the DARC-Seg method route;
no descriptor, seed, group-count, backbone, or risk-target search follows it.

## Frozen protocol

- SegFormer-B0 was trained once for 60 fixed epochs on the leakage-screened
  UIIS train partition.
- CRC controllers were fit only on 508 calibration-image clusters and frozen
  before confirmation was opened.
- Confirmation comprised 511 independent image clusters, evaluated at
  alpha = 0.10 over the clean plus 12 deterministic degradation conditions.
- Bootstrap used 1,000 paired resamples of original-image clusters.
- SUIM official TEST was not evaluated.

## Primary comparison

Global CRC selected calibration coverage 0.22. On UIIS confirmation, the
quality-group controller had lower aggregate selective risk but also lower
coverage for every frozen KMeans seed:

| Seed | Coverage difference vs. Global CRC | 95% CI | Global risk | Quality risk |
| ---: | ---: | ---: | ---: | ---: |
| 20260722 | -1.63 pp | [-1.80, -1.45] pp | 0.0740 | 0.0713 |
| 20260723 | -2.14 pp | [-2.32, -1.92] pp | 0.0740 | 0.0705 |
| 20260724 | -1.64 pp | [-1.81, -1.48] pp | 0.0740 | 0.0713 |

Thus the primary requirement (coverage improvement of at least 3 pp with a
positive 95% lower confidence bound) fails for all three seeds. The aggregate
risk, severe low-light safety, and worst-quality-group safety checks did not
show a violation, but safety alone cannot rescue a method whose primary
coverage endpoint is negative.

## Decision

The exploratory SUIM signal at alpha = 0.10 did not replicate on the
independent, leakage-screened UIIS confirmation set. The scientifically
correct conclusion is that the current quality-conditioned CRC formulation is
not supported as a general or moderate-risk coverage-improvement method.

The project retains the reliable segmentation, calibration, degradation, and
selective-risk analysis infrastructure. Future work must begin from a new,
independently motivated hypothesis rather than tuning this failed route.
