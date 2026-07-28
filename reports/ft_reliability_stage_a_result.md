# FT-Reliability v1.2 Stage-A Result

**Scientific run ID:** `ftdev-20260727T200249-8d058c6`  
**Final implementation:** `f6425c59951060ff260b9b9fcf3b2647b241fbdf`  
**Decision:** **FAIL — FT-Reliability / TCCR is permanently stopped as a method route.**

## Frozen evaluation record

- Five SegFormer final-epoch checkpoints (A--E) were fixed before development evaluation.
- The 231-image `method_development` role was evaluated once over clean plus twelve registered degradation conditions.
- The 65 model-condition prediction caches were generated atomically; the subsequent metric-domain audit contained 45,045 per-image-condition-region records.
- The audit found zero non-finite occurrences for required metrics. The only undefined per-image diagnostics were the declared mathematical cases: no positive errors for Error AUPRC/top-10% error recall, and one-class error targets for Error AUROC.
- The final aggregation used cached predictions, direct pixel-level sufficient statistics, 1,000 paired sample-ID cluster bootstrap replicates, and no further model inference or training.
- Validation, calibration, official SUIM TEST, and external datasets remained unaccessed.

## Pre-registered primary endpoint

The unique primary endpoint was

\[
I_{\mathrm{eAURC}}=\mathrm{eAURC}_{C}-\mathrm{eAURC}_{E},
\]

evaluated on the full region with a 1,000-replicate paired scene-cluster bootstrap. The required criterion was a strictly positive 95% CI lower bound.

| Comparison | Region | Metric | Mean improvement | 95% CI | Result |
| --- | --- | ---: | ---: | ---: | --- |
| E vs C | full | eAURC | -0.003671 | [-0.007097, -0.000275] | Fail |

The CI is wholly negative: E worsened the pre-registered error-ranking endpoint relative to C.

## Supporting fixed comparisons

| Comparison | Region | Metric | Mean improvement | 95% CI |
| --- | --- | ---: | ---: | ---: |
| E vs C | boundary | eAURC | -0.011092 | [-0.014887, -0.007377] |
| E vs C | full | Error AUPRC | -0.011474 | [-0.020716, -0.002931] |
| E vs B | full | eAURC | -0.003189 | [-0.007346, 0.001172] |

These secondary outcomes do not rescue the method: the boundary result is directionally adverse and the full-region AUPRC result is also directionally adverse.

## Consequences

1. Do not retune the TCCR loss, boundary sampling, warm-up, retention term, margin, seed, or stopping rule.
2. Do not run the planned DeepLab replication for this method.
3. Do not access formal SUIM validation/calibration/TEST to seek a more favorable result.
4. Preserve the code, cached predictions, domain audit, checkpoints, attempt records, result-file SHA-256 manifest, and this report as a negative method result.
5. Retain the broader UWR-Bench reliability findings as the active research asset; this negative result may be cited as evidence that transition-conditioned ranking did not improve error ranking under the locked protocol.
