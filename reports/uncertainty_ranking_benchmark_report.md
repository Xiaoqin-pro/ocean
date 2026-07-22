# Frozen-Logit Uncertainty-Ranking Benchmark

## Protocol

- Source: the frozen `v0.4-temperature-scaling` validation logits and clean-global temperature `T = 2.1971989`.
- Scope: 13 registered conditions × 7 uncertainty scores × full/boundary/interior strata = 273 aggregate rows.
- Official TEST was not accessed and no model or calibration network was trained.
- Boundary labels are evaluation strata only; no score receives ground-truth boundary information.
- Scores: raw MSP, calibrated MSP, entropy, probability margin, logit margin, energy, and 3×3 local prediction disagreement.

## Ranking definitions and validation

- All metrics are tie-aware. A cutoff inside a tied score group uses the expected error count implied by that group's error proportion; it never uses raster order.
- AURC uses the same tie-aware expectation, and eAURC is `AURC - oracle AURC`.
- AUPRC is validated against `sklearn.metrics.average_precision_score` for no-tie, tie-heavy, all-equal, rare-error, and balanced-error cases.
- Local disagreement has nine discrete levels and is evaluated from a score-level histogram rather than a full pixel sort.
- Scores are calculated from float32 logits after the frozen AMP prediction is recorded; an assertion verifies identical argmax predictions.
- The complete unit suite passed: 33 tests.

## Runtime and provenance

- All 13 condition files are atomically written under `outputs/uncertainty_ranking/conditions/` and the runner supports `--resume`.
- Peak process working set was approximately 3.41 GB.
- Typical non-discrete score sorts took about 1.3–2.8 seconds per condition; local disagreement's histogram path took about 0.5–0.9 seconds.
- `metadata.json` records `official_test_evaluated: false`, `model_retrained: false`, the checkpoint hash, and the degradation-config hash.

## Main validation finding

The calibration baseline and the best error-ranking score are not the same:

| Candidate vs calibrated MSP | Full-stratum clustered result | Interpretation |
| --- | --- | --- |
| Raw MSP | eAURC improvement `+0.001090`, 95% CI `[+0.000306, +0.001890]`; AUPRC `+0.008874`, CI `[+0.005507, +0.012394]`; top-10% error recall `+0.005104`, CI `[+0.001788, +0.008469]` | Reliable but modest ranking gain after temperature scaling is removed. |
| Probability margin | eAURC `+0.000723`, CI `[+0.000208, +0.001227]`; AUPRC CI crosses zero | Weak evidence only. |
| Logit margin | eAURC and AUPRC CIs cross zero | No stable gain. |
| Entropy, energy, local disagreement | Negative full-stratum eAURC/AUPRC/recall intervals | Reliably worse than calibrated MSP. |

Raw MSP has roughly a 2–3% average full-stratum relative eAURC improvement over calibrated MSP, below the pre-registered 5% threshold. Its boundary eAURC interval crosses zero, and its boundary top-10% recall gain is only about 0.25 percentage points, far below the required 5 points.

Low-light severity 3 and blur severity 3 show the same pattern: raw MSP has a small full-stratum eAURC advantage, while no candidate achieves a stable joint gain in eAURC, AUPRC, and boundary top-10% recall.

## Decision

No single uncertainty score qualifies for a new learned uncertainty module. Do not add a boundary head or train a neural refinement model.

The only justified next experiment is a calibration-only, regularized logistic-regression fusion of `calibrated_msp`, local disagreement, and margin features. It must be fitted on calibration caches only and evaluated on the already frozen validation conditions, with the same tie-aware ranking metrics and no official TEST access.
