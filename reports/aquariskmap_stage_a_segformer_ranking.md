# AquaRiskMap Stage-A — SegFormer ranking result

**Status:** frozen ranking-only Stage-A sub-result; CRC and latency gates remain pending.  
**Evaluation commit:** `9fa53f9aad24df211a208902bcd38f47f929a50c`  
**Risk-head checkpoint SHA-256:** `4ED7018BE3A9291DD69368CB9922DCFABD8E8B636DA147755F09C389F5E438EB`  
**Pilot config SHA-256:** `3EDEFCA3A02115521C35A109D95E8231D9D6D4EE914EB4165945DF4EC3A5579D`

## Protocol integrity

- Evaluation split: frozen SUIM `val` only, 146 original-image clusters.
- Conditions: the fixed clean plus 12 registered degradations.
- Scores: raw MSP, six previously frozen secondary scores, and AquaRiskMap.
- Regions: full, GT boundary radius 3, and interior.
- Statistics: 1,000 paired `sample_id` cluster-bootstrap replicates; every draw retains all 13 conditions for an image.
- Calibration was not read; SUIM official TEST was not read; no model was retrained.
- The result table contains 45,552 unique rows with no non-finite numeric values.

## Fixed raw-MSP comparison

| Region | Metric | Raw MSP | AquaRiskMap | Candidate − raw |
| --- | --- | ---: | ---: | ---: |
| Full | eAURC | 0.044391 | 0.046349 | +0.001958 |
| Full | Error AUPRC | 0.452498 | 0.439976 | -0.012522 |
| Full | Top-10% error recall | 0.502789 | 0.498006 | -0.004784 |
| Boundary | eAURC | 0.194032 | 0.198355 | +0.004323 |
| Boundary | Error AUPRC | 0.500110 | 0.488729 | -0.011381 |
| Boundary | Top-10% error recall | 0.160501 | 0.153650 | -0.006851 |

The fixed ranking sub-gate does **not** pass for SegFormer: the full and boundary eAURC directions are negative (candidate eAURC is higher), and AUPRC/recall are lower.  This is an intermediate frozen observation, not a Stage-A decision: the preregistered DeepLab ranking result, Global CRC result, and latency measurement must still be completed without changing the method.

## Local result artifacts

The complete local, ignored result set is under:

`outputs/aquariskmap_suim_pilot_v1/stage_a/segformer/`

It contains `per_image_metrics.csv`, `aggregate_metrics.csv`, `clustered_bootstrap.csv`, and `metadata.json`.
