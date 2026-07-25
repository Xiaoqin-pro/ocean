# AquaRiskMap Stage-A — DeepLab ranking result

**Status:** frozen ranking-only Stage-A sub-result; CRC and latency gates remain pending.  
**Evaluation-code commit:** `9fa53f9aad24df211a208902bcd38f47f929a50c`  
**Result-freeze parent commit:** `0f3c859b5589515d938d02604bb8f2c2ba1f64a2`  
**Risk-head checkpoint SHA-256:** `5B1BAB00A0061112648ECD223D4BF67D06A6E976B904FAC35727B22CC64B06F4`  
**Pilot config SHA-256:** `3EDEFCA3A02115521C35A109D95E8231D9D6D4EE914EB4165945DF4EC3A5579D`

## Protocol integrity

- Evaluation split: frozen SUIM `val` only, 146 original-image clusters.
- Conditions: the fixed clean plus 12 registered degradations.
- Scores: raw MSP, six previously frozen secondary scores, and AquaRiskMap.
- Regions: full, GT boundary radius 3, and interior.
- Statistics: 1,000 paired `sample_id` cluster-bootstrap replicates; every draw retains all 13 conditions for an image.
- Calibration was not read; SUIM official TEST was not read; no model was retrained.
- The result table contains 45,552 unique rows; all numeric output values are finite.

## Fixed raw-MSP comparison

| Region | Metric | Raw MSP | AquaRiskMap | Candidate − raw |
| --- | --- | ---: | ---: | ---: |
| Full | eAURC | 0.055130 | 0.055026 | -0.000104 |
| Full | Error AUPRC | 0.461027 | 0.471518 | +0.010491 |
| Full | Top-10% error recall | 0.414341 | 0.421686 | +0.007345 |
| Boundary | eAURC | 0.240860 | 0.231721 | -0.009139 |
| Boundary | Error AUPRC | 0.538582 | 0.555073 | +0.016491 |
| Boundary | Top-10% error recall | 0.136582 | 0.142419 | +0.005836 |

The fixed ranking sub-gate does **not** pass for DeepLab.  The eAURC direction is positive but far below the required 10% relative decrease (full: 0.19%; boundary: 3.79%); full AUPRC and top-10% recall also remain below the preregistered +3pp and +5pp thresholds.  This does not authorize a method change: the Global-CRC and latency gates remain required for a complete, transparent Stage-A failure record.

## Local result artifacts

The complete local, ignored result set is under:

`outputs/aquariskmap_suim_pilot_v1/stage_a/deeplab/`

It contains `per_image_metrics.csv`, `aggregate_metrics.csv`, `clustered_bootstrap.csv`, and `metadata.json`.
