# UIIS alpha=0.10 calibration-only freeze

## Training provenance

- Model: SegFormer-B0, fixed 60 epochs, 384x384, AMP, batch size 2.
- Completed checkpoint: epoch 60, global step 71,160.
- Checkpoint SHA-256:
  `C2EAD86D324A02AD874C5DBDF88CC5238F9ABC0C9E673A1EE08D5CFE99DB1B52`.
- Final train loss: 0.214086.
- The training checkpoint records
  `official_suim_test_evaluated=false` and
  `confirmation_evaluated=false`.

## Calibration-only controller fit

The following values were fitted before opening UIIS confirmation:

- Target risk: alpha = 0.10.
- Measurement resolution: 192x192.
- Conditions: clean plus four frozen degradation families at three severities
  (13 conditions total).
- Independent calibration clusters: 508 original images, with all
  deterministic condition variants retained inside each cluster.
- Quality groups: three KMeans groups, fitted from 30,823 train-only
  input-descriptor rows; no labels or predictions entered grouping.

The global CRC controller selected coverage 0.22. Its empirical calibration
risk was 0.09754 and its finite-sample corrected risk was 0.09932.

All three frozen KMeans seeds retained three usable quality groups. The
smallest group had 426 independent calibration images, above the preregistered
minimum of 80. Across the seeds, selected group coverages were 0.17, 0.18,
0.19, or 0.25, with the global 0.22 controller recorded as fallback.

## Boundary

This is a calibration parameter freeze, not a confirmation result. No
confirmation logits, labels, risks, coverage, bootstrap interval, or model
selection result was read. The SUIM official TEST remains locked.

The next permissible step is to commit this freeze and then open UIIS
confirmation exactly once using these fixed parameters.
