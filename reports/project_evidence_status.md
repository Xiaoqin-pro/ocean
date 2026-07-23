# Project Evidence Status

## Succeeded

- Scene-grouped, leakage-audited SUIM protocol with an isolated calibration split and locked official TEST.
- Controlled 13-condition underwater degradation reliability benchmark.
- Clean-global temperature scaling for probability calibration.
- Boundary residual-risk diagnosis using ground truth only as an evaluation stratum.
- Tie-aware frozen-logit uncertainty-ranking benchmark.

## Not supported

- Per-degradation scalar temperature scaling.
- Entropy, energy, or local-disagreement error ranking as a new method.
- Calibration-only logistic ranking fusion.
- Boundary, spatial-temperature, or error-ranking neural refinement.

## Exploratory signal requiring independent confirmation

The broad DARC-Seg hypothesis did not pass its preregistered three-risk-target
gate.  At the exploratory operating point `alpha=0.10`, however, train-only
quality grouping improved validation coverage by 4.52--5.90pp over global CRC
without a low-light safety deterioration.  This is not a final claim and must
not be tuned further on SUIM validation.

The only permitted continuation is a separately preregistered UIIS confirmation
at `alpha=0.10`, after exact and near-duplicate auditing against all SUIM
partitions.  The SUIM official TEST remains locked.
