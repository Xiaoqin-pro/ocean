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
- DARC-Seg quality-group CRC as a coverage-improving method.

## Closed negative-control result

The broad DARC-Seg hypothesis did not pass its preregistered three-risk-target
gate. Its sole exploratory operating point, alpha = 0.10, was then evaluated
once on the independent, leakage-screened UIIS confirmation protocol.

All three frozen quality-group seeds reduced coverage relative to Global CRC
(-1.63pp, -2.14pp, and -1.64pp); every paired bootstrap interval was below
zero. Risk was slightly lower because the candidate was more conservative,
not because it improved risk-coverage efficiency. DARC-Seg is retained as a
negative control and must not be tuned further.

## Next evidence gate

The project now tests whether its reliability observations are architecture
dependent. The minimum gate is a DeepLabV3-MobileNetV3 replication on the
formal SUIM protocol, covering calibration versus ranking, boundary versus
interior risk, and Global versus quality-group CRC. The UIIS CNN evaluation is
permitted only if the predefined cross-architecture gate is met.
