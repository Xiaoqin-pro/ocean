# Project Evidence Status

## Succeeded

- Scene-grouped, leakage-audited SUIM protocol with an isolated calibration split and locked official TEST.
- Controlled 13-condition underwater degradation reliability benchmark.
- Clean-global temperature scaling for probability calibration.
- Boundary residual-risk diagnosis using ground truth only as an evaluation stratum.
- Tie-aware frozen-logit uncertainty-ranking benchmark.
- DeepLabV3-MobileNetV3 replication on the formal SUIM protocol.  The CNN
  architecture gate passed: scalar temperature scaling improved calibration
  without improving ranking, boundary risk exceeded interior risk in every
  registered condition, and quality-group CRC did not provide stable
  risk--coverage gains.

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

## CNN architecture gate: PASSED

The SUIM reliability decomposition now holds for a Transformer baseline and a
lightweight CNN baseline.  DARC-Seg remains rejected; this result supports the
UWR-Bench diagnostic framing, not a new selection method.  The official SUIM
TEST remains locked.

## Next evidence gate

Extend the pre-registered reliability matrix to the leakage-screened UIIS
protocol.  First add the registered benchmark metrics to the already trained
UIIS SegFormer-B0 without retraining or method development.  UIIS
DeepLabV3-MobileNetV3 training is then permitted under the fixed 60-epoch
protocol.  UIIS confirmation is an external benchmark extension, not a new
blind confirmation, because it was previously opened for the DARC negative
control.
