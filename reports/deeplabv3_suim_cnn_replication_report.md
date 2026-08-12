# DeepLabV3-MobileNetV3 SUIM Reliability Replication

## Protocol

- Architecture: DeepLabV3-MobileNetV3-Large, ImageNet backbone initialization.
- Data: formal scene-grouped SUIM v2 split; validation and calibration only.
- Checkpoint: epoch 85, val mIoU 0.5606678204; SHA-256 `041655E0793D0F5DF792571E2F1FF9322082540971AE1A12709FC960DE7AAF59`.
- Conditions: clean plus four fixed degradation families at three severities.
- Official SUIM TEST: not evaluated.
- No model retraining occurred after the frozen 100-epoch baseline.

## Replication findings

1. **Calibration improves without changing segmentation decisions.** On clean validation, clean-global scalar temperature reduced ECE from 0.09070 to 0.02886 and NLL from 0.75664 to 0.56642, while mIoU remained exactly 0.56067. The 104-row cached evaluation verified argmax and segmentation invariants for all methods, conditions and splits.
2. **Calibration does not improve error ranking.** Across 13 validation conditions, raw MSP had a small but positive eAURC improvement over calibrated MSP: +0.000897 (scene-cluster bootstrap 95% CI [+0.000225, +0.001634]).
3. **Residual error concentrates at boundaries.** For raw MSP, boundary error rate exceeded interior error rate in every one of 13 conditions. The mean gap was 0.2939 (29.39 percentage points); the minimum was 20.89 percentage points.
4. **Quality-group CRC is not stable across risk targets.** At alpha=0.10, quality-group coverage changes were non-significant or negative across all three frozen KMeans seeds. At alpha=0.15, all seeds significantly reduced coverage by 4.72--5.43 pp. The small alpha=0.05 gain (~0.35 pp) occurred while global CRC retained only ~1% coverage, a floor regime without usable method headroom.

## Decision

The CNN independently reproduces all three benchmark-level observations needed for the architecture gate: calibration/ranking separation, boundary-dominated residual risk, and instability of quality-conditioned CRC. It does **not** revive the rejected DARC-Seg method hypothesis; the result strengthens the leakage-audited UWR-Bench reliability benchmark framing instead.
