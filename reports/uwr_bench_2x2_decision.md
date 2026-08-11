# UWR-Bench 2×2 Fixed-Protocol Decision

## Completed matrix

| Dataset | SegFormer-B0 | DeepLabV3-MobileNetV3-Large |
| --- | --- | --- |
| SUIM v2 | completed | completed |
| Leakage-audited UIIS | completed | completed |

The UIIS DeepLab cell was completed without retraining after its fixed 60-epoch
checkpoint and without accessing the locked SUIM official TEST.

## Evidence decision

| Benchmark-level question | Decision | Fixed evidence |
| --- | --- | --- |
| Does scalar temperature scaling improve probability calibration without changing segmentation? | PASS | Temperature scaling lowers NLL/ECE while argmax, mIoU, Dice, and accuracy remain invariant in all completed cells. |
| Does improved probability calibration imply improved error ranking? | PASS: separation | UIIS DeepLab shows no stable eAURC gain and raw MSP has higher full/interior AUPRC and top-10% recall; the other frozen cells do not provide a stable joint ranking gain from calibration. |
| Are residual errors boundary concentrated? | PASS | UIIS DeepLab has a +24.40pp mean boundary-minus-interior error gap, with positive 95% CIs in all 13 conditions; the prior three cells show the same qualitative pattern. |
| Does visual-quality grouped CRC stably improve selective coverage? | PASS: negative result | The quality CRC direction changes by target/model/dataset; UIIS DeepLab gains at two targets have adverse risk components and loses coverage at α=0.15. |

## Consequence

The evidence supports UWR-Bench as a reliability benchmark paper, not a
quality-conditioned CRC method paper. The rejected DARC-Seg formulation remains
a documented negative control. No third backbone, new quality descriptor, new
cluster count, selector change, or test-set tuning is warranted.

## Remaining work before the one final SUIM TEST evaluation

1. Add semantic-validity and engineering summaries using already frozen outputs:
   foreground/background coverage, per-class selective risk, boundary coverage,
   small-object coverage, and post-processing cost.
2. Freeze figures, main tables, metrics, and paper claims.
3. Only then evaluate the locked SUIM official TEST exactly once under the fixed
   final protocol.
