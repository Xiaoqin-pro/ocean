# UWR-Bench Result Matrix

This table is a reporting template. Cells marked **pending** must not be interpreted as evidence.

## Base segmentation performance

| Dataset | Model | Evaluation partition | Clean mIoU | Input size | Status |
| --- | --- | --- | ---: | ---: | --- |
| SUIM | SegFormer-B0 | v2 validation | 0.602939 | 384 | Frozen formal baseline |
| SUIM | DeepLabV3-MobileNetV3-Large | v2 validation | 0.560668 | 384 | Frozen architecture replication |
| UIIS | SegFormer-B0 | confirmation | 0.518948 | 384 | Fixed-protocol benchmark extension |
| UIIS | DeepLabV3-MobileNetV3-Large | confirmation | 0.467299 | 384 | Frozen fixed-protocol benchmark extension |

## Reliability evidence

| Dataset | Model | Raw ECE | Clean-global TS ECE | Ranking benchmark | Boundary gap | Quality CRC |
| --- | --- | ---: | ---: | --- | --- | --- |
| SUIM | SegFormer-B0 | Frozen in baseline report | Frozen in temperature report | Frozen | Frozen | Exploratory negative control |
| SUIM | DeepLabV3-MobileNetV3-Large | Frozen in replication report | Frozen in replication report | Frozen | Frozen | Architecture-gate negative control |
| UIIS | SegFormer-B0 | 0.120271 | 0.025261 | Raw MSP vs calibrated MSP eAURC CI crosses 0 | 22.76pp mean boundary-minus-interior error rate | Fixed three-alpha sensitivity: unstable / negative at alpha 0.10 and 0.15 |
| UIIS | DeepLabV3-MobileNetV3-Large | 0.110327 | 0.027354 | Raw MSP has higher full/interior error AUPRC and top-10% error recall; eAURC has no stable raw advantage | 24.40pp mean boundary-minus-interior error rate; all 13 CIs positive | Fixed three-alpha sensitivity: target-dependent; quality grouping is not stably beneficial |

## Interpretation constraints

- UIIS SegFormer confirmation was previously opened for the DARC negative control. Its fixed benchmark extension is not a second blind confirmation.
- Temperature scaling preserves predicted labels and segmentation metrics; only probability-based reliability measures may change.
- Oracle condition CRC is a diagnostic upper-bound comparison, not a deployable method.
- SUIM official TEST remains locked until the final evaluation protocol is frozen.
- The completed UIIS DeepLab cell is an external fixed-protocol benchmark
  extension, not a new blind confirmation, because confirmation was previously
  opened for the DARC negative control.
