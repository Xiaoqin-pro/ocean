# DTS-Seg Gate-0 result

## Frozen decision

**FAIL.** The exact degradation-trajectory straightening loss registered in
`dts_seg_gate0_preregistration.md` is closed. Its weight, tolerance, mask,
training duration, and checkpoint rule must not be tuned on the 231-image
development role.

| Metric (DTS - F4) | Change (mIoU points) |
|---|---:|
| 13-condition mean | -0.509 |
| four severity-3 conditions | -0.491 |
| clean | -0.534 |
| color severity 3 | -0.568 |
| turbidity severity 3 | -0.472 |
| low-light severity 3 | -0.482 |
| blur severity 3 | -0.443 |

All four severe degradation families were negative. Formal validation,
calibration, and official SUIM test data were not evaluated.

## Interpretation

The loss supplied dense supervision (roughly 33k--35k eligible pixels per
batch) and optimized stably, but its effect was a nearly uniform reduction in
segmentation quality. A constraint on the curvature of semantic margins asks
the backbone not to collapse without giving it a mechanism that can undo the
feature distortion. The next hypothesis must therefore add a learnable,
input-conditioned recovery operation rather than another scalar penalty on
the same logits.

The raw, condition-level evidence and machine-readable decision are stored
under `outputs/dts_seg_gate0/evaluation/`.
