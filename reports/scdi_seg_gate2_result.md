# SCDI-Seg Gate-2 result

## Frozen decision

**FAIL on magnitude, with a positive structural signal.**

| Metric (SCDI - F4) | Change (mIoU points) |
|---|---:|
| 13-condition mean | +0.272 |
| four severity-3 conditions | +0.276 |
| clean | +0.257 |
| color severity 3 | +0.345 |
| turbidity severity 3 | +0.349 |
| low-light severity 3 | +0.250 |
| blur severity 3 | +0.159 |

All severe families and all 13 individual conditions improved, but the frozen
minimum was +0.50 mean or +1.00 severe. Formal validation, calibration, and
official test roles were not evaluated.

## Mechanistic conclusion

Semantic experts amplified the shared SDTC gain from +0.045 to +0.272 point,
but the inverse-tangent training loss remained approximately 1.0 through
epoch eight. The routers learned (four-view router loss 8.49 to 3.69), whereas
the feature-reconstruction target did not. Therefore feature-tangent recovery
is closed, while semantic routing remains supported.

The weakest baseline class, PF, has only 19.456% mean IoU and gained 0.674
point with SCDI. RF and HD gained 0.486 and 0.471 point. This motivates a new
decision-level method: semantic experts trained with online group robustness
over class-by-severity losses, directly matching the class-balanced mIoU goal
instead of reconstructing encoder features.
