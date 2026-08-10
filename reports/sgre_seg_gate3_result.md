# SGRE-Seg Gate-3 result

## Frozen decision

**FAIL.** Online group robustness over class-by-view losses damaged both clean
and degraded segmentation.

| Metric (SGRE - F4) | Change (mIoU points) |
|---|---:|
| 13-condition mean | -0.865 |
| four severity-3 conditions | -0.899 |
| clean | -0.905 |
| color severity 3 | -0.454 |
| turbidity severity 3 | -0.719 |
| low-light severity 3 | -0.995 |
| blur severity 3 | -1.429 |

All four severe families were negative. The exact group-robust formulation is
closed; its temperature, EMA, mixture, and router weights must not be tuned on
this development role.

## Interpretation

The group objective reduced its training scalar smoothly, but that scalar did
not correspond to generalization. Online weights over sparse class/view groups
over-focused unstable rare groups and weakened the shared representation. This
rules out class-severity group DRO as the primary method for this project.

SCDI remains the strongest candidate because it improved all 13 conditions and
all four severe families (+0.272 mean mIoU points) without clean loss. The next
phase will test that structural method on the second public dataset and on
unseen compound stress conditions, with SCDI's failed tangent loss and SGRE's
group objective retained only as ablations.
