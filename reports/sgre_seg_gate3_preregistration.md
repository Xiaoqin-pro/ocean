# SGRE-Seg Gate-3 preregistration

## Research hypothesis

SCDI showed that semantic routing is useful (+0.272 mean mIoU point, all 13
conditions positive), but its feature-tangent reconstruction did not optimize.
The weakest class PF remains at only 19.456% IoU under F4, while dominant BW
is 86.388%. Pixel-averaged four-view CE therefore underweights exactly the
groups that determine mean IoU.

SGRE-Seg keeps SCDI's class-routed low-rank spatial experts but removes all
feature-tangent and feature-identity losses. It maintains online exponential
moving losses for 32 semantic degradation groups (clean/s1/s2/s3 by eight
classes). A temperature-softmax group distribution upweights persistently hard
class-severity groups. Training uses a fixed 50/50 mixture of standard pixel CE
and the group-robust objective, plus the same semantic router supervision.

This creates a coupled method: semantic routing determines *which expert*
corrects a location, while group robustness determines *which semantic
degradation failures* receive optimization pressure. Inference remains one RGB
forward pass with no labels, clean image, degradation type, or severity input.

## Frozen Gate-3

The candidate starts from the same variant-B checkpoint and uses the same 936
method-training scenes, seed, four views, eight epochs, and final checkpoint as
F4. Base/expert learning rates are `1e-5`/`1e-4`; group EMA momentum is 0.9,
temperature 0.5, CE/group mixture 0.5/0.5, and router weight 0.02.

Pass requires: +0.50 point 13-condition mean or +1.00 point severity-3 mean;
clean decrease no worse than 0.30 point; and at least three of four severe
families non-negative. Formal validation/calibration/test remain locked.

Failure closes this exact group-robust formulation. It does not end the overall
research search, but it rules out feature adapters/semantic experts as the
paper's primary contribution and forces a different method family.
