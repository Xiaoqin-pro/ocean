# RCR Gate R0: Full-image Repairability Oracle Screen

## Scope

R0 is the last low-cost oracle screen using existing assets.  It uses only the
already-open 231-image `risk_head_development` role, two frozen SUIM models,
and the fixed clean plus twelve degradation conditions.  It performs no
training, calibration, crop refinement, communication simulation, selector
learning, or access to SUIM validation, calibration, official test, or
external data.

## Fixed candidates

The local/remote candidates are SegFormer to DeepLab, DeepLab to SegFormer,
SegFormer to equal-probability ensemble, and DeepLab to equal-probability
ensemble.  The ensemble is exactly
`argmax(0.5 * softmax(segformer_logits) + 0.5 * softmax(deeplab_logits))`;
weights cannot be selected on development data.

For every valid pixel, R0 reports repair rate, damage rate, net correction
mass, full-image local/remote metrics, and a pixelwise GT-oracle union.  The
oracle replaces local output only where the local model is wrong and the fixed
remote candidate is correct.  It is an upper bound, never a deployable fusion
rule.

## R0 screen

At least one candidate must satisfy all frozen requirements:

1. Repair rate at least 15%.
2. Pixelwise-oracle macro mIoU gain over its local model at least 2pp.
3. Foreground macro-IoU gain is positive.
4. Boundary oracle macro-IoU gain is positive.
5. At least three degradation families have positive oracle mIoU gain.
6. The 1,000-replicate scene-cluster bootstrap lower 95% bound for full oracle
   mIoU gain is positive.

Failure closes RCR: no GT crop oracle, communication simulation, or a new
repairability selector may be developed.  Success only authorizes separately
preregistered R1a (uncompressed GT-crop local re-inference), then R1b (actual
codec byte budget).  GT may select crop regions in R1a but may never decide
whether a remote central prediction is accepted.
