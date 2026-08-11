# DTS-Seg Gate-0 preregistration

## Publication hypothesis

Standard degradation augmentation treats sampled underwater conditions as
independent views. DTS-Seg instead uses their physical severity order. It
penalizes only an accelerating collapse of the true-class logit margin along
`clean -> s1 -> s2 -> s3`, while leaving constant-rate degradation, recovery,
and already stable pixels unconstrained. The method adds no inference module,
parameter, input modality, or test-time view.

This is a robust-segmentation method, not a calibration or error-ranking
method. The earlier FT-Reliability result is not reused as positive evidence:
its confidence-ranking objective failed. Its variant-B checkpoint is used only
as a strong common initialization because three-view degradation CE improved
13-condition mIoU by 3.70 points and severe-condition mean mIoU by 7.87 points
over clean-only training on the already-open development role.

## Gate-0 comparison

Both variants start from the exact same frozen variant-B checkpoint and use
the same optimizer, data order, four-view trajectory, eight epochs, and final
checkpoint rule.

- `F4`: mean CE over clean, s1, s2, and s3.
- `DTS`: F4 plus the one-sided degradation-acceleration loss at s2 and s3.

For margins `m0,m1,m2,m3`, the two terms are

`relu((m1-m2) - (m0-m1) - delta)` and
`relu((m2-m3) - (m1-m2) - delta)`.

Earlier margins are stop-gradient values. A term is eligible only where the
previous-severity prediction is correct. Thus the loss acts before semantic
failure instead of training on sparse correct-to-wrong event labels. The fixed
values are `delta=0.05` and loss weight `0.05`.

## Decision rule

DTS-Seg proceeds to full from-scratch training and UIIS replication only if:

1. DTS exceeds F4 by at least +0.50 points in 13-condition mean mIoU, **or** by
   at least +1.00 point over the four severity-3 conditions;
2. clean mIoU decreases by no more than 0.30 point; and
3. at least three of four severity-3 degradation families are non-negative.

Failure closes this exact loss and prevents weight, tolerance, epoch, or mask
retuning on the development role. Passing Gate-0 does not authorize the SUIM
official test. Formal validation, calibration, and official test remain locked.

## Literature distinction fixed before training

- SEA-Net divides underwater images into high/low severity groups and learns
  visual prompts/adapters; DTS models a continuous within-scene severity path.
- Restoration-adaptation methods add image-restoration components; DTS acts
  only on semantic margins during training.
- Generic consistency methods align augmented predictions; DTS is one-sided,
  order-aware, and constrains the discrete second derivative only when semantic
  degradation accelerates.

