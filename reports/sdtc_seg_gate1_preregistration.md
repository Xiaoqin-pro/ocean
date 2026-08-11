# SDTC-Seg Gate-1 preregistration

## Method hypothesis

DTS-Seg failed because a logit-trajectory penalty constrained collapse but
provided no operation capable of undoing feature distortion. SDTC-Seg adds
four identity-initialized spatial rectifiers before the SegFormer decoder.
For each aligned clean/degraded scene and each semantic class present at a
feature scale, training computes the class centroid displacement

`t(c,s) = mean(F_degraded | y=c) - mean(F_clean | y=c)`.

The rectifier residual is trained to cancel this displacement. The target is
stop-gradient, so the encoder cannot satisfy the loss by moving both
endpoints. Clean-view residual energy is penalized, preserving identity on
already clean inputs. Pixel labels and clean references are privileged
training information only; inference uses one RGB image, one forward pass,
and no degradation family/severity label.

This differs from Layer-Wise Feature Adjustor (WACV 2023), which aligns
degraded features globally through adjustors, and from class-conditional
domain adaptation, which aligns source/target distributions. SDTC explicitly
supervises the *generated correction residual* with paired, per-scene,
per-class inverse degradation tangents at four scales. It also differs from
underwater enhancement/restoration systems because it never reconstructs an
image.

## Frozen Gate-1 protocol

SDTC and the already-frozen F4 comparator start from the same variant-B
checkpoint and use the same 936 method-training scenes, four ordered views,
seed, eight epochs, and final-epoch checkpoint rule. The base SegFormer uses
learning rate `1e-5`; newly initialized rectifiers use `1e-4`. Tangent and
clean-identity weights are both fixed at `0.02`.

SDTC passes only if all of the following hold on the already-open 231-image
development role:

1. 13-condition mean mIoU improves by at least 0.50 point, or four-condition
   severity-3 mean improves by at least 1.00 point;
2. clean mIoU decreases by no more than 0.30 point; and
3. at least three of four severity-3 families are non-negative.

Failure closes this exact architecture and objective without hyperparameter
tuning on development. Passing authorizes ablations and replication on a
second public dataset, but does not authorize SUIM formal validation,
calibration, or official test evaluation.
