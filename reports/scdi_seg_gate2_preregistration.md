# SCDI-Seg Gate-2 preregistration

## Final structural hypothesis

SDTC's shared rectifier improved all 13 conditions but only by 0.045 mIoU
point. Its class-level analysis showed unequal effects (HD +0.148, RF -0.008,
RI -0.016), so the shared correction is the likely bottleneck.

SCDI-Seg replaces each shared rectifier with eight class-specific low-rank
spatial inverse experts and a learned semantic router. The router is supervised
by downsampled public-dataset labels during training. At inference it predicts
soft semantic routing weights from the image itself; labels, clean references,
degradation types, and severity values are not inputs. Expert mixtures generate
the residual passed to the standard SegFormer decoder. The same per-scene,
per-class inverse tangent target used by SDTC supervises the routed correction.

The contribution is therefore not generic mixture-of-experts or global feature
alignment: it is semantic routing of degradation-inversion fields, trained from
paired class-conditional feature displacements. SCDI adds no image restoration
decoder and uses one RGB forward pass at deployment.

## Frozen final gate

SCDI starts from the same variant-B checkpoint as F4, uses the same 936 scenes,
four views, seed, eight epochs, and final-checkpoint rule. Base learning rate is
`1e-5`; expert/router learning rate is `1e-4`. Tangent, clean-identity, and
router weights are fixed at `0.02` each.

It passes only if:

1. mean 13-condition mIoU improves by at least 0.50 point, or severity-3 mean
   improves by at least 1.00 point;
2. clean loss is no worse than 0.30 point; and
3. at least three of four severity-3 families are non-negative.

This is the final method-selection gate on the already-open development role.
Failure stops this research line rather than triggering weight, width, loss,
epoch, or router tuning. A pass authorizes ablations, unseen compound-stress
evaluation, and replication on a second public dataset, while formal SUIM
validation/calibration/test remain locked.
