# FT-Reliability protocol v1.2

**Status:** frozen before real smoke, formal training, or development access.

The pilot uses only the fixed 936-image method_train role. The 231-image
method_development role, formal SUIM validation/calibration/TEST, and external
data remain locked until all final-epoch checkpoints are frozen.

Variants B/C/D/E share the identical first five epochs of three-view
Degradation-CE. C, D, and E branch from that completed epoch-5 checkpoint;
E enables correct-only clean retention only from epoch 6. All runs use final
epoch checkpoints.

The transition loss is the fixed top-1/top-2 gap objective on same-pixel
s1-correct to s3-wrong transitions, with margin 0.20, weight 0.10, 4,096
terms per batch, and a 50/50 boundary/interior target with replacement
fallback. C uses the same budget for generic correct/wrong ranking.

The unique E-vs-C primary endpoint is eAURC_C - eAURC_E; its 1,000-replicate
paired scene-cluster bootstrap 95% lower confidence bound must exceed zero.
ECE is a 15-bin equal-width diagnostic reported with classwise ECE, NLL, and
Brier; it is not a standalone veto.

The strong-result criterion is either macro mIoU +0.5 pp or mean severity-3
mIoU +1.0 pp, after the core reliability gates pass.
