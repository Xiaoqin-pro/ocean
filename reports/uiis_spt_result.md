# UIIS SPT replication result

## Protocol

SPT (Semantic Prototype Transport) was trained on the frozen 2,371-image UIIS
train role for eight epochs from the completed 60-epoch UIIS SegFormer-B0
checkpoint.  The fixed F4 comparator is the paired eight-epoch train-only
replication run.  Each view uses the same clean/s1/s2/s3 degradation registry;
the SPT objective adds multi-scale class-prototype transport, semantic router
cross-entropy, and clean identity regularization.  No UIIS calibration or
confirmation image was read during training.

The 508-image calibration role was used for the development check.  After that
check the SPT weights were frozen, and the 511-image confirmation role was read
once for final evidence.  The confirmation role was not used for tuning,
selection, or checkpoint writing.

## Results

| Split | F4 mean mIoU | SPT mean mIoU | Gain |
|---|---:|---:|---:|
| calibration (5 conditions) | 0.433976 | 0.435904 | +0.193 pp |
| confirmation (5 conditions) | 0.494378 | 0.495418 | +0.104 pp |

Confirmation condition gains were clean -0.102 pp, color_s3 +0.010 pp,
turbidity_s3 +0.296 pp, lowlight_s3 +0.150 pp, and blur_s3 +0.161 pp.

## Decision

SPT is a mechanistically meaningful improvement over SCDI: it replaces direct
feature-tangent regression with class-prototype alignment and retains a
class-routed low-rank correction.  The confirmation result is positive on the
mean and on four of five conditions, but the +0.104 pp gain is still too small
to claim a strong standalone paper contribution.  SPT is therefore retained as
the current best UIIS result and as the semantic anchor for the next route;
further development must target cross-dataset magnitude and blur robustness
without revisiting this frozen confirmation result.

`confirmation_evaluated=true`; `official_suim_test_evaluated=false`.
