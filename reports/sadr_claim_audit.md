# SADR claim audit

| proposed sentence | evidence | safe wording |
|---|---|---|
| SADR improves robustness | Three 8-epoch seeds: +0.607, +0.760, +0.872 pp on the locked 13-condition UIIS confirmation split | Safe. Report mean +0.746 pp and seed spread. |
| SADR is statistically reliable | Three seed gains, one-sample t-test p=0.0105 | Use as a stability indication, not a large-sample significance claim. |
| SADR generalizes to SUIM | SUIM changes are -0.130, -0.197, -0.207 pp | Unsafe. Say source-domain cost is small but external transfer is not improved. |
| SADR is better than generic enhancement | GrayWorld +0.221 pp and CLAHE -1.596 pp on confirmation | Safe against these two controls only; do not claim all enhancement methods. |
| SADR is the first task-driven underwater enhancement method | TFUIE, STSC, and HSRUIE predate this work | Unsafe. Claim frozen-expert parameter efficiency and robustness protocol instead. |
| SADR handles all underwater degradations | Blur-s3 mean is about -0.43 pp | Unsafe. State color/low-light gains and blur limitation. |
| UVMulti validates real-world generalization | Available subset has only a small held-out frame sample and no improvement | Unsafe. Use only as a negative sanity check. |
| SADR is lightweight | 11,012 trainable vs 3,716,200 frozen parameters; +21.4% measured latency at 384x384 | Safe if both parameter and latency costs are reported. |
| Extra modules are necessary | Frequency, routing, and matched feature consistency did not exceed ordinary SADR | Unsafe. The evidence favors the simple semantic residual adapter. |
| SADR transfers to unseen degradation compositions | Eight ordered compositions on the same confirmation images; all 3 seeds positive, mean +1.302 pp | Safe as a post-freeze compositional stress test; do not call it an independent split or real-world generalization. |
| SADR corrections are exactly additive | Binary cosine is 0.785, triple cosine is 0.535; direct additive regularization reduces the primary gain | Unsafe. Claim order-stable/partially compositional semantic corrections, not a linear law. |
| Compositionality regularization is the source of the gain | Additive, logit-order, probability-KL, and composite-distillation controls all fall to about +0.36 pp | Unsafe. Keep these as negative controls; the reported transfer is emergent from task-supervised SADR. |

## Minimum defensible abstract sentence

“On a locked UIIS robustness protocol with 13 deterministic color, turbidity,
low-light, blur, and clean conditions, the proposed adapter improves mean mIoU
by 0.746 percentage points over a frozen UIIS-F4 expert across three seeds,
while adding 0.296% trainable parameters; the method improves synthetic
cross-condition robustness but incurs a small mean cost on the untouched SUIM
source-domain check.”
