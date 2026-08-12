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
| SADR transfers to unseen degradation compositions | Confirmation stress test: eight ordered compositions on reused scenes, mean +1.302 pp; calibration audit: +0.278 pp, 15/24 positive | Safe only as a post-freeze diagnostic with family-selective transfer; do not call it an independent or universal generalization result. |
| SADR corrections are exactly additive | All-family screen: add cosine falls 0.877/0.583/0.367 from severity 1/2/3; correct-add beats shuffled null in 54/54 cases | Unsafe. Claim severity-dependent, family-selective partial compositionality, not a linear law. |
| SADR order stability is universal | Order cosine is 0.995/0.952/0.859 for matched severity 1/2/3, with family outliers | Unsafe. State that order consistency is strongest at mild severity and degrades with severity. |
| Held-out scenes establish robust composition generalization | Calibration audit mean +0.278 pp; lowlight/blur is -0.590 pp across orders | Unsafe. Call it a held-out audit and report the failed family explicitly. |
| Compositionality regularization is the source of the gain | Additive, logit-order, probability-KL, and composite-distillation controls all fall to about +0.36 pp | Unsafe. Keep these as negative controls; the reported transfer is emergent from task-supervised SADR. |

## Minimum defensible abstract sentence

“On a locked UIIS robustness protocol with 13 deterministic color, turbidity,
low-light, blur, and clean conditions, the proposed adapter improves mean mIoU
by 0.746 percentage points over a frozen UIIS-F4 expert across three seeds,
while adding 0.296% trainable parameters; the method improves synthetic
cross-condition robustness but incurs a small mean cost on the untouched SUIM
source-domain check.”
