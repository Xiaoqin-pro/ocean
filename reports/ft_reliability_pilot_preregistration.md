# FT-Reliability Seg pilot preregistration

**Status:** preregistered protocol only; no FT-Reliability code, cache, Baseline-936, FT model, or `method_development` result exists.  
**Branch:** `experiment/ft-reliability-pilot`  
**Date:** 2026-07-25  
**UWR-Bench frozen parent:** `8d24fdf`  
**AquaRiskMap final failure:** `5d18be8` (permanently stopped; not reopened here).

## Revision record

* **v1 / `00595ba`:** initial preregistration, committed before implementation.
* **v1.1:** corrects a sign error identified in the original truth-class-margin
  objective, specifies the generic-ranking comparator and all variant schedules,
  defines clean retention precisely, fixes memory-safe backward order, and
  resolves the full-eAURC gate to 8%. This amendment is made before any method
  code, training, or `method_development` access; no result was inspected.

## 1. Aim and falsifiable hypothesis

This is an independent, train-time reliability-method pilot. It does not alter UWR-Bench, retry DARC-Seg, or rescue AquaRiskMap. The method is provisionally named **FT-Reliability Seg: Failure-Transition Reliability Learning for Underwater Semantic Segmentation**.

Its hypothesis is narrow: when the same pixel changes from correct under a mild degradation to incorrect under a severe degradation from the same family, directly training the confidence gap of the severe wrong prediction to fall by a fixed gap can improve degradation robustness and error ranking without sacrificing clean segmentation quality.

This is not a universal monotonic-confidence objective. It constrains only observed correct-to-wrong transitions and adds no risk head, attention module, decoder, adapter, backbone, or inference-time computation.

## 2. Frozen data roles and access boundary

The existing inner scene-aware split is reused without changing sample IDs or scene groups:

| frozen source role | FT role | images | allowed use |
| --- | --- | ---: | --- |
| `risk_head_train` | `method_train` | 936 | all model and teacher training |
| `risk_head_development` | `method_development` | 231 | one pilot evaluation only, after code and all choices are frozen |

Before training, an integrity check must prove zero sample-ID overlap, exact-image-duplicate overlap, and scene-group overlap between these roles. The former 1,167-image models cannot act as method baselines. New `Baseline-936` models are required and every comparison within an architecture uses the same initialization, seed, 936 images, augmentation protocol, and schedule.

Until all variants are trained and frozen, this branch must not read, cache, predict on, inspect, or evaluate the formal SUIM `val`, `calibration`, or official `TEST` splits. UIIS, DUT-USEG, USIS10K, and other external data are prohibited. The official SUIM TEST remains locked.

## 3. Fixed degradation trajectory

For scene `sample_id` in epoch `epoch`, choose exactly one degradation family:

```python
family_index = (stable_hash(sample_id) + epoch) % 4
```

The fixed family order is `color`, `turbidity`, `lowlight`, `blur`. The scene trajectory is always `clean + family_s1 + family_s3`; cross-family and prediction-driven trajectory selection are forbidden. Four consecutive epochs therefore cover all four families for every scene.

## 4. FT objective

For pixel \(i\) and logit vector \(z\), define the **top-1/top-2** prediction-confidence gap:

\[
q_i(x) = z_{i,(1)}(x) - z_{i,(2)}(x),
\]

where \(z_{i,(1)}\) and \(z_{i,(2)}\) are the largest and second-largest
logits. A larger \(q_i\) means that the current predicted class is more
confident, independent of its semantic identity.

Define a failure transition only when mild degradation is correct and severe degradation is wrong:

\[
T_i = \mathbf{1}[\hat y_i^{s1}=y_i \land \hat y_i^{s3}\ne y_i].
\]

For transition pixels only:

\[
L_{FT}=\frac{1}{|T|}\sum_{i\in T}\operatorname{softplus}(\delta-\operatorname{sg}(q_i^{s1})+q_i^{s3}).
\]

The `s1` gap and predictions used to construct the transition mask are detached. Minimization imposes \(q_i^{s3}\le q_i^{s1}-\delta\): an incorrect severe-degradation prediction becomes less overconfident without pushing the true-class logit further down or opposing severe-view cross entropy. A batch with no transition pixels has \(L_{FT}=0\). No constraint is imposed on other pixels or on a full severity sequence.

The three-view segmentation component is \(L_{seg}=[CE(clean,y)+CE(s1,y)+CE(s3,y)]/3\). This avoids a three-fold loss-scale change relative to clean-only training. Clean retention is defined on valid 384-by-384 pixels as \(L_{retain}=\frac{1}{|\Omega|}\sum_{i\in\Omega}KL(p_{teacher,i}^{clean}\Vert p_{student,i}^{clean})\), at temperature 1.0. The teacher is a same-architecture `Baseline-936`, trained only on `method_train`, frozen, in evaluation mode, and evaluated under `no_grad`; its checkpoint is never used to initialize an FT student. Every student starts from the same ImageNet-pretrained initialization and seed as its same-architecture comparators.

The fixed values are: margin 0.20, FT weight 0.10, clean-retention weight 0.10, warm-up 5 epochs, at most 4,096 transition pixels per batch, GT boundary radius 3, and 50%/50% boundary/interior transition sampling. If one region is short, all of it is used and the other region fills the remainder; replacement is allowed when fewer than 4,096 total transitions exist. Boundary data are used only for this training-time sampler and never at inference. Boundary/interior transition counts are logged every epoch.

Variant C is an explicitly fixed generic comparator rather than an underspecified label. It uses only the same `family_s3` batch, with no same-pixel trajectory: \(L_{CR}=\frac{1}{N}\sum\operatorname{softplus}(\delta-q_{correct}^{s3}+q_{wrong}^{s3})\). It uses the same 0.20 margin, 0.10 weight, five-epoch warm-up, 4,096 correct/wrong pairs per batch, and 50% boundary target. Correct or wrong pools that are empty yield zero loss; undersized pools are sampled with replacement using `seed + 100003 * epoch + batch_index`.

## 5. Fixed training and comparison plan

SegFormer-B0 and DeepLabV3-MobileNetV3-Large retain their existing baseline optimizer, learning rate, weight decay, 100-epoch schedule, 384-pixel input, and AMP setting. Their only data change is replacement of the formal train partition with the 936-image `method_train`. If memory requires it, only the base-scene batch size may be reduced and gradient accumulation restores the same effective batch; image resolution may not change. All variants select the final epoch, never `method_development`.

For memory safety on 8GB VRAM, D and E use three sequential forwards with gradient accumulation: (1) clean forward, clean CE and retention, backward, release the clean graph; (2) `s1` forward, `s1` CE plus detached `q_s1` and correctness, backward, release the `s1` graph; (3) `s3` forward, `s3` CE, detached transition mask, and FT loss, backward; then perform one optimizer step. This is objective-equivalent because FT explicitly detaches the `s1` branch. C calculates its generic ranking loss from `s3` only.

SegFormer is the sole initial pilot architecture:

| label | variant |
| --- | --- |
| A | Baseline-936: clean CE only |
| B | Degradation-CE: CE on clean, `s1`, and `s3` |
| C | B + generic correctness ranking without a same-pixel trajectory |
| D | B + failure-transition loss |
| E | B + failure-transition loss + clean retention |

The fixed schedules are:

| variant | epochs 1--5 | epochs 6--100 |
| --- | --- | --- |
| A | clean CE | clean CE |
| B | mean three-view CE | mean three-view CE |
| C | mean three-view CE | mean three-view CE + 0.10 \(L_{CR}\) |
| D | mean three-view CE | mean three-view CE + 0.10 \(L_{FT}\) |
| E | mean three-view CE + 0.10 \(L_{retain}\) | mean three-view CE + 0.10 \(L_{FT}\) + 0.10 \(L_{retain}\) |

The principal comparisons are **E vs B** and **E vs C**. No loss weight, margin, warm-up, sampler, condition, seed, or training length may be changed after the first `method_development` result.

## 6. One-time pilot evaluation and gates

After all SegFormer code, configuration, checkpoint-selection rule, and seed are frozen, `method_development` is evaluated exactly once under clean plus the 12 registered degradations. Full, GT-boundary-radius-3, and interior regions report mIoU, NLL, ECE, Brier, eAURC, error AUPRC, and top-10% error recall. All comparisons use 1,000 paired scene-cluster bootstrap replicates.

SegFormer may proceed to an identical DeepLab replication only if **all** criteria below hold:

| comparison | required criterion |
| --- | --- |
| E vs B | 13-condition macro mIoU improvement at least +1.0 pp |
| E vs B | clean mIoU decrease no greater than 0.5 pp |
| E vs B | full eAURC relative decrease at least 8% |
| E vs B | boundary eAURC relative decrease at least 5% |
| E vs B | error AUPRC improvement at least +1.5 pp |
| E vs B | ECE absolute worsening no greater than 0.01 |
| E vs C | better direction for full eAURC and error AUPRC |
| E vs C | at least one corresponding paired-bootstrap 95% CI excludes zero |

Failure of any gate permanently stops FT-Reliability: it may not be retuned, retrained to rescue a result, or advanced to formal `val`, `calibration`, official TEST, UIIS, or another external dataset.

## 7. Initial implementation boundary

This commit authorizes only a future implementation review. At commit time:

* no FT-Reliability method code has been implemented;
* no Baseline-936 or FT-Reliability model has been trained;
* `method_development` has not been read;
* formal `val`, `calibration`, and official TEST have not been read;
* AquaRiskMap remains permanently stopped; and
* no external dataset is being used to tune this method.
