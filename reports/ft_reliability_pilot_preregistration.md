# FT-Reliability Seg pilot preregistration

**Status:** preregistered protocol only; no FT-Reliability code, cache, Baseline-936, FT model, or `method_development` result exists.  
**Branch:** `experiment/ft-reliability-pilot`  
**Date:** 2026-07-25  
**UWR-Bench frozen parent:** `8d24fdf`  
**AquaRiskMap final failure:** `5d18be8` (permanently stopped; not reopened here).

## 1. Aim and falsifiable hypothesis

This is an independent, train-time reliability-method pilot. It does not alter UWR-Bench, retry DARC-Seg, or rescue AquaRiskMap. The method is provisionally named **FT-Reliability Seg: Failure-Transition Reliability Learning for Underwater Semantic Segmentation**.

Its hypothesis is narrow: when the same pixel changes from correct under a mild degradation to incorrect under a severe degradation from the same family, directly training its native segmentation margin to fall by a fixed gap can improve degradation robustness and error ranking without sacrificing clean segmentation quality.

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

For pixel \(i\), true class \(y_i\), and logit vector \(z\), define the truth-class margin:

\[
m_i(x) = z_{i,y_i}(x) - \max_{k \ne y_i} z_{i,k}(x).
\]

Define a failure transition only when mild degradation is correct and severe degradation is wrong:

\[
T_i = \mathbf{1}[\hat y_i^{s1}=y_i \land \hat y_i^{s3}\ne y_i].
\]

For transition pixels only:

\[
L_{FT}=\frac{1}{|T|}\sum_{i\in T}\operatorname{softplus}(\delta-\operatorname{sg}(m_i^{s1})+m_i^{s3}).
\]

The `s1` margin is detached. A batch with no transition pixels has \(L_{FT}=0\). No constraint is imposed on other pixels or on a full severity sequence.

The fixed objective is \(L=L_{seg}+0.10L_{FT}+0.10L_{retain}\), where \(L_{seg}=CE(clean,y)+CE(s1,y)+CE(s3,y)\) and \(L_{retain}=KL(p_{Baseline936}^{clean}\Vert p_{FT}^{clean})\). The teacher is a same-architecture `Baseline-936`, trained only on `method_train`, frozen, and detached. During epochs 1--5 the objective omits \(L_{FT}\); from epoch 6 onward it includes it.

The fixed values are: margin 0.20, FT weight 0.10, clean-retention weight 0.10, warm-up 5 epochs, at most 4,096 transition pixels per batch, GT boundary radius 3, and 50%/50% boundary/interior transition sampling. If one region is short, all of it is used and the other region fills the remainder; replacement is allowed when fewer than 4,096 total transitions exist. Boundary data are used only for this training-time sampler and never at inference. Boundary/interior transition counts are logged every epoch.

## 5. Fixed training and comparison plan

SegFormer-B0 and DeepLabV3-MobileNetV3-Large retain their existing baseline optimizer, learning rate, weight decay, 100-epoch schedule, 384-pixel input, and AMP setting. Their only data change is replacement of the formal train partition with the 936-image `method_train`. If memory requires it, only the base-scene batch size may be reduced and gradient accumulation restores the same effective batch; image resolution may not change. Each clean, `s1`, and `s3` forward pass is performed sequentially, then one joint backward/update is executed. All variants select the final epoch, never `method_development`.

SegFormer is the sole initial pilot architecture:

| label | variant |
| --- | --- |
| A | Baseline-936: clean CE only |
| B | Degradation-CE: CE on clean, `s1`, and `s3` |
| C | B + generic correctness ranking without a same-pixel trajectory |
| D | B + failure-transition loss |
| E | B + failure-transition loss + clean retention |

The principal comparisons are **E vs B** and **E vs C**. No loss weight, margin, warm-up, sampler, condition, seed, or training length may be changed after the first `method_development` result.

## 6. One-time pilot evaluation and gates

After all SegFormer code, configuration, checkpoint-selection rule, and seed are frozen, `method_development` is evaluated exactly once under clean plus the 12 registered degradations. Full, GT-boundary-radius-3, and interior regions report mIoU, NLL, ECE, Brier, eAURC, error AUPRC, and top-10% error recall. All comparisons use 1,000 paired scene-cluster bootstrap replicates.

SegFormer may proceed to an identical DeepLab replication only if **all** criteria below hold:

| comparison | required criterion |
| --- | --- |
| E vs B | 13-condition macro mIoU improvement at least +1.0 pp |
| E vs B | clean mIoU decrease no greater than 0.5 pp |
| E vs B | full eAURC relative decrease at least 5% |
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
