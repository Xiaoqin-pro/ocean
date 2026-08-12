# AquaRiskMap Stage-A decision

**Decision:** **FAIL — AquaRiskMap is permanently stopped under its preregistered Stage-A rule.**  
**Decision-code commit:** `33d2fda` (the final report is committed immediately after this record).  
**Scope:** SUIM Stage A only; this is not a claim about UWR-Bench, which remains frozen and valid.

## Protocol completion

- Both independently initialized risk heads completed their fixed 20 final-epoch training schedules.
- Both used the fixed 14-channel class-agnostic input, 15,615-parameter risk head, three fixed losses, and unchanged 13-condition registry.
- Ranking was evaluated once on the frozen 146-image SUIM validation split with 1,000 paired `sample_id` cluster-bootstrap replicates.
- Global CRC was fit only on the frozen calibration split at `alpha=0.10` using the fixed 1%–100% grid and monotone envelope, then evaluated once on validation.
- Latency used the fixed 50-image warm-up plus 50-image measurement protocol on `risk_head_development`.
- SUIM official TEST was not accessed.  No UIIS or DUT-USEG AquaRiskMap run was started.

## Preregistered gate table

| Requirement | SegFormer | DeepLab | Required outcome |
| --- | ---: | ---: | --- |
| Full eAURC relative decrease | -4.41% | +0.19% | >=10% for a complete pass; both backbones positive direction |
| Boundary eAURC relative decrease | -2.23% | +3.79% | >=10% for a complete pass; both backbones positive direction |
| Full error-AUPRC change | -1.25pp | +1.05pp | >=+3pp |
| Full top-10% error-recall change | -0.48pp | +0.73pp | >=+5pp |
| Global-CRC coverage change | +4pp | +2pp | >=+3pp and no higher selective risk |
| Global-CRC selective-risk change | +0.476pp | +0.318pp | <=0pp |
| Added latency / base segmentation | 20.10% | 29.29% | <10% |
| Risk-head parameters | 15,615 | 15,615 | <200,000 |

Both backbones fail the ranking complete-pass gate, both fail the CRC safety condition, and both exceed the latency limit.  In particular, SegFormer violates the required positive direction for both full and boundary eAURC.  DeepLab has small positive ranking directions, but none approaches the fixed magnitude thresholds.

## Consequence

The branch must not change features, architecture, losses, schedule, score direction, CRC implementation, or thresholds in an attempt to rescue AquaRiskMap.  It must not proceed to Stage B, UIIS, DUT-USEG, another segmentation backbone, or SUIM official TEST for this method.

The negative result may be retained as a carefully controlled UWR-Bench diagnostic: a lightweight learned error map did not deliver stable ranking, safe selective coverage, or low-overhead deployment across two frozen underwater segmentation backbones.  The primary UWR-Bench benchmark paper remains the active publication route.

## Local provenance

Ignored detailed artifacts remain under `outputs/aquariskmap_suim_pilot_v1/stage_a/`, including ranking tables, CRC curves/parameters, and latency JSON for both backbones.  The compact machine-readable summary is committed as `experiments/aquariskmap_stage_a_summary.csv`.
