# DTH Gate-1 Conditional Horizon Signal Screen: Frozen Result

## Decision

**FAIL — DTH Gate 2 must not start.**  The pre-registered Gate-1 condition
requires at least three degradation families to have no single
H1/H2/H3/censored state above 90% of eligible pixels.  All four families
failed this requirement because the censored state dominated the label space.
No DTH head was trained, no horizon parameterization was selected, and no
threshold or screening rule was modified after inspecting the result.

| Family | H1 | H2 | H3 | Censored | Post-failure recovery |
| --- | ---: | ---: | ---: | ---: | ---: |
| Color attenuation | 0.1987% | 0.2379% | 0.3017% | 99.2617% | 8.7630% |
| Turbidity | 0.1089% | 0.1781% | 0.2859% | 99.4271% | 9.2861% |
| Low light | 0.1770% | 0.2443% | 0.4353% | 99.1434% | 14.7603% |
| Blur | 0.2684% | 0.5030% | 0.6313% | 98.5972% | 12.9503% |

Every family had 936 eligible scenes and more than 30 H2 and H3 event scenes,
and the recovery rule passed.  These facts do not overcome the pre-registered
pixel-level state-concentration failure: the largest state was 98.5972% or
higher in every family, versus the 90% maximum.  The resulting conditional
onset labels are therefore too censored for the frozen Gate-2 learning task.

## Protocol and provenance

- Parent commit: `8ccc2d97757b1ba477aed64fa1b79eb4f847abbb`.
- Gate-1 implementation commit: `b2f6f07bd1a77b3c08ffb52d38d16a854a31407a`.
- Frozen model: SegFormer-B0 FT Variant B final checkpoint.
- Checkpoint SHA-256:
  `F9C8D08B39ED04295F63006381384DBD987CAC3B65BFF11A7C8D736DE23430F6`.
- Source role: frozen 936-image `risk_head_train.csv` only.
- Source CSV SHA-256:
  `7CF81D0B822A70414764E71BD596DBF8500B19D3D9667A5F9F066C7AF7656D07`.
- Deterministic resize-only geometry; no crop or flip; each per-family
  s1/s2/s3 trajectory shared its latent seed and spatial pattern.
- Registry SHA-256:
  `706A1DA10870C0659CD9CC35D8472F22FC727B2292933FEBD26961BB7666CF93`.

## Access audit

The output manifest records all of the following as false:

- DTH head trained
- method-development evaluated
- SUIM validation evaluated
- SUIM calibration evaluated
- SUIM official test evaluated

The raw Gate-1 artifacts remain local under
`outputs/dth_horizon_signal_screen/`; their SHA-256 manifest is
`outputs/dth_horizon_signal_screen/manifest.json`.  They are not committed,
because they are generated result tables rather than source or protocol.

## Consequence

The DTH-Seg conditional degradation-onset route is closed under this
pre-registered design.  It must not be rescued by redefining eligibility,
filtering censored pixels, changing severity levels, adding a class-specific
screen, or training a head on the 231-image development role.  TCCR remains
closed independently.  Any future work must be a separately authorized,
independent research route rather than a post-hoc DTH revision.
