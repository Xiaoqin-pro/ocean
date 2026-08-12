# SADR manuscript v0.1 — internal review checklist

This checklist is for reviewing the manuscript against the paper-freeze
manifest, not for reopening experiments.

## Scientific consistency

- [ ] Every primary number matches `reports/sadr_paper_freeze.md`.
- [ ] The primary claim is the three-seed, 13-condition UIIS result: +0.746 pp
      mean and 0.296% trainable parameters.
- [ ] LoRA/head/last/full FT are comparative controls; full FT is explicitly
      stronger and is not hidden.
- [ ] Composition numbers are labelled post-freeze diagnostics and are not
      called independent generalization.
- [ ] The manuscript does not claim composition transfer is SADR-exclusive.
- [ ] SUIM -0.178 pp mean, blur-s3 weakness, calibration composition failure,
      +21.4% latency, and UVMulti exclusion appear in the limitations.

## Evidence hierarchy

1. Primary: three-seed locked UIIS robustness.
2. Comparative: rank-2 LoRA, head-only, last-block-only, and full FT.
3. Mechanism: severity/family correction geometry and shuffled null.
4. Boundary: SUIM, calibration, blur, latency, and UVMulti audits.

The main text should not present frequency, routing, basis, selector, or every
failed composition regularizer as separate contributions. Those controls may
be summarized in one appendix table.

## Reviewer attack points

- Does the title overstate compositionality? It should use “parameter-efficient
  task-supervised residual adaptation.”
- Does “same protocol” incorrectly imply identical resolution, batch size, or
  optimizer? It must say shared split/exposure/budget/evaluation, with
  method-specific optimization settings.
- Does “parameter-efficient” accidentally imply lower inference compute? The
  paper must report the measured +21.4% SADR latency.
- Does the abstract imply SUIM improvement or real-world UVMulti validation?
  It must not.
- Does the paper imply the +1.141 pp matched composition diagnostic is the
  same as the original +1.302 pp stress test? These are different protocols.

## Figure/table freeze

- Fig. 1: frozen expert plus zero-initialized residual adapter.
- Fig. 2: 13-condition family/severity gain plot, including blur-s3.
- Fig. 3: UIIS target-robustness versus trainable-parameter trade-off.
- Fig. 4: severity-dependent correction geometry and shuffled null.
- Table 1: three-seed primary result and SUIM cost.
- Table 2: SADR, LoRA, head, last, and full FT parameter comparison.
- Table 3: post-freeze confirmation/calibration composition diagnostic.
- Table 4: compact ablation and conventional controls.

## Freeze rule

After this checklist is completed, only prose, figure rendering, table
formatting, citation, and non-numerical bug fixes are permitted. Any new
training or baseline requires a new freeze version and must not silently
change manuscript v0.1.
