# 19_PAPER_WRITING_PLAN.md

This file maps paper claims to required evidence.

## Claim–evidence map

### Claim C1: ScrubID provides a practical audit certificate

Evidence:

- Synthetic suite results show RR and CC spike in planted redundancy regimes.
- Audit certificate files include `reason_codes` distinguishing RR/CC-driven non-identifiability from SSS-driven discovery instability.

Artifacts:

- Table T1
- Example certificates (appendix)

### Claim C2: Scrubbed Solution Stability (SSS) captures discovery instability

Evidence:

- Replicate runs on the same suite produce varying circuits when the discovery process is unstable.
- SSS quantifies this variation as mean pairwise overlap.

Artifacts:

- Table T1 (SSS columns)
- Table T3 (SSS sensitivity)

### Claim C3: Contradiction Coefficient (CC) captures incompatible necessity claims

Evidence:

- In redundant regimes, distinct near-optimal circuits disagree about which components are necessary.
- CC increases with redundancy and correlates with certificate emission.

Artifacts:

- Table T1 (CC columns)
- Certificate summaries

### Claim C4: Real case studies show when mechanistic explanations are stable or unstable

Evidence:

- On IOI/greater-than/induction, ScrubID produces circuits whose RR/SSS/CC indicate stability vs discovery instability under the paper’s chosen intervention family and generators.

Artifacts:

- Table T2
- Reproducibility tests (spec/16)

## Mechanical closure

All claim→evidence links used in the paper must be encoded in `paper_results_manifest.json` and validated by `CLI_CMD_VALIDATE_PAPER_MANIFEST`.
