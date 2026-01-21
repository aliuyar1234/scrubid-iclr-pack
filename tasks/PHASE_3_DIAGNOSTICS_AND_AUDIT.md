# PHASE_3_DIAGNOSTICS_AND_AUDIT

## Objective

Compute identifiability diagnostics and emit audit certificates that make results robust to drift.

## Dependencies

- `CANONICAL.IDS.PHASE_IDS.PHASE_2_INTERVENTIONS_AND_GENERATORS` completed.

## Deliverables

- Diagnostic implementations for:
  - `CANONICAL.SYMBOLS.RR_SYMBOL`
  - `CANONICAL.SYMBOLS.SSS_SYMBOL`
  - `CANONICAL.SYMBOLS.CC_SYMBOL`
- Audit certificate emission per `spec/04_IDENTIFIABILITY_DEFINITIONS.md` and `spec/06_DIAGNOSTICS_RR_SSS_CC.md`.
- Citation hygiene: `bib/citation_audit_table.md` and `bib/citations_verified.md` updated.

## Definition of Done

- RR uses the **max-distance over the near-optimal faithful set** as defined in `spec/06_DIAGNOSTICS_RR_SSS_CC.md`.
- SSS measures **replicate discovery stability** (not within-circuit scrub stability).
- CC measures **contradiction rate** (conflicting necessity claims across near-optimal circuits).

## Acceptance tests

- Synthetic identifiability sanity:
  - On `CANONICAL.IDS.SUITE_IDS.SUITE_SYNTH_V1` with redundancy_factor = 1, RR and CC are low and SSS is high.
  - Increasing redundancy_factor increases RR and CC.
- Certificate generation:
  - When RR exceeds `CANONICAL.DIAGNOSTICS.RR_THRESHOLD_HIGH`, a certificate JSON is emitted and contains:
    - reason_codes indicating the failing diagnostic(s)
    - S_near circuits
    - the evaluation seed and dataset hash
    - the intervention family ID
