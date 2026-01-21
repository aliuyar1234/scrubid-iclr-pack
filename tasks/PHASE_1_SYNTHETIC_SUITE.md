# PHASE_1_SYNTHETIC_SUITE.md

## Goal

Implement the synthetic suite generator and use it to validate the diagnostics RR, SSS, CC.

## Dependencies

- PHASE_0_FOUNDATION.md

## Deliverables

- `scrubid/datasets/synthetic.py` implements the synthetic generator specified in `spec/07_SYNTHETIC_SUITE_GENERATOR.md`.
- Synthetic D_ref generation matches `CANONICAL.REFERENCE.REFDIST_DEFAULT_BY_SUITE`.
- Diagnostics modules compute RR/SSS/CC per `spec/06_DIAGNOSTICS_RR_SSS_CC.md`.

## Definition of done

- Synthetic suite can be generated deterministically.
- Planted redundancy produces measurable changes in RR and CC.

## Acceptance tests

- `spec/16_TEST_PLAN.md` test T3 passes.
