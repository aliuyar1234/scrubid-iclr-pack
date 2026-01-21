# SPEC.md (SSOT entrypoint)

This is the entrypoint specification for the ScrubID implementation.

## Precedence

1. `spec/00_CANONICAL.md` is the single source of truth for all constants, IDs, paths, CLI literals, thresholds, and enums.
2. `spec/02_FORMAL_DEFINITIONS.md` through `spec/08_REAL_MODEL_CASE_STUDIES.md` define the core method and must not contradict each other.
3. `configs/*.yaml` define experiment configuration using canonical keys from `spec/00_CANONICAL.md`.
4. `tasks/*` define the implementation plan and acceptance criteria.
5. `paper/*` uses produced artifacts and must not claim results that are not backed by run IDs.

## Invariants

- SDR: all literals defined once in `spec/00_CANONICAL.md`.
- Every CLI invocation emits a `run_record.json` that conforms to `spec/14_RUN_RECORD_SCHEMA.md`.
- Every run writes an append-only JSONL log conforming to `spec/15_LOGGING_SCHEMA.md`.
- All outputs are written under `PATH_OUTPUT_ROOT` and are immutable.

## Quality gates (PASS criteria)

Quality gates are defined as named gates and must be evaluated in the implementation.

- **G0 Spec coherence:**
  - No undefined IDs.
  - Config resolution is total.
  - All references to constants are by canonical name.

- **G1 Determinism smoke:**
  - Running the determinism smoke test twice yields identical `run_record_hash`.

- **G2 Synthetic ground-truth sanity:**
  - Synthetic generator emits ground-truth labels.
  - RR increases with redundancy factor.
  - CC correlates with planted circuit size.

- **G3 Real-model reproducibility:**
  - RR/SSS/CC are stable across seeds.
  - RR/SSS/CC are stable across intervention families.

- **G4 Reproducibility:**
  - All tables and plots are generated only from `run_record.json`, `logs.jsonl`, and config files.

- **G5 Paper evidence:**
  - Each central claim in `paper/` is linked to a table or plot that is reproducibly generated.

## Acceptance criteria

The implementation is accepted when the phase plan in `tasks/` is PASS through PHASE_4 and the writing plan in `spec/19_PAPER_WRITING_PLAN.md` is satisfied.
