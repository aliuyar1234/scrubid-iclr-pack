# PHASE_0_FOUNDATION.md

## Goal

Implement the core plumbing required by all later phases:

- Canonical loader and config resolution
- Deterministic run directory creation
- Structured logging and run_record.json emission
- Minimal end-to-end pipeline stub

## Dependencies

None.

## Deliverables

- `scrubid/canonical.py` loads the canonical block from `spec/00_CANONICAL.md`.
- `scrubid/config.py` resolves `${CANONICAL.*}` references.
- `scrubid/io/run_record.py` writes `CANONICAL.FILES.RUN_RECORD_FILENAME`.
- `scrubid/io/logging.py` emits JSONL logs with event types in `CANONICAL.ENUMS.LOG_EVENT_TYPES`.

## Definition of done

- Running the canonical spec validator succeeds.
- Any CLI run creates an immutable run directory under `CANONICAL.PATHS.PATH_RUNS_ROOT` and writes:
  - `CANONICAL.FILES.RUN_RECORD_FILENAME`
  - `CANONICAL.FILES.LOG_JSONL_FILENAME`

## Acceptance tests

- `spec/16_TEST_PLAN.md` tests T1 and T2 pass.
