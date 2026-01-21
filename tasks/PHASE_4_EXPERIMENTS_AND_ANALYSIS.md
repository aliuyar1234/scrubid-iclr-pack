# PHASE_4_EXPERIMENTS_AND_ANALYSIS

## Objective

Run the full experimental matrix and produce analysis artifacts for the paper.

## Dependencies

- `CANONICAL.IDS.PHASE_IDS.PHASE_3_DIAGNOSTICS_AND_AUDIT` completed.

## Deliverables

- Synthetic sweep results for `CANONICAL.IDS.SUITE_IDS.SUITE_SYNTH_V1`.
- Real-model case study results for suites in `CANONICAL.IDS.SUITE_IDS`.
- Aggregated tables matching `spec/18_RESULTS_TABLE_SKELETONS.md`.

## Definition of Done

- All planned runs:
  - write `CANONICAL.FILES.RUN_RECORD_FILENAME` per `spec/14_RUN_RECORD_SCHEMA.md`.
  - log per `spec/15_LOGGING_SCHEMA.md`.
- Analysis scripts produce stable summary tables:
  - RR, SSS, CC statistics
  - effect sizes with `CANONICAL.STATS.CONFIDENCE_LEVEL` intervals

## Acceptance tests

- Real-model reproduction:
  - Re-running a full run with identical config and deterministic flag reproduces the same run hash.
- Paper tables:
  - The produced tables match the column skeletons in `spec/18_RESULTS_TABLE_SKELETONS.md`.

