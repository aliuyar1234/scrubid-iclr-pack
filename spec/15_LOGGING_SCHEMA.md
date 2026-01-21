# 15_LOGGING_SCHEMA.md

This file defines required structured logging events.

All event type identifiers are defined in `CANONICAL.ENUMS.LOG_EVENT_TYPES` (in `spec/00_CANONICAL.md`).

## Logging rules

- Every log line is a single JSON object.
- Every log event must include:
  - `event_type`
  - `timestamp_utc`
  - `run_id`
  - `schema_version` (must equal `CANONICAL.SCHEMAS.LOG_SCHEMA_VERSION`)

## Required events

### EVENT_RUN_START

Must include:

- suite_id, experiment_id, intervention_family_id, candidate_generator_id, component_granularity
- deterministic_mode, seed_global
- reference_distribution_id, reference_assignment_id

### EVENT_DATASET_WRITTEN

Must include:

- dataset_fingerprint
- reference_dataset_fingerprint
- sizes: num_examples_eval, num_examples_ref

### EVENT_CANDIDATES_WRITTEN

Must include:

- num_candidates
- topk_summary: summary of the lowest-MDL candidates

### EVENT_DIAGNOSTICS_WRITTEN

Must include:

- RR, SSS, CC and their verdicts
- whether an audit certificate was emitted
- diagnostics_fingerprint

### EVENT_RUN_END

Must include:

- overall_verdict
- pointers to run_record.json and diagnostics.json
