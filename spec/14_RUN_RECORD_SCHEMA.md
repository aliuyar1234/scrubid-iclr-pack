# 14_RUN_RECORD_SCHEMA.md

This file defines the schema for the required `run_record.json` emitted by every CLI run.

- The filename is `CANONICAL.FILES.RUN_RECORD_FILENAME`.
- The schema version is `CANONICAL.SCHEMAS.RUN_RECORD_SCHEMA_VERSION`.

## Required fields

All fields are required unless explicitly marked optional.

### Top-level

- `schema_version` (integer): must equal `CANONICAL.SCHEMAS.RUN_RECORD_SCHEMA_VERSION`.
- `project_id` (string): must equal `CANONICAL.PROJECT_ID`.
- `project_version` (string): must equal `CANONICAL.PROJECT_VERSION`.

### Identifiers

- `run_id` (string): the directory name for the run (see `CANONICAL.OUTPUT_NAMING`).
- `suite_id` (string): one of `CANONICAL.IDS.SUITE_IDS`.
- `experiment_id` (string): one of `CANONICAL.IDS.EXPERIMENT_IDS`.
- `model_id` (string): a resolvable model identifier (for example a HuggingFace model ID).
- `model_revision` (string): immutable model revision identifier; use `"unknown"` only if unavailable.
- `model_local_path` (string, optional): local filesystem directory used to load weights, when different from `model_id`.
- `intervention_family_id` (string): one of `CANONICAL.IDS.INTERVENTION_FAMILY_IDS`.
- `candidate_generator_id` (string): one of `CANONICAL.IDS.CANDIDATE_GENERATOR_IDS`.
- `component_granularity` (string): one of `CANONICAL.COMPONENT_GRANULARITY.GRANULARITY_OPTIONS`.

### Reference distribution

- `reference_distribution_id` (string): one of the `CANONICAL.REFERENCE` REFDIST IDs.
- `reference_assignment_id` (string): one of the `CANONICAL.REFERENCE` REFASSIGN IDs.

### Environment and determinism

- `timestamp_utc` (string): ISO-8601 UTC timestamp.
- `git_commit` (string): length `CANONICAL.GIT.GIT_COMMIT_HEX_LEN`.
- `python_version` (string)
- `platform` (string)
- `deterministic_mode` (boolean)

### Seeds

- `seed_global` (integer)
- `seed_suite` (integer)
- `seed_reference_pairing` (integer)

### Data provenance

- `dataset_fingerprint` (string): deterministic hash of prompts and labels used for D_eval.
- `reference_dataset_fingerprint` (string): deterministic hash of D_ref.

### Core results

- `baseline_score_s0` (number)
- `epsilon` (number)
- `best_circuit_mdl` (number)
- `best_circuit_size` (integer)
- `faithfulness_delta` (number): Δ(C_best)

### Diagnostics

- `RR` (number)
- `RR_verdict` (string): one of `CANONICAL.ENUMS.VERDICT_ENUM`.
- `SSS` (number)
- `SSS_verdict` (string): one of `CANONICAL.ENUMS.VERDICT_ENUM`.
- `CC` (number)
- `CC_verdict` (string): one of `CANONICAL.ENUMS.VERDICT_ENUM`.

### Overall gating

- `overall_verdict` (string): one of `CANONICAL.ENUMS.VERDICT_ENUM`.
- `quality_gates_passed` (boolean)

### Artifact pointers

Paths are relative to the run directory.

- `paths` (object):
  - `logs_dir` (string)
  - `results_dir` (string)
  - `diagnostics_file` (string)
  - `certificate_file` (string, optional)

## Immutability requirement

A `run_record.json` is immutable once written. If a run is re-executed, the implementation must create a new run directory with a new run_id.

## Deterministic subset for `run_record_hash`

`run_record_hash` is computed as `sha256_hex(canonical_json_bytes(deterministic_subset))` (see `spec/11_DETERMINISM_REPRODUCIBILITY.md`).

Normative rule for v1.0.3 baseline:

- The deterministic subset includes all top-level fields **except**:
  - `run_id`
  - `timestamp_utc`
  - `git_commit`
  - `python_version`
  - `platform`
  - `paths`
  - `model_local_path` (optional local-only field)
  - `CANONICAL.OUTPUT_NAMING.RUN_RECORD_HASH_FIELD`
- Scalar floating-point fields are serialized deterministically using the canonical decimal-string convention described in `CANONICAL.HASHING`.
