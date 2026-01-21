# 11_DETERMINISM_REPRODUCIBILITY.md

## SDR note

All determinism flags and seeds are defined in `spec/00_CANONICAL.md`.

## Deterministic mode

Deterministic mode is enabled by `CLI.CANONICAL_FLAGS.FLAG_DETERMINISTIC`.

Requirements:

- All randomness must be seeded deterministically from `CANONICAL.SEEDS.SEED_GLOBAL` and from the deterministic `run_key`.
- Any sampling step (reference pairing, replicate seeds, bootstrap) must use a local RNG initialized from an explicitly recorded seed.
- GPU nondeterminism must be minimized by enabling deterministic algorithms and disabling benchmarking.
- Fail closed: if deterministic mode is enabled and the implementation cannot guarantee determinism (for example, unsupported non-deterministic ops on the selected device), it must abort and write a `run_record.json` with `overall_verdict = HARD_FAIL`.

## Canonical hashing primitives

All hashing constants are defined in `CANONICAL.HASHING`.

### sha256_hex

Definition:

- Input: a byte string b.
- Output: the lowercase hex SHA-256 digest of b.

### sha256_uint32

Definition:

- Input: a UTF-8 string s.
- Compute d = sha256_hex(utf8(s)).
- Output: interpret the first 8 hex characters of d as a big-endian unsigned 32-bit integer.

This method is pinned by `CANONICAL.HASHING.UINT32_FROM_SHA256_HEX_METHOD`.

## Canonical JSON serialization

To compute deterministic hashes of structured objects, ScrubID uses a canonical JSON byte representation.

### canonical_json_bytes(obj)

Definition:

- obj must be a JSON-serializable object containing only:
  - null, booleans, integers, strings, lists, and dicts.
- Floating point numbers are forbidden in objects that are hashed for determinism. If a value is conceptually real-valued, store it as a decimal string with fixed formatting.
- Serialize using:
  - sort_keys = `CANONICAL.HASHING.CANONICAL_JSON.SORT_KEYS`
  - ensure_ascii = `CANONICAL.HASHING.CANONICAL_JSON.ENSURE_ASCII`
  - separators = `CANONICAL.HASHING.CANONICAL_JSON.SEPARATORS`
  - allow_nan = `CANONICAL.HASHING.CANONICAL_JSON.ALLOW_NAN`
- Encode the resulting JSON string in UTF-8.

## run_key

`run_key` is the stable identifier of the deterministic inputs to a run.

Definition:

1. Build a dict `run_key_obj` with exactly the fields listed below, in any order (ordering is handled by canonical JSON):
   - project_id, project_version
   - suite_id, experiment_id
   - model_id
   - model_revision (string; use "unknown" if not available)
   - component_granularity
   - intervention_family_id
   - candidate_generator_ids (sorted list of ids)
   - reference_distribution_id
   - reference_assignment_id
   - resolved_config_hashes (dict of config file sha256)
   - dataset_fingerprints (dict keyed by split id)
   - thresholds (epsilon_abs, tau_abs, rr_near_optimal_mdl_rel_frac)
   - budgets (resolved numeric budgets)
2. Let b = canonical_json_bytes(run_key_obj).
3. Set run_key = sha256_hex(b).

All runs that share the same deterministic inputs must have the same run_key.

## Seed derivation

All seed derivation salts are defined in `CANONICAL.DETERMINISM.SEED_DERIVATION`.

### seed_effective

Definition:

- seed_effective = sha256_uint32(
  f"{CANONICAL.SEEDS.SEED_GLOBAL}|{CANONICAL.DETERMINISM.SEED_DERIVATION.SALT_SEED_EFFECTIVE}|{run_key}"
  )

### seed_reference_pairing

Definition:

- seed_reference_pairing = sha256_uint32(
  f"{seed_effective}|{CANONICAL.DETERMINISM.SEED_DERIVATION.SALT_REFERENCE_PAIRING}"
  )

### replicate seeds

Replicate seeds are used for replicate discovery runs (SSS) and for any randomized candidate generator steps.

Definition:

- seed_replicate(r) = sha256_uint32(
  f"{seed_effective}|{CANONICAL.DETERMINISM.SEED_DERIVATION.SALT_REPLICATE}|{r}"
  )

### bootstrap seed

Definition:

- seed_bootstrap = sha256_uint32(
  f"{seed_effective}|{CANONICAL.DETERMINISM.SEED_DERIVATION.SALT_BOOTSTRAP}"
  )

## run_key and run_record_hash

- `run_key` must be deterministic and computed from:
  - project id and version
  - resolved config content hashes
  - dataset hashes
  - model identifier and weight hashes
  - intervention family id(s)
  - generator id(s)
  - seed

`run_record_hash` is the sha256 hash of a canonical JSON serialization of the deterministic subset of the run record.

Normative rule:

- The deterministic subset is defined in `spec/14_RUN_RECORD_SCHEMA.md`.
- The byte representation must be `canonical_json_bytes(deterministic_subset)`.
- The hash must be sha256_hex(canonical_json_bytes(deterministic_subset)).

This allows multiple immutable runs with different `run_id` values to share the same `run_record_hash` when their deterministic inputs match.

## Determinism smoke test

The smoke test defined by `CLI_CMD_DETERMINISM_SMOKE_TEST` must:

1. Run a minimal synthetic generation and diagnostics pipeline twice with the same `run_key`.
2. Assert that the two runs have identical `run_record_hash`.
3. Assert that the two runs have identical diagnostics JSON and identical report table CSVs when regenerated only from `logs.jsonl`.

The smoke test must write a dedicated report under `PATH_REPORTS_ROOT`.
