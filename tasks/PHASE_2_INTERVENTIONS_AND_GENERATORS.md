# PHASE_2_INTERVENTIONS_AND_GENERATORS

## Objective

Implement intervention families and candidate generators, end-to-end, against the synthetic suite.

## Dependencies

- `CANONICAL.IDS.PHASE_IDS.PHASE_0_FOUNDATION` completed.
- `CANONICAL.IDS.PHASE_IDS.PHASE_1_SYNTHETIC_SUITE` completed.

## Deliverables

- Implementations for `CANONICAL.IDS.INTERVENTION_FAMILY_IDS` following `spec/03_INTERVENTION_FAMILIES.md` and `spec/22_IMPLEMENTATION_CONTRACT.md`.
- Candidate generator adapters for the enabled generator set in `configs/generators.yaml`.
- Smoke-run scripts using:
  - `CANONICAL.CLI.CANONICAL_COMMANDS.CLI_CMD_SYNTH_RUN_CANDIDATE_GENERATORS`, and
  - `CANONICAL.CLI.CANONICAL_COMMANDS.CLI_CMD_SYNTH_RUN_DIAGNOSTICS`.

## Definition of Done

- For each intervention family:
  - The scrubbed model f^(C,I) matches `spec/02_FORMAL_DEFINITIONS.md`.
  - D_ref is consumed using the suite’s `CANONICAL.REFERENCE.REF_ASSIGNMENT_DEFAULT_BY_SUITE`.
  - Hookpoints and tensor slicing match `CANONICAL.HOOKS`.
- For each enabled candidate generator:
  - Produces a finite candidate set S with deterministic ordering under `CANONICAL.CLI.CANONICAL_FLAGS.FLAG_DETERMINISTIC`.

## Acceptance tests

- Run `CANONICAL.CLI.CANONICAL_COMMANDS.CLI_CMD_DETERMINISM_SMOKE_TEST` and confirm:
  - Hashes of artifacts match across repeated runs.
  - `MANIFEST.sha256` remains unchanged for the pack.
- Run the synthetic pipeline on `CANONICAL.IDS.SUITE_IDS.SUITE_SYNTH_V1`:
  1) `CANONICAL.CLI.CANONICAL_COMMANDS.CLI_CMD_SYNTH_RUN_CANDIDATE_GENERATORS`
  2) `CANONICAL.CLI.CANONICAL_COMMANDS.CLI_CMD_SYNTH_RUN_DIAGNOSTICS`
  and confirm:
  - At least one ε-faithful circuit is found.
  - `CANONICAL.FILES.RUN_RECORD_FILENAME` is written and validates against `spec/14_RUN_RECORD_SCHEMA.md`.

