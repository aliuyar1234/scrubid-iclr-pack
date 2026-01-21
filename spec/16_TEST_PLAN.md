# 16_TEST_PLAN.md

This file defines required tests and acceptance criteria.

## Test suite overview

Tests must be runnable via the canonical CLI commands in `CANONICAL.CLI.CANONICAL_COMMANDS`.

At minimum, the implementation must provide the following test entrypoints:

- `CANONICAL.CLI.CANONICAL_COMMANDS.CLI_CMD_VALIDATE_SPEC`
- `CANONICAL.CLI.CANONICAL_COMMANDS.CLI_CMD_DETERMINISM_SMOKE_TEST`

All tests are deterministic under `CANONICAL.DETERMINISM.DETERMINISTIC_MODE_STRICT`.

## T1: Spec conformance

Goal: ensure the SSOT spec pack is internally consistent.

Acceptance:

- `CANONICAL.CLI.CANONICAL_COMMANDS.CLI_CMD_VALIDATE_SPEC` returns success.
- No placeholder tokens exist and no ellipsis markers are present.

## T2: Determinism smoke test

Goal: verify bitwise determinism for a fixed configuration.

Protocol:

- Run the determinism smoke test twice with identical inputs and `CANONICAL.DETERMINISM.DETERMINISTIC_MODE_STRICT`.
- Compare:
  - `CANONICAL.FILES.RUN_RECORD_FILENAME`
  - `CANONICAL.FILES.DIAGNOSTICS_FILENAME`
  - selected circuit file

Acceptance:

- All artifacts match bit-for-bit.

## T3: Synthetic suite sanity

Goal: verify that the synthetic suite produces known redundancy regimes.

Protocol:

- Run on `CANONICAL.IDS.SUITE_IDS.SUITE_SYNTH_V1` with the canonical synthetic budgets.
- Evaluate RR, SSS, CC across settings that include:
  - a uniquely identifiable regime (expected RR low, CC low)
  - a redundant regime with multiple equivalent explanations (expected RR high and CC high)

Acceptance:

- RR increases monotonically with planted redundancy.
- CC increases with planted redundancy.
- SSS remains high when discovery randomness is controlled.

## T4: Real case study reproducibility

Goal: ensure the real-case pipeline produces stable metrics.

Protocol:

- For each real suite:
  - run twice under deterministic mode
  - compute diagnostic deltas between runs

Acceptance:

- Faithfulness and diagnostics match exactly in deterministic mode.
- Under non-deterministic mode, SSS meets the stable threshold for at least one candidate generator.
