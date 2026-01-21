# 13_CLI_RUNBOOK.md

## SDR note

All command strings and flags are defined in `spec/00_CANONICAL.md`.

## Required commands

The pipeline is executed by the canonical command IDs listed below. The exact strings are stored in `CLI.CANONICAL_COMMANDS`.

1. `CLI_CMD_VALIDATE_SPEC`
   - Validates SDR, config schemas, and ID resolution.

2. `CLI_CMD_SYNTH_GENERATE_SUITE`
   - Generates the synthetic suite `SUITE_SYNTH_V1` under an immutable run directory.

3. `CLI_CMD_SYNTH_RUN_CANDIDATE_GENERATORS`
   - Runs all enabled candidate generators on the synthetic suite (enabled set is defined in `configs/generators.yaml`).

4. `CLI_CMD_SYNTH_RUN_DIAGNOSTICS`
   - Computes RR/SSS/CC and emits certificates as needed.

5. `CLI_CMD_REAL_IOI_RUN`
6. `CLI_CMD_REAL_GREATERTHAN_YN_RUN`
7. `CLI_CMD_REAL_INDUCTION_RUN`
   - Runs real-model case studies and produces diagnostics.

8. `CLI_CMD_AGGREGATE_RESULTS`
   - Aggregates run outputs into canonical tables.

9. `CLI_CMD_BUILD_PAPER_ARTIFACTS`
   - Produces paper-ready tables and plots and stores them under `PATH_REPORTS_ROOT`.

10. `CLI_CMD_VALIDATE_PAPER_MANIFEST`
   - Validates `paper_results_manifest.json` claim→evidence entries and verifies referenced artifact hashes.

11. `CLI_CMD_DETERMINISM_SMOKE_TEST`
   - Runs the determinism smoke test and writes a report.

12. `CLI_CMD_REPRODUCE_PAPER`
   - Runs the full paper pipeline into a fresh output bundle and verifies artifact hashes against `paper_results_manifest.json`.

## Output rules

- All run-producing commands write a new `run_id` directory under `PATH_RUNS_ROOT`.
- Runs must fail if a target directory already exists.
- All run-producing commands write a `run_record.json` and `logs.jsonl`.
