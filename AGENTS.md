# AGENTS.md (Codex operating rules)

This pack is designed for an implementation agent that must avoid drift.

## Read order

1. `SPEC.md`
2. `spec/00_CANONICAL.md`
3. `spec/01_VISION_SCOPE.md`
4. `spec/02_FORMAL_DEFINITIONS.md` through `spec/08_REAL_MODEL_CASE_STUDIES.md`
5. `spec/09_CANDIDATE_GENERATORS.md` through `spec/22_IMPLEMENTATION_CONTRACT.md`
6. `tasks/` (phase-by-phase plan)
7. `paper/` (paper text and evidence mapping)
8. `bib/` (verified citations)

## Single Definition Rule (SDR)

- Every constant, ID, CLI literal, path, threshold, event type, and symbol definition must be defined *exactly once* in `spec/00_CANONICAL.md`.
- Any other file must reference it by name only.
- If you need a new constant, add it to `spec/00_CANONICAL.md` and update any dependent docs.

## Determinism and immutability

- The pipeline must support deterministic mode (see `spec/11_DETERMINISM_REPRODUCIBILITY.md`).
- Runs and artifacts are immutable: never overwrite any prior run directory. If a target output path exists, the run must fail.

## Change protocol

When implementing or modifying anything:

1. Update `spec/00_CANONICAL.md` if any literal, threshold, path, ID, or enum changes.
2. Update the relevant spec file(s) and ensure they reference canonical names.
3. Update `configs/*.yaml` only with canonical keys (no raw literals that violate SDR).
4. Run `CLI_CMD_VALIDATE_SPEC`.
5. Run the determinism smoke test `CLI_CMD_DETERMINISM_SMOKE_TEST`.
6. Regenerate `MANIFEST.sha256` and rebuild the zip artifact if packaging.

## Phase workflow

- Follow phases in `tasks/PHASE_0_FOUNDATION.md` through `tasks/PHASE_6_RELEASE_AND_REPRO.md`.
- A phase may be marked PASS only if its acceptance criteria are met and recorded with a concrete run ID.
- Any divergence from spec requires updating the spec first.
