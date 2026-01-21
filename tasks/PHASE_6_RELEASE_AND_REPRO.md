# PHASE_6_RELEASE_AND_REPRO

## Objective

Finalize packaging, reproducibility, and release artifacts.

## Dependencies

- `CANONICAL.IDS.PHASE_IDS.PHASE_5_PAPER_CAMERA_READY` completed.

## Deliverables

- Release ZIP (this pack) with an up-to-date `MANIFEST.sha256`.
- Reproducibility report documenting determinism and stability checks.

## Definition of Done

- All quality gates referenced in `spec/16_TEST_PLAN.md` pass.
- `MANIFEST.sha256` matches all files (excluding itself) byte-for-byte.
- `AUDIT_REPORT.md` and `PATCH_REPORT.md` are included at the repository root.

## Acceptance tests

- Run the determinism smoke test (`CANONICAL.CLI.CANONICAL_COMMANDS.CLI_CMD_DETERMINISM_SMOKE_TEST`) and confirm:
  - identical config produces identical run hash and artifacts.
- Verify `MANIFEST.sha256` by recomputing sha256 hashes and comparing.

