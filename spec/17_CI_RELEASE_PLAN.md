# 17_CI_RELEASE_PLAN.md

## SDR note

All command IDs are defined in `spec/00_CANONICAL.md`.

## Continuous integration

CI must run on every commit affecting:

- spec files
- configs
- implementation code

Required CI steps:

1. Spec validation: run `CLI_CMD_VALIDATE_SPEC`.
2. Unit tests: run the test suite including determinism and schema tests.
3. Integration tests: run the end-to-end toy pipeline.
4. Determinism smoke: run `CLI_CMD_DETERMINISM_SMOKE_TEST`.

## Release process

A release is created when:

- all CI steps pass
- all quality gates G0 through G4 are PASS

Release artifacts:

- a versioned tarball or zip of code
- a reproducibility report under `PATH_REPORTS_ROOT`
- a frozen copy of `bib/references.bib`
