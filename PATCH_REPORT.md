# PATCH_REPORT.md

This report summarizes the v1.0.2 → v1.0.3 patch.

## What changed

### Paper–spec alignment

- Updated paper text to match the normative diagnostic semantics:
  - SSS is stability across replicate discovery runs.
  - CC is a contradiction score over component necessity across the near-optimal circuit set.
- Files patched: `paper/01_ABSTRACT.md`, `paper/02_INTRO_NOVELTY.md`, `paper/04_METHOD.md`, `paper/05_EXPERIMENTS.md`, `paper/06_LIMITATIONS_ETHICS.md`, `paper/00_PAPER_SNAPSHOT.md`.

### Logging schema coherence

- Updated `spec/15_LOGGING_SCHEMA.md` to require the same event names used in the SSOT and configs (the WRITTEN convention).

### Intervention family list hardened

- Removed the undefined feature-level patching intervention from `spec/03_INTERVENTION_FAMILIES.md`.
- Added an explicit fail-closed rule for the `feature` and `neuron` granularities (v1.0.3 is head and MLP only).

### Real-case metric definitions made single-source

- Treated `spec/08_REAL_MODEL_CASE_STUDIES.md` as the normative definition of m(x; f) for real tasks.
- Updated `spec/22_IMPLEMENTATION_CONTRACT.md` to match the Greater-Than and Induction metrics in `spec/08_REAL_MODEL_CASE_STUDIES.md`.

### Determinism pinned to an explicit procedure

- Added SSOT constants required to make determinism non-ambiguous:
  - `DETERMINISM.SEED_DERIVATION.*` salts
  - `HASHING.*` constants for canonical JSON and sha256-to-uint32 conversion
- Expanded `spec/11_DETERMINISM_REPRODUCIBILITY.md` to specify:
  - canonical JSON byte encoding
  - run_key construction
  - seed derivation functions (seed_effective, seed_reference_pairing, seed_replicate, seed_bootstrap)
- Updated `spec/06_DIAGNOSTICS_RR_SSS_CC.md` to use `seed_replicate(r)`.

### SSOT key typo fixes

- Fixed induction dataset size key references in `spec/08_REAL_MODEL_CASE_STUDIES.md`.

### Version markers

- Updated `README.md`, `paper/00_PAPER_SNAPSHOT.md`, and `spec/00_CANONICAL.md` to v1.0.3 identifiers.

## Rationale

v1.0.2 was close to implementable, but several contradictions required implementers to guess the intended meaning of diagnostics, logging events, and real-task metrics. v1.0.3 resolves these contradictions by selecting a single normative definition per concept and enforcing SSOT-backed identifiers.

## Backwards compatibility

- This pack remains documentation-only; there are no code APIs to break.
- SSOT keys were extended (new `HASHING` and `DETERMINISM.SEED_DERIVATION` blocks). Implementations targeting v1.0.3 must follow the updated SSOT.
