# AUDIT_REPORT.md

Scope: audit of `scrubid_iclr_pack_v1_0_2` to produce a drift-proof `scrubid_iclr_pack_v1_0_3`.

This report focuses on implementation drift: places where an implementer (human or Codex) would need to guess the intended meaning, invent constants, or reconcile contradictions.

## Summary of findings

### BLOCKER 1 — Paper vs spec mismatch on SSS and CC

- Problem: `paper/01_ABSTRACT.md`, `paper/02_INTRO_NOVELTY.md`, `paper/05_EXPERIMENTS.md`, and `paper/06_LIMITATIONS_ETHICS.md` described earlier meanings of SSS and CC that conflict with the normative diagnostics in `spec/06_DIAGNOSTICS_RR_SSS_CC.md`.
- Why this causes drift: implementers tend to follow the paper narrative when the paper and spec disagree. This can silently change what SSS and CC compute, invalidating claims and tables.
- Fix in v1.0.3:
  - Paper updated to match the spec:
    - SSS is stability across replicate discovery runs.
    - CC is a contradiction score over component necessity across near-optimal faithful circuits.
  - Files patched: `paper/01_ABSTRACT.md`, `paper/02_INTRO_NOVELTY.md`, `paper/04_METHOD.md`, `paper/05_EXPERIMENTS.md`, `paper/06_LIMITATIONS_ETHICS.md`, `paper/00_PAPER_SNAPSHOT.md`.

### BLOCKER 2 — Logging event names conflicted with SSOT and configs

- Problem: `spec/15_LOGGING_SCHEMA.md` required event types named with a READY / COMPUTED convention, while `spec/00_CANONICAL.md` and `configs/logging_schema.yaml` used the WRITTEN convention.
- Why this causes drift: logging is part of the determinism and immutability guarantees. If event names differ, CI and downstream aggregation will fail or, worse, partially work with silently missing fields.
- Fix in v1.0.3:
  - `spec/15_LOGGING_SCHEMA.md` updated to require the SSOT-backed WRITTEN event names.

### BLOCKER 3 — Feature patching intervention was described but not defined

- Problem: `spec/03_INTERVENTION_FAMILIES.md` described a feature-level patching intervention family without an SSOT ID and without config support.
- Why this causes drift: implementers will either invent an ID, implement an ad hoc version, or guess the intended feature dictionary interface.
- Fix in v1.0.3:
  - Removed the undefined feature-level intervention from the normative intervention family list.
  - Added an explicit fail-closed rule for `component_granularity` values `feature` and `neuron`.

### BLOCKER 4 — Real-task metric definitions diverged between real-case specs

- Problem: `spec/08_REAL_MODEL_CASE_STUDIES.md` and `spec/22_IMPLEMENTATION_CONTRACT.md` disagreed on the metrics for Greater-Than and Induction.
- Why this causes drift: the behavior metric m(x; f) is the object that defines Δ(C), faithfulness, and all diagnostics. Metric drift breaks reproducibility and undermines any claimed results.
- Fix in v1.0.3:
  - `spec/22_IMPLEMENTATION_CONTRACT.md` updated to align exactly with the normative definitions in `spec/08_REAL_MODEL_CASE_STUDIES.md`.

### BLOCKER 5 — Determinism was underspecified for hashing and seed derivation

- Problem: v1.0.2 stated that replicate seeds and hashes are derived deterministically, but did not pin down:
  - the canonical JSON serialization used for hashing,
  - the exact sha256-to-uint32 conversion,
  - the seed derivation function (including salts),
  - and the forbidden-types rule for hashed objects.
- Why this causes drift: bitwise determinism can fail across machines or Python versions if hashing and JSON canonicalization are not pinned.
- Fix in v1.0.3:
  - Added SSOT hashing constants under `HASHING` and seed derivation salts under `DETERMINISM.SEED_DERIVATION` in `spec/00_CANONICAL.md`.
  - Fully specified canonical JSON hashing and seed derivation in `spec/11_DETERMINISM_REPRODUCIBILITY.md`.
  - Updated `spec/06_DIAGNOSTICS_RR_SSS_CC.md` to use `seed_replicate(r)` rather than an abstract hash placeholder.

### MAJOR 6 — SSOT key typos and suite-id references in the real-case spec

- Problem: `spec/08_REAL_MODEL_CASE_STUDIES.md` referenced non-existent dataset size keys for induction and used a non-SSOT suite-id namespace label.
- Why this causes drift: typos in SSOT keys are a direct implementation blocker.
- Fix in v1.0.3:
  - Updated induction prompt-count keys to match `spec/00_CANONICAL.md`.
  - Standardized suite-id references to use `IDS.SUITE_IDS.*`.

### MINOR 7 — Version sync markers were inconsistent

- Problem: the pack title and paper snapshot referenced v1.0.2.
- Fix in v1.0.3: updated version markers to v1.0.3.

## Validation checks performed

- Forbidden-token sweep: verified that no file contains placeholder markers or a three-dot ellipsis.
- Canonical reference sweep:
  - All SSOT path references that begin with `CANONICAL.` resolve against the YAML block in `spec/00_CANONICAL.md` (for example, `CANONICAL.IDS.SUITE_IDS.SUITE_SYNTH_V1`).
  - All dotted config keys in `configs/*.yaml` resolve to a path in the same YAML block.
- Logging schema check: the required event names in `spec/15_LOGGING_SCHEMA.md` match `configs/logging_schema.yaml` and the SSOT.
