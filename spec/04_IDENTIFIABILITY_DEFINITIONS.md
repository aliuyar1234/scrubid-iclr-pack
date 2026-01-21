# 04_IDENTIFIABILITY_DEFINITIONS.md

## SDR note

All constants referenced here are defined in `spec/00_CANONICAL.md`.

## Circuit equivalence under an intervention family

Fix a model `f`, dataset `D`, behavior metric `m`, tolerance `ε`, and intervention family `𝓘`.

Define the *scrubbed behavior functional*:

- `B(C) = { m(x; f^(C,𝓘)) : x ∈ D }`.

Two circuits `C1` and `C2` are **𝓘-equivalent at tolerance ε** (written `C1 ≈_{𝓘,ε} C2`) if:

- `mean_{x∈D} |m(x; f^(C1,𝓘)) - m(x; f^(C2,𝓘))| ≤ ε`.

This equivalence is task- and intervention-family-specific.

## Identifiability of a circuit explanation

A circuit `C*` is **identifiable under `𝓘` at tolerance `ε`** if every `ε`-faithful circuit `C` satisfies:

- `C ≈_{𝓘,ε} C*`.

Operationally, identifiability fails if the search procedure finds at least two circuits that:

1. are both `ε`-faithful, and
2. are not `𝓘`-equivalent.

## Audit certificate

A ScrubID audit certificate is a JSON artifact emitted by the diagnostic stage when one or more diagnostic verdicts are FAIL (see `spec/06_DIAGNOSTICS_RR_SSS_CC.md`).

The audit certificate must be sufficient for a reviewer to determine whether the failure is driven by:

- **Non-identifiability** under the chosen intervention family (RR/CC failure), and/or
- **Discovery instability** (SSS failure).

Required contents:

- `reason_codes`: one or more values from `CANONICAL.ENUMS.CERTIFICATE_REASON_CODES` indicating which diagnostic(s) are FAIL.
- model identifier and provenance (at minimum `model_id` and `model_revision`).
- dataset provenance (at minimum deterministic fingerprints for D_eval and D_ref).
- intervention family id and reference distribution provenance (`reference_distribution_id`, `reference_assignment_id`).
- tolerance `ε`, baseline scale `S0`, and necessity threshold `τ`.
- the diagnostic values RR/SSS/CC and their verdicts.
- the near-optimal set `S_near` used for RR/CC (possibly empty), with per-circuit:
  - component lists
  - MDL values
  - faithfulness loss `Δ(C)`
  - necessity labels `N(v; C)` when computed
- the replicate-selected circuits {C_r} used for SSS.

## Relation to prior identifiability work

`[@meloux2025identifiability]` demonstrates systematic non-identifiability even in small networks under mechanistic interpretability criteria.

ScrubID adopts the identifiability framing but provides an end-to-end operational certificate for transformer circuits, and adds stability and complexity diagnostics to distinguish:

- multiple faithful circuits that are functionally equivalent under the chosen `𝓘`, and
- multiple faithful circuits that yield incompatible mechanistic claims.
