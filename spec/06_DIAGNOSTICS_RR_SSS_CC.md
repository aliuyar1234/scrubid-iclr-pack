# 06_DIAGNOSTICS_RR_SSS_CC.md

This file defines ScrubID’s three diagnostics:

- **RR**: Redundancy Ratio (worst-case disagreement among near-optimal circuits).
- **SSS**: Scrubbed Solution Stability (stability of discovered circuits across replicate discovery runs).
- **CC**: Contradiction Coefficient (fraction of components whose necessity status is inconsistent across near-optimal circuits).

All thresholds and constants referenced here are defined in `spec/00_CANONICAL.md`.

## Inputs

A diagnostic computation is parameterized by:

- suite_id
- intervention_family_id
- component_granularity
- candidate_generator_id
- seed_global

The candidate generation stage must produce a list of candidate circuits with:

- Δ(C): faithfulness loss (defined in `spec/02_FORMAL_DEFINITIONS.md`)
- MDL(C): complexity proxy (defined in `spec/02_FORMAL_DEFINITIONS.md`)

Let S be the set of candidate circuits considered for diagnostics.

Define:

- ε using `CANONICAL.DIAGNOSTICS.EPSILON_ABS_MIN` and `CANONICAL.DIAGNOSTICS.EPSILON_REL_FRAC`.
- F = {C ∈ S : Δ(C) ≤ ε}.

## Deterministic tie-break ordering

Whenever ScrubID needs to select or truncate a set of circuits deterministically, it must sort by:

1. MDL(C) ascending.
2. |C| ascending.
3. Lexicographic order of the sorted component IDs in C.

## RR (Redundancy Ratio)

1. If F is empty, define RR = 0 and set RR_verdict = `CANONICAL.ENUMS.VERDICTS.VERDICT_FAIL`.
2. Otherwise, let C* be the first element of F under the deterministic tie-break ordering.
3. Define the near-optimal set:
   - S_near = {C ∈ F : MDL(C) ≤ (1 + CANONICAL.DIAGNOSTICS.RR_NEAR_OPTIMAL_MDL_REL_FRAC) · MDL(C*)}.
4. If |S_near| exceeds `CANONICAL.DIAGNOSTICS.RR_NUM_CIRCUITS_SET`, keep only the first `CANONICAL.DIAGNOSTICS.RR_NUM_CIRCUITS_SET` circuits under the deterministic ordering.
5. Define Jaccard similarity between two circuits A and B as:
   - J(A, B) = |A ∩ B| / |A ∪ B|.
   - If A ∪ B is empty, define J(A, B) = 1.
6. Define distance d(A, B) = 1 − J(A, B).
7. RR = max_{A, B ∈ S_near} d(A, B). If |S_near| < 2, RR = 0.

Verdicting:

- PASS if RR < `CANONICAL.DIAGNOSTICS.RR_THRESHOLD_LOW`.
- WARN if `CANONICAL.DIAGNOSTICS.RR_THRESHOLD_LOW` ≤ RR < `CANONICAL.DIAGNOSTICS.RR_THRESHOLD_HIGH`.
- FAIL if RR ≥ `CANONICAL.DIAGNOSTICS.RR_THRESHOLD_HIGH`.

## SSS (Scrubbed Solution Stability)

SSS measures how stable the discovered explanation is under replicate discovery runs.

Let R = `CANONICAL.DIAGNOSTICS.SSS_NUM_REPLICATES`.

1. Run the full discovery pipeline R times, holding suite_id, intervention_family_id, model, and datasets fixed.
2. Each replicate r uses a deterministic derived seed defined in `spec/11_DETERMINISM_REPRODUCIBILITY.md`:
   - seed_r = seed_replicate(r)
   - where seed_replicate is derived from run_key and `CANONICAL.SEEDS.SEED_GLOBAL`.
3. Let the selected circuit from replicate r be C_r.
4. Define SSS as the mean pairwise Jaccard similarity:
   - SSS = mean over all unordered pairs (r, s) with r < s of J(C_r, C_s).

If any replicate fails to produce a faithful circuit, that replicate must output C_r = empty set; this deterministically reduces SSS.

Verdicting:

- PASS if SSS ≥ `CANONICAL.DIAGNOSTICS.SSS_THRESHOLD_STABLE`.
- WARN if `CANONICAL.DIAGNOSTICS.SSS_THRESHOLD_BORDERLINE` ≤ SSS < `CANONICAL.DIAGNOSTICS.SSS_THRESHOLD_STABLE`.
- FAIL if SSS < `CANONICAL.DIAGNOSTICS.SSS_THRESHOLD_BORDERLINE`.

## CC (Contradiction Score)

CC measures whether near-optimal circuits disagree about which components are necessary.

Preliminaries:

- Use S_near from RR (after truncation to `CANONICAL.DIAGNOSTICS.RR_NUM_CIRCUITS_SET`).
- Define the necessity threshold:
  - τ = max(`CANONICAL.DIAGNOSTICS.EPSILON_ABS_MIN`, `CANONICAL.DIAGNOSTICS.TAU_REL_FRAC_NECESSITY` · S0)
  - S0 is defined in `spec/02_FORMAL_DEFINITIONS.md`.

Necessity labeling:

- For each circuit C in S_near, and each component v in C:
  1. Compute Δ(C).
  2. Compute Δ(C \ {v}).
  3. Mark v as necessary for C if Δ(C \ {v}) − Δ(C) ≥ τ.

Contradiction definition:

- Let U = union of all component sets in S_near.
- A component v in U is contradictory if there exist two circuits C_a and C_b in S_near such that:
  - v is necessary for C_a, and
  - either v ∉ C_b, or v is not necessary for C_b.

CC = (# contradictory components) / |U|.

If U is empty, define CC = 0.

Verdicting:

- PASS if CC ≤ `CANONICAL.DIAGNOSTICS.CC_THRESHOLD_GOOD`.
- WARN if `CANONICAL.DIAGNOSTICS.CC_THRESHOLD_GOOD` < CC ≤ `CANONICAL.DIAGNOSTICS.CC_THRESHOLD_MODERATE`.
- FAIL if CC > `CANONICAL.DIAGNOSTICS.CC_THRESHOLD_MODERATE`.

## Audit certificate trigger

Emit an audit certificate (see `spec/04_IDENTIFIABILITY_DEFINITIONS.md`) if any of the following hold:

- RR_verdict is FAIL.
- SSS_verdict is FAIL.
- CC_verdict is FAIL.

The certificate must include:

- `reason_codes`: a list of one or more values from `CANONICAL.ENUMS.CERTIFICATE_REASON_CODES`, in deterministic order (RR, then SSS, then CC) corresponding to which verdicts are FAIL.
- The diagnostic values RR/SSS/CC and their verdicts.
- The set S_near used for RR/CC (possibly empty if no faithful circuits exist).
- The replicate circuits {C_r} used for SSS.
- The reference distribution ID and pairing algorithm ID used to construct D_ref.
