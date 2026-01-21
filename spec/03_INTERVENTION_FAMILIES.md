# 03_INTERVENTION_FAMILIES.md

This file normatively defines ScrubID intervention families and the reference distribution D_ref that makes scrubbing well-defined.

**Single Definition Rule:** all IDs (suite IDs, intervention IDs, reference distribution IDs, pairing algorithm IDs) are defined only in `spec/00_CANONICAL.md`.

## What an intervention family is

An intervention family I specifies how to construct the scrubbed model f^(C, I) given:

- A circuit C ⊆ V.
- A target prompt x.
- A paired reference prompt x_ref sampled from D_ref using a deterministic pairing rule.

Formally, an intervention family fixes:

- `hook_backend` (default: `CANONICAL.HOOKS.BACKEND_DEFAULT`).
- A hookpoint set, expressed using templates in `CANONICAL.HOOKS`.
- A patch operator that describes how reference activations replace (or constrain) target activations.
- A reference distribution ID `reference_distribution_id` and a deterministic pairing rule ID `reference_assignment_id`.

## Reference distributions D_ref

ScrubID requires a reference distribution D_ref for every suite.

The default mapping is defined in:

- `CANONICAL.REFERENCE.REFDIST_DEFAULT_BY_SUITE`

Intuition:

- Real suites use a *corrupted* prompt dataset aligned index-by-index with the clean prompt dataset.
- The synthetic suite uses an unpaired reference distribution created by shuffling examples (a deterministic derangement).

## Deterministic pairing: x ↦ x_ref

ScrubID supports two canonical deterministic pairing algorithms:

### REFASSIGN_INDEX_ALIGNED_V1

- Assumes the dataset provides a clean split D_clean and a corrupted split D_corrupted with the same length and aligned indices.
- Pairing rule: if x is the i-th element of D_clean, then x_ref is the i-th element of D_corrupted.

### REFASSIGN_DERANGEMENT_SHUFFLE_V1

- Assumes D_ref is an ordered dataset of length N.
- Computes a deterministic permutation π over indices 0 through N-1 using the run seed and rejects permutations with fixed points.
- Pairing rule: if x is the i-th element of D, then x_ref is the π(i)-th element of D_ref.

The seed used for pairing must be recorded in `run_record.json`.

## Intervention family IDs

Intervention family IDs are enumerated by `CANONICAL.IDS.INTERVENTION_FAMILY_IDS`.

The required semantics per ID are:

### I_ACTPATCH (activation patching)

Goal: preserve activations for components inside C and patch activations for components outside C.

- Hook backend: `CANONICAL.HOOKS.BACKEND_DEFAULT`.
- Head patch hookpoint template: `CANONICAL.HOOKS.TL_HOOK_HEAD_RESULT_TEMPLATE`.
- MLP patch hookpoint template: `CANONICAL.HOOKS.TL_HOOK_MLP_OUT_TEMPLATE`.

Patch operator (component-level):

- For an attention head component (layer l, head h), replace the slice corresponding to head h at the head hookpoint (pre-W_O, i.e. TransformerLens `hook_z`) with the reference activation.
- For an MLP block component (layer l), replace the MLP-out activation with the reference activation.

The exact tensor slicing and shapes are specified in `spec/22_IMPLEMENTATION_CONTRACT.md`.

### I_PATHPATCH (path patching)

Goal: localize behavior to a set of directed information paths.

**v1.0.3 scope note (Path B):** this pack does not implement true per-edge contribution blocking. `I_PATHPATCH` is provided as a lightweight approximation and is excluded from main paper claims.

Operational definition for v1.0.3:

- Uses the same reference pairing as activation patching.
- If a circuit provides no edges, `I_PATHPATCH` reduces exactly to `I_ACTPATCH`.
- If edges are provided, define the active component set as: all components that can reach a sink component in the maximum layer present in the circuit (reachability in the directed edge set, restricted to components listed in the circuit). Then apply `I_ACTPATCH` to that active component set.

### I_CAUSAL_SCRUB (causal scrubbing)

Goal: test a causal hypothesis about a high-level decomposition by resampling parts of the computation.

**v1.0.3 scope note (Path B):** suite-specific causal-variable resampling is out of scope in this pack. `I_CAUSAL_SCRUB` is implemented as the same operator as `I_ACTPATCH` (same hookpoints, same patching semantics, same reference distribution selection), and is excluded from main paper claims.

## Out of scope: feature-level patching

This pack does not include a feature dictionary (for example an SAE) and therefore does not define a feature-level patching intervention family.

Fail-closed rule:

- If `component_granularity` is set to `feature` or `neuron`, or if an implementation is asked to run a feature-level intervention, the implementation must raise a deterministic error that is recorded in `logs.jsonl` and `run_record.json`.

## Constraints

- All intervention implementations must be deterministic under `CANONICAL.DETERMINISM` requirements.
- The reference dataset(s), pairing algorithm, and any corruption/resampling parameters must be recorded in `run_record.json`.
