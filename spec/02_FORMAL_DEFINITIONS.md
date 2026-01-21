# 02_FORMAL_DEFINITIONS.md

This file defines ScrubID’s core formal objects.

**Single Definition Rule:** all constants, IDs, thresholds, symbols, canonical CLI strings, and canonical paths are defined only in `spec/00_CANONICAL.md`. This file only references them by name.

## Model, dataset, and behavior metric

- Let **f** be a fixed autoregressive Transformer language model.
- A suite defines a dataset **D**: an ordered sequence of prompts x_0 through x_{N-1}, with deterministic order.
- A suite also defines a scalar behavior metric **m(x; f) → ℝ**.
  - Examples: a logit-difference at a specific token position, or a probability assigned to a target token.
  - The metric definition is suite-specific and is normatively specified for each real case study in `spec/08_REAL_MODEL_CASE_STUDIES.md` and `spec/22_IMPLEMENTATION_CONTRACT.md`.

## Component set V

Let **g** be the component granularity used in the run.

- The allowed granularities are defined in `CANONICAL.COMPONENT_GRANULARITY`.
- The default is `CANONICAL.COMPONENT_GRANULARITY.GRANULARITY_DEFAULT`.

For a given model **f** and granularity **g**, define the component set **V** as:

- If g = `head`: V contains one component per attention head (layer, head_index).
- If g = `mlp`: V contains one component per MLP block (layer).
- If g = `head_mlp`: V is the union of attention-head components and MLP-block components.
- If g = `node`: V is the set of synthetic graph nodes for `SUITE_SYNTH_V1` as specified in `spec/07_SYNTHETIC_SUITE_GENERATOR.md`. Implementations must fail closed if `node` is selected for any real suite.
- If g = `neuron` or `feature`: V is defined by the implementation contract (indices must be stable and deterministic). For v1.0.3, `neuron` and `feature` are out of scope and implementations must fail closed if selected.

Each component v ∈ V must have a deterministic string ID (see `spec/22_IMPLEMENTATION_CONTRACT.md`).

## Circuit C

A **circuit** is a subset C ⊆ V.

ScrubID may also use an edge set E(C) when running edge-aware intervention families. When edges are used, E(C) must be a deterministic function of C and the chosen candidate generator.

## Intervention family I and the reference distribution D_ref

An **intervention family** is identified by `intervention_family_id ∈ CANONICAL.IDS.INTERVENTION_FAMILY_IDS`.

Each intervention family instantiation fixes:

- A hook backend (default: `CANONICAL.HOOKS.BACKEND_DEFAULT`).
- A set of hookpoints and a patch operator (defined in `spec/03_INTERVENTION_FAMILIES.md` and `spec/22_IMPLEMENTATION_CONTRACT.md`).
- A reference distribution **D_ref** used to supply “reference” activations during patching.

For any run, D_ref is selected via:

- `reference_distribution_id = CANONICAL.REFERENCE.REFDIST_DEFAULT_BY_SUITE[suite_id]`.
- `reference_assignment_id = CANONICAL.REFERENCE.REF_ASSIGNMENT_DEFAULT_BY_SUITE[suite_id]`.

The deterministic pairing rule x ↦ x_ref is defined in `spec/03_INTERVENTION_FAMILIES.md`.

## Scrubbed model f^(C, I)

Given a circuit C and an intervention family I, define the **scrubbed model** f^(C, I) as a model that:

1. Runs a reference forward pass on x_ref to capture activations at the intervention hookpoints.
2. Runs a target forward pass on x while applying I’s patch operator to every component v ∉ C at the designated hookpoints, using the captured reference activations.

The output logits (and any downstream metric m) of the patched target pass define f^(C, I)(x).

## Faithfulness loss Δ(C) and tolerance ε

Let D_eval be the deterministic evaluation subset of D (defined by the suite and budgets).

Define the faithfulness loss as:

- Δ(C) = mean_{x ∈ D_eval} | m(x; f) − m(x; f^(C, I)) |.

Define the tolerance ε as:

- S0 = mean_{x ∈ D_eval} |m(x; f)|.
- ε = max(CANONICAL.DIAGNOSTICS.EPSILON_ABS_MIN, CANONICAL.DIAGNOSTICS.EPSILON_REL_FRAC · S0).

A circuit C is **ε-faithful** if Δ(C) ≤ ε.

## Complexity and mechanistic description length MDL(C)

ScrubID selects among ε-faithful circuits by minimizing a mechanistic description length proxy.

For component-only circuits, MDL is defined as:

- MDL(C) = CANONICAL.DIAGNOSTICS.MDL_WEIGHT_NODE · |C|.

If a run uses explicit edges or sparse-feature components, MDL adds additional terms using `CANONICAL.DIAGNOSTICS.MDL_WEIGHT_EDGE` and `CANONICAL.DIAGNOSTICS.MDL_WEIGHT_FEATURE`.

## Near-optimal circuit set

Given a candidate set S of circuits, let:

- F = {C ∈ S : Δ(C) ≤ ε} be the faithful set.
- C* = argmin_{C ∈ F} MDL(C) be the minimum-MDL faithful circuit.

Define the near-optimal set:

- S_near = {C ∈ F : MDL(C) ≤ (1 + CANONICAL.DIAGNOSTICS.RR_NEAR_OPTIMAL_MDL_REL_FRAC) · MDL(C*)}.

If |S_near| exceeds `CANONICAL.DIAGNOSTICS.RR_NUM_CIRCUITS_SET`, keep the lowest-MDL circuits under the deterministic tie-break described in `spec/06_DIAGNOSTICS_RR_SSS_CC.md`.

The ScrubID diagnostics RR, SSS, and CC are defined in `spec/06_DIAGNOSTICS_RR_SSS_CC.md`.
