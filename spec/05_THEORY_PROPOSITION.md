# 05_THEORY_PROPOSITION.md

## SDR note

All constants referenced here are defined in `spec/00_CANONICAL.md`.

## Proposition (theory-lite)

This proposition is intended to be checkable on the synthetic suite, and to clarify what assumptions would be needed for identifiability.

### Definitions

Let `V` be the component set and `𝓘` an intervention family.

Call `𝓘` **separating** for `V` if for every component `v ∈ V` there exists an intervention in `𝓘` that changes `v` while leaving the activation values of `V\{v}` unchanged.

Call the task metric `m(x; f)` **component-separable** if there exists a subset `C* ⊆ V` such that `m(x; f)` depends only on activations in `C*` and is invariant to interventions on `V\C*`.

### Proposition

Assume:

1. (Separating interventions) `𝓘` is separating for `V`.
2. (Component-separable metric) there exists a minimal circuit `C*` such that for all `x ∈ D` and for all interventions on `V\C*`, the metric is invariant.
3. (Margin) For any circuit `C` that omits at least one element of `C*`, the faithfulness loss satisfies `Δ(C) > ε`.

Then `C*` is identifiable under `𝓘` at tolerance `ε`.

### Proof sketch expectations

- By separating interventions, any omitted component in `C*` can be independently perturbed.
- By component-separability and the margin assumption, omitting any `v ∈ C*` yields a detectable increase in `Δ(C)` beyond `ε`.
- Any `ε`-faithful circuit must therefore include all of `C*`.
- Minimality of `C*` and the invariance to interventions outside `C*` implies any additional components outside `C*` are unnecessary, so all minimal `ε`-faithful circuits are `𝓘`-equivalent to `C*`.

### How this is used

- The synthetic suite explicitly constructs instances where assumption (2) holds and where it fails by injecting redundant parallel pathways.
- ScrubID diagnostics are designed to detect departures from the proposition’s conditions.
