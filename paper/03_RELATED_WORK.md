# 03_RELATED_WORK.md

We position ScrubID at the intersection of mechanistic interpretability practice and causal identifiability.

## Mechanistic interpretability and circuit discovery

Activation patching and its variants are widely used to localize behavior to internal components, but different intervention choices can yield different explanations. Recent methodological work has clarified best practices and pitfalls for activation patching and related procedures [@heimersheim2024activationpatching; @zhang2023bestpractices; @makelov2024subspaceillusion].

Automated circuit discovery methods such as ACDC and feature-based circuit extraction have improved scalability, yet still leave open the question of when an explanation is uniquely supported by the intervention family [@conmy2023acdc; @marks2025sparsefeaturecircuits].

Path patching offers a more fine-grained localization primitive that can be viewed as edge-aware scrubbing [@goldowsky_dill2023pathpatching].

## Causal identifiability and causal abstraction

From a causal perspective, a mechanistic explanation is only meaningful if it is identifiable under the interventions available. In causal discovery, interventional Markov equivalence classes formalize when multiple causal graphs are indistinguishable [@hauser2012imec; @eberhardt2012experiments].

In mechanistic interpretability, causal abstraction provides a formal foundation for reasoning about interventions and explanations, and motivates certifying when multiple explanations are equally compatible with the same intervention family [@geiger2025foundation; @geiger2021causalabstractions].

The identifiability framing for mechanistic interpretability has recently been made explicit, including impossibility results under restricted interventions [@meloux2025identifiability].

## Benchmarks and robustness across scale

Causal interpretability benchmarks aim to evaluate intervention methods under controlled task settings [@arora2024causalgym]. Empirically, some circuit analyses appear consistent across training and scale, but this can depend on methodology and task choice [@tigges2024consistent; @lieberum2023scale].

## ScrubID’s contribution

ScrubID introduces a compact, implementation-ready protocol to:

- quantify worst-case redundancy among near-optimal circuits (RR),
- quantify solution stability across replicate discovery runs (SSS), and
- quantify contradictory necessity claims (CC),

and to emit an audit certificate when explanations are not reliably supported (e.g., non-identifiability or discovery instability under the chosen protocol).
