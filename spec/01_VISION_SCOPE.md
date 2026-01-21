# 01_VISION_SCOPE.md

## SDR note

All literals and thresholds referenced here are defined in `spec/00_CANONICAL.md`.

## Vision

Mechanistic interpretability often claims a specific circuit explains a behavior. ScrubID turns this into an auditable statement:

- given a component universe `V`
- under an intervention family `I`
- and a tolerance `ε`

how many distinct circuits are `ε`-faithful, how stable is the explanation across scrub variants, and how complex must an explanation be?

ScrubID is explicitly designed to address the kind of non-identifiability documented in `[@meloux2025identifiability]` by providing a practical auditing workflow for transformer models.

## Scope

In scope:

- Formal definitions for circuits, interventions, scrubbed models, `ε`-faithfulness, and identifiability under an intervention family.
- Diagnostics RR/SSS/CC and an audit certificate format (including non-identifiability and discovery-instability cases).
- A deterministic synthetic benchmark suite with ground-truth equivalence class size.
- Real-model case studies on IOI, greater-than, and induction-style prompts.

Out of scope:

- Inventing a new circuit discovery algorithm that supersedes existing methods.
- Claiming a single true mechanistic explanation for all distributions.

## Deliverables

- A reproducible pipeline that produces immutable run artifacts and logs.
- Tables and plots that support the paper claims, and are regenerable from logs.
- A reviewer-resistant narrative that distinguishes faithful behavior replication from identifiability.
