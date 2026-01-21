# 04_METHOD.md

This section summarizes ScrubID’s method. The normative definitions live in `spec/02_FORMAL_DEFINITIONS.md`, `spec/03_INTERVENTION_FAMILIES.md`, and `spec/06_DIAGNOSTICS_RR_SSS_CC.md`.

## Setup

Given:

- a model f,
- a suite-defined dataset D and behavior metric m(x; f),
- a component granularity defining a component set V,
- an intervention family I with reference distribution D_ref,

ScrubID searches for ε-faithful circuits C ⊆ V with small mechanistic description length MDL(C).

## Candidate generation and selection

ScrubID uses one or more candidate generators to propose circuits. Each candidate is scored by:

- faithfulness loss Δ(C), and
- complexity proxy MDL(C).

ScrubID selects the minimum-MDL ε-faithful circuit C* and defines a near-optimal set S_near of circuits close to C* in MDL.

## Diagnostics

ScrubID reports three diagnostics:

- RR (Redundancy Ratio): the maximum Jaccard distance between any two circuits in S_near.
- SSS (Scrubbed Solution Stability): the mean pairwise overlap between circuits recovered across replicate discovery runs.
- CC (Contradiction Score): the fraction of components whose necessity status is inconsistent across S_near.

When any diagnostic indicates non-identifiability, ScrubID emits a certificate containing the near-optimal circuits, replicate circuits, and the full reference distribution provenance.
