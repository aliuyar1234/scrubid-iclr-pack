# 09_CANDIDATE_GENERATORS.md

## SDR note

All generator IDs are defined in `spec/00_CANONICAL.md`.

## Goal

Candidate generators propose circuits `C ⊆ V` that are likely to be `ε`-faithful for a given task.

ScrubID treats generators as adapters. The audit logic (RR/SSS/CC) must be independent of generator choice.

## Canonical generator set

- `G_ACDC` (Automated circuit discovery) `[@conmy2023acdc]`
- `G_ATTR_PATCH` (Attribution patching) `[@syed2024attributionpatching]`
- `G_ATPSTAR` (AtP*) `[@kramar2024atpstar]`
- `G_SPARSE_FEATURE` (Sparse feature circuits) `[@marks2025sparsefeaturecircuits]`
- `G_MANUAL_SEED` (Seed circuits from literature) `[@wang2023ioi; @hanna2023greaterthan; @olsson2022inductionheads]`
- `G_RANDOM_K` (Stratified random size-matched circuits baseline)

## v1.0.3 default enabled generators

The v1.0.3 pack is designed to be implementable from scratch without requiring re-implementation of multiple external circuit-discovery stacks.

The default configuration (`configs/generators.yaml`) enables only:

- `G_MANUAL_SEED`
- `G_ATTR_PATCH`
- `G_RANDOM_K`

All other canonical generators are optional extensions. Implementations may add them, but they are not required for the deterministic synthetic validation and the baseline real-model case studies.

## Required interface

Each generator must implement a deterministic function:

- `generate_candidates(model, task_spec, intervention_family_id, budget_spec) -> CandidateSet`

where `CandidateSet` contains:

- `ranked_components`: list of `(component_id, score)`
- `ranked_edges` (optional): list of `(edge_id, score)`
- `candidate_circuits`: list of circuits with:
  - `components`
  - `edges` (optional)
  - `score`
  - `metadata` (generator-specific)

All IDs must be canonical, and all results must be logged.

## Determinism requirements

- In deterministic mode, generators must use a fixed pseudo-random stream seeded from `SEEDS.SEED_GLOBAL` and the experiment id.
- Tie-breaking rules must be deterministic and logged.
