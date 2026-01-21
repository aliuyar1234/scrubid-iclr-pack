# 18_RESULTS_TABLE_SKELETONS.md

This file defines the canonical results tables to include in the paper.

All metric symbols and thresholds are defined in `spec/00_CANONICAL.md`.

## Table T1: Synthetic suite redundancy sweep

Rows: synthetic settings ordered by planted redundancy.

Columns:

- `setting_id`
- `planted_redundancy_factor`
- `RR_mean`, `RR_ci`
- `SSS_mean`, `SSS_ci`
- `CC_mean`, `CC_ci`
- `non_identifiability_rate`

## Table T2: Real case study summary

Rows: each real suite.

Columns:

- `suite_id`
- `intervention_family_id`
- `candidate_generator_id`
- `N`
- `best_circuit_size_mean`, `best_circuit_size_ci`
- `faithfulness_delta_mean`, `faithfulness_delta_ci`
- `RR_mean`, `RR_ci`
- `SSS_mean`, `SSS_ci`
- `CC_mean`, `CC_ci`
- `overall_verdict`

## Table T3: Ablation and sensitivity

Rows: ablations (intervention family, granularity, candidate generator).

Columns:

- `variant_id`
- `Δ_size`
- `Δ_faithfulness`
- `Δ_RR`
- `Δ_SSS`
- `Δ_CC`

## Table T4: Real case study summary (OOD)

Rows: each real suite, OOD split.

Columns:

- `suite_id`
- `experiment_id`
- `intervention_family_id`
- `candidate_generator_id`
- `N`
- `best_circuit_size_mean`, `best_circuit_size_ci`
- `faithfulness_delta_mean`, `faithfulness_delta_ci`
- `RR_mean`, `RR_ci`
- `SSS_mean`, `SSS_ci`
- `CC_mean`, `CC_ci`
- `overall_verdict`
