# Experiments

ScrubID evaluation follows two tracks: a deterministic synthetic benchmark that stress-tests non-identifiability, and real-model case studies that probe stability across intervention families and candidate generators.

## Reproducible artifacts for this draft (2026-01-20)

All numeric results referenced in this draft are generated from immutable run directories under:

- `outputs/paper_ready_gpt2_20260120_v6c/runs`

and aggregated into a paper-artifacts report directory under:

- `outputs/paper_ready_gpt2_20260120_v6c/reports/report_20260120T171815Z_0000`

Artifacts:

- Table T1 (synthetic redundancy sweep): `outputs/paper_ready_gpt2_20260120_v6c/reports/report_20260120T171815Z_0000/table_T1.csv`
- Table T2 (real case study summary; ID split): `outputs/paper_ready_gpt2_20260120_v6c/reports/report_20260120T171815Z_0000/table_T2.csv`
- Table T3 (sensitivity deltas): `outputs/paper_ready_gpt2_20260120_v6c/reports/report_20260120T171815Z_0000/table_T3.csv`
- Table T4 (real case study summary; OOD split): `outputs/paper_ready_gpt2_20260120_v6c/reports/report_20260120T171815Z_0000/table_T4.csv`
- Figure (synthetic trends): `outputs/paper_ready_gpt2_20260120_v6c/reports/report_20260120T171815Z_0000/fig_synth.png`

## Synthetic benchmark

Use the deterministic synthetic suite `SUITE_SYNTH_V1` with planted redundancy factors and ground-truth equivalence class sizes.

Evaluate:

- RR monotonicity with planted redundancy_factor
- CC growth with planted redundancy_factor (contradictory necessity across interchangeable subcircuits)
- SSS degradation as redundancy_factor grows (multiple near-optimal solutions reduce replicate stability)

In Table T1, planted redundancy factor 1 yields RR=0 and CC=0 with no certificates. For redundancy factors ≥2, RR rises to ≈0.67, CC rises to ≈0.67–0.82, and certificate emission rate is 1.0 across all three synthetic templates (XOR, COMPARE, INDUCTION).

## Real-model case studies

Use GPT-2 (`gpt2`) and three case studies:

- IOI `[@wang2023ioi]`
- greater-than year-span comparison (yes/no) `[@hanna2023greaterthan]`
- induction-style prompts `[@olsson2022inductionheads]`

For each suite, we evaluate under activation patching (`I_ACTPATCH`) at `head_mlp` granularity and compare three candidate generators (`G_ATTR_PATCH`, `G_MANUAL_SEED`, `G_RANDOM_K`) across N=5 suite seeds. ID results are reported in Table T2 and OOD results in Table T4; Table T3 reports sensitivity deltas against a per-suite baseline configuration.

## Ablations

Report:

- seed sensitivity
- generator sensitivity
- intervention family sensitivity
- granularity sensitivity (head vs MLP)
- tolerance sensitivity (vary `EPSILON_REL_FRAC` within canonical ablation set)

Table T3 summarizes per-suite deltas against a fixed baseline configuration (lexicographically smallest intervention family and generator per suite). For IOI and induction, variants that select the full circuit have larger size but lower faithfulness loss relative to the baseline near-full circuit, while RR/SSS/CC deltas remain zero.
