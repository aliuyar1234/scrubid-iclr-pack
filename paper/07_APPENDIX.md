# Appendix

## Diagnostic formulas

RR/SSS/CC are defined in `spec/06_DIAGNOSTICS_RR_SSS_CC.md`.

## Implementation notes

The implementation contract in `spec/22_IMPLEMENTATION_CONTRACT.md` defines:

- component ID formats
- hook ordering
- behavior metrics
- tolerance computation
- seed handling

## Certificate format

See `spec/04_IDENTIFIABILITY_DEFINITIONS.md`.

## Example certificate artifact (synthetic)

A representative audit certificate from the synthetic suite (redundant regime; non-identifiability case) is:

- Run ID: `run_20260120T115715Z_0009`
- Suite/setting: `SUITE_SYNTH_V1` / `setting_XOR_2_1` (planted redundancy factor 2)
- Intervention family / generator: `I_ACTPATCH` / `G_ATTR_PATCH`
- Certificate: `outputs/paper_ready_gpt2_20260120_v6c/runs/run_20260120T115715Z_0009/certificate.json`
- Best circuit: `outputs/paper_ready_gpt2_20260120_v6c/runs/run_20260120T115715Z_0009/best_circuit.json`

## Example certificate artifact (real)

A representative audit certificate from the real suite (induction OOD RR+SSS failure) is:

- Run ID: `run_20260120T160424Z_0000`
- Suite/experiment: `SUITE_REAL_INDUCTION_V1` / `EXP_REAL_INDUCTION_OOD_V1`
- Intervention family / generator: `I_ACTPATCH` / `G_ATTR_PATCH`
- Certificate: `outputs/paper_ready_gpt2_20260120_v6c/runs/run_20260120T160424Z_0000/certificate.json`
- Best circuit: `outputs/paper_ready_gpt2_20260120_v6c/runs/run_20260120T160424Z_0000/best_circuit.json`
