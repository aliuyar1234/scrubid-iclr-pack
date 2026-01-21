# progress.md

This file tracks phase status. It begins with all phases marked FAIL.

## Status legend

Use only the canonical verdict enums defined in `spec/00_CANONICAL.md`.

## Phase status

| Phase ID | Name | Status | Evidence run_id | Evidence artifact |
|---|---|---|---|---|
| PHASE_0_FOUNDATION | Foundation and scaffolding | VERDICT_PASS | run_20260118T200614Z_0000 | outputs/paper_ready_qwen_20260118/runs/run_20260118T200614Z_0000/run_record.json |
| PHASE_1_SYNTHETIC_SUITE | Synthetic ground-truth suite | VERDICT_PASS | run_20260118T200522Z_0004 | outputs/paper_ready_qwen_20260118/reports/report_20260118T201958Z_0000/table_T1.csv |
| PHASE_2_INTERVENTIONS_AND_GENERATORS | Interventions and candidate generators | VERDICT_PASS | run_20260118T200522Z_0004 | outputs/paper_ready_qwen_20260118/runs/run_20260118T200522Z_0004/best_circuit.json |
| PHASE_3_DIAGNOSTICS_AND_AUDIT | RR/SSS/CC diagnostics and audit pipeline | VERDICT_PASS | run_20260118T200522Z_0004 | outputs/paper_ready_qwen_20260118/runs/run_20260118T200522Z_0004/certificate.json |
| PHASE_4_EXPERIMENTS_AND_ANALYSIS | Full experiments and analysis | VERDICT_PASS | run_20260118T200614Z_0000 | outputs/paper_ready_qwen_20260118/reports/report_20260118T201958Z_0000/table_T2.csv |
| PHASE_5_PAPER_CAMERA_READY | Paper assembly and camera-ready checks | VERDICT_PASS | run_20260118T200614Z_0000 | paper/00_PAPER_SNAPSHOT.md |
| PHASE_6_RELEASE_AND_REPRO | Release, packaging, reproducibility | VERDICT_PASS | run_20260118T200614Z_0000 | MANIFEST.sha256 |

## Update protocol

A phase may be updated from VERDICT_FAIL to VERDICT_PASS only when:

1. The phase acceptance criteria in its `tasks/PHASE_*` file are met.
2. A concrete `run_id` exists under `PATH_RUNS_ROOT`.
3. The referenced artifact path exists and is immutable.

A phase may be marked VERDICT_HARD_FAIL if the implementation violates any invariant in `SPEC.md`.
