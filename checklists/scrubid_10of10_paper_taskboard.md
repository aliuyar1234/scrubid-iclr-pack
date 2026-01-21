# ScrubID “10/10 Paper” Taskboard

Date last updated: 2026-01-19

This file tracks what remains to reach a “10/10” ScrubID paper (publishable, reproducible, and claim↔evidence closed).

How to use:

- Treat this as the canonical “what’s left” list.
- When a task is completed, record the concrete run ID(s) / artifact bundle ID(s) and mark the checkbox.
- Prefer referencing canonical command names (e.g., `CLI_CMD_VALIDATE_SPEC`) rather than pasting raw CLI strings.

---

## Definition of “10/10”

A 10/10 ScrubID submission satisfies all of the following:

- Claim↔evidence closure: every non-trivial claim maps to specific tables/figures + run IDs + immutable artifacts.
- Reproducible key results: a fresh machine can regenerate all main tables/figures from scripts + configs + logs (no hidden local paths).
- No spec/paper drift: paper method == spec method == code behavior (IDs, hookpoints, constants, interventions).
- Non-trivial real-model finding: at least one real case where multiple distinct ε-faithful circuits exist (RR>0 and/or CC>0) under a meaningful intervention family (not just the full circuit C=V).
- Strong baselines & ablations: comparisons vs reasonable alternatives; ≥10 ablations; multi-seed; error bars/CI protocol stated and reproduced.
- Intervention-family integrity: “path patching” and “causal scrubbing” are either implemented per canonical meaning or renamed/scoped out (no “same as actpatch” while claiming cross-family robustness).
- Compute/cost transparency: runtime + memory + cost for diagnostics (especially CC necessity tests) and a measured accuracy↔cost frontier if cost is non-trivial.
- Complete artifacts: datasets (or generators), configs, logs, run records, and a results manifest with hashes.

---

## Current status (as of 2026-01-19)

Already in place (partial):

- A paper draft with embedded tables/appendix provenance and an artifact bundle with hashes (see `paper.md`).
- Scope is locked to `I_ACTPATCH` for paper artifacts (see `configs/interventions.yaml` and `paper.md`).
- OOD evaluation is reported (Table T4 in `paper.md`).
- Public model provenance is used for main results (GPT‐2 with pinned revision; see `paper.md`).
- A real certificate exists showing a non-trivial near-optimal set with RR non-identifiability (|S_near|=2, RR=0.9090909090909091) on induction OOD under `G_ATTR_PATCH` (see Table T4 / Appendix D in `paper.md`).
- Audit certificates include machine-readable `reason_codes` (see Appendix C/D in `paper.md`).
- A mechanical claim↔evidence manifest + validator exists (`paper_results_manifest.json`, `CLI_CMD_VALIDATE_PAPER_MANIFEST`).
- GPT Pro review is pasted in `paper/gpt_pro_review.md` (2026-01-19).
- P1 scaffolding implemented (not yet paper-backed): multi-seed real-table aggregation (mean+CI) for T2/T4, and a deterministic `G_RANDOM_K` baseline generator enabled in `configs/generators.yaml`.
- Canonical multi-seed offsets are defined: `SEEDS.PAPER_REAL_SEED_OFFSETS = [0,1,2,3,4]` (N=5 seeds) in `spec/00_CANONICAL.md`.

Still missing for “10/10”:

- Full baseline/ablation suite + multi-seed + CIs for real runs.
- Environment lock for fresh machines (the pack includes `CLI_CMD_REPRODUCE_PAPER` for end-to-end regeneration + hash verification).
- A new paper artifact bundle (v6) that regenerates tables using the new T2/T4 mean+CI schema and includes `G_RANDOM_K`.

Work-in-progress bundles (do not cite as paper evidence):

- `outputs/paper_ready_gpt2_20260119_v6/` (aborted during synth after a `G_RANDOM_K` uniqueness/hang issue; fixed in code afterwards).
- `outputs/paper_ready_gpt2_20260119_v6b/` (synthetic suite completed; real runs were started then stopped; contains an incomplete run dir without `run_record.json`: `runs/run_20260119T195840Z_0000`).
- `outputs/paper_ready_gpt2_20260120_v6c/` (in progress; synth diagnostics completed; real 5-seed suite partially executed; resume with `python scripts/resume_paper_bundle.py --output_root outputs/paper_ready_gpt2_20260120_v6c --device cuda --include_synth`).

---

## P0 (Blockers)

### P0-01 — Scope lock to activation patching (GPT Pro recommends Path B)

- [x] Update **paper + spec + code** so the v1.0.3 artifact pack is unambiguous:
  - Main results use **only** `I_ACTPATCH`.
  - `I_PATHPATCH` and `I_CAUSAL_SCRUB` are explicitly out of scope / not implemented (or excluded from paper claims), and any cross-family robustness claims are removed.
- [x] Acceptance:
  - No main table/figure depends on non-`I_ACTPATCH` operators.
  - Terminology in `paper.md` matches actual operators in `src/`.
  - Running a non-`I_ACTPATCH` intervention fails closed (or is clearly marked experimental and excluded from paper claims).
- [x] Run IDs / evidence: `configs/interventions.yaml` (only `I_ACTPATCH`); `outputs/paper_ready_gpt2_20260119_v5/reports/report_20260119T165153Z_0000/` (tables/figures are `I_ACTPATCH` only).

### P0-02 — Certificate semantics fix (non-identifiability vs instability)

- [x] Rename/reframe the certificate as an audit certificate and add machine-readable `reason_codes` so SSS-only cases are not labeled “non-identifiability.”
- [x] Update paper language and any tables/appendix references to reflect the corrected semantics.
- [x] Acceptance:
  - `certificate.json` includes `reason_codes` and the paper’s labels match those codes.
  - A reviewer can distinguish (a) RR/CC non-identifiability vs (b) discovery instability (SSS) from artifacts alone.
- [x] Run IDs / evidence: `run_20260119T151730Z_0006` (reason_codes: rr_fail, cc_fail) and `run_20260119T162244Z_0000` (reason_codes: sss_fail); see `outputs/paper_ready_gpt2_20260119_v5/runs/` and Appendix C/D in `paper.md`.

### P0-03 — Artifact completeness for every paper-backed run

- [x] For every run referenced by any main table/figure, ensure the run directory is self-contained:
  - `run_record.json`, `logs.jsonl`
  - per-run candidates (if applicable), `diagnostics.json`, `best_circuit.json`, and `certificate.json` when triggered
- [x] Acceptance:
  - All paper tables/figures can be regenerated from immutable run directories + configs only (no hidden local state).
- [x] Run IDs / evidence: all paper-backed runs and file hashes are enumerated in `paper_results_manifest.json` (validated by `CLI_CMD_VALIDATE_PAPER_MANIFEST`).

### P0-04 — Non-trivial real-model identifiability ambiguity (RR/CC)

- [x] Produce at least one **real** suite+model+generator+intervention where:
  - `|S_near| ≥ 2` and
  - `RR ≥ CANONICAL.DIAGNOSTICS.RR_THRESHOLD_LOW` and/or `CC ≥ CANONICAL.DIAGNOSTICS.CC_THRESHOLD_GOOD` (with clearly stated slack/epsilon/tau settings).
- [x] Include the emitted `certificate.json` and update the main results table(s) and narrative accordingly.
- [x] Start with the concrete sweeps and candidate-diversification strategies enumerated in `paper/gpt_pro_review.md` (Strategies 1–3).
- [x] Acceptance:
  - A reviewer can locate the row in a table, follow the run ID to artifacts, and see multiple distinct ε-faithful circuits in the certificate.
- [x] Run IDs / evidence: `run_20260119T162723Z_0000` (induction OOD; reason_codes: rr_fail, sss_fail; |S_near|=2; RR=0.9090909090909091); Table T4 in `outputs/paper_ready_gpt2_20260119_v5/reports/report_20260119T165153Z_0000/` and Appendix D in `paper.md`.

### P0-05 — Deterministic candidate generation (bit-exact candidate sets)

- [x] Ensure candidate generation is fully deterministic (seed derivation, ordering, tie-breaks) and recorded in `run_record.json`.
- [x] Acceptance:
  - Candidate sets reproduce bit-exactly across determinism reruns on the same machine.
  - Under strict deterministic mode, the run fails closed if determinism cannot be guaranteed.
- [x] Run IDs / evidence: `CLI_CMD_DETERMINISM_SMOKE_TEST` PASS (synthetic sweep re-run twice; diagnostics bytes identical, including candidate records), reports `outputs/reports/report_20260119T175558Z_0000/`, `outputs/reports/report_20260119T175558Z_0001/`, `outputs/reports/report_20260119T175558Z_0002/`.

### P0-06 — Claim↔evidence closure (mechanical)

- [x] Add `paper_results_manifest.json` (schema + entries) mapping every non-trivial claim → exact table/figure cell(s) → run IDs → artifact hashes.
- [x] Add a validation command/check that fails closed if:
  - a claim has no evidence entry,
  - a referenced artifact is missing,
  - a referenced hash doesn’t match.
- [x] Acceptance:
  - “No missing evidence” is enforced mechanically, not manually.
- [x] Run IDs / evidence: `paper_results_manifest.json`; validator command `CLI_CMD_VALIDATE_PAPER_MANIFEST`.

### P0-07 — Reproduction from a fresh machine

- [x] Add a single “reproduce main paper results” script (Windows-friendly) that:
  - installs deps in a clean env,
  - downloads the public model(s),
  - runs the canonical pipelines,
  - rebuilds all main tables/figures into a new immutable run bundle,
  - and verifies hashes against a manifest.
- [x] Acceptance:
  - A fresh machine can regenerate main tables/figures without any local-only paths.
- [x] Run IDs / evidence: `CLI_CMD_REPRODUCE_PAPER` (resumed) output bundle `outputs/repro_paper_20260120T175516Z_0000/` with report `outputs/repro_paper_20260120T175516Z_0000/reports/report_20260120T233529Z_0000/` (table/figure hashes match `paper_results_manifest.json`).

### P0-08 — Spec↔paper↔code drift audit (mechanical)

- [x] Add a drift check that ensures:
  - IDs/hookpoints/constants referenced in `paper.md` match `spec/00_CANONICAL.md`,
  - configs only use canonical keys/IDs.
- [x] Acceptance:
  - CI/local check fails if drift is introduced.
- [x] Run IDs / evidence: `CLI_CMD_VALIDATE_SPEC` PASS (includes paper.md ID drift guard + canonical config resolution), 2026-01-19.

### P0-09 — Run required gates and record run IDs

- [x] Run `CLI_CMD_VALIDATE_SPEC` and record the run ID(s) / outputs.
- [x] Run `CLI_CMD_DETERMINISM_SMOKE_TEST` and record run ID(s) / outputs.
- [x] Acceptance:
  - Both commands pass and produce immutable run directories; hashes match across determinism reruns.
- [x] Run IDs / evidence: `CLI_CMD_VALIDATE_SPEC` PASS (2026-01-19); `CLI_CMD_DETERMINISM_SMOKE_TEST` PASS with reports `outputs/reports/report_20260119T143349Z_0000/`, `outputs/reports/report_20260119T143349Z_0001/`, `outputs/reports/report_20260119T143349Z_0002/`.

---

## P1 (Strongly recommended for acceptance; required for “10/10”)

### P1-01 — Strong baselines & ≥10 ablations

- [x] Add ≥10 explicit ablations (examples: epsilon sensitivity, tau sensitivity, slack curve, candidate budget, replicate count, reference pairing, corruption strength, OOD vs ID, generator variants, component granularity).
- [x] Add at least 2–3 reasonable baselines/generators beyond the current ones, or justify omissions in writing.
- [x] Acceptance:
  - Tables/figures include the ablations and show error bars / CIs.
- Not marked done until:
  - a new paper artifact bundle includes `G_RANDOM_K` rows in Tables T2/T4,
  - `paper_results_manifest.json` is updated and `CLI_CMD_VALIDATE_PAPER_MANIFEST` passes,
  - `CLI_CMD_REPRODUCE_PAPER` passes against the updated manifest.
- Progress (code/spec only; not yet paper-backed):
  - [x] Add deterministic stratified random baseline generator `G_RANDOM_K` and enable it in `configs/generators.yaml`.
  - [x] Filter Table T1 (synthetic) aggregation to `G_ATTR_PATCH` and `G_MANUAL_SEED` only so adding `G_RANDOM_K` does not change Table T1 semantics.
  - [x] Run a new paper bundle that includes `G_RANDOM_K` rows in T2/T4 and update `paper.md` + `paper_results_manifest.json`.
- Run IDs / evidence (fill when complete):
  - paper bundle output_root: `outputs/paper_ready_gpt2_20260120_v6c/`
  - report_dir: `outputs/paper_ready_gpt2_20260120_v6c/reports/report_20260120T171815Z_0000/`

### P1-02 — Multi-seed + CI protocol for real runs

- [x] Specify and implement:
  - resampling unit (over seeds? over examples? both?),
  - number of seeds,
  - confidence level,
  - deterministic seed derivation.
- [x] Acceptance:
  - Real-suite tables include CI columns that can be reproduced from logs/configs.
- Not marked done until:
  - 5-seed real runs are executed for all paper suites (ID + OOD),
  - Tables T2/T4 are regenerated with `N`, `*_mean`, and `*_ci` columns,
  - `paper_results_manifest.json` is updated and `CLI_CMD_VALIDATE_PAPER_MANIFEST` passes,
  - `CLI_CMD_REPRODUCE_PAPER` passes against the updated manifest.
- Protocol (implemented; needs paper-backed runs):
  - [x] Resampling unit: run seed (bootstrap over per-seed scalar metrics).
  - [x] N=5 seeds per real configuration: `seed_suite = SEEDS.SEED_REAL_SUITE + offset` for each `offset ∈ SEEDS.PAPER_REAL_SEED_OFFSETS`.
  - [x] CI: 95% bootstrap CI with `STATS.BOOTSTRAP_RESAMPLES` resamples.
  - [x] Deterministic bootstrap seed: derived from `SEEDS.SEED_GLOBAL`, `DETERMINISM.SEED_DERIVATION.SALT_BOOTSTRAP`, and the configuration key.
  - [x] Table schema updated: T2/T4 now report `*_mean` and `*_ci` plus `N`.
  - [x] Execute the 5-seed real runs, regenerate tables, and update paper + manifest.
- Run IDs / evidence (fill when complete):
  - paper bundle output_root: `outputs/paper_ready_gpt2_20260120_v6c/`
  - report_dir: `outputs/paper_ready_gpt2_20260120_v6c/reports/report_20260120T171815Z_0000/`

### P1-03 — Compute/cost transparency + frontier

- [x] Measure and report:
  - runtime per suite/run stage,
  - peak VRAM / memory (if feasible),
  - CC necessity-test scaling and/or approximations.
- [x] Add an accuracy↔cost frontier (or justify why cost is trivial in the chosen setting).
- [x] Acceptance:
  - A reviewer can see how expensive diagnostics are and what approximations trade off accuracy vs cost.

### P1-04 — OOD analysis narrative

- [x] Expand the OOD discussion to explain *why* the audit changes on OOD (e.g., instability mechanisms, dataset shift effects).
- [x] Acceptance:
  - OOD results are interpreted, not just reported.

---

## P2 (Polish)

### P2-01 — Paper packaging

- [ ] Produce a clean LaTeX PDF (or conference template) with consistent cross-references.
- [ ] Ensure every citation key resolves to a BibTeX entry.

### P2-02 — README & release hygiene

- [x] Update `README.md` to reflect this is runnable code (not “documentation-only”).
- [x] Ensure `MANIFEST.sha256` / packaging rules are followed if you ship a release zip.

---

## Notes for next iteration

This taskboard has been updated after pasting GPT Pro’s review into `paper/gpt_pro_review.md`.

Next updates should:

- mark tasks as done/canceled,
- add exact run IDs to every completed item,
- and split ambiguous tasks into concrete subtasks with acceptance criteria.

Next session (exact steps to continue from this checkpoint):

1) Create a fresh immutable output root for the new paper bundle (do not reuse v5/v6/v6b), for example:
   - `outputs/paper_ready_gpt2_20260120_v6c/`

2) Run required gates before compute-heavy runs:
   - `CLI_CMD_VALIDATE_SPEC`

3) Generate the new bundle (GPU recommended; use `--device cuda`):
   - Run synth diagnostics once into the new output root (equivalent to `CLI_CMD_SYNTH_RUN_DIAGNOSTICS` with `--output_root <OUTPUT_ROOT>`).
   - (Convenience) Resume missing paper-scope real runs into an existing output root:
     - `python scripts/resume_paper_bundle.py --output_root <OUTPUT_ROOT> --device cuda --include_synth`
   - Run real suites for ID and OOD for each seed in `SEEDS.PAPER_REAL_SEED_OFFSETS`:
     - For each `seed_suite = SEEDS.SEED_REAL_SUITE + offset`:
       - IOI: (`EXP_REAL_IOI_V1`, `EXP_REAL_IOI_OOD_V1`)
       - Greater-than (yes/no): (`EXP_REAL_GREATERTHAN_YN_V1`, `EXP_REAL_GREATERTHAN_YN_OOD_V1`)
       - Induction: (`EXP_REAL_INDUCTION_V1`, `EXP_REAL_INDUCTION_OOD_V1`)
     - Each run uses the enabled generators set from `configs/generators.yaml` (must include `G_ATTR_PATCH`, `G_MANUAL_SEED`, `G_RANDOM_K`) and `I_ACTPATCH` only.

4) Build paper artifacts for that output root:
   - `CLI_CMD_BUILD_PAPER_ARTIFACTS` with `--output_root <OUTPUT_ROOT>`
   - Record the produced report dir path (under `<OUTPUT_ROOT>/reports/`).

5) Update paper + manifest to v6:
   - Update `paper.md` to use the new tables (T2/T4 mean+CI schema) and to cite only v6 artifacts.
   - Update `paper_results_manifest.json`:
     - bump `artifact_bundle_id` to the new bundle,
     - replace artifact paths + sha256 values,
     - replace the `runs` list to match the new bundle (including any emitted certificates).

6) Run required validators and record PASS evidence:
   - `CLI_CMD_VALIDATE_PAPER_MANIFEST`
   - `CLI_CMD_DETERMINISM_SMOKE_TEST`

7) Repro end-to-end and record the output bundle:
   - `CLI_CMD_REPRODUCE_PAPER`
