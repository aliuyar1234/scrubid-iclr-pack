# ScrubID (ICLR Pack v1.0.3)

ScrubID is an identifiability-aware auditing pipeline for mechanistic interpretability claims in transformer language models.

It operationalizes a simple question that standard circuit papers rarely answer explicitly:

> If you can find one faithful circuit, how many other distinct circuits are also faithful under your intervention family, and how sensitive is your conclusion to implementation choices?

ScrubID produces (i) a scrubbed model construction, (ii) three diagnostics (RR, SSS, CC) that quantify redundancy, scrub-sensitivity, and complexity, and (iii) a non-identifiability certificate when multiple incompatible explanations survive the same validation criteria.

Author: **Ali Uyar** (Independent Researcher)

## Paper

- Paper PDF (download): https://github.com/aliuyar1234/scrubid-iclr-pack/releases/latest/download/scrubid_preprint.pdf
- LaTeX source bundle (download): https://github.com/aliuyar1234/scrubid-iclr-pack/releases/latest/download/scrubid_latex_sources.zip

Local copies (same files, repo paths):

- `paper/scrubid_preprint.pdf`
- `paper/scrubid_latex_sources.zip`

This repository is a runnable, deterministic artifact pack: it includes the reference implementation, a spec (Single Definition Rule), and end-to-end reproduction commands. All constants, IDs, CLI literals, thresholds, and paths live in `spec/00_CANONICAL.md`.

## Citation

See `CITATION.cff` (GitHub will also surface this under “Cite this repository”).

## Repository layout

- `paper/`: preprint PDF and LaTeX source bundle.
- `paper.md`: canonical manuscript source (Markdown).
- `src/`: reference implementation (Python).
- `configs/`: experiment configuration (YAML; canonical keys only).
- `spec/`: formal definitions + Single Definition Rule (SSOT constants/IDs).
- `outputs/`: immutable run artifacts (large; used for provenance / verification).
- `tasks/`, `checklists/`: phase plan and release checklists.

## Quickstart (12 commands)

Each line below is a *canonical command ID*. The exact command string is defined once (and only once) in `spec/00_CANONICAL.md` under `CLI.CANONICAL_COMMANDS`.

1. `CLI_CMD_VALIDATE_SPEC`
2. `CLI_CMD_SYNTH_GENERATE_SUITE`
3. `CLI_CMD_SYNTH_RUN_CANDIDATE_GENERATORS`
4. `CLI_CMD_SYNTH_RUN_DIAGNOSTICS`
5. `CLI_CMD_REAL_IOI_RUN`
6. `CLI_CMD_REAL_GREATERTHAN_YN_RUN`
7. `CLI_CMD_REAL_INDUCTION_RUN`
8. `CLI_CMD_AGGREGATE_RESULTS`
9. `CLI_CMD_BUILD_PAPER_ARTIFACTS`
10. `CLI_CMD_VALIDATE_PAPER_MANIFEST`
11. `CLI_CMD_DETERMINISM_SMOKE_TEST`
12. `CLI_CMD_REPRODUCE_PAPER`

## Convenience helpers

- `scripts/resume_paper_bundle.py` resumes missing paper-scope real runs into an existing `--output_root` (useful if a long GPU run was interrupted).

## What "publishable-ready" means here

A run is considered publishable-ready when all quality gates defined in `SPEC.md` are PASS:

- **G0 Spec coherence:** Single Definition Rule holds; configs resolve; no undefined IDs.
- **G1 Determinism smoke:** same seed and same resolved config produce identical `run_record_hash`.
- **G2 Synthetic ground-truth sanity:** RR monotonically increases with planted redundancy; CC tracks planted minimal size.
- **G3 Real-model reproducibility:** RR/SSS/CC are stable across seeds and across intervention families.
- **G4 Reproducibility:** tables and plots are regenerated *only* from logs and configs.
- **G5 Paper evidence:** every main claim maps to concrete run IDs and generated artifacts.

## Where to start reading

- `SPEC.md` (entrypoint and invariants)
- `spec/00_CANONICAL.md` (all constants and IDs)
- `spec/02_FORMAL_DEFINITIONS.md` through `spec/08_REAL_MODEL_CASE_STUDIES.md` (the core method)
- `tasks/TASK_INDEX.md` (implementation plan)
