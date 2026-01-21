# Outputs

This directory contains generated, immutable run artifacts produced by the ScrubID pipeline.

Included bundles:

- `paper_ready_gpt2_20260120_v6c/`: the paper artifact bundle referenced by `paper_results_manifest.json`.
- `repro_paper_20260120T175516Z_0000/`: a fresh reproduction bundle (hash-verified).
- `reports/`: validation and smoke-test reports produced by the CLI quality gates.

The pipeline is designed to never overwrite an existing run directory; new runs must write to a fresh output root.
