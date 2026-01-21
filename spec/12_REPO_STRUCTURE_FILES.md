# 12_REPO_STRUCTURE_FILES.md

This file specifies the **concrete** repository structure that implements this spec pack.

This pack (`scrubid_iclr_pack_v1_0_3/`) is documentation-only. The implementation repository is a separate codebase that must follow the structure below.

All file and module names are normative. Public interfaces listed here are required.

## Top-level tree

```
repo_root/
  pyproject.toml
  README.md
  paper_results_manifest.json
  src/
    scrubid/
      __init__.py
      cli.py
      canonical.py
      config.py
      determinism.py
      hashing.py
      io/
        __init__.py
        manifest.py
        run_record.py
        logging.py
      datasets/
        __init__.py
        registry.py
        synthetic.py
        real_ioi.py
        real_greaterthan.py
        real_induction.py
        reference.py
      interventions/
        __init__.py
        hooks.py
        actpatch.py
        pathpatch.py
        causal_scrub.py
      candidates/
        __init__.py
        registry.py
        generator_manual.py
        generator_acdc.py
        generator_attr_patch.py
        generator_atpstar.py
        generator_sparse_feature.py
      scoring/
        __init__.py
        behavior_metrics.py
        faithfulness.py
        mdl.py
      diagnostics/
        __init__.py
        rr.py
        sss.py
        cc.py
        certificates.py
      experiments/
        __init__.py
        runner.py
  tests/
    test_spec_conformance.py
    test_determinism_smoke.py
    test_synth_suite_sanity.py
    test_real_case_repro.py
```

## Module responsibilities and required public interfaces

### `scrubid/cli.py`

- Provides the CLI entrypoint specified by `CANONICAL.CLI.CLI_ENTRYPOINT`.
- Must implement canonical commands listed in `CANONICAL.CLI.CANONICAL_COMMANDS`.

Public interface:

- `def main() -> int`

### `scrubid/canonical.py`

- Loads and validates the canonical block from `spec/00_CANONICAL.md`.
- Provides accessors for canonical constants.

Public interface:

- `def load_canonical(spec_root: str) -> dict`

### `scrubid/config.py`

- Loads YAML configs from `configs/`.
- Resolves `${CANONICAL.*}` references using `scrubid.canonical`.

Public interface:

- `def load_config(config_path: str, canonical: dict) -> dict`

### `scrubid/datasets/registry.py`

- Registers suites by suite_id.

Public interface:

- `def get_suite(suite_id: str)`

### `scrubid/datasets/reference.py`

- Builds D_ref and a deterministic pairing function for a given suite.

Public interface:

- `def build_reference(suite_id: str, canonical: dict, seed: int) -> tuple[list, callable]`

### `scrubid/interventions/hooks.py`

- Defines hookpoint resolution using `CANONICAL.HOOKS`.
- Provides deterministic hook registration and patch application.

Public interface:

- `def make_hooks(model, canonical: dict)`

### `scrubid/interventions/actpatch.py`

- Implements I_ACTPATCH semantics from `spec/03` and `spec/22`.

Public interface:

- `def apply_actpatch(model, circuit, hooks, reference_cache)`

### `scrubid/candidates/registry.py`

- Registers candidate generators by generator_id.

Public interface:

- `def get_generator(generator_id: str)`

### `scrubid/scoring/behavior_metrics.py`

- Implements suite-specific behavior metrics m(x; f).

Public interface:

- `def compute_metric(suite_id: str, model, batch) -> list[float]`

### `scrubid/scoring/faithfulness.py`

- Computes Δ(C) and ε-faithfulness.

Public interface:

- `def compute_delta(suite_id: str, model, scrubbed_model, dataset_eval) -> float`
- `def compute_epsilon(suite_id: str, model, dataset_eval, canonical: dict) -> float`

### `scrubid/scoring/mdl.py`

- Computes MDL(C) using canonical weights.

Public interface:

- `def compute_mdl(circuit, canonical: dict) -> float`

### `scrubid/diagnostics/rr.py`, `sss.py`, `cc.py`

- Compute RR, SSS, CC exactly as defined in `spec/06_DIAGNOSTICS_RR_SSS_CC.md`.

Public interfaces:

- `def compute_rr(candidate_records, canonical: dict) -> dict`
- `def compute_sss(replicate_circuits, canonical: dict) -> dict`
- `def compute_cc(near_optimal_records, canonical: dict) -> dict`

### `scrubid/io/run_record.py`

- Writes `run_record.json` matching `spec/14_RUN_RECORD_SCHEMA.md`.

Public interface:

- `def write_run_record(path: str, record: dict) -> None`

### `scrubid/experiments/runner.py`

- Orchestrates end-to-end runs.
- Enforces immutability: never overwrites run directories.

Public interface:

- `def run_experiment(config: dict, canonical: dict) -> dict`

## Test coverage obligations

- `tests/test_determinism_smoke.py` must implement the determinism smoke test described in `spec/11_DETERMINISM_REPRODUCIBILITY.md`.
- `tests/test_synth_suite_sanity.py` must validate planted redundancy detection (RR/CC) and stability (SSS) per `spec/07_SYNTHETIC_SUITE_GENERATOR.md` and `spec/16_TEST_PLAN.md`.
- `tests/test_real_case_repro.py` must validate that real case study metrics reproduce within thresholds under deterministic mode.
