# 10_METRICS_STATS_PROTOCOL.md

## SDR note

All constants referenced here are defined in `spec/00_CANONICAL.md`.

## Metrics

ScrubID reports:

- Faithfulness loss `Δ(C)` (defined in `spec/02_FORMAL_DEFINITIONS.md`)
- Necessity statistics `N(v; C)`
- RR / SSS / CC (defined in `spec/06_DIAGNOSTICS_RR_SSS_CC.md`)

## Confidence intervals

### Within-run (over examples)

For any dataset-level scalar statistic `z` computed as a mean over examples within a run, compute a paired bootstrap confidence interval:

- Use a deterministic bootstrap seed derived from `SEEDS.SEED_GLOBAL` and the run key.
- Use a fixed number of bootstrap resamples specified as a canonical constant in `spec/00_CANONICAL.md`.

Report:

- point estimate
- quantiles `STATS.CI_LO` and `STATS.CI_HI`

### Across-run (multi-seed; over run seeds)

For paper-facing real-suite summary tables that aggregate multiple runs of the same configuration, compute a bootstrap CI where the resampling unit is the **run seed**.

Normative v1.0.3 paper protocol:

- For a fixed `(suite_id, experiment_id, intervention_family_id, candidate_generator_id)` configuration, execute multiple runs whose `seed_suite` values are:
  - `SEEDS.SEED_REAL_SUITE + offset` for each `offset ∈ SEEDS.PAPER_REAL_SEED_OFFSETS`.
- Compute the mean of each reported scalar metric across these runs.
- Compute a bootstrap CI over the per-run metric values using:
  - `STATS.BOOTSTRAP_RESAMPLES` resamples,
  - quantiles `STATS.CI_LO` and `STATS.CI_HI`,
  - a deterministic bootstrap RNG seed derived from `SEEDS.SEED_GLOBAL` and the configuration key.

## Multiple comparisons

When comparing multiple circuits on the same dataset, control false discovery using a deterministic Benjamini-Hochberg procedure.

The exact procedure is deterministic given p-values.

## Logging requirements

Every reported metric must be derivable from:

- `run_record.json`
- `logs.jsonl`
- config files

No table or plot may rely on unstored intermediate state.
