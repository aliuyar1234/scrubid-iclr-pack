# 07_SYNTHETIC_SUITE_GENERATOR.md

## SDR note

All constants and suite IDs referenced here are defined in `spec/00_CANONICAL.md`.

## Goal

The synthetic suite provides ground-truth circuits and ground-truth non-identifiability structure. It is used to validate that RR and CC respond to planted redundancy (non-identifiability), and that MDL-based selection recovers the planted minimal circuit size in identifiable settings.

## Suite ID

- `IDS.SUITE_IDS.SUITE_SYNTH_V1`

## Generator inputs

- `SEEDS.SEED_SYNTH_SUITE`
- `DATASETS.SYNTH_NUM_INSTANCES`
- `DATASETS.SYNTH_REDUNDANCY_FACTORS`
- `DATASETS.SYNTH_TEMPLATES`

## Synthetic model class

Each synthetic instance defines a deterministic computational graph `G = (V, E)` where each node computes a scalar activation.

- Nodes are named deterministically as `node_{template}_{instance_id}_{local_id}`.
- The output node emits logits for a binary classification task.

The ground-truth circuit is a subset of nodes known by construction.

## Templates

Each template constructs a base circuit and then injects redundancy.

### XOR template

Inputs: three binary variables `(a, b, c)`.

Target behavior: `y = XOR(a, b)`.

Base circuit:

- compute `h1 = XOR(a, b)`
- output logits depend only on `h1`

Redundancy injection:

- create `r` parallel copies `h1^{(1)} through h1^{(r)}` computing the same function
- define an aggregator `h_aggr = mean_i h1^{(i)}`
- output depends on `h_aggr`

Ground-truth minimal circuits:

- any circuit that includes exactly one `h1^{(i)}` plus the aggregator is sufficient
- equivalence class size equals the redundancy factor `r`

### COMPARE template

Inputs: two integers `(p, q)` represented as scalars.

Target behavior: `y = 1[p > q]`.

Base circuit computes `h_cmp = p - q` and thresholds it.

Redundancy injection duplicates `h_cmp` into `r` parallel copies combined by a fixed linear map.

### INDUCTION template

Inputs: token ids `(A, B)` from a small vocabulary.

Target behavior: predict `B` given sequence `[A, B, A]`.

Base circuit uses two feature nodes:

- `h_key` encodes `A`
- `h_val` encodes `B`

Redundancy injection duplicates either `h_key` or `h_val` groups.

## Ground-truth labels

For each instance, write a metadata record containing:

- `template_id`
- `redundancy_factor`
- `ground_truth_circuit_primary`
- `redundant_groups`: a list of sets of nodes that are interchangeable
- `equivalence_class_size`: computed as the product of redundant group sizes
- `planted_mdl_best`: MDL of the minimal circuit computed from the construction

## Determinism requirements

- The generator must be fully deterministic given `SEEDS.SEED_SYNTH_SUITE`.
- Output order is deterministic.
- Hashes of produced files are recorded in the `run_record.json`.

## Diagnostics expectations

For this suite, the implementation must satisfy the following sanity assertions:

- When redundancy_factor equals 1 (unique planted mechanism), RR should be near zero, CC should be near zero, and SSS should be high.
- When redundancy_factor is greater than 1 (multiple interchangeable subcircuits), RR and CC should increase as redundancy_factor increases.
- When candidate generation has stochastic tie-breaks, SSS may drop; this is expected and should be visible in the reported SSS.

