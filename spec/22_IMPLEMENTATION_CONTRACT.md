# 22_IMPLEMENTATION_CONTRACT.md

This document is the normative bridge between the mathematical spec and an implementation.

The goal is to eliminate ambiguity about:

- The concrete component set V used in code.
- Hookpoints and tensor semantics for patching.
- Suite-specific behavior metrics.
- Reference distribution generation and pairing.

All constants and IDs are defined in `spec/00_CANONICAL.md`.

## 1. Component IDs and canonical ordering

### Component ID strings

The implementation must represent each component v ∈ V by a deterministic string ID.

For the default granularity `CANONICAL.COMPONENT_GRANULARITY.GRANULARITY_DEFAULT`, the required ID formats are:

- Attention head: `H{layer}:{head}` where `layer` is the 0-indexed Transformer block index, and `head` is the 0-indexed head index.
- MLP block: `M{layer}` where `layer` is the 0-indexed Transformer block index.

Examples:

- `H3:7` means layer 3, head 7.
- `M5` means the MLP block at layer 5.

The canonical ordering of component IDs is lexicographic by this string representation.

For the synthetic-only granularity `CANONICAL.COMPONENT_GRANULARITY.GRANULARITY_SYNTH_NODE`, the required ID format is:

- Synthetic node: `node_{template}_{instance_id}_{local_id}`

where:

- `template` is one of `DATASETS.SYNTH_TEMPLATES`.
- `instance_id` is a 0-indexed integer identifier for the synthetic instance.
- `local_id` is a deterministic, template-specific local name.

## 2. Hook backend

ScrubID’s canonical backend is `CANONICAL.HOOKS.BACKEND_DEFAULT`.

Implementations must support TransformerLens-style hookpoints.

## 3. Hookpoints and tensor semantics

ScrubID requires the following hookpoints (templates are defined in `CANONICAL.HOOKS`):

### 3.1 Attention head hook (pre-W_O)

Hookpoint template: `CANONICAL.HOOKS.TL_HOOK_HEAD_RESULT_TEMPLATE`.

Semantics:

- The hooked tensor must have shape (batch, position, head, d_head).
- The head dimension index must correspond to the `head` in the component ID `H{layer}:{head}`.

Patching semantics for I_ACTPATCH:

- For component `H{layer}:{head}` that is outside the circuit C, replace:
  - target[:, :, head, :] ← reference[:, :, head, :]

### 3.2 MLP output hook

Hookpoint template: `CANONICAL.HOOKS.TL_HOOK_MLP_OUT_TEMPLATE`.

Semantics:

- The hooked tensor must have shape (batch, position, d_model).

Patching semantics for I_ACTPATCH:

- For component `M{layer}` that is outside the circuit C, replace:
  - target[:, :, :] ← reference[:, :, :]

## 4. Reference distribution D_ref

The implementation must:

- Build D_ref according to `CANONICAL.REFERENCE.REFDIST_DEFAULT_BY_SUITE[suite_id]`.
- Apply pairing according to `CANONICAL.REFERENCE.REF_ASSIGNMENT_DEFAULT_BY_SUITE[suite_id]`.

The pairing seed must be recorded as `seed_reference_pairing` in the run record.

## 5. Suite-specific behavior metrics

All real suites in `CANONICAL.IDS.SUITE_IDS` used for experiments must implement a scalar metric m(x; f).

### 5.1 IOI (SUITE_REAL_IOI_V1)

- Define the indirect object token as the correct answer and the competitor token as the incorrect answer (as in the IOI prompt template).
- Metric: logit_diff = logit(correct_token) − logit(competitor_token) at the final position.

### 5.2 Greater-than (SUITE_REAL_GREATERTHAN_V1)

Normative definition: this suite and its metric are fully specified in `spec/08_REAL_MODEL_CASE_STUDIES.md`.

Implementation must compute the scalar behavior metric as:

- Let YY be the two-digit year fragment used in the prompt.
- Define T_good(x) as the set of single-token completions representing two-digit ZZ satisfying ZZ > YY.
- Define T_bad(x) as the set of single-token completions representing two-digit ZZ satisfying ZZ ≤ YY.
- Metric:
  - m_GT(x; f) = logsumexp_{t∈T_good(x)} logit_f(x, t) − logsumexp_{t∈T_bad(x)} logit_f(x, t)

The construction of YY, the candidate set of ZZ tokens, and the deterministic token filtering rules are defined in `spec/08_REAL_MODEL_CASE_STUDIES.md`.

### 5.3 Greater-than (SUITE_REAL_GREATERTHAN_YN_V1)

Normative definition: this suite and its metric are fully specified in `spec/08_REAL_MODEL_CASE_STUDIES.md`.

Implementation must compute the scalar behavior metric as:

- Let t_yes and t_no be the single-token encodings of `TOKEN_LISTS.GREATERTHAN_ANSWER_YES` and `TOKEN_LISTS.GREATERTHAN_ANSWER_NO`.
- For each prompt x, define t_correct = t_yes if ZZ > YY else t_no, and t_incorrect as the other.
- Metric:
  - m_GT_YN(x; f) = logit_f(x, t_correct) − logit_f(x, t_incorrect)

### 5.4 Induction (SUITE_REAL_INDUCTION_V1)

Normative definition: this suite and its metric are fully specified in `spec/08_REAL_MODEL_CASE_STUDIES.md`.

Implementation must compute the scalar behavior metric as:

- The clean prompt is of the form `A B A` (and OOD variants include a distractor token).
- Let t_correct be the token for B.
- Let t_distract be the deterministic distractor token D used by Corr_IND.
- Metric:
  - m_IND(x; f) = logit_f(x, t_correct) − logit_f(x, t_distract)

The deterministic rule for selecting D and the prompt construction rules are defined in `spec/08_REAL_MODEL_CASE_STUDIES.md`.

## 6. Faithfulness loss Δ(C)

Given a dataset D_eval and metric m(x; f):

- Compute the baseline per-example metric values under the unmodified model.
- Compute per-example metric values under the scrubbed model f^(C, I).
- Faithfulness loss is the mean absolute difference across D_eval.

## 7. Determinism obligations

- Under deterministic mode (`CANONICAL.DETERMINISM.DETERMINISTIC_MODE_STRICT`), all outputs must be bitwise identical given the same inputs and seeds.
- Replicate discovery runs used for SSS must differ only by the derived replicate seed (see `spec/06_DIAGNOSTICS_RR_SSS_CC.md`).

## 8. Required artifacts

Every run directory must include:

- `CANONICAL.FILES.RUN_RECORD_FILENAME`
- `CANONICAL.FILES.DIAGNOSTICS_FILENAME`
- `CANONICAL.FILES.BEST_CIRCUIT_FILENAME`
- `CANONICAL.FILES.CERTIFICATE_FILENAME` (required only if the audit certificate trigger fires)
