# 08_REAL_MODEL_CASE_STUDIES.md

## SDR note

All constants, dataset sizes, token lists, and suite identifiers referenced here are defined in `spec/00_CANONICAL.md`.

## What this spec fixes

Real-model case studies are where ScrubID is most likely to drift, because small implementation choices (prompt generation, corrupted-reference construction, pairing rules, and tokenization constraints) materially change results.

This file therefore specifies, for each suite:

- the clean dataset D
- the reference dataset D_ref (the “corrupted” split)
- the deterministic pairing rule between D and D_ref
- the scalar behavior metric m(x; f)

The intervention mechanics are defined in `spec/03_INTERVENTION_FAMILIES.md` and the hook-level contract is defined in `spec/22_IMPLEMENTATION_CONTRACT.md`.

## Common requirements

- Use HuggingFace model identifiers from `MODELS.*`.
- Record the exact model revision and weight file hashes in `run_record.json`.
- Prompt generation is deterministic given `SEEDS.SEED_REAL_SUITE`.
- Token lists are filtered deterministically to enforce single-token constraints where required.

## Reference distribution D_ref and pairing

ScrubID interventions require a reference distribution D_ref that supplies activations for “scrubbed” components.

- The default reference distribution ID for each suite is `REFERENCE.REFDIST_DEFAULT_BY_SUITE[suite_id]`.
- The required pairing rule (how each clean prompt selects its reference prompt) is `REFERENCE.REF_ASSIGNMENT_DEFAULT_BY_SUITE[suite_id]`.

For the real-model suites in this pack, D_ref is a *corrupted prompt set aligned 1:1 with the clean prompts*, and the pairing rule is index-aligned.

Concretely: if D has prompts x_0 through x_{N-1} and D_ref has prompts x_ref_0 through x_ref_{N-1}, then for clean prompt x_i, the reference prompt is x_ref_i.

### Corruption rule

Each suite defines a deterministic corruption function Corr(x) that maps a clean prompt to a corrupted prompt of the same length and tokenization shape (as far as token constraints allow).

The corrupted dataset is built as:

- D_ref[i] = Corr(D[i]) for each index i.

The corruption rules below are chosen to *reduce the clean-model metric* while preserving superficial prompt structure, matching standard practice in the activation patching literature.

## Suite: IOI (Indirect Object Identification)

Suite id: `IDS.SUITE_IDS.SUITE_REAL_IOI_V1`.

Primary reference: `[@wang2023ioi]`.

### Prompt template

For names A and B and strings place and obj:

- Clean: `When A and B went to the place, A gave a obj to`

Correct next token is B (the indirect object). Incorrect token is A.

### Dataset construction

- Candidate names: `TOKEN_LISTS.IOI_NAME_CANDIDATES`
- Places: `TOKEN_LISTS.IOI_PLACES`
- Objects: `TOKEN_LISTS.IOI_OBJECTS`

IID split size: `DATASETS.IOI_NUM_PROMPTS_ID`.

OOD split size: `DATASETS.IOI_NUM_PROMPTS_OOD`.

Construction rule:

- Enumerate (A, B, place, obj) tuples in a deterministic order (lexicographic over the source lists, excluding A=B).
- Apply a deterministic shuffle using `SEEDS.SEED_REAL_SUITE`.
- Take the first N tuples for each split size.

### Reference dataset D_ref for IOI

Define Corr_IOI on a clean prompt by swapping the final-clause subject:

- Corr_IOI(`When A and B went to the place, A gave a obj to`) = `When A and B went to the place, B gave a obj to`

This flips which name is the indirect object on the clean prompt.

### Metric

Let t_correct and t_incorrect be the single-token encodings of the correct and incorrect names with leading whitespace.

Define:

- m_IOI(x; f) = logit_f(x, t_correct) − logit_f(x, t_incorrect)

## Suite: Greater-than via year-span prediction

Suite id: `IDS.SUITE_IDS.SUITE_REAL_GREATERTHAN_V1`.

Primary reference: `[@hanna2023greaterthan]`.

### Prompt template

For noun, century prefix, and two-digit YY:

- Clean: `The noun lasted from the year {prefix}{YY} to the year {prefix}`

The model should assign higher probability mass to completions ZZ such that ZZ > YY.

### Dataset construction

- Nouns: `TOKEN_LISTS.GREATERTHAN_NOUNS`
- Century prefix (IID): `TOKEN_LISTS.GREATERTHAN_CENTURY_PREFIX_ID`
- Century prefix (OOD): `TOKEN_LISTS.GREATERTHAN_CENTURY_PREFIX_OOD`
- Two-digit values: `DATASETS.GT_YY_VALUES_ID`

IID split size: `DATASETS.GT_NUM_PROMPTS_ID`.

OOD split size: `DATASETS.GT_NUM_PROMPTS_OOD`.

Construction rule:

- Enumerate (noun, YY) tuples deterministically.
- Deterministically shuffle using `SEEDS.SEED_REAL_SUITE`.
- Assign the century prefix according to split.

### Reference dataset D_ref for greater-than

Define Corr_GT by swapping the implicit comparison direction. For clean prompt built with YY, choose a deterministic “mirror” value YY_mirror from the same candidate set such that YY_mirror < YY.

Operationally:

- Build an ordered list of YY candidates from `DATASETS.GT_YY_VALUES_ID`.
- For a given YY at index k in that list, define YY_mirror as the candidate at index (k − 1) with wraparound.

Then:

- Corr_GT(prompt with placeholder {YY}) = same prompt with YY replaced by YY_mirror.

This forces the correct set of completions (ZZ > YY) to differ between clean and corrupted prompts while keeping surface form similar.

### Metric

Let T_good(x) be the set of single-token completions representing two-digit ZZ satisfying ZZ > YY.

Let T_bad(x) be the set of single-token completions representing ZZ ≤ YY.

Define:

- m_GT(x; f) = logsumexp_{t∈T_good(x)} logit_f(x, t) − logsumexp_{t∈T_bad(x)} logit_f(x, t)

## Suite: Greater-than (yes/no) for tokenizer-agnostic evaluation

Suite id: `IDS.SUITE_IDS.SUITE_REAL_GREATERTHAN_YN_V1`.

Primary reference: `[@hanna2023greaterthan]` (task inspiration; this variant avoids single-token two-digit completion constraints).

### Prompt template

For noun, century prefix, and two-digit YY and ZZ:

- Clean: `The noun lasted from the year {prefix}{YY} to the year {prefix}{ZZ}. Is the end year greater than the start year? Answer:`

The correct next token is `TOKEN_LISTS.GREATERTHAN_ANSWER_YES` if ZZ > YY, else `TOKEN_LISTS.GREATERTHAN_ANSWER_NO`.

### Dataset construction

- Nouns: `TOKEN_LISTS.GREATERTHAN_NOUNS`
- Century prefix (IID): `TOKEN_LISTS.GREATERTHAN_CENTURY_PREFIX_ID`
- Century prefix (OOD): `TOKEN_LISTS.GREATERTHAN_CENTURY_PREFIX_OOD`
- Two-digit values for both YY and ZZ: `DATASETS.GT_YY_VALUES_ID` (with YY != ZZ)

IID split size: `DATASETS.GT_NUM_PROMPTS_ID`.

OOD split size: `DATASETS.GT_NUM_PROMPTS_OOD`.

Construction rule:

- Enumerate (noun, YY, ZZ) tuples deterministically (YY != ZZ).
- Deterministically shuffle using `SEEDS.SEED_REAL_SUITE`.
- Assign the century prefix according to split.

### Reference dataset D_ref for greater-than (yes/no)

Define Corr_GT_YN by swapping start/end years:

- Corr_GT_YN(`The noun lasted from the year {prefix}{YY} to the year {prefix}{ZZ}. Is the end year greater than the start year? Answer:`) = same prompt with YY and ZZ swapped.

This flips the correct answer while keeping surface form highly similar.

### Metric

Let t_yes and t_no be the single-token encodings of `TOKEN_LISTS.GREATERTHAN_ANSWER_YES` and `TOKEN_LISTS.GREATERTHAN_ANSWER_NO`.

Let t_correct be t_yes if ZZ > YY else t_no, and t_incorrect be the other.

Define:

- m_GT_YN(x; f) = logit_f(x, t_correct) − logit_f(x, t_incorrect)

## Suite: Induction-style next-token prediction

Suite id: `IDS.SUITE_IDS.SUITE_REAL_INDUCTION_V1`.

Primary reference: `[@olsson2022inductionheads]`.

### Prompt template

Choose distinct tokens A and B from `TOKEN_LISTS.INDUCTION_TOKENS_CANDIDATES` after deterministic single-token filtering.

- Clean: `A B A`

Correct next token is B.

OOD prompts: use `A B C A` where C is a distractor token not equal to A or B.

### Dataset construction

IID split size: `DATASETS.IND_NUM_PROMPTS_ID`.

OOD split size: `DATASETS.IND_NUM_PROMPTS_OOD`.

Construction rule:

- Deterministically filter the token list to single-token candidates.
- Deterministically enumerate (A, B) pairs.
- Deterministically shuffle using `SEEDS.SEED_REAL_SUITE`.
- For each (A, B), form the clean prompt.

### Reference dataset D_ref for induction

Define Corr_IND by breaking the repeated-token pattern:

- Corr_IND(`A B A`) = `A B D`

where D is a deterministic distractor token selected from the filtered token list, with D ≠ A and D ≠ B.

### Metric

Let t_correct be the token for B and t_distract be the deterministic distractor token D.

Define:

- m_IND(x; f) = logit_f(x, t_correct) − logit_f(x, t_distract)

## Reproducibility reporting

Each ScrubID run computes the diagnostics RR, SSS, and CC as defined in `spec/06_DIAGNOSTICS_RR_SSS_CC.md`.

For paper reporting and debugging, also compute *between-run overlap* across different random seeds and candidate generators:

- For each pair of runs on the same suite and intervention family, compute Jaccard overlap between their selected best circuits.

This between-run overlap is a derived analysis quantity and must not be conflated with SSS (which is computed from replicates inside a single run).
