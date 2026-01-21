# 00_CANONICAL.md

This file is the Single Source of Truth for:

- all constants (numeric, boolean, and string)
- all IDs (phases, tasks, suites, experiments)
- all CLI literals
- all paths
- all thresholds
- all enums and event types
- all symbol definitions

Any other file must reference these items by name only.

## Canonical machine-readable block

```yaml
PROJECT_ID: scrubid_iclr_pack_v1_0_3
PROJECT_VERSION: "1.0.4"
PACK_CREATED_DATE_UTC: 2026-01-18

DOCS:
  CITATION_STYLE: pandoc
  BIB_PATH: bib/references.bib

PAPER:
  # Canonical suite set used for paper tables/figures (excluding optional suites
  # implemented in the codebase but not used in the paper results).
  #
  # Values are canonical dotted paths, resolved via get_canonical at runtime.
  PAPER_SUITE_ID_KEYS:
    - IDS.SUITE_IDS.SUITE_SYNTH_V1
    - IDS.SUITE_IDS.SUITE_REAL_IOI_V1
    - IDS.SUITE_IDS.SUITE_REAL_GREATERTHAN_YN_V1
    - IDS.SUITE_IDS.SUITE_REAL_INDUCTION_V1

ERRORS:
  # Hard-fail message when non-actpatch intervention families are requested in
  # the v1.0.4 paper reproduction scope.
  NOT_IMPLEMENTED_FOR_PAPER_REPRO: NOT_IMPLEMENTED_FOR_PAPER_REPRO

BOOL:
  "TRUE": true
  "FALSE": false

PATHS:
  PATH_SPEC_DIR: spec
  PATH_TASKS_DIR: tasks
  PATH_PAPER_DIR: paper
  PATH_BIB_DIR: bib
  PATH_CONFIG_DIR: configs
  PATH_CHECKLIST_DIR: checklists
  PATH_OUTPUT_ROOT: outputs
  PATH_RUNS_ROOT: outputs/runs
  PATH_REPORTS_ROOT: outputs/reports
  PATH_DATA_CACHE_ROOT: outputs/data_cache

FILES:
  RUN_RECORD_FILENAME: run_record.json
  LOG_JSONL_FILENAME: logs.jsonl
  DIAGNOSTICS_FILENAME: diagnostics.json
  CERTIFICATE_FILENAME: certificate.json
  BEST_CIRCUIT_FILENAME: best_circuit.json
  REPORT_TABLE_T1_FILENAME: table_T1.csv
  REPORT_TABLE_T2_FILENAME: table_T2.csv
  REPORT_TABLE_T3_FILENAME: table_T3.csv
  REPORT_TABLE_T4_FILENAME: table_T4.csv
  REPORT_FIG_SYNTH_FILENAME: fig_synth.png
  CONFIG_INTERVENTIONS: configs/interventions.yaml
  CONFIG_GENERATORS: configs/generators.yaml
  CONFIG_EXPERIMENTS: configs/experiments.yaml
  CONFIG_DETERMINISM: configs/determinism.yaml
  CONFIG_LOGGING_SCHEMA: configs/logging_schema.yaml
  CONFIG_BUDGETS: configs/budgets.yaml
  PAPER_RESULTS_MANIFEST: paper_results_manifest.json

DETERMINISM:
  # If true, implementations must enforce bitwise determinism (where supported)
  # and must fail closed when determinism cannot be guaranteed.
  DETERMINISTIC_MODE_STRICT: true

  # Canonical salts used in seed derivation. These are treated as literals.
  # Implementations must not invent additional salts.
  SEED_DERIVATION:
    SALT_SEED_EFFECTIVE: seed_effective_v1
    SALT_REFERENCE_PAIRING: reference_pairing_v1
    SALT_REPLICATE: replicate_v1
    SALT_BOOTSTRAP: bootstrap_v1

HASHING:
  HASH_ALGORITHM: sha256
  # sha256_uint32(x): compute SHA-256 over UTF-8 bytes of x, take the first 8 hex
  # characters as a big-endian unsigned 32-bit integer.
  UINT32_FROM_SHA256_HEX_METHOD: first8hex_be
  # Canonical JSON serialization settings for deterministic hashing.
  CANONICAL_JSON:
    SORT_KEYS: true
    ENSURE_ASCII: false
    SEPARATORS: [",", ":"]
    ALLOW_NAN: false

ENUMS:
  VERDICT_ENUM: [PASS, WARN, FAIL, HARD_FAIL]
  VERDICTS:
    VERDICT_PASS: PASS
    VERDICT_WARN: WARN
    VERDICT_FAIL: FAIL
    VERDICT_HARD_FAIL: HARD_FAIL
  LOG_EVENT_TYPES:
    EVENT_RUN_START: run_start
    EVENT_RUN_END: run_end
    EVENT_CONFIG_RESOLVED: config_resolved
    EVENT_DATASET_WRITTEN: dataset_written
    EVENT_CANDIDATES_WRITTEN: candidates_written
    EVENT_INTERVENTION_APPLIED: intervention_applied
    EVENT_SCRUBBED_MODEL_EVAL: scrubbed_model_eval
    EVENT_DIAGNOSTICS_WRITTEN: diagnostics_written
    EVENT_CERTIFICATE_WRITTEN: certificate_written
    EVENT_REPORT_WRITTEN: report_written

  CERTIFICATE_REASON_CODES:
    REASON_RR_FAIL: rr_fail
    REASON_SSS_FAIL: sss_fail
    REASON_CC_FAIL: cc_fail

IDS:
  PHASE_IDS:
    PHASE_0_FOUNDATION: PHASE_0_FOUNDATION
    PHASE_1_SYNTHETIC_SUITE: PHASE_1_SYNTHETIC_SUITE
    PHASE_2_INTERVENTIONS_AND_GENERATORS: PHASE_2_INTERVENTIONS_AND_GENERATORS
    PHASE_3_DIAGNOSTICS_AND_AUDIT: PHASE_3_DIAGNOSTICS_AND_AUDIT
    PHASE_4_EXPERIMENTS_AND_ANALYSIS: PHASE_4_EXPERIMENTS_AND_ANALYSIS
    PHASE_5_PAPER_CAMERA_READY: PHASE_5_PAPER_CAMERA_READY
    PHASE_6_RELEASE_AND_REPRO: PHASE_6_RELEASE_AND_REPRO

  SUITE_IDS:
    SUITE_SYNTH_V1: SUITE_SYNTH_V1
    SUITE_REAL_IOI_V1: SUITE_REAL_IOI_V1
    SUITE_REAL_GREATERTHAN_V1: SUITE_REAL_GREATERTHAN_V1
    SUITE_REAL_GREATERTHAN_YN_V1: SUITE_REAL_GREATERTHAN_YN_V1
    SUITE_REAL_INDUCTION_V1: SUITE_REAL_INDUCTION_V1

  EXPERIMENT_IDS:
    EXP_SYNTH_SANITY_V1: EXP_SYNTH_SANITY_V1
    EXP_SYNTH_REDUNDANCY_SWEEP_V1: EXP_SYNTH_REDUNDANCY_SWEEP_V1
    EXP_REAL_IOI_V1: EXP_REAL_IOI_V1
    EXP_REAL_IOI_OOD_V1: EXP_REAL_IOI_OOD_V1
    EXP_REAL_GREATERTHAN_V1: EXP_REAL_GREATERTHAN_V1
    EXP_REAL_GREATERTHAN_OOD_V1: EXP_REAL_GREATERTHAN_OOD_V1
    EXP_REAL_GREATERTHAN_YN_V1: EXP_REAL_GREATERTHAN_YN_V1
    EXP_REAL_GREATERTHAN_YN_OOD_V1: EXP_REAL_GREATERTHAN_YN_OOD_V1
    EXP_REAL_INDUCTION_V1: EXP_REAL_INDUCTION_V1
    EXP_REAL_INDUCTION_OOD_V1: EXP_REAL_INDUCTION_OOD_V1

  INTERVENTION_FAMILY_IDS:
    I_ACTPATCH: I_ACTPATCH
    I_PATHPATCH: I_PATHPATCH
    I_CAUSAL_SCRUB: I_CAUSAL_SCRUB

  CANDIDATE_GENERATOR_IDS:
    G_ACDC: G_ACDC
    G_ATTR_PATCH: G_ATTR_PATCH
    G_ATPSTAR: G_ATPSTAR
    G_SPARSE_FEATURE: G_SPARSE_FEATURE
    G_MANUAL_SEED: G_MANUAL_SEED
    G_RANDOM_K: G_RANDOM_K

HOOKS:
  # Canonical mechanistic interpretability backend.
  BACKEND_DEFAULT: transformer_lens

  # TransformerLens hookpoint templates used by ScrubID.
  # These templates are resolved by formatting {layer} with an integer layer index.
  # NOTE: We patch at hook_z (per-head, pre-W_O) to avoid enabling use_attn_result,
  # which can be prohibitively memory-expensive on modern large models.
  TL_HOOK_HEAD_RESULT_TEMPLATE: blocks.{layer}.attn.hook_z
  TL_HOOK_MLP_OUT_TEMPLATE: blocks.{layer}.hook_mlp_out

REFERENCE:
  # Reference distributions D_ref used by intervention families.
  # Each value is an ID string.
  REFDIST_SYNTH_PAIRED_V1: REFDIST_SYNTH_PAIRED_V1
  REFDIST_IOI_CORRUPTED_V1: REFDIST_IOI_CORRUPTED_V1
  REFDIST_GREATERTHAN_CORRUPTED_V1: REFDIST_GREATERTHAN_CORRUPTED_V1
  REFDIST_GREATERTHAN_YN_CORRUPTED_V1: REFDIST_GREATERTHAN_YN_CORRUPTED_V1
  REFDIST_INDUCTION_CORRUPTED_V1: REFDIST_INDUCTION_CORRUPTED_V1

  REFASSIGN_INDEX_ALIGNED_V1: REFASSIGN_INDEX_ALIGNED_V1
  REFASSIGN_DERANGEMENT_SHUFFLE_V1: REFASSIGN_DERANGEMENT_SHUFFLE_V1

  # Default reference distribution per suite.
  REFDIST_DEFAULT_BY_SUITE:
    SUITE_SYNTH_V1: REFDIST_SYNTH_PAIRED_V1
    SUITE_REAL_IOI_V1: REFDIST_IOI_CORRUPTED_V1
    SUITE_REAL_GREATERTHAN_V1: REFDIST_GREATERTHAN_CORRUPTED_V1
    SUITE_REAL_GREATERTHAN_YN_V1: REFDIST_GREATERTHAN_YN_CORRUPTED_V1
    SUITE_REAL_INDUCTION_V1: REFDIST_INDUCTION_CORRUPTED_V1

  # Default pairing algorithm per suite.
  REF_ASSIGNMENT_DEFAULT_BY_SUITE:
    SUITE_SYNTH_V1: REFASSIGN_DERANGEMENT_SHUFFLE_V1
    SUITE_REAL_IOI_V1: REFASSIGN_INDEX_ALIGNED_V1
    SUITE_REAL_GREATERTHAN_V1: REFASSIGN_INDEX_ALIGNED_V1
    SUITE_REAL_GREATERTHAN_YN_V1: REFASSIGN_INDEX_ALIGNED_V1
    SUITE_REAL_INDUCTION_V1: REFASSIGN_INDEX_ALIGNED_V1

SYMBOLS:
  EPSILON_SYMBOL: "ε"
  RR_SYMBOL: RR
  SSS_SYMBOL: SSS
  CC_SYMBOL: CC

DIAGNOSTICS:
  EPSILON_REL_FRAC: 0.10
  EPSILON_ABS_MIN: 0.0001
  TAU_REL_FRAC_NECESSITY: 0.05

  RR_NUM_CIRCUITS_SET: 20
  RR_NEAR_OPTIMAL_MDL_REL_FRAC: 0.00
  RR_THRESHOLD_LOW: 0.20
  RR_THRESHOLD_HIGH: 0.50
  SSS_NUM_REPLICATES: 5
  SSS_THRESHOLD_STABLE: 0.80
  SSS_THRESHOLD_BORDERLINE: 0.60

  MDL_WEIGHT_NODE: 1.00
  MDL_WEIGHT_EDGE: 0.50
  MDL_WEIGHT_FEATURE: 0.25

  CC_THRESHOLD_GOOD: 0.20
  CC_THRESHOLD_MODERATE: 0.50

BUDGETS:
  BUDGET_DEFAULT:
    EVAL_BATCH_SIZE: 8
    ATTR_PATCH_SCORE_SUBSET_N: 16
    ATTR_PATCH_MAX_COMPONENTS_SCORED: 256
    MAX_CANDIDATES_PER_GENERATOR_SYNTH: 500
    MAX_CANDIDATES_PER_GENERATOR_REAL: 800
    MAX_CIRCUIT_SIZE_COMPONENTS: 200
  BUDGET_SYNTH_CANDIDATES:
    EVAL_BATCH_SIZE: 8
    ATTR_PATCH_SCORE_SUBSET_N: 16
    ATTR_PATCH_MAX_COMPONENTS_SCORED: 256
    MAX_CANDIDATES_PER_GENERATOR: 500
    RANDOM_K_MAX_CANDIDATES: 250
    MAX_CIRCUIT_SIZE_COMPONENTS: 150
  BUDGET_REAL_CANDIDATES:
    EVAL_BATCH_SIZE: 128
    ATTR_PATCH_SCORE_SUBSET_N: 8
    ATTR_PATCH_MAX_COMPONENTS_SCORED: 128
    MAX_CANDIDATES_PER_GENERATOR: 800
    RANDOM_K_MAX_CANDIDATES: 250
    MAX_CIRCUIT_SIZE_COMPONENTS: 250


COMPONENT_GRANULARITY:
  GRANULARITY_OPTIONS:
    - head_mlp
    - node
    - head
    - mlp
    - neuron
    - feature
  GRANULARITY_DEFAULT: head_mlp
  GRANULARITY_SYNTH_NODE: node

SEEDS:
  SEED_GLOBAL: 1337
  SEED_SYNTH_SUITE: 20260116
  SEED_REAL_SUITE: 20260117
  PAPER_REAL_SEED_OFFSETS: [0, 1, 2, 3, 4]

DATASETS:
  SYNTH_NUM_INSTANCES: 200
  SYNTH_REDUNDANCY_FACTORS: [1, 2, 4, 8]
  SYNTH_TEMPLATES: [XOR, COMPARE, INDUCTION]

  IOI_NUM_PROMPTS_ID: 512
  IOI_NUM_PROMPTS_OOD: 512
  GT_NUM_PROMPTS_ID: 1024
  GT_NUM_PROMPTS_OOD: 1024
  GT_YY_VALUES_ID: [5, 12, 32, 48, 73, 91]
  IND_NUM_PROMPTS_ID: 512
  IND_NUM_PROMPTS_OOD: 512

MODELS:
  HF_MODEL_GPT2_SMALL: gpt2
  HF_MODEL_GPT2_MEDIUM: gpt2-medium
  HF_MODEL_QWEN25_7B_INSTRUCT: Qwen/Qwen2.5-7B-Instruct
  HF_MODEL_QWEN25_CODER_7B_INSTRUCT: Qwen/Qwen2.5-Coder-7B-Instruct
  # Immutable revisions (git SHAs) for HuggingFace-hosted models used in the paper.
  # These are used as deterministic defaults for run_key/run_record provenance.
  HF_MODEL_REVISIONS:
    gpt2: 607a30d783dfa663caf39e06633721c8d4cfcd7e
    gpt2-medium: 6dcaa7a952f72f9298047fd5137cd6e4f05f41da
    Qwen/Qwen2.5-7B-Instruct: a09a35458c702b33eeacc393d103063234e8bc28
    Qwen/Qwen2.5-Coder-7B-Instruct: c03e6d358207e414f1eca0bb1891e29f1db0e242

TOKEN_LISTS:
  IOI_NAME_CANDIDATES:
    - Mary
    - John
    - Alice
    - Bob
    - Sarah
    - Michael
    - Laura
    - James
    - Linda
    - Robert
    - David
    - Susan
    - Karen
    - Daniel
    - Emma
    - Olivia
    - Liam
    - Noah
    - Ava
    - Mia
  IOI_PLACES:
    - store
    - park
    - school
    - office
    - library
    - restaurant
  IOI_OBJECTS:
    - drink
    - book
    - ball
    - letter
    - sandwich
    - gift

  GREATERTHAN_NOUNS:
    - war
    - festival
    - drought
    - campaign
    - strike
    - ceremony
  GREATERTHAN_ANSWER_YES: " Yes"
  GREATERTHAN_ANSWER_NO: " No"

  GREATERTHAN_CENTURY_PREFIX_ID: "17"
  GREATERTHAN_CENTURY_PREFIX_OOD: "18"

  INDUCTION_TOKENS_CANDIDATES:
    - red
    - blue
    - green
    - yellow
    - cat
    - dog
    - apple
    - banana
    - river
    - mountain

STATS:
  BOOTSTRAP_RESAMPLES: 10000
  CI_LO: 0.025
  CI_HI: 0.975
  CONFIDENCE_LEVEL: 0.95
  BH_FDR_ALPHA: 0.05

GIT:
  GIT_COMMIT_HEX_LEN: 40

CLI:
  CLI_ENTRYPOINT: "python -m scrubid.cli"
  CANONICAL_FLAGS:
    FLAG_CONFIG: "--config"
    FLAG_SUITE_ID: "--suite_id"
    FLAG_EXPERIMENT_ID: "--experiment_id"
    FLAG_MODEL_ID: "--model_id"
    FLAG_SEED: "--seed"
    FLAG_DETERMINISTIC: "--deterministic"
    FLAG_DEVICE: "--device"
    FLAG_OUTPUT_ROOT: "--output_root"

  CANONICAL_COMMANDS:
    CLI_CMD_VALIDATE_SPEC: "python -m scrubid.cli spec validate --config configs/experiments.yaml --deterministic"
    CLI_CMD_SYNTH_GENERATE_SUITE: "python -m scrubid.cli synth generate --suite_id SUITE_SYNTH_V1 --config configs/experiments.yaml --deterministic"
    CLI_CMD_SYNTH_RUN_CANDIDATE_GENERATORS: "python -m scrubid.cli synth candidates --suite_id SUITE_SYNTH_V1 --config configs/experiments.yaml --deterministic"
    CLI_CMD_SYNTH_RUN_DIAGNOSTICS: "python -m scrubid.cli synth diagnostics --suite_id SUITE_SYNTH_V1 --config configs/experiments.yaml --deterministic"
    CLI_CMD_REAL_IOI_RUN: "python -m scrubid.cli real run --suite_id SUITE_REAL_IOI_V1 --experiment_id EXP_REAL_IOI_V1 --config configs/experiments.yaml --deterministic"
    CLI_CMD_REAL_IOI_OOD_RUN: "python -m scrubid.cli real run --suite_id SUITE_REAL_IOI_V1 --experiment_id EXP_REAL_IOI_OOD_V1 --config configs/experiments.yaml --deterministic"
    CLI_CMD_REAL_GREATERTHAN_RUN: "python -m scrubid.cli real run --suite_id SUITE_REAL_GREATERTHAN_V1 --experiment_id EXP_REAL_GREATERTHAN_V1 --config configs/experiments.yaml --deterministic"
    CLI_CMD_REAL_GREATERTHAN_OOD_RUN: "python -m scrubid.cli real run --suite_id SUITE_REAL_GREATERTHAN_V1 --experiment_id EXP_REAL_GREATERTHAN_OOD_V1 --config configs/experiments.yaml --deterministic"
    CLI_CMD_REAL_GREATERTHAN_YN_RUN: "python -m scrubid.cli real run --suite_id SUITE_REAL_GREATERTHAN_YN_V1 --experiment_id EXP_REAL_GREATERTHAN_YN_V1 --config configs/experiments.yaml --deterministic"
    CLI_CMD_REAL_GREATERTHAN_YN_OOD_RUN: "python -m scrubid.cli real run --suite_id SUITE_REAL_GREATERTHAN_YN_V1 --experiment_id EXP_REAL_GREATERTHAN_YN_OOD_V1 --config configs/experiments.yaml --deterministic"
    CLI_CMD_REAL_INDUCTION_RUN: "python -m scrubid.cli real run --suite_id SUITE_REAL_INDUCTION_V1 --experiment_id EXP_REAL_INDUCTION_V1 --config configs/experiments.yaml --deterministic"
    CLI_CMD_REAL_INDUCTION_OOD_RUN: "python -m scrubid.cli real run --suite_id SUITE_REAL_INDUCTION_V1 --experiment_id EXP_REAL_INDUCTION_OOD_V1 --config configs/experiments.yaml --deterministic"
    CLI_CMD_AGGREGATE_RESULTS: "python -m scrubid.cli report aggregate --config configs/experiments.yaml --deterministic"
    CLI_CMD_BUILD_PAPER_ARTIFACTS: "python -m scrubid.cli report paper_artifacts --config configs/experiments.yaml --deterministic"
    CLI_CMD_VALIDATE_PAPER_MANIFEST: "python -m scrubid.cli report validate_paper_manifest --config configs/experiments.yaml --deterministic"
    CLI_CMD_DETERMINISM_SMOKE_TEST: "python -m scrubid.cli test determinism_smoke --config configs/experiments.yaml --deterministic"
    CLI_CMD_REPRODUCE_PAPER: "python -m scrubid.cli repro paper --config configs/experiments.yaml --deterministic"
    CLI_CMD_RESUME_PAPER_BUNDLE: "python scripts/resume_paper_bundle.py --config configs/experiments.yaml --device cuda --include_synth"

SCHEMAS:
  RUN_RECORD_SCHEMA_VERSION: 3
  LOG_SCHEMA_VERSION: 2
  CONFIG_SCHEMA_VERSION: 2
  PAPER_RESULTS_MANIFEST_SCHEMA_VERSION: 1

OUTPUT_NAMING:
  RUN_DIR_PREFIX: run_
  REPORT_DIR_PREFIX: report_
  REPRO_OUTPUT_ROOT_PREFIX: repro_paper_
  RUN_RECORD_HASH_FIELD: run_record_hash
  RUN_KEY_FIELD: run_key

```

## Canonical interpretation notes

- `EPSILON_REL_FRAC` defines task tolerance as a fraction of the baseline behavior scale.
- Diagnostics thresholds are interpreted in `spec/06_DIAGNOSTICS_RR_SSS_CC.md`.
- `run_key` is deterministic; `run_id` is unique. Determinism checks compare `run_record_hash`, which is computed from deterministic fields.
