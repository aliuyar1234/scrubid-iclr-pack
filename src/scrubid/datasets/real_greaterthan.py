from __future__ import annotations

import random
from typing import Any


def build_greaterthan_suite(*, canonical: dict[str, Any], seed: int, model_id: str) -> dict[str, Any]:
    nouns: list[str] = list(canonical["TOKEN_LISTS"]["GREATERTHAN_NOUNS"])
    yy_values: list[int] = [int(x) for x in canonical["DATASETS"]["GT_YY_VALUES_ID"]]
    prefix_id = str(canonical["TOKEN_LISTS"]["GREATERTHAN_CENTURY_PREFIX_ID"])
    prefix_ood = str(canonical["TOKEN_LISTS"]["GREATERTHAN_CENTURY_PREFIX_OOD"])
    n_id = int(canonical["DATASETS"]["GT_NUM_PROMPTS_ID"])
    n_ood = int(canonical["DATASETS"]["GT_NUM_PROMPTS_OOD"])

    base: list[tuple[str, int]] = []
    for noun in sorted(nouns):
        for yy in yy_values:
            base.append((noun, int(yy)))

    # Repeat deterministically to reach required sizes.
    total_needed = n_id + n_ood
    repeated: list[tuple[str, int]] = [base[i % len(base)] for i in range(total_needed)]
    rng = random.Random(int(seed))
    rng.shuffle(repeated)
    id_rows = repeated[:n_id]
    ood_rows = repeated[n_id : n_id + n_ood]

    yy_index = {yy: i for i, yy in enumerate(yy_values)}

    def yy_to_str(yy: int) -> str:
        return f"{yy:02d}"

    def corr_yy(yy: int) -> int:
        # Operational rule per spec/08: YY_mirror is candidate at index (k-1) with wraparound.
        k = yy_index[int(yy)]
        return int(yy_values[(k - 1) % len(yy_values)])

    def make_prompt(noun: str, prefix: str, yy: int) -> str:
        return f"The {noun} lasted from the year {prefix}{yy_to_str(yy)} to the year {prefix}"

    def rows(tups: list[tuple[str, int]], *, prefix: str) -> list[dict[str, Any]]:
        out: list[dict[str, Any]] = []
        for noun, yy in tups:
            out.append(
                {
                    "prompt_clean": make_prompt(noun, prefix, yy),
                    "prompt_ref": make_prompt(noun, prefix, corr_yy(yy)),
                    "yy": int(yy),
                    "prefix": prefix,
                }
            )
        return out

    return {
        "suite_id": "SUITE_REAL_GREATERTHAN_V1",
        "model_id": model_id,
        "seed": int(seed),
        "splits": {
            "id": rows(id_rows, prefix=prefix_id),
            "ood": rows(ood_rows, prefix=prefix_ood),
        },
        "yy_values": yy_values,
        "reference_assignment_id": canonical["REFERENCE"]["REF_ASSIGNMENT_DEFAULT_BY_SUITE"]["SUITE_REAL_GREATERTHAN_V1"],
        "reference_distribution_id": canonical["REFERENCE"]["REFDIST_DEFAULT_BY_SUITE"]["SUITE_REAL_GREATERTHAN_V1"],
    }


def build_greaterthan_yn_suite(*, canonical: dict[str, Any], seed: int, model_id: str) -> dict[str, Any]:
    nouns: list[str] = list(canonical["TOKEN_LISTS"]["GREATERTHAN_NOUNS"])
    yy_values: list[int] = [int(x) for x in canonical["DATASETS"]["GT_YY_VALUES_ID"]]
    prefix_id = str(canonical["TOKEN_LISTS"]["GREATERTHAN_CENTURY_PREFIX_ID"])
    prefix_ood = str(canonical["TOKEN_LISTS"]["GREATERTHAN_CENTURY_PREFIX_OOD"])
    n_id = int(canonical["DATASETS"]["GT_NUM_PROMPTS_ID"])
    n_ood = int(canonical["DATASETS"]["GT_NUM_PROMPTS_OOD"])

    base: list[tuple[str, int, int]] = []
    for noun in sorted(nouns):
        for yy in yy_values:
            for zz in yy_values:
                if int(zz) == int(yy):
                    continue
                base.append((noun, int(yy), int(zz)))

    # Repeat deterministically to reach required sizes.
    total_needed = n_id + n_ood
    repeated: list[tuple[str, int, int]] = [base[i % len(base)] for i in range(total_needed)]
    rng = random.Random(int(seed))
    rng.shuffle(repeated)
    id_rows = repeated[:n_id]
    ood_rows = repeated[n_id : n_id + n_ood]

    def yy_to_str(yy: int) -> str:
        return f"{yy:02d}"

    def make_prompt(noun: str, prefix: str, yy: int, zz: int) -> str:
        # Tokenizer-agnostic: the model predicts a single-token yes/no answer.
        return (
            f"The {noun} lasted from the year {prefix}{yy_to_str(yy)} to the year {prefix}{yy_to_str(zz)}."
            f" Is the end year greater than the start year? Answer:"
        )

    def rows(tups: list[tuple[str, int, int]], *, prefix: str) -> list[dict[str, Any]]:
        out: list[dict[str, Any]] = []
        for noun, yy, zz in tups:
            out.append(
                {
                    "prompt_clean": make_prompt(noun, prefix, yy, zz),
                    # Corruption: swap the years to flip the answer while keeping surface form similar.
                    "prompt_ref": make_prompt(noun, prefix, zz, yy),
                    "yy": int(yy),
                    "zz": int(zz),
                    "prefix": prefix,
                    "is_greater": bool(int(zz) > int(yy)),
                }
            )
        return out

    return {
        "suite_id": "SUITE_REAL_GREATERTHAN_YN_V1",
        "model_id": model_id,
        "seed": int(seed),
        "splits": {
            "id": rows(id_rows, prefix=prefix_id),
            "ood": rows(ood_rows, prefix=prefix_ood),
        },
        "yy_values": yy_values,
        "reference_assignment_id": canonical["REFERENCE"]["REF_ASSIGNMENT_DEFAULT_BY_SUITE"]["SUITE_REAL_GREATERTHAN_YN_V1"],
        "reference_distribution_id": canonical["REFERENCE"]["REFDIST_DEFAULT_BY_SUITE"]["SUITE_REAL_GREATERTHAN_YN_V1"],
    }
