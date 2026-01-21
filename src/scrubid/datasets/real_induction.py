from __future__ import annotations

import random
from typing import Any


def build_induction_suite(*, canonical: dict[str, Any], seed: int, model_id: str) -> dict[str, Any]:
    tokens: list[str] = list(canonical["TOKEN_LISTS"]["INDUCTION_TOKENS_CANDIDATES"])
    n_id = int(canonical["DATASETS"]["IND_NUM_PROMPTS_ID"])
    n_ood = int(canonical["DATASETS"]["IND_NUM_PROMPTS_OOD"])

    # We'll filter to single-token candidates later (tokenizer-dependent). For now keep list.
    base_pairs: list[tuple[str, str]] = []
    for a in tokens:
        for b in tokens:
            if a == b:
                continue
            base_pairs.append((a, b))

    total_needed = n_id + n_ood
    repeated: list[tuple[str, str]] = [base_pairs[i % len(base_pairs)] for i in range(total_needed)]
    rng = random.Random(int(seed))
    rng.shuffle(repeated)
    id_pairs = repeated[:n_id]
    ood_pairs = repeated[n_id : n_id + n_ood]

    def choose_distractor(a: str, b: str) -> str:
        for t in tokens:
            if t != a and t != b:
                return t
        raise RuntimeError("No distractor token available")

    def make_clean(a: str, b: str) -> str:
        return f"{a} {b} {a}"

    def make_ood(a: str, b: str) -> str:
        c = choose_distractor(a, b)
        return f"{a} {b} {c} {a}"

    def corr_ood(a: str, b: str) -> str:
        # OOD corruption: keep length fixed and break the final "a -> b" induction cue
        # by replacing the last token with the distractor.
        d = choose_distractor(a, b)
        return f"{a} {b} {d} {d}"

    def corr_clean(a: str, b: str) -> str:
        d = choose_distractor(a, b)
        return f"{a} {b} {d}"

    def rows(pairs: list[tuple[str, str]], *, ood: bool) -> list[dict[str, Any]]:
        out: list[dict[str, Any]] = []
        for a, b in pairs:
            out.append(
                {
                    "prompt_clean": make_ood(a, b) if ood else make_clean(a, b),
                    "prompt_ref": corr_clean(a, b) if not ood else corr_ood(a, b),
                    "a": a,
                    "b": b,
                    "distractor": choose_distractor(a, b),
                    "ood": bool(ood),
                }
            )
        return out

    return {
        "suite_id": "SUITE_REAL_INDUCTION_V1",
        "model_id": model_id,
        "seed": int(seed),
        "splits": {
            "id": rows(id_pairs, ood=False),
            "ood": rows(ood_pairs, ood=True),
        },
        "reference_assignment_id": canonical["REFERENCE"]["REF_ASSIGNMENT_DEFAULT_BY_SUITE"]["SUITE_REAL_INDUCTION_V1"],
        "reference_distribution_id": canonical["REFERENCE"]["REFDIST_DEFAULT_BY_SUITE"]["SUITE_REAL_INDUCTION_V1"],
    }
