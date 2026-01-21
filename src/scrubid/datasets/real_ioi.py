from __future__ import annotations

import random
from typing import Any


def _deterministic_shuffle(items: list[tuple[Any, ...]], seed: int) -> list[tuple[Any, ...]]:
    rng = random.Random(int(seed))
    items2 = list(items)
    rng.shuffle(items2)
    return items2


def build_ioi_suite(*, canonical: dict[str, Any], seed: int, model_id: str) -> dict[str, Any]:
    names: list[str] = list(canonical["TOKEN_LISTS"]["IOI_NAME_CANDIDATES"])
    places: list[str] = list(canonical["TOKEN_LISTS"]["IOI_PLACES"])
    objects: list[str] = list(canonical["TOKEN_LISTS"]["IOI_OBJECTS"])
    n_id = int(canonical["DATASETS"]["IOI_NUM_PROMPTS_ID"])
    n_ood = int(canonical["DATASETS"]["IOI_NUM_PROMPTS_OOD"])

    tuples: list[tuple[str, str, str, str]] = []
    for a in sorted(names):
        for b in sorted(names):
            if a == b:
                continue
            for place in sorted(places):
                for obj in sorted(objects):
                    tuples.append((a, b, place, obj))

    tuples = _deterministic_shuffle(tuples, seed)
    # Deterministically allocate ID then OOD without overlap.
    id_tuples = tuples[:n_id]
    ood_tuples = tuples[n_id : n_id + n_ood]

    def make_clean(a: str, b: str, place: str, obj: str) -> str:
        return f"When {a} and {b} went to the {place}, {a} gave a {obj} to"

    def make_corr(a: str, b: str, place: str, obj: str) -> str:
        # Corr_IOI: swap the final-clause subject.
        return f"When {a} and {b} went to the {place}, {b} gave a {obj} to"

    def rows(tups: list[tuple[str, str, str, str]]) -> list[dict[str, Any]]:
        out: list[dict[str, Any]] = []
        for (a, b, place, obj) in tups:
            out.append(
                {
                    "prompt_clean": make_clean(a, b, place, obj),
                    "prompt_ref": make_corr(a, b, place, obj),
                    "name_correct": b,
                    "name_incorrect": a,
                }
            )
        return out

    return {
        "suite_id": "SUITE_REAL_IOI_V1",
        "model_id": model_id,
        "seed": int(seed),
        "splits": {
            "id": rows(id_tuples),
            "ood": rows(ood_tuples),
        },
        "reference_assignment_id": canonical["REFERENCE"]["REF_ASSIGNMENT_DEFAULT_BY_SUITE"]["SUITE_REAL_IOI_V1"],
        "reference_distribution_id": canonical["REFERENCE"]["REFDIST_DEFAULT_BY_SUITE"]["SUITE_REAL_IOI_V1"],
    }
