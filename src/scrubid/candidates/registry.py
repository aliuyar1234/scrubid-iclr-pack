from __future__ import annotations

from typing import Any, Callable

from scrubid.candidates.generator_acdc import generate_candidates_acdc
from scrubid.candidates.generator_attr_patch import generate_candidates_attr_patch
from scrubid.candidates.generator_atpstar import generate_candidates_atpstar
from scrubid.candidates.generator_manual import generate_candidates_manual_seed
from scrubid.candidates.generator_random_k import generate_candidates_random_k
from scrubid.candidates.generator_sparse_feature import generate_candidates_sparse_feature


def get_generator(generator_id: str) -> Callable[..., dict[str, Any]]:
    mapping = {
        "G_ACDC": generate_candidates_acdc,
        "G_ATTR_PATCH": generate_candidates_attr_patch,
        "G_ATPSTAR": generate_candidates_atpstar,
        "G_SPARSE_FEATURE": generate_candidates_sparse_feature,
        "G_MANUAL_SEED": generate_candidates_manual_seed,
        "G_RANDOM_K": generate_candidates_random_k,
    }
    if generator_id not in mapping:
        raise KeyError(f"Unknown generator_id: {generator_id}")
    return mapping[generator_id]
