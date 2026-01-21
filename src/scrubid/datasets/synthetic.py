from __future__ import annotations

import random
from typing import Any


def _rng(seed: int) -> random.Random:
    return random.Random(int(seed))


def _node_id(*, template_id: str, instance_id: int, local_id: str) -> str:
    return f"node_{template_id}_{int(instance_id)}_{str(local_id)}"


def _setting_id(*, template_id: str, redundancy_factor: int, instance_id: int) -> str:
    return f"setting_{template_id}_{int(redundancy_factor)}_{int(instance_id)}"


def _xor_examples(*, n: int, rng: random.Random) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for _ in range(int(n)):
        a = int(rng.randrange(2))
        b = int(rng.randrange(2))
        c = int(rng.randrange(2))
        y = int((a ^ b) & 1)
        out.append({"a": a, "b": b, "c": c, "y": y})
    return out


def _compare_examples(*, n: int, rng: random.Random, redundancy_factor: int) -> list[dict[str, Any]]:
    # Deterministic integer range keyed to redundancy_factor to avoid inventing a new constant.
    lo = -int(redundancy_factor)
    hi = int(redundancy_factor)
    out: list[dict[str, Any]] = []
    for _ in range(int(n)):
        p = int(rng.randint(lo, hi))
        q = int(rng.randint(lo, hi))
        y = int(p > q)
        out.append({"p": p, "q": q, "y": y})
    return out


def _induction_examples(*, n: int, rng: random.Random, tokens: list[str]) -> list[dict[str, Any]]:
    # Synthetic induction proxy: binary label depends on token equality.
    out: list[dict[str, Any]] = []
    for _ in range(int(n)):
        a = str(rng.choice(tokens))
        b = str(rng.choice(tokens))
        y = int(a == b)
        out.append({"A": a, "B": b, "y": y})
    return out


def _build_setting(
    *,
    canonical: dict[str, Any],
    template_id: str,
    redundancy_factor: int,
    instance_id: int,
    seed: int,
) -> dict[str, Any]:
    n = int(canonical["DATASETS"]["SYNTH_NUM_INSTANCES"])
    rng = _rng(int(seed) + int(instance_id))

    aggr = _node_id(template_id=template_id, instance_id=instance_id, local_id="aggr")
    redundants = [
        _node_id(template_id=template_id, instance_id=instance_id, local_id=f"r{i}") for i in range(int(redundancy_factor))
    ]
    components = [aggr, *redundants]

    if template_id == "XOR":
        examples = _xor_examples(n=n, rng=rng)
    elif template_id == "COMPARE":
        examples = _compare_examples(n=n, rng=rng, redundancy_factor=redundancy_factor)
    elif template_id == "INDUCTION":
        tokens = list(canonical["TOKEN_LISTS"]["INDUCTION_TOKENS_CANDIDATES"])
        examples = _induction_examples(n=n, rng=rng, tokens=tokens)
    else:
        raise KeyError(f"Unknown synthetic template_id: {template_id}")

    # Deterministic "clean/reference" markers to support reference dataset generation.
    eval_rows = [{**r, "is_reference": False} for r in examples]
    ref_rows = [{**r, "is_reference": True} for r in examples]

    return {
        "setting_id": _setting_id(template_id=template_id, redundancy_factor=redundancy_factor, instance_id=instance_id),
        "template_id": template_id,
        "redundancy_factor": int(redundancy_factor),
        "instance_id": int(instance_id),
        "aggregator_id": aggr,
        "redundant_ids": redundants,
        "components": components,
        "ground_truth_circuit_primary": sorted([aggr, redundants[0]]) if redundants else sorted([aggr]),
        "redundant_groups": [sorted(redundants)] if redundants else [],
        "equivalence_class_size": int(redundancy_factor),
        # For node-only circuits, planted MDL is proportional to |C|.
        "planted_mdl_best_components": 2,
        "splits": {"eval": eval_rows, "ref": ref_rows},
    }


def build_synth_suite(*, canonical: dict[str, Any], seed: int, model_id: str) -> dict[str, Any]:
    templates: list[str] = list(canonical["DATASETS"]["SYNTH_TEMPLATES"])
    factors: list[int] = [int(x) for x in canonical["DATASETS"]["SYNTH_REDUNDANCY_FACTORS"]]

    settings: list[dict[str, Any]] = []
    instance_id = 0
    for t in templates:
        for r in factors:
            settings.append(
                _build_setting(
                    canonical=canonical,
                    template_id=str(t),
                    redundancy_factor=int(r),
                    instance_id=int(instance_id),
                    seed=int(seed),
                )
            )
            instance_id += 1

    suite_id = "SUITE_SYNTH_V1"
    return {
        "suite_id": suite_id,
        "model_id": model_id,
        "seed": int(seed),
        "settings": settings,
        "reference_assignment_id": canonical["REFERENCE"]["REF_ASSIGNMENT_DEFAULT_BY_SUITE"][suite_id],
        "reference_distribution_id": canonical["REFERENCE"]["REFDIST_DEFAULT_BY_SUITE"][suite_id],
    }
