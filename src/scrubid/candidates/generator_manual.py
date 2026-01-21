from __future__ import annotations

import random
from typing import Any


def generate_candidates_manual_seed(
    model: Any,
    task_spec: dict[str, Any],
    intervention_family_id: str,
    budget_spec: dict[str, Any],
    *,
    seed: int,
) -> dict[str, Any]:
    suite_id = str(task_spec.get("suite_id", ""))

    component_granularity = str(task_spec["component_granularity"])
    if component_granularity == "node" or suite_id == "SUITE_SYNTH_V1":
        setting: dict[str, Any] = task_spec["synth_setting"]
        components: list[str] = [str(x) for x in setting.get("components", [])]
        aggr = str(setting.get("aggregator_id"))
        redundant_ids: list[str] = [str(x) for x in setting.get("redundant_ids", [])]

        rng = random.Random(int(seed))
        # Sample a deterministic subset of redundant choices to allow SSS variation across replicate seeds.
        k = min(len(redundant_ids), 4)
        sampled = list(redundant_ids)
        rng.shuffle(sampled)
        sampled = sorted(sampled[:k])

        circuits: list[dict[str, Any]] = [
            {"components": [], "edges": [], "score": 0.0, "metadata": {"kind": "empty"}},
            {"components": [aggr], "edges": [], "score": 0.0, "metadata": {"kind": "aggr_only"}},
            {"components": sorted(components), "edges": [], "score": 0.0, "metadata": {"kind": "full"}},
        ]
        for rid in sampled:
            circuits.append(
                {
                    "components": sorted([aggr, rid]),
                    "edges": [],
                    "score": 1.0,
                    "metadata": {"kind": "minimal_sampled"},
                }
            )

        # Deterministic random larger circuit (for near-optimal set diversity).
        if redundant_ids:
            extra_k = min(len(redundant_ids), max(1, len(redundant_ids) // 2))
            extra = list(redundant_ids)
            rng.shuffle(extra)
            circuits.append(
                {
                    "components": sorted([aggr, *extra[:extra_k]]),
                    "edges": [],
                    "score": 2.0,
                    "metadata": {"kind": "aggr_plus_subset"},
                }
            )

        # Deduplicate deterministically.
        seen: set[tuple[str, ...]] = set()
        deduped: list[dict[str, Any]] = []
        for c in circuits:
            key = tuple(c["components"])
            if key in seen:
                continue
            seen.add(key)
            deduped.append(c)

        ranked = [(cid, 0.0) for cid in sorted(components)]
        return {"ranked_components": ranked, "ranked_edges": [], "candidate_circuits": deduped}

    hooks: dict[str, Any] = task_spec["hooks"]
    n_layers = int(hooks["n_layers"])
    n_heads = int(hooks["n_heads"])

    if component_granularity != "head_mlp":
        raise RuntimeError(f"Only head_mlp granularity supported in v1.0.3 baseline, got {component_granularity}")

    all_components: list[str] = [f"H{l}:{h}" for l in range(n_layers) for h in range(n_heads)] + [f"M{l}" for l in range(n_layers)]

    # Heuristic "manual" seeds: last-layer heads, last-layer MLP, and full model.
    last_layer = n_layers - 1
    head_last = [f"H{last_layer}:{h}" for h in range(n_heads)]
    mlp_last = [f"M{last_layer}"]

    circuits = [
        {"components": [], "edges": [], "score": 0.0, "metadata": {"kind": "empty"}},
        {"components": sorted(head_last), "edges": [], "score": 1.0, "metadata": {"kind": "last_layer_heads"}},
        {"components": sorted(head_last + mlp_last), "edges": [], "score": 2.0, "metadata": {"kind": "last_layer_head_mlp"}},
        {"components": sorted(all_components), "edges": [], "score": 3.0, "metadata": {"kind": "full"}},
    ]

    return {"ranked_components": [(c, 0.0) for c in all_components], "ranked_edges": [], "candidate_circuits": circuits}
