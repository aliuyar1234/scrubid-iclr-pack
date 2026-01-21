from __future__ import annotations

import math
from typing import Any

import numpy as np


def generate_candidates_random_k(
    model: Any,
    task_spec: dict[str, Any],
    intervention_family_id: str,
    budget_spec: dict[str, Any],
    *,
    seed: int,
) -> dict[str, Any]:
    """
    Deterministic Random-k baseline candidate generator.

    This generator ignores model internals (no attribution/gradients) and proposes
    circuits by sampling random subsets of components at a range of sizes.
    """
    # Synthetic suite: sample over the provided component list.
    suite_id = str(task_spec.get("suite_id"))
    component_granularity = str(task_spec.get("component_granularity"))

    rng = np.random.default_rng(int(seed))

    if component_granularity == "node" or suite_id == "SUITE_SYNTH_V1":
        setting: dict[str, Any] = task_spec["synth_setting"]
        all_components = sorted([str(x) for x in setting.get("components", [])])
        if not all_components:
            raise RuntimeError("Synthetic random_k generator requires non-empty setting['components']")

        max_candidates = int(
            budget_spec.get("RANDOM_K_MAX_CANDIDATES", budget_spec.get("MAX_CANDIDATES_PER_GENERATOR", 500))
        )
        max_candidates = max(2, max_candidates)
        max_circuit = int(budget_spec.get("MAX_CIRCUIT_SIZE_COMPONENTS", len(all_components)))
        max_circuit = max(0, min(max_circuit, len(all_components)))

        ladder = sorted({k for k in [0, 1, 2, 4, 8, 16, 32, 64, max_circuit] if 0 <= k <= max_circuit})

        candidate_circuits: list[dict[str, Any]] = []
        candidate_circuits.append({"components": sorted(all_components), "edges": [], "metadata": {"kind": "full"}})
        candidate_circuits.append({"components": [], "edges": [], "metadata": {"kind": "empty"}})

        seen: set[tuple[str, ...]] = {tuple(c["components"]) for c in candidate_circuits}
        sizes = [k for k in ladder if 0 < k < len(all_components)] or [min(1, len(all_components))]

        # Cap requested candidates by the number of unique circuits representable
        # by this sampler (full + empty + the allowed subset sizes).
        n = int(len(all_components))
        max_unique = 2
        for k in sizes:
            max_unique += int(math.comb(n, int(k)))
            if max_unique >= max_candidates:
                max_unique = max_candidates
                break
        max_candidates = min(max_candidates, max_unique)

        while len(candidate_circuits) < max_candidates:
            k = int(rng.choice(sizes))
            idx = rng.choice(len(all_components), size=int(k), replace=False)
            comps = sorted([all_components[int(i)] for i in idx])
            key = tuple(comps)
            if key in seen:
                continue
            seen.add(key)
            candidate_circuits.append({"components": comps, "edges": [], "metadata": {"kind": "random_k", "k": int(k)}})

        ranked_components = [(cid, 0.0) for cid in all_components]
        return {"ranked_components": ranked_components, "ranked_edges": [], "candidate_circuits": candidate_circuits}

    # Real suites (head_mlp): sample over all head + mlp components derived from hooks.
    hooks: dict[str, Any] = task_spec["hooks"]
    if component_granularity != "head_mlp":
        raise RuntimeError(f"Only head_mlp supported for real random_k baseline, got {component_granularity}")

    n_layers = int(hooks["n_layers"])
    n_heads = int(hooks["n_heads"])
    all_components: list[str] = (
        [f"H{l}:{h}" for l in range(n_layers) for h in range(n_heads)] + [f"M{l}" for l in range(n_layers)]
    )
    all_components = sorted(all_components)
    if not all_components:
        raise RuntimeError("Real random_k generator requires non-empty component set")

    max_candidates = int(
        budget_spec.get("RANDOM_K_MAX_CANDIDATES", budget_spec.get("MAX_CANDIDATES_PER_GENERATOR", 800))
    )
    max_candidates = max(2, max_candidates)
    max_circuit = int(budget_spec.get("MAX_CIRCUIT_SIZE_COMPONENTS", len(all_components)))
    max_circuit = max(0, min(max_circuit, len(all_components)))

    ladder = sorted({k for k in [0, 1, 2, 4, 8, 16, 32, 64, 128, max_circuit] if 0 <= k <= max_circuit})

    candidate_circuits: list[dict[str, Any]] = []
    candidate_circuits.append({"components": sorted(all_components), "edges": [], "metadata": {"kind": "full"}})
    candidate_circuits.append({"components": [], "edges": [], "metadata": {"kind": "empty"}})

    seen: set[tuple[str, ...]] = {tuple(c["components"]) for c in candidate_circuits}
    sizes = [k for k in ladder if 0 < k < len(all_components)] or [min(1, len(all_components))]

    n = int(len(all_components))
    max_unique = 2
    for k in sizes:
        max_unique += int(math.comb(n, int(k)))
        if max_unique >= max_candidates:
            max_unique = max_candidates
            break
    max_candidates = min(max_candidates, max_unique)

    while len(candidate_circuits) < max_candidates:
        k = int(rng.choice(sizes))
        idx = rng.choice(len(all_components), size=int(k), replace=False)
        comps = sorted([all_components[int(i)] for i in idx])
        key = tuple(comps)
        if key in seen:
            continue
        seen.add(key)
        candidate_circuits.append({"components": comps, "edges": [], "metadata": {"kind": "random_k", "k": int(k)}})

    ranked_components = [(cid, 0.0) for cid in all_components]
    return {"ranked_components": ranked_components, "ranked_edges": [], "candidate_circuits": candidate_circuits}
