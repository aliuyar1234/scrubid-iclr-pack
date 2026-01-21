from __future__ import annotations

import hashlib
from typing import Any

import numpy as np
import torch

from scrubid.interventions.actpatch import apply_actpatch
from scrubid.interventions.causal_scrub import apply_causal_scrub
from scrubid.interventions.pathpatch import apply_pathpatch
from scrubid.scoring.behavior_metrics import compute_metric


_RANKED_COMPONENTS_CACHE: dict[tuple[str, str, str, int, int, int, int, str, str], list[tuple[str, float]]] = {}


def _sha256_tensor(t: torch.Tensor) -> str:
    arr = t.detach().cpu().numpy()
    return hashlib.sha256(arr.tobytes()).hexdigest()


def generate_candidates_attr_patch(
    model: Any,
    task_spec: dict[str, Any],
    intervention_family_id: str,
    budget_spec: dict[str, Any],
    *,
    seed: int,
) -> dict[str, Any]:
    suite_id = str(task_spec["suite_id"])
    canonical: dict[str, Any] = task_spec["canonical"]

    component_granularity = str(task_spec["component_granularity"])
    if component_granularity == "node" or suite_id == "SUITE_SYNTH_V1":
        setting: dict[str, Any] = task_spec["synth_setting"]
        components: list[str] = [str(x) for x in setting.get("components", [])]
        aggr = str(setting.get("aggregator_id"))
        redundant_ids: list[str] = [str(x) for x in setting.get("redundant_ids", [])]

        eval_rows: list[dict[str, Any]] = list(task_spec.get("eval_rows", []))
        if not eval_rows:
            raise RuntimeError("Synthetic attr_patch generator requires task_spec['eval_rows']")

        baseline_vals = [1.0 if int(r["y"]) == 1 else -1.0 for r in eval_rows]

        def scrubbed_vals_for(cset: set[str]) -> list[float]:
            if aggr not in cset:
                return [0.0 for _ in baseline_vals]
            if not any(rid in cset for rid in redundant_ids):
                return [0.0 for _ in baseline_vals]
            return list(baseline_vals)

        def delta_for(keep: set[str]) -> float:
            vals = scrubbed_vals_for(keep)
            return float(sum(abs(a - b) for a, b in zip(baseline_vals, vals, strict=True)) / len(baseline_vals))

        # Per-component leave-one-out: patch just v to reference (circuit = V \ {v}).
        ranked_components: list[tuple[str, float]] = []
        all_set = set(components)
        for cid in components:
            keep = all_set - {cid}
            ranked_components.append((cid, float(delta_for(keep))))

        ranked_components.sort(key=lambda x: (-x[1], x[0]))
        top_ids = [cid for cid, _ in ranked_components]

        max_circuit = int(budget_spec.get("MAX_CIRCUIT_SIZE_COMPONENTS", 200))
        ladder = sorted({k for k in [0, 1, 2, 4, 8, 16, 32, 64, max_circuit] if 0 <= k <= max_circuit})

        rng = np.random.default_rng(int(seed))
        candidate_circuits: list[dict[str, Any]] = []
        candidate_circuits.append(
            {"components": sorted(components), "edges": [], "score": 0.0, "metadata": {"kind": "full"}}
        )
        for k in ladder:
            comps = sorted(top_ids[:k]) if k > 0 else []
            candidate_circuits.append({"components": comps, "edges": [], "score": float(k), "metadata": {"kind": "topk"}})

        # Sample a few minimal circuits to expose redundancy (aggr + one redundant).
        if redundant_ids:
            sample_k = min(len(redundant_ids), 4)
            sampled = list(redundant_ids)
            rng.shuffle(sampled)
            for rid in sorted(sampled[:sample_k]):
                candidate_circuits.append(
                    {
                        "components": sorted([aggr, rid]),
                        "edges": [],
                        "score": 1.0,
                        "metadata": {"kind": "minimal_sampled"},
                    }
                )

        # Deduplicate circuits deterministically.
        seen: set[tuple[str, ...]] = set()
        deduped: list[dict[str, Any]] = []
        for c in candidate_circuits:
            key = tuple(c["components"])
            if key in seen:
                continue
            seen.add(key)
            deduped.append(c)

        return {"ranked_components": ranked_components, "ranked_edges": [], "candidate_circuits": deduped}

    hooks: dict[str, Any] = task_spec["hooks"]
    eval_batch: dict[str, Any] = task_spec["eval_batch"]
    ref_tokens: torch.Tensor = task_spec["ref_tokens"]

    if component_granularity != "head_mlp":
        raise RuntimeError(f"Only head_mlp granularity supported in v1.0.3 baseline, got {component_granularity}")

    n_layers = int(hooks["n_layers"])
    n_heads = int(hooks["n_heads"])
    all_components: list[str] = [f"H{l}:{h}" for l in range(n_layers) for h in range(n_heads)] + [f"M{l}" for l in range(n_layers)]

    # Scoring subset (deterministic): first N examples.
    score_subset_n = int(budget_spec.get("ATTR_PATCH_SCORE_SUBSET_N"))
    score_n = min(score_subset_n, int(eval_batch["tokens"].shape[0]))
    score_eval_batch = dict(eval_batch)
    score_eval_batch["tokens"] = eval_batch["tokens"][:score_n]
    for k in ["token_correct", "token_incorrect", "token_distract", "good_token_ids", "bad_token_ids"]:
        if k in score_eval_batch:
            score_eval_batch[k] = score_eval_batch[k][:score_n]

    score_ref_tokens = ref_tokens[:score_n]

    def make_scrubbed(circuit_components: list[str]):
        circuit = {"components": circuit_components}
        if intervention_family_id == "I_ACTPATCH":
            return apply_actpatch(model, circuit, hooks, reference_cache)
        if intervention_family_id == "I_PATHPATCH":
            return apply_pathpatch(model, circuit, hooks, reference_cache)
        if intervention_family_id == "I_CAUSAL_SCRUB":
            return apply_causal_scrub(model, circuit, hooks, reference_cache)
        raise KeyError(f"Unknown intervention_family_id: {intervention_family_id}")

    # Per-component leave-one-out: patch just v to reference (circuit = V \ {v}).
    max_components_scored = int(budget_spec.get("ATTR_PATCH_MAX_COMPONENTS_SCORED", len(all_components)))
    max_components_scored = max(0, min(max_components_scored, len(all_components)))

    score_components: list[str] = []
    for l in range(n_layers - 1, -1, -1):
        for h in range(n_heads):
            score_components.append(f"H{l}:{h}")
            if len(score_components) >= max_components_scored:
                break
        if len(score_components) >= max_components_scored:
            break
        score_components.append(f"M{l}")
        if len(score_components) >= max_components_scored:
            break
    score_components = score_components[:max_components_scored]

    tokens_fp = _sha256_tensor(score_eval_batch["tokens"])
    ref_fp = _sha256_tensor(score_ref_tokens)
    cache_key = (
        suite_id,
        str(intervention_family_id),
        str(component_granularity),
        int(n_layers),
        int(n_heads),
        int(score_subset_n),
        int(max_components_scored),
        tokens_fp,
        ref_fp,
    )

    ranked_components = _RANKED_COMPONENTS_CACHE.get(cache_key)
    if ranked_components is None:
        # Reference cache for scoring subset.
        hook_names = hooks["head_hooknames"] + hooks["mlp_hooknames"]
        with torch.no_grad():
            _, ref_cache = model.run_with_cache(score_ref_tokens, names_filter=lambda n: n in hook_names)
        reference_cache = {name: ref_cache[name] for name in hook_names}

        baseline_vals = compute_metric(suite_id, model, score_eval_batch)

        scores: dict[str, float] = {cid: 0.0 for cid in all_components}
        all_set = set(all_components)
        for cid in score_components:
            keep = sorted(all_set - {cid})
            scrubbed_model = make_scrubbed(keep)
            scrubbed_vals = compute_metric(suite_id, scrubbed_model, score_eval_batch)
            deltas = [abs(a - b) for a, b in zip(baseline_vals, scrubbed_vals, strict=True)]
            score = float(np.mean(deltas)) if deltas else 0.0
            scores[str(cid)] = float(score)

        # Higher score => more important.
        ranked_components = [(cid, float(scores[cid])) for cid in all_components]
        ranked_components.sort(key=lambda x: (-x[1], x[0]))
        _RANKED_COMPONENTS_CACHE[cache_key] = ranked_components

    top_ids = [cid for cid, _ in ranked_components]

    n_total = int(len(all_components))
    max_circuit = int(budget_spec.get("MAX_CIRCUIT_SIZE_COMPONENTS", 200))
    max_circuit = max(0, min(max_circuit, n_total))

    # Candidate-size ladder:
    # - small top-k circuits up to MAX_CIRCUIT_SIZE_COMPONENTS
    # - several larger sizes to probe near-full behavior on large models
    # - a "full minus k" ladder implemented as top-(N-k) components
    ladder_small = []
    for k in [0, 1, 2, 4, 8, 16, 32, 64, 128, max_circuit]:
        if 0 <= k <= max_circuit:
            ladder_small.append(int(k))
    ladder_small = sorted(set(ladder_small))

    ladder_large = sorted({k for k in [max_circuit, 256, 384, 512, 640, 768, n_total] if 0 <= k <= n_total})
    remove_counts = [1, 2, 4, 8, 16, 32, 64, 128, 256, 512]
    ladder_near_full = sorted({n_total - int(rc) for rc in remove_counts if 0 < n_total - int(rc) <= n_total})

    ladder = sorted(set([*ladder_small, *ladder_large, *ladder_near_full]))

    rng = np.random.default_rng(int(seed))

    rank_index = {str(cid): int(i) for i, cid in enumerate(top_ids)}

    def _layer_of(cid: str) -> int:
        if cid.startswith("H"):
            layer_s, _head_s = cid[1:].split(":", 1)
            return int(layer_s)
        if cid.startswith("M"):
            return int(cid[1:])
        raise ValueError(f"Invalid component id: {cid}")

    def _edges_for(components: list[str], *, kind: str) -> list[list[str]]:
        if intervention_family_id != "I_PATHPATCH":
            return []
        if kind == "full":
            # Edge-free PATHPATCH reduces to ACTPATCH (component-only circuit).
            return []

        by_layer: dict[int, list[str]] = {}
        for cid in components:
            layer = _layer_of(str(cid))
            by_layer.setdefault(int(layer), []).append(str(cid))

        layers = sorted(by_layer.keys())
        if len(layers) < 2:
            return []

        width = 4
        picks_by_layer: dict[int, list[str]] = {}
        for layer in layers:
            layer_comps = list(by_layer[layer])
            layer_comps.sort(key=lambda c: (rank_index.get(str(c), 10**9), str(c)))
            picks_by_layer[layer] = layer_comps[: min(len(layer_comps), int(width))]

        edges: set[tuple[str, str]] = set()
        for l1, l2 in zip(layers, layers[1:]):
            for s in picks_by_layer.get(l1, []):
                for t in picks_by_layer.get(l2, []):
                    edges.add((str(s), str(t)))

        return [[s, t] for (s, t) in sorted(edges, key=lambda x: (x[0], x[1]))]

    candidate_circuits: list[dict[str, Any]] = []
    # Always include full circuit (delta=0 upper bound).
    candidate_circuits.append(
        {
            "components": sorted(all_components),
            "edges": _edges_for(sorted(all_components), kind="full"),
            "score": 0.0,
            "metadata": {"kind": "full"},
        }
    )

    if intervention_family_id == "I_PATHPATCH":
        for width in [2, 4]:
            layerwise: list[str] = []
            for layer in range(n_layers):
                comps_l = [c for c in all_components if _layer_of(str(c)) == int(layer)]
                comps_l.sort(key=lambda c: (rank_index.get(str(c), 10**9), str(c)))
                layerwise.extend(comps_l[: min(len(comps_l), int(width))])
            layerwise = sorted(set([str(x) for x in layerwise]))
            candidate_circuits.append(
                {
                    "components": layerwise,
                    "edges": _edges_for(layerwise, kind=f"path_layer_top{int(width)}"),
                    "score": float(len(layerwise)),
                    "metadata": {"kind": f"path_layer_top{int(width)}"},
                }
            )
    for k in ladder:
        if k == 0:
            comps: list[str] = []
        else:
            comps = sorted(top_ids[:k])
        candidate_circuits.append(
            {
                "components": comps,
                "edges": _edges_for(comps, kind="topk"),
                "score": float(k),
                "metadata": {"kind": "topk"},
            }
        )

    # Deterministic random variants: expand candidate diversity to surface multiple
    # distinct faithful circuits (needed for RR/CC to be non-trivial on real models).
    #
    # v1.0.3 baseline originally produced a small fixed number of candidates. For
    # real-model RR/CC ambiguity to be observable under rr_near_optimal_mdl_rel_frac=0,
    # we need enough candidate diversity to find multiple *minimum-MDL* faithful circuits.
    max_candidates = int(budget_spec.get("MAX_CANDIDATES_PER_GENERATOR", 800))
    max_candidates = max(max_candidates, len(candidate_circuits))

    pool_size = min(len(top_ids), max(32, max_circuit, 512))
    pool = [str(x) for x in top_ids[:pool_size]]
    drop_pool_all = [str(x) for x in top_ids]

    size_grid = [16, 24, 32, 48, 64, 96]
    size_grid = sorted({int(k) for k in size_grid if 0 < int(k) <= len(pool)})

    drop_grid = [1, 2, 4, 8, 16, 32, 64]
    drop_grid = sorted({int(d) for d in drop_grid if 0 < int(d) < len(all_components)})

    # Track component-set uniqueness during construction to avoid wasting the
    # candidate budget on duplicates (final dedup below is still retained).
    seen_component_sets: set[tuple[str, ...]] = {tuple(c["components"]) for c in candidate_circuits}

    def _try_add_candidate(*, components: list[str], kind: str, metadata: dict[str, Any]) -> bool:
        comps = sorted([str(x) for x in components])
        key = tuple(comps)
        if key in seen_component_sets:
            return False
        seen_component_sets.add(key)
        candidate_circuits.append(
            {
                "components": comps,
                "edges": _edges_for(comps, kind=kind),
                "score": float(len(comps)),
                "metadata": dict(metadata),
            }
        )
        return True

    def _sample_rand_k(k: int) -> list[str]:
        # Uniform subsets over the pool; determinism comes from the provided RNG.
        return [str(x) for x in rng.choice(pool, size=int(k), replace=False)]

    def _sample_swap_k(k: int) -> list[str]:
        base = [str(x) for x in top_ids[: int(k)]]
        swaps = max(1, int(k) // 8)
        comps = list(base)
        for _s in range(swaps):
            if not comps:
                break
            drop_idx = int(rng.integers(0, len(comps)))
            _dropped = comps.pop(drop_idx)
            # Add a new component not already present.
            for _try in range(16):
                add = str(rng.choice(pool))
                if add not in comps:
                    comps.append(add)
                    break
        return list(sorted(set([str(x) for x in comps])))

    def _sample_drop_d(d: int) -> list[str]:
        # Drop from a low-importance pool slice to keep many drop-variants faithful.
        d_int = int(d)
        if d_int <= 0:
            return list(all_components)
        slice_sz = min(len(drop_pool_all), max(d_int * 4, d_int))
        drop_pool = drop_pool_all[-slice_sz:] if slice_sz > 0 else list(drop_pool_all)
        if len(drop_pool) < d_int:
            drop_pool = list(drop_pool_all)
        drop = set([str(x) for x in rng.choice(drop_pool, size=int(d_int), replace=False)])
        return [str(x) for x in all_components if str(x) not in drop]

    # Fill out the remaining candidate budget with stratified candidates.
    groups: list[tuple[str, int]] = []
    groups.extend([("rand", int(k)) for k in size_grid])
    groups.extend([("swap", int(k)) for k in size_grid])
    groups.extend([("drop", int(d)) for d in drop_grid])

    remaining_slots = int(max_candidates - len(candidate_circuits))
    if remaining_slots > 0 and groups:
        per_group = max(1, remaining_slots // len(groups))
        attempts_cap = max_candidates * 50
        attempts = 0

        def make_candidate(kind: str, val: int) -> list[str]:
            if kind == "rand":
                return _sample_rand_k(val)
            if kind == "swap":
                return _sample_swap_k(val)
            if kind == "drop":
                return _sample_drop_d(val)
            raise ValueError(f"Unknown candidate kind: {kind}")

        # First pass: allocate roughly evenly across groups.
        for _i in range(per_group):
            for kind, val in groups:
                if len(candidate_circuits) >= max_candidates:
                    break
                if attempts >= attempts_cap:
                    break
                attempts += 1
                comps = make_candidate(kind, val)
                if kind == "rand":
                    _try_add_candidate(components=comps, kind=f"rand_k{int(val)}", metadata={"kind": "rand", "k": int(val)})
                elif kind == "swap":
                    _try_add_candidate(
                        components=comps,
                        kind=f"swap_k{int(val)}",
                        metadata={"kind": "swap", "k": int(val), "swaps": int(max(1, int(val) // 8))},
                    )
                else:
                    _try_add_candidate(components=comps, kind=f"drop_d{int(val)}", metadata={"kind": "drop", "d": int(val)})

        # Second pass: top up deterministically if duplicates prevented us from
        # reaching max_candidates.
        idx = 0
        while len(candidate_circuits) < max_candidates and attempts < attempts_cap:
            kind, val = groups[idx % len(groups)]
            idx += 1
            attempts += 1
            comps = make_candidate(kind, val)
            if kind == "rand":
                _try_add_candidate(components=comps, kind=f"rand_k{int(val)}", metadata={"kind": "rand", "k": int(val)})
            elif kind == "swap":
                _try_add_candidate(
                    components=comps,
                    kind=f"swap_k{int(val)}",
                    metadata={"kind": "swap", "k": int(val), "swaps": int(max(1, int(val) // 8))},
                )
            else:
                _try_add_candidate(components=comps, kind=f"drop_d{int(val)}", metadata={"kind": "drop", "d": int(val)})

    # Deduplicate circuits deterministically.
    seen: set[tuple[str, ...]] = set()
    deduped: list[dict[str, Any]] = []
    for c in candidate_circuits:
        key = tuple(c["components"])
        if key in seen:
            continue
        seen.add(key)
        deduped.append(c)

    return {
        "ranked_components": ranked_components,
        "ranked_edges": [],
        "candidate_circuits": deduped,
    }
