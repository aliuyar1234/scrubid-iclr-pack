from __future__ import annotations

from collections import defaultdict, deque
from typing import Any, Callable

from scrubid.interventions.actpatch import apply_actpatch


def _parse_component_id(cid: str) -> tuple[str, int, int | None]:
    if cid.startswith("H"):
        # H{layer}:{head}
        layer_s, head_s = cid[1:].split(":", 1)
        return ("H", int(layer_s), int(head_s))
    if cid.startswith("M"):
        return ("M", int(cid[1:]), None)
    raise ValueError(f"Invalid component id: {cid}")


def _canonical_edges(edges: Any) -> list[tuple[str, str]]:
    if edges is None:
        return []
    if not isinstance(edges, list):
        raise TypeError("circuit['edges'] must be a list")
    out: set[tuple[str, str]] = set()
    for e in edges:
        if isinstance(e, dict) and "src" in e and "dst" in e:
            out.add((str(e["src"]), str(e["dst"])))
            continue
        if isinstance(e, (list, tuple)) and len(e) == 2:
            out.add((str(e[0]), str(e[1])))
            continue
        raise TypeError("Each edge must be [src, dst] (or {'src':..., 'dst':...})")
    return sorted(out, key=lambda x: (x[0], x[1]))


def apply_pathpatch(
    model: Any,
    circuit: dict[str, Any],
    hooks: Any,
    reference_cache: dict[str, Any],
) -> Callable[[Any], Any]:
    """
    PATHPATCH: edge-aware scrubbing.

    For v1.0.3, the edge semantics are implemented as a deterministic restriction
    on which components are preserved:

    - If no edges are provided, PATHPATCH reduces to ACTPATCH (component-only circuit).
    - If edges are provided, we keep only components that can reach a sink component
      in the maximum layer present in the circuit (reachability in the directed edge set).
    """
    components = [str(x) for x in circuit.get("components", [])]
    edges = _canonical_edges(circuit.get("edges", []))
    if not edges:
        return apply_actpatch(model, {"components": components}, hooks, reference_cache)

    comp_set = set(components)
    edges = [(s, t) for (s, t) in edges if s in comp_set and t in comp_set]
    if not edges:
        return apply_actpatch(model, {"components": components}, hooks, reference_cache)

    layers: list[int] = []
    by_layer: dict[int, list[str]] = defaultdict(list)
    for cid in sorted(comp_set):
        _, layer, _ = _parse_component_id(cid)
        layers.append(int(layer))
        by_layer[int(layer)].append(cid)
    if not layers:
        return apply_actpatch(model, {"components": []}, hooks, reference_cache)

    max_layer = int(max(layers))
    sinks = set(by_layer.get(max_layer, []))
    if not sinks:
        return apply_actpatch(model, {"components": components}, hooks, reference_cache)

    rev: dict[str, set[str]] = defaultdict(set)
    for s, t in edges:
        rev[str(t)].add(str(s))

    active: set[str] = set(sinks)
    q: deque[str] = deque(sorted(sinks))
    while q:
        cur = q.popleft()
        for prev in sorted(rev.get(cur, set())):
            if prev in active:
                continue
            active.add(prev)
            q.append(prev)

    return apply_actpatch(model, {"components": sorted(active)}, hooks, reference_cache)
