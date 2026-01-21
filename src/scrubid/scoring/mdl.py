from __future__ import annotations

from typing import Any


def compute_mdl(circuit: dict[str, Any], canonical: dict[str, Any]) -> float:
    # MDL proxy (spec/02): MDL(C) = w_node*|C| + w_edge*|E| + w_feature*|F| (when present).
    w_node = float(canonical["DIAGNOSTICS"]["MDL_WEIGHT_NODE"])
    w_edge = float(canonical["DIAGNOSTICS"]["MDL_WEIGHT_EDGE"])
    w_feature = float(canonical["DIAGNOSTICS"]["MDL_WEIGHT_FEATURE"])
    components = circuit.get("components", [])
    if not isinstance(components, list):
        raise TypeError("circuit['components'] must be a list")
    edges = circuit.get("edges", [])
    if edges is None:
        edges = []
    if not isinstance(edges, list):
        raise TypeError("circuit['edges'] must be a list")
    features = circuit.get("features", [])
    if features is None:
        features = []
    if not isinstance(features, list):
        raise TypeError("circuit['features'] must be a list")

    return (
        float(w_node) * float(len(components))
        + float(w_edge) * float(len(edges))
        + float(w_feature) * float(len(features))
    )
