from __future__ import annotations

from typing import Any


def generate_candidates_sparse_feature(
    model: Any,
    task_spec: dict[str, Any],
    intervention_family_id: str,
    budget_spec: dict[str, Any],
    *,
    seed: int,
) -> dict[str, Any]:
    raise RuntimeError("G_SPARSE_FEATURE is optional and not implemented in v1.0.3 baseline")

