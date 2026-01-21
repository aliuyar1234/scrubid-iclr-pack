from __future__ import annotations

from typing import Any


def generate_candidates_acdc(
    model: Any,
    task_spec: dict[str, Any],
    intervention_family_id: str,
    budget_spec: dict[str, Any],
    *,
    seed: int,
) -> dict[str, Any]:
    raise RuntimeError("G_ACDC is optional and not implemented in v1.0.3 baseline")

