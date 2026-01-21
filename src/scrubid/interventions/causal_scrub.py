from __future__ import annotations

from typing import Any

from scrubid.interventions.actpatch import apply_actpatch


def apply_causal_scrub(model: Any, circuit: dict[str, Any], hooks: Any, reference_cache: dict[str, Any]) -> Any:
    # v1.0.3 baseline: suite-specific causal-variable resampling is approximated
    # by the same reference dataset used for activation patching.
    return apply_actpatch(model, circuit, hooks, reference_cache)
