from __future__ import annotations

from typing import Any

import numpy as np

from scrubid.scoring.behavior_metrics import compute_metric


def compute_delta(suite_id: str, model: Any, scrubbed_model: Any, dataset_eval: Any) -> float:
    baseline_vals = compute_metric(suite_id, model, dataset_eval)
    scrubbed_vals = compute_metric(suite_id, scrubbed_model, dataset_eval)
    if len(baseline_vals) != len(scrubbed_vals):
        raise ValueError("Baseline and scrubbed metric lengths differ")
    diffs = [abs(a - b) for a, b in zip(baseline_vals, scrubbed_vals, strict=True)]
    return float(np.mean(diffs)) if diffs else 0.0


def compute_epsilon(suite_id: str, model: Any, dataset_eval: Any, canonical: dict[str, Any]) -> float:
    vals = compute_metric(suite_id, model, dataset_eval)
    s0 = float(np.mean([abs(v) for v in vals])) if vals else 0.0
    eps_abs = float(canonical["DIAGNOSTICS"]["EPSILON_ABS_MIN"])
    eps_rel = float(canonical["DIAGNOSTICS"]["EPSILON_REL_FRAC"])
    return max(eps_abs, eps_rel * s0)
