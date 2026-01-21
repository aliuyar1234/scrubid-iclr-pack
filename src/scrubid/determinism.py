from __future__ import annotations

import os
import platform
from dataclasses import dataclass
from typing import Any

import numpy as np
import torch

from scrubid.canonical import get_canonical
from scrubid.hashing import canonical_json_bytes, decimal_str, sha256_hex, sha256_uint32


class DeterminismError(RuntimeError):
    pass


@dataclass(frozen=True)
class SeedBundle:
    seed_global: int
    seed_effective: int
    seed_reference_pairing: int
    seed_bootstrap: int

    def seed_replicate(self, r: int) -> int:
        return self._seed_replicate(r)

    def _seed_replicate(self, r: int) -> int:
        raise RuntimeError("seed_replicate not bound; use derive_seeds()")


def configure_determinism(*, deterministic_mode: bool, canonical: dict[str, Any]) -> None:
    if not deterministic_mode:
        return

    # Needed for deterministic CuBLAS (PyTorch requirement). Some shells may
    # define the variable as an empty string; treat that as unset.
    if not os.environ.get("CUBLAS_WORKSPACE_CONFIG"):
        os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"

    # These settings are required by configs/determinism.yaml (resolved via canonical keys).
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark = False
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    # Ensure hash randomization does not affect any unordered iteration in code.
    os.environ.setdefault("PYTHONHASHSEED", "0")


def _run_key_thresholds(canonical: dict[str, Any], *, epsilon_abs: float, tau_abs: float) -> dict[str, Any]:
    rr_rel = float(get_canonical(canonical, "DIAGNOSTICS.RR_NEAR_OPTIMAL_MDL_REL_FRAC"))
    return {
        "epsilon_abs": decimal_str(epsilon_abs),
        "tau_abs": decimal_str(tau_abs),
        "rr_near_optimal_mdl_rel_frac": decimal_str(rr_rel),
    }


def compute_run_key(
    *,
    canonical: dict[str, Any],
    suite_id: str,
    experiment_id: str,
    model_id: str,
    model_revision: str,
    component_granularity: str,
    intervention_family_id: str,
    candidate_generator_ids: list[str],
    reference_distribution_id: str,
    reference_assignment_id: str,
    resolved_config_hashes: dict[str, str],
    dataset_fingerprints: dict[str, str],
    budgets: dict[str, Any],
    epsilon_abs: float,
    tau_abs: float,
) -> str:
    run_key_obj: dict[str, Any] = {
        "project_id": get_canonical(canonical, "PROJECT_ID"),
        "project_version": get_canonical(canonical, "PROJECT_VERSION"),
        "suite_id": suite_id,
        "experiment_id": experiment_id,
        "model_id": model_id,
        "model_revision": model_revision or "unknown",
        "component_granularity": component_granularity,
        "intervention_family_id": intervention_family_id,
        "candidate_generator_ids": sorted(candidate_generator_ids),
        "reference_distribution_id": reference_distribution_id,
        "reference_assignment_id": reference_assignment_id,
        "resolved_config_hashes": dict(sorted(resolved_config_hashes.items())),
        "dataset_fingerprints": dict(sorted(dataset_fingerprints.items())),
        "thresholds": _run_key_thresholds(canonical, epsilon_abs=epsilon_abs, tau_abs=tau_abs),
        "budgets": budgets,
    }
    b = canonical_json_bytes(run_key_obj, canonical)
    return sha256_hex(b)


def derive_seeds(*, canonical: dict[str, Any], seed_global: int, run_key: str) -> SeedBundle:
    salt_effective = str(get_canonical(canonical, "DETERMINISM.SEED_DERIVATION.SALT_SEED_EFFECTIVE"))
    salt_ref = str(get_canonical(canonical, "DETERMINISM.SEED_DERIVATION.SALT_REFERENCE_PAIRING"))
    salt_rep = str(get_canonical(canonical, "DETERMINISM.SEED_DERIVATION.SALT_REPLICATE"))
    salt_boot = str(get_canonical(canonical, "DETERMINISM.SEED_DERIVATION.SALT_BOOTSTRAP"))

    seed_effective = sha256_uint32(f"{seed_global}|{salt_effective}|{run_key}")
    seed_reference_pairing = sha256_uint32(f"{seed_effective}|{salt_ref}")
    seed_bootstrap = sha256_uint32(f"{seed_effective}|{salt_boot}")

    bundle = SeedBundle(
        seed_global=int(seed_global),
        seed_effective=int(seed_effective),
        seed_reference_pairing=int(seed_reference_pairing),
        seed_bootstrap=int(seed_bootstrap),
    )

    def seed_replicate(r: int) -> int:
        return sha256_uint32(f"{seed_effective}|{salt_rep}|{r}")

    object.__setattr__(bundle, "_seed_replicate", seed_replicate)  # type: ignore[attr-defined]
    return bundle


def seed_everything(seed: int) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def environment_fingerprint() -> dict[str, str]:
    return {
        "python_version": platform.python_version(),
        "platform": platform.platform(),
        "torch_version": getattr(torch, "__version__", "unknown"),
        "cuda_available": str(torch.cuda.is_available()),
        "numpy_version": getattr(np, "__version__", "unknown"),
    }
