from __future__ import annotations

import csv
import json
import re
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Sequence

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformer_lens import HookedTransformer

from scrubid.candidates.registry import get_generator
from scrubid.canonical import get_canonical, load_canonical
from scrubid.config import load_config
from scrubid.datasets.registry import get_suite
from scrubid.determinism import (
    compute_run_key,
    configure_determinism,
    derive_seeds,
    environment_fingerprint,
    seed_everything,
)
from scrubid.hashing import canonical_json_bytes, decimal_str, sha256_hex
from scrubid.interventions.hooks import make_hooks
from scrubid.interventions.actpatch import apply_actpatch
from scrubid.interventions.pathpatch import apply_pathpatch
from scrubid.interventions.causal_scrub import apply_causal_scrub
from scrubid.scoring.behavior_metrics import compute_metric
from scrubid.scoring.mdl import compute_mdl
from scrubid.io.logging import make_logger
from scrubid.io.run_record import write_run_record
from scrubid.diagnostics.rr import compute_rr
from scrubid.diagnostics.sss import compute_sss
from scrubid.diagnostics.cc import compute_cc
from scrubid.diagnostics.certificates import build_certificate


class RunError(RuntimeError):
    pass


def _enforce_paper_repro_interventions(*, interventions: Sequence[str], canonical: dict[str, Any]) -> None:
    """
    v1.0.3 paper scope is activation patching only.

    Even though the codebase contains placeholder intervention adapters for other
    IDs, paper reproduction must fail closed if non-actpatch families are
    configured.
    """
    actpatch_id = str(get_canonical(canonical, "IDS.INTERVENTION_FAMILY_IDS.I_ACTPATCH"))
    if any(str(x) != actpatch_id for x in interventions):
        raise RunError(str(get_canonical(canonical, "ERRORS.NOT_IMPLEMENTED_FOR_PAPER_REPRO")))


def _sha256_file(path: Path) -> str:
    return sha256_hex(path.read_bytes())


def _audit_certificate_reason_codes(
    *,
    canonical: dict[str, Any],
    rr_verdict: str,
    sss_verdict: str,
    cc_verdict: str,
) -> list[str]:
    fail = canonical["ENUMS"]["VERDICTS"]["VERDICT_FAIL"]
    out: list[str] = []
    if rr_verdict == fail:
        out.append(str(get_canonical(canonical, "ENUMS.CERTIFICATE_REASON_CODES.REASON_RR_FAIL")))
    if sss_verdict == fail:
        out.append(str(get_canonical(canonical, "ENUMS.CERTIFICATE_REASON_CODES.REASON_SSS_FAIL")))
    if cc_verdict == fail:
        out.append(str(get_canonical(canonical, "ENUMS.CERTIFICATE_REASON_CODES.REASON_CC_FAIL")))
    return out


def _list_repo_text_files(repo_root: Path) -> Iterable[Path]:
    # Spec-conformance scans target the spec pack text artifacts (not the implementation code).
    allow_dirs = {"spec", "configs", "tasks", "paper", "bib", "checklists"}
    allow_files = {
        "SPEC.md",
        "README.md",
        "AGENTS.md",
        "AUDIT_REPORT.md",
        "PATCH_REPORT.md",
        "progress.md",
        "paper.md",
        "MANIFEST.sha256",
    }
    allow_ext = {".md", ".yaml", ".yml", ".bib", ".sha256"}

    for p in repo_root.rglob("*"):
        if p.is_dir():
            continue
        rel = p.relative_to(repo_root).as_posix()
        top = rel.split("/", 1)[0]
        if top not in allow_dirs and p.name not in allow_files:
            continue
        if p.suffix and p.suffix not in allow_ext and p.name not in allow_files:
            continue
        if rel.startswith("outputs/") or rel.startswith(".git/") or rel.startswith(".venv/"):
            continue
        yield p


def validate_spec_pack(config_path: str) -> int:
    repo_root = Path.cwd()
    canonical = load_canonical(str(repo_root))

    # Load all referenced configs and ensure canonical references resolve.
    _ = load_config(config_path, canonical)
    for cfg_rel in [
        get_canonical(canonical, "FILES.CONFIG_INTERVENTIONS"),
        get_canonical(canonical, "FILES.CONFIG_GENERATORS"),
        get_canonical(canonical, "FILES.CONFIG_DETERMINISM"),
        get_canonical(canonical, "FILES.CONFIG_LOGGING_SCHEMA"),
        get_canonical(canonical, "FILES.CONFIG_BUDGETS"),
    ]:
        _ = load_config(str(repo_root / str(cfg_rel)), canonical)

    # Paper drift guard: ensure all canonical IDs referenced in paper.md exist.
    def _collect_str_values(obj: Any) -> set[str]:
        out: set[str] = set()
        if isinstance(obj, str):
            out.add(obj)
            return out
        if isinstance(obj, dict):
            for v in obj.values():
                out |= _collect_str_values(v)
            return out
        if isinstance(obj, list):
            for v in obj:
                out |= _collect_str_values(v)
            return out
        return out

    allowed_ids = _collect_str_values(canonical.get("IDS", {})) | _collect_str_values(canonical.get("REFERENCE", {}))
    paper_path = repo_root / "paper.md"
    if paper_path.exists():
        paper_text = paper_path.read_text(encoding="utf-8")
        id_patterns = [
            re.compile(r"\bSUITE_[A-Z0-9_]+\b"),
            re.compile(r"\bEXP_[A-Z0-9_]+\b"),
            re.compile(r"\bI_[A-Z0-9_]+\b"),
            re.compile(r"\bG_[A-Z0-9_]+\b"),
            re.compile(r"\bREFDIST_[A-Z0-9_]+\b"),
            re.compile(r"\bREFASSIGN_[A-Z0-9_]+\b"),
        ]
        unknown: set[str] = set()
        for pat in id_patterns:
            for m in pat.finditer(paper_text):
                tok = m.group(0)
                if tok not in allowed_ids:
                    unknown.add(tok)
        if unknown:
            raise RunError(f"paper.md references unknown canonical IDs: {sorted(unknown)[:20]}")

    # Forbidden token sweep (T1).
    forbidden_patterns = [
        re.compile(r"\bTODO\b"),
        re.compile(r"\bTBD\b"),
        re.compile(r"\.\.\."),  # three-dot ellipsis
    ]
    for p in _list_repo_text_files(repo_root):
        try:
            text = p.read_text(encoding="utf-8")
        except UnicodeDecodeError:
            continue
        for pat in forbidden_patterns:
            if pat.search(text):
                raise RunError(f"Forbidden token found ({pat.pattern}) in {p}")
    return 0


def _utc_compact() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def _torch_dtype_from_str(dtype: str) -> torch.dtype:
    if dtype == "bfloat16":
        return torch.bfloat16
    if dtype == "float16":
        return torch.float16
    if dtype == "float32":
        return torch.float32
    raise RunError(f"Unsupported dtype: {dtype}")


def _load_model(
    *,
    canonical: dict[str, Any],
    model_id: str,
    device: str,
    dtype: str,
) -> tuple[HookedTransformer, str, str, str | None]:
    """
    Load a TransformerLens HookedTransformer, supporting both:
      - official model IDs (e.g. "gpt2")
      - local HF directories (e.g. "D:\\models\\Qwen2.5-Coder-7B-Instruct")

    Returns (model, resolved_model_id, model_revision, model_local_path).
    """

    def _revision_for(resolved_id: str) -> str:
        revs = canonical.get("MODELS", {}).get("HF_MODEL_REVISIONS", {})
        if isinstance(revs, dict):
            r = revs.get(resolved_id)
            if isinstance(r, str) and r:
                return r
        return "unknown"

    model_path = Path(model_id)
    if model_path.exists():
        # TransformerLens requires an official model name even when providing hf_model.
        # Map a local directory to a resolvable public model ID when possible.
        model_local_path = str(model_path)
        model_local_path_l = model_local_path.lower()
        if "qwen" in model_local_path_l and "coder" in model_local_path_l:
            resolved_id = str(get_canonical(canonical, "MODELS.HF_MODEL_QWEN25_CODER_7B_INSTRUCT"))
        elif "qwen" in model_local_path_l and "instruct" in model_local_path_l:
            resolved_id = str(get_canonical(canonical, "MODELS.HF_MODEL_QWEN25_7B_INSTRUCT"))
        else:
            resolved_id = model_id

        tok = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
        hf_model = AutoModelForCausalLM.from_pretrained(
            model_id, torch_dtype=_torch_dtype_from_str(dtype), trust_remote_code=True
        )

        # NOTE: TransformerLens only supports a fixed allowlist of "official model names".
        # For local Qwen2.5 variants (including coder checkpoints), we route through the
        # closest supported Qwen2.5-7B-Instruct identifier while keeping provenance in
        # (resolved_id, model_revision, model_local_path).
        tl_model_name = resolved_id
        qwen25_7b_instruct = str(get_canonical(canonical, "MODELS.HF_MODEL_QWEN25_7B_INSTRUCT"))
        if resolved_id.startswith("Qwen/") and resolved_id != qwen25_7b_instruct:
            tl_model_name = qwen25_7b_instruct
        model = HookedTransformer.from_pretrained(
            tl_model_name,
            hf_model=hf_model,
            tokenizer=tok,
            device=device,
            dtype=dtype,
            trust_remote_code=True,
        )
        del hf_model
        return model, resolved_id, _revision_for(resolved_id), model_local_path

    trust_remote_code = "/" in model_id
    model = HookedTransformer.from_pretrained(
        model_id, device=device, dtype=dtype, trust_remote_code=trust_remote_code
    )
    return model, model_id, _revision_for(model_id), None


def _make_run_dir(*, runs_root: Path, canonical: dict[str, Any]) -> tuple[str, Path]:
    prefix = str(get_canonical(canonical, "OUTPUT_NAMING.RUN_DIR_PREFIX"))
    for attempt in range(10000):
        run_id = f"{prefix}{_utc_compact()}_{attempt:04d}"
        run_dir = runs_root / run_id
        if run_dir.exists():
            continue
        run_dir.mkdir(parents=True, exist_ok=False)
        return run_id, run_dir
    raise RunError("Unable to allocate a fresh immutable run directory")


def _make_report_dir(*, reports_root: Path, canonical: dict[str, Any]) -> tuple[str, Path]:
    prefix = str(get_canonical(canonical, "OUTPUT_NAMING.REPORT_DIR_PREFIX"))
    for attempt in range(10000):
        report_id = f"{prefix}{_utc_compact()}_{attempt:04d}"
        report_dir = reports_root / report_id
        if report_dir.exists():
            continue
        report_dir.mkdir(parents=True, exist_ok=False)
        return report_id, report_dir
    raise RunError("Unable to allocate a fresh immutable report directory")


def _make_repro_output_root(*, base_output_root: Path, canonical: dict[str, Any]) -> Path:
    prefix = str(get_canonical(canonical, "OUTPUT_NAMING.REPRO_OUTPUT_ROOT_PREFIX"))
    for attempt in range(10000):
        repro_id = f"{prefix}{_utc_compact()}_{attempt:04d}"
        repro_root = base_output_root / repro_id
        if repro_root.exists():
            continue
        repro_root.mkdir(parents=True, exist_ok=False)
        return repro_root
    raise RunError("Unable to allocate a fresh immutable repro output root")


def _runs_root_for_output_root(*, output_root: Path, canonical: dict[str, Any]) -> Path:
    runs_rel = Path(str(get_canonical(canonical, "PATHS.PATH_RUNS_ROOT")))
    out_rel = Path(str(get_canonical(canonical, "PATHS.PATH_OUTPUT_ROOT")))
    try:
        runs_sub = runs_rel.relative_to(out_rel)
    except ValueError:
        runs_sub = Path(runs_rel.name)
    return output_root / runs_sub


def _reports_root_for_output_root(*, output_root: Path, canonical: dict[str, Any]) -> Path:
    reports_rel = Path(str(get_canonical(canonical, "PATHS.PATH_REPORTS_ROOT")))
    out_rel = Path(str(get_canonical(canonical, "PATHS.PATH_OUTPUT_ROOT")))
    try:
        reports_sub = reports_rel.relative_to(out_rel)
    except ValueError:
        reports_sub = Path(reports_rel.name)
    return output_root / reports_sub


def _load_run_records(*, runs_root: Path, canonical: dict[str, Any]) -> list[dict[str, Any]]:
    prefix = str(get_canonical(canonical, "OUTPUT_NAMING.RUN_DIR_PREFIX"))
    rr_name = str(get_canonical(canonical, "FILES.RUN_RECORD_FILENAME"))
    run_record_hash_field = str(get_canonical(canonical, "OUTPUT_NAMING.RUN_RECORD_HASH_FIELD"))
    out: list[dict[str, Any]] = []
    seen_hashes: set[str] = set()
    if not runs_root.exists():
        return out
    for run_dir in sorted([p for p in runs_root.iterdir() if p.is_dir() and p.name.startswith(prefix)]):
        rr_path = run_dir / rr_name
        if not rr_path.exists():
            continue
        try:
            rec = json.loads(rr_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            continue
        if isinstance(rec, dict):
            rrh = rec.get(run_record_hash_field)
            if isinstance(rrh, str) and rrh:
                if rrh in seen_hashes:
                    continue
                seen_hashes.add(rrh)
            out.append(rec)
    return out


def _write_csv(*, path: Path, fieldnames: list[str], rows: list[dict[str, Any]]) -> None:
    import csv

    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore", lineterminator="\n")
        w.writeheader()
        for row in rows:
            w.writerow({k: row.get(k, "") for k in fieldnames})


def _bootstrap_mean_ci(
    *,
    values: list[float],
    seed: int,
    resamples: int,
    ci_lo: float,
    ci_hi: float,
) -> tuple[float, float, float]:
    import numpy as np

    if not values:
        return 0.0, 0.0, 0.0
    arr = np.asarray(values, dtype=float)
    mean = float(arr.mean())
    if arr.size == 1:
        return mean, mean, mean
    rng = np.random.default_rng(int(seed))
    idx = rng.integers(0, arr.size, size=(int(resamples), arr.size))
    samples = arr[idx].mean(axis=1)
    lo = float(np.quantile(samples, float(ci_lo)))
    hi = float(np.quantile(samples, float(ci_hi)))
    return mean, lo, hi


def _format_ci(lo: float, hi: float) -> str:
    return f"[{decimal_str(lo)},{decimal_str(hi)}]"


def _worst_verdict(verdicts: list[str], canonical: dict[str, Any]) -> str:
    order = list(canonical["ENUMS"]["VERDICT_ENUM"])
    if not order:
        return str(canonical["ENUMS"]["VERDICTS"]["VERDICT_FAIL"])
    rank = {str(v): int(i) for i, v in enumerate(order)}
    worst_rank = max([rank.get(str(v), len(order)) for v in verdicts], default=0)
    worst_rank = min(int(worst_rank), len(order) - 1)
    return str(order[int(worst_rank)])


def _build_table_t1_synth(*, canonical: dict[str, Any], run_records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    from scrubid.hashing import sha256_uint32

    sweep_id = str(get_canonical(canonical, "IDS.EXPERIMENT_IDS.EXP_SYNTH_REDUNDANCY_SWEEP_V1"))
    synth_id = str(get_canonical(canonical, "IDS.SUITE_IDS.SUITE_SYNTH_V1"))
    allowed_generators = {
        str(get_canonical(canonical, "IDS.CANDIDATE_GENERATOR_IDS.G_ATTR_PATCH")),
        str(get_canonical(canonical, "IDS.CANDIDATE_GENERATOR_IDS.G_MANUAL_SEED")),
    }

    seed_global = int(canonical["SEEDS"]["SEED_GLOBAL"])
    salt = str(get_canonical(canonical, "DETERMINISM.SEED_DERIVATION.SALT_BOOTSTRAP"))
    resamples = int(get_canonical(canonical, "STATS.BOOTSTRAP_RESAMPLES"))
    ci_lo = float(get_canonical(canonical, "STATS.CI_LO"))
    ci_hi = float(get_canonical(canonical, "STATS.CI_HI"))

    grouped: dict[str, list[dict[str, Any]]] = {}
    for r in run_records:
        if str(r.get("suite_id")) != synth_id:
            continue
        if str(r.get("experiment_id")) != sweep_id:
            continue
        if str(r.get("candidate_generator_id")) not in allowed_generators:
            continue
        setting_id = str(r.get("setting_id", ""))
        if not setting_id:
            continue
        grouped.setdefault(setting_id, []).append(r)

    rows: list[dict[str, Any]] = []
    for setting_id in sorted(grouped.keys()):
        recs = grouped[setting_id]
        template_id = str(recs[0].get("template_id", ""))
        redundancy_factor = int(recs[0].get("redundancy_factor", 0))
        rr_vals = [float(x.get("RR", 0.0)) for x in recs]
        sss_vals = [float(x.get("SSS", 0.0)) for x in recs]
        cc_vals = [float(x.get("CC", 0.0)) for x in recs]
        certs = [1.0 if "certificate_file" in dict(x.get("paths", {})) else 0.0 for x in recs]

        rr_mean, rr_lo, rr_hi = _bootstrap_mean_ci(
            values=rr_vals,
            seed=sha256_uint32(f"{seed_global}|{salt}|{setting_id}|RR"),
            resamples=resamples,
            ci_lo=ci_lo,
            ci_hi=ci_hi,
        )
        sss_mean, sss_lo, sss_hi = _bootstrap_mean_ci(
            values=sss_vals,
            seed=sha256_uint32(f"{seed_global}|{salt}|{setting_id}|SSS"),
            resamples=resamples,
            ci_lo=ci_lo,
            ci_hi=ci_hi,
        )
        cc_mean, cc_lo, cc_hi = _bootstrap_mean_ci(
            values=cc_vals,
            seed=sha256_uint32(f"{seed_global}|{salt}|{setting_id}|CC"),
            resamples=resamples,
            ci_lo=ci_lo,
            ci_hi=ci_hi,
        )
        non_id_rate = float(sum(certs) / len(certs)) if certs else 0.0

        rows.append(
            {
                "setting_id": setting_id,
                "template_id": template_id,
                "planted_redundancy_factor": int(redundancy_factor),
                "RR_mean": decimal_str(rr_mean),
                "RR_ci": _format_ci(rr_lo, rr_hi),
                "SSS_mean": decimal_str(sss_mean),
                "SSS_ci": _format_ci(sss_lo, sss_hi),
                "CC_mean": decimal_str(cc_mean),
                "CC_ci": _format_ci(cc_lo, cc_hi),
                "non_identifiability_rate": decimal_str(non_id_rate),
            }
        )

    rows.sort(key=lambda r: (int(r["planted_redundancy_factor"]), str(r["template_id"]), str(r["setting_id"])))
    return rows


def _build_table_t2_real(*, canonical: dict[str, Any], run_records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    # ID-split real-model runs only (OOD is reported separately in T4).
    #
    # P1: aggregate across multiple suite seeds (multi-seed protocol) and report
    # mean + bootstrap CI over seeds.
    from scrubid.hashing import sha256_uint32

    real_ids = {
        str(get_canonical(canonical, "IDS.SUITE_IDS.SUITE_REAL_IOI_V1")),
        str(get_canonical(canonical, "IDS.SUITE_IDS.SUITE_REAL_GREATERTHAN_V1")),
        str(get_canonical(canonical, "IDS.SUITE_IDS.SUITE_REAL_GREATERTHAN_YN_V1")),
        str(get_canonical(canonical, "IDS.SUITE_IDS.SUITE_REAL_INDUCTION_V1")),
    }
    id_experiments = {
        str(get_canonical(canonical, "IDS.EXPERIMENT_IDS.EXP_REAL_IOI_V1")),
        str(get_canonical(canonical, "IDS.EXPERIMENT_IDS.EXP_REAL_GREATERTHAN_V1")),
        str(get_canonical(canonical, "IDS.EXPERIMENT_IDS.EXP_REAL_GREATERTHAN_YN_V1")),
        str(get_canonical(canonical, "IDS.EXPERIMENT_IDS.EXP_REAL_INDUCTION_V1")),
    }

    seed_global = int(canonical["SEEDS"]["SEED_GLOBAL"])
    salt = str(get_canonical(canonical, "DETERMINISM.SEED_DERIVATION.SALT_BOOTSTRAP"))
    resamples = int(get_canonical(canonical, "STATS.BOOTSTRAP_RESAMPLES"))
    ci_lo = float(get_canonical(canonical, "STATS.CI_LO"))
    ci_hi = float(get_canonical(canonical, "STATS.CI_HI"))

    grouped: dict[tuple[str, str, str], list[dict[str, Any]]] = {}
    for r in run_records:
        suite_id = str(r.get("suite_id"))
        if suite_id not in real_ids:
            continue
        if str(r.get("experiment_id")) not in id_experiments:
            continue
        key = (suite_id, str(r.get("intervention_family_id")), str(r.get("candidate_generator_id")))
        grouped.setdefault(key, []).append(r)

    rows: list[dict[str, Any]] = []
    for (suite_id, intervention_family_id, candidate_generator_id) in sorted(grouped.keys()):
        recs = grouped[(suite_id, intervention_family_id, candidate_generator_id)]
        n = int(len(recs))

        size_vals = [float(r.get("best_circuit_size", 0)) for r in recs]
        delta_vals = [float(r.get("faithfulness_delta", 0.0)) for r in recs]
        rr_vals = [float(r.get("RR", 0.0)) for r in recs]
        sss_vals = [float(r.get("SSS", 0.0)) for r in recs]
        cc_vals = [float(r.get("CC", 0.0)) for r in recs]

        key_prefix = f"{suite_id}|{intervention_family_id}|{candidate_generator_id}"
        size_mean, size_lo, size_hi = _bootstrap_mean_ci(
            values=size_vals,
            seed=sha256_uint32(f"{seed_global}|{salt}|{key_prefix}|size"),
            resamples=resamples,
            ci_lo=ci_lo,
            ci_hi=ci_hi,
        )
        delta_mean, delta_lo, delta_hi = _bootstrap_mean_ci(
            values=delta_vals,
            seed=sha256_uint32(f"{seed_global}|{salt}|{key_prefix}|delta"),
            resamples=resamples,
            ci_lo=ci_lo,
            ci_hi=ci_hi,
        )
        rr_mean, rr_lo, rr_hi = _bootstrap_mean_ci(
            values=rr_vals,
            seed=sha256_uint32(f"{seed_global}|{salt}|{key_prefix}|RR"),
            resamples=resamples,
            ci_lo=ci_lo,
            ci_hi=ci_hi,
        )
        sss_mean, sss_lo, sss_hi = _bootstrap_mean_ci(
            values=sss_vals,
            seed=sha256_uint32(f"{seed_global}|{salt}|{key_prefix}|SSS"),
            resamples=resamples,
            ci_lo=ci_lo,
            ci_hi=ci_hi,
        )
        cc_mean, cc_lo, cc_hi = _bootstrap_mean_ci(
            values=cc_vals,
            seed=sha256_uint32(f"{seed_global}|{salt}|{key_prefix}|CC"),
            resamples=resamples,
            ci_lo=ci_lo,
            ci_hi=ci_hi,
        )

        rows.append(
            {
                "suite_id": suite_id,
                "intervention_family_id": intervention_family_id,
                "candidate_generator_id": candidate_generator_id,
                "N": int(n),
                "best_circuit_size_mean": decimal_str(size_mean),
                "best_circuit_size_ci": _format_ci(size_lo, size_hi),
                "faithfulness_delta_mean": decimal_str(delta_mean),
                "faithfulness_delta_ci": _format_ci(delta_lo, delta_hi),
                "RR_mean": decimal_str(rr_mean),
                "RR_ci": _format_ci(rr_lo, rr_hi),
                "SSS_mean": decimal_str(sss_mean),
                "SSS_ci": _format_ci(sss_lo, sss_hi),
                "CC_mean": decimal_str(cc_mean),
                "CC_ci": _format_ci(cc_lo, cc_hi),
                "overall_verdict": _worst_verdict([str(r.get("overall_verdict")) for r in recs], canonical),
            }
        )

    rows.sort(key=lambda r: (str(r["suite_id"]), str(r["intervention_family_id"]), str(r["candidate_generator_id"])))
    return rows


def _build_table_t4_ood(*, canonical: dict[str, Any], run_records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """
    OOD-split real-model runs (one-to-one companion to Table T2).
    """
    from scrubid.hashing import sha256_uint32

    real_ids = {
        str(get_canonical(canonical, "IDS.SUITE_IDS.SUITE_REAL_IOI_V1")),
        str(get_canonical(canonical, "IDS.SUITE_IDS.SUITE_REAL_GREATERTHAN_V1")),
        str(get_canonical(canonical, "IDS.SUITE_IDS.SUITE_REAL_GREATERTHAN_YN_V1")),
        str(get_canonical(canonical, "IDS.SUITE_IDS.SUITE_REAL_INDUCTION_V1")),
    }
    ood_experiments = {
        str(get_canonical(canonical, "IDS.EXPERIMENT_IDS.EXP_REAL_IOI_OOD_V1")),
        str(get_canonical(canonical, "IDS.EXPERIMENT_IDS.EXP_REAL_GREATERTHAN_OOD_V1")),
        str(get_canonical(canonical, "IDS.EXPERIMENT_IDS.EXP_REAL_GREATERTHAN_YN_OOD_V1")),
        str(get_canonical(canonical, "IDS.EXPERIMENT_IDS.EXP_REAL_INDUCTION_OOD_V1")),
    }

    seed_global = int(canonical["SEEDS"]["SEED_GLOBAL"])
    salt = str(get_canonical(canonical, "DETERMINISM.SEED_DERIVATION.SALT_BOOTSTRAP"))
    resamples = int(get_canonical(canonical, "STATS.BOOTSTRAP_RESAMPLES"))
    ci_lo = float(get_canonical(canonical, "STATS.CI_LO"))
    ci_hi = float(get_canonical(canonical, "STATS.CI_HI"))

    grouped: dict[tuple[str, str, str, str], list[dict[str, Any]]] = {}
    for r in run_records:
        suite_id = str(r.get("suite_id"))
        if suite_id not in real_ids:
            continue
        experiment_id = str(r.get("experiment_id"))
        if experiment_id not in ood_experiments:
            continue
        key = (
            suite_id,
            experiment_id,
            str(r.get("intervention_family_id")),
            str(r.get("candidate_generator_id")),
        )
        grouped.setdefault(key, []).append(r)

    rows: list[dict[str, Any]] = []
    for (suite_id, experiment_id, intervention_family_id, candidate_generator_id) in sorted(grouped.keys()):
        recs = grouped[(suite_id, experiment_id, intervention_family_id, candidate_generator_id)]
        n = int(len(recs))

        size_vals = [float(r.get("best_circuit_size", 0)) for r in recs]
        delta_vals = [float(r.get("faithfulness_delta", 0.0)) for r in recs]
        rr_vals = [float(r.get("RR", 0.0)) for r in recs]
        sss_vals = [float(r.get("SSS", 0.0)) for r in recs]
        cc_vals = [float(r.get("CC", 0.0)) for r in recs]

        key_prefix = f"{suite_id}|{experiment_id}|{intervention_family_id}|{candidate_generator_id}"
        size_mean, size_lo, size_hi = _bootstrap_mean_ci(
            values=size_vals,
            seed=sha256_uint32(f"{seed_global}|{salt}|{key_prefix}|size"),
            resamples=resamples,
            ci_lo=ci_lo,
            ci_hi=ci_hi,
        )
        delta_mean, delta_lo, delta_hi = _bootstrap_mean_ci(
            values=delta_vals,
            seed=sha256_uint32(f"{seed_global}|{salt}|{key_prefix}|delta"),
            resamples=resamples,
            ci_lo=ci_lo,
            ci_hi=ci_hi,
        )
        rr_mean, rr_lo, rr_hi = _bootstrap_mean_ci(
            values=rr_vals,
            seed=sha256_uint32(f"{seed_global}|{salt}|{key_prefix}|RR"),
            resamples=resamples,
            ci_lo=ci_lo,
            ci_hi=ci_hi,
        )
        sss_mean, sss_lo, sss_hi = _bootstrap_mean_ci(
            values=sss_vals,
            seed=sha256_uint32(f"{seed_global}|{salt}|{key_prefix}|SSS"),
            resamples=resamples,
            ci_lo=ci_lo,
            ci_hi=ci_hi,
        )
        cc_mean, cc_lo, cc_hi = _bootstrap_mean_ci(
            values=cc_vals,
            seed=sha256_uint32(f"{seed_global}|{salt}|{key_prefix}|CC"),
            resamples=resamples,
            ci_lo=ci_lo,
            ci_hi=ci_hi,
        )

        rows.append(
            {
                "suite_id": suite_id,
                "experiment_id": experiment_id,
                "intervention_family_id": intervention_family_id,
                "candidate_generator_id": candidate_generator_id,
                "N": int(n),
                "best_circuit_size_mean": decimal_str(size_mean),
                "best_circuit_size_ci": _format_ci(size_lo, size_hi),
                "faithfulness_delta_mean": decimal_str(delta_mean),
                "faithfulness_delta_ci": _format_ci(delta_lo, delta_hi),
                "RR_mean": decimal_str(rr_mean),
                "RR_ci": _format_ci(rr_lo, rr_hi),
                "SSS_mean": decimal_str(sss_mean),
                "SSS_ci": _format_ci(sss_lo, sss_hi),
                "CC_mean": decimal_str(cc_mean),
                "CC_ci": _format_ci(cc_lo, cc_hi),
                "overall_verdict": _worst_verdict([str(r.get("overall_verdict")) for r in recs], canonical),
            }
        )

    rows.sort(
        key=lambda r: (
            str(r["suite_id"]),
            str(r["experiment_id"]),
            str(r["intervention_family_id"]),
            str(r["candidate_generator_id"]),
        )
    )
    return rows


def _build_table_t3_deltas(*, canonical: dict[str, Any], run_records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """
    Sensitivity deltas against a per-suite baseline (lexicographically smallest intervention/generator).
    """
    real_rows = _build_table_t2_real(canonical=canonical, run_records=run_records)
    by_suite: dict[str, list[dict[str, Any]]] = {}
    for r in real_rows:
        by_suite.setdefault(str(r["suite_id"]), []).append(r)

    out: list[dict[str, Any]] = []
    for suite_id in sorted(by_suite.keys()):
        rows = sorted(by_suite[suite_id], key=lambda r: (str(r["intervention_family_id"]), str(r["candidate_generator_id"])))
        base = rows[0] if rows else None
        if base is None:
            continue
        base_size = float(base.get("best_circuit_size_mean", 0.0))
        base_delta = float(base.get("faithfulness_delta_mean", 0.0))
        base_rr = float(base.get("RR_mean", 0.0))
        base_sss = float(base.get("SSS_mean", 0.0))
        base_cc = float(base.get("CC_mean", 0.0))

        for r in rows[1:]:
            vid = f"{suite_id}|{r['intervention_family_id']}|{r['candidate_generator_id']}"
            out.append(
                {
                    "variant_id": vid,
                    "Δ_size": decimal_str(float(r.get("best_circuit_size_mean", 0.0)) - base_size),
                    "Δ_faithfulness": decimal_str(float(r.get("faithfulness_delta_mean", 0.0)) - base_delta),
                    "Δ_RR": decimal_str(float(r.get("RR_mean", 0.0)) - base_rr),
                    "Δ_SSS": decimal_str(float(r.get("SSS_mean", 0.0)) - base_sss),
                    "Δ_CC": decimal_str(float(r.get("CC_mean", 0.0)) - base_cc),
                }
            )
    out.sort(key=lambda r: str(r["variant_id"]))
    return out


def _write_report_tables(*, report_dir: Path, canonical: dict[str, Any], run_records: list[dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    t1 = _build_table_t1_synth(canonical=canonical, run_records=run_records)
    t2 = _build_table_t2_real(canonical=canonical, run_records=run_records)
    t3 = _build_table_t3_deltas(canonical=canonical, run_records=run_records)
    t4 = _build_table_t4_ood(canonical=canonical, run_records=run_records)

    t1_path = report_dir / str(get_canonical(canonical, "FILES.REPORT_TABLE_T1_FILENAME"))
    t2_path = report_dir / str(get_canonical(canonical, "FILES.REPORT_TABLE_T2_FILENAME"))
    t3_path = report_dir / str(get_canonical(canonical, "FILES.REPORT_TABLE_T3_FILENAME"))
    t4_path = report_dir / str(get_canonical(canonical, "FILES.REPORT_TABLE_T4_FILENAME"))

    _write_csv(
        path=t1_path,
        fieldnames=[
            "setting_id",
            "planted_redundancy_factor",
            "RR_mean",
            "RR_ci",
            "SSS_mean",
            "SSS_ci",
            "CC_mean",
            "CC_ci",
            "non_identifiability_rate",
        ],
        rows=t1,
    )
    _write_csv(
        path=t2_path,
        fieldnames=[
            "suite_id",
            "intervention_family_id",
            "candidate_generator_id",
            "N",
            "best_circuit_size_mean",
            "best_circuit_size_ci",
            "faithfulness_delta_mean",
            "faithfulness_delta_ci",
            "RR_mean",
            "RR_ci",
            "SSS_mean",
            "SSS_ci",
            "CC_mean",
            "CC_ci",
            "overall_verdict",
        ],
        rows=t2,
    )
    _write_csv(
        path=t3_path,
        fieldnames=["variant_id", "Δ_size", "Δ_faithfulness", "Δ_RR", "Δ_SSS", "Δ_CC"],
        rows=t3,
    )

    _write_csv(
        path=t4_path,
        fieldnames=[
            "suite_id",
            "experiment_id",
            "intervention_family_id",
            "candidate_generator_id",
            "N",
            "best_circuit_size_mean",
            "best_circuit_size_ci",
            "faithfulness_delta_mean",
            "faithfulness_delta_ci",
            "RR_mean",
            "RR_ci",
            "SSS_mean",
            "SSS_ci",
            "CC_mean",
            "CC_ci",
            "overall_verdict",
        ],
        rows=t4,
    )

    return {"T1": t1, "T2": t2, "T3": t3, "T4": t4}


def _write_fig_synth(*, canonical: dict[str, Any], report_dir: Path, table_t1: list[dict[str, Any]]) -> None:
    import matplotlib

    matplotlib.use("Agg")
    matplotlib.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "axes.unicode_minus": False,
        }
    )
    import matplotlib.pyplot as plt

    by_template: dict[str, list[dict[str, Any]]] = {}
    for r in table_t1:
        by_template.setdefault(str(r["template_id"]), []).append(r)
    for t in by_template:
        by_template[t].sort(key=lambda r: int(r["planted_redundancy_factor"]))

    fig, axes = plt.subplots(1, 3, figsize=(12, 3.6), constrained_layout=True)
    metrics = [("RR_mean", "RR"), ("SSS_mean", "SSS"), ("CC_mean", "CC")]
    for ax, (col, title) in zip(axes, metrics, strict=True):
        for template_id, rows in sorted(by_template.items()):
            xs = [int(r["planted_redundancy_factor"]) for r in rows]
            ys = [float(r[col]) for r in rows]
            ax.plot(xs, ys, marker="o", label=template_id)
        ax.set_title(title)
        ax.set_xlabel("Planted redundancy factor")
        ax.set_ylim(0.0, 1.0)
        ax.grid(True, alpha=0.3)
    axes[0].set_ylabel("Score")
    axes[-1].legend(loc="best", fontsize=8)

    fig_path = report_dir / str(get_canonical(canonical, "FILES.REPORT_FIG_SYNTH_FILENAME"))
    fig.savefig(fig_path, dpi=200)
    plt.close(fig)


def _resolve_output_root(repo_root: Path, canonical: dict[str, Any], cli_output_root: str | None) -> Path:
    if cli_output_root:
        return Path(cli_output_root)
    return repo_root / str(get_canonical(canonical, "PATHS.PATH_OUTPUT_ROOT"))


def _git_commit_hex(*, canonical: dict[str, Any], repo_root: Path) -> str:
    # Best-effort: this pack may not live in a git repo.
    git_len = int(get_canonical(canonical, "GIT.GIT_COMMIT_HEX_LEN"))
    head = repo_root / ".git" / "HEAD"
    if not head.exists():
        return "0" * git_len
    try:
        text = head.read_text(encoding="utf-8").strip()
        if text.startswith("ref:"):
            ref = text.split(":", 1)[1].strip()
            ref_path = repo_root / ".git" / ref
            if ref_path.exists():
                h = ref_path.read_text(encoding="utf-8").strip()
                return (h[:git_len]).ljust(git_len, "0")
        return (text[:git_len]).ljust(git_len, "0")
    except Exception:
        return "0" * git_len


def _deterministic_subset_for_hash(record: dict[str, Any], canonical: dict[str, Any]) -> dict[str, Any]:
    """
    Deterministic subset used for run_record_hash.

    Excludes per-run identifiers and environment fields that are allowed to vary
    across immutable re-executions (run_id, timestamp, git/python/platform).
    """
    scalar_float_fields = {
        "baseline_score_s0",
        "epsilon",
        "best_circuit_mdl",
        "faithfulness_delta",
        "RR",
        "SSS",
        "CC",
    }
    out: dict[str, Any] = {}
    run_record_hash_field = str(get_canonical(canonical, "OUTPUT_NAMING.RUN_RECORD_HASH_FIELD"))
    for k, v in record.items():
        if k in {"run_id", "timestamp_utc", "git_commit", "python_version", "platform", "paths", "model_local_path", run_record_hash_field}:
            continue
        if k in scalar_float_fields and isinstance(v, (float, int)):
            out[k] = decimal_str(float(v))
        else:
            out[k] = v
    return out


def compute_run_record_hash(record: dict[str, Any], canonical: dict[str, Any]) -> str:
    subset = _deterministic_subset_for_hash(record, canonical)
    b = canonical_json_bytes(subset, canonical)
    return sha256_hex(b)


@dataclass(frozen=True)
class CommonCliArgs:
    config_path: str
    deterministic: bool
    device: str | None
    output_root: str | None


def _common_args(args: Any) -> CommonCliArgs:
    return CommonCliArgs(
        config_path=str(args.config_path),
        deterministic=bool(args.deterministic),
        device=getattr(args, "device", None),
        output_root=getattr(args, "output_root", None),
    )


def _synth_settings_for_experiment(
    suite: dict[str, Any], *, experiment_id: str, canonical: dict[str, Any]
) -> list[dict[str, Any]]:
    settings = list(suite.get("settings", []))
    if experiment_id == "EXP_SYNTH_REDUNDANCY_SWEEP_V1":
        return settings
    if experiment_id == "EXP_SYNTH_SANITY_V1":
        factors = [int(x) for x in canonical["DATASETS"]["SYNTH_REDUNDANCY_FACTORS"]]
        if not factors:
            return []
        lo = int(min(factors))
        hi = int(max(factors))
        # Keep a minimal, deterministic sanity subset.
        return [
            s
            for s in settings
            if str(s.get("template_id")) == "XOR" and int(s.get("redundancy_factor")) in {lo, hi}
        ]
    raise RunError(f"Unknown synthetic experiment_id: {experiment_id}")


def _synth_metric_values(eval_rows: Sequence[dict[str, Any]]) -> list[float]:
    out: list[float] = []
    for r in eval_rows:
        y = int(r["y"])
        out.append(1.0 if y == 1 else -1.0)
    return out


def _synth_scrubbed_metric_values(
    *,
    setting: dict[str, Any],
    circuit_components: set[str],
    baseline_vals: Sequence[float],
) -> list[float]:
    aggr = str(setting.get("aggregator_id"))
    redundant_ids = [str(x) for x in setting.get("redundant_ids", [])]
    if aggr not in circuit_components:
        return [0.0 for _ in baseline_vals]
    if not any(rid in circuit_components for rid in redundant_ids):
        return [0.0 for _ in baseline_vals]
    return [float(x) for x in baseline_vals]


def _synth_delta(
    *,
    baseline_vals: Sequence[float],
    scrubbed_vals: Sequence[float],
) -> float:
    if len(baseline_vals) != len(scrubbed_vals):
        raise RunError("Baseline and scrubbed metric lengths differ")
    if not baseline_vals:
        return 0.0
    return float(sum(abs(float(a) - float(b)) for a, b in zip(baseline_vals, scrubbed_vals, strict=True)) / len(baseline_vals))


def _candidate_key(rec: dict[str, Any]) -> tuple[float, int, tuple[str, ...]]:
    comps = tuple(sorted([str(x) for x in rec.get("components", [])]))
    return (float(rec.get("mdl", 0.0)), int(len(comps)), comps)


def _select_best_faithful(candidate_records: list[dict[str, Any]]) -> dict[str, Any] | None:
    faithful = [r for r in candidate_records if bool(r.get("faithful", False))]
    if not faithful:
        return None
    return sorted(faithful, key=_candidate_key)[0]


def _write_json(path: Path, obj: Any) -> None:
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def _run_one_synth(
    *,
    repo_root: Path,
    canonical: dict[str, Any],
    runs_root: Path,
    suite_id: str,
    experiment_id: str,
    intervention_family_id: str,
    candidate_generator_id: str,
    component_granularity: str,
    model_id: str,
    model_revision: str,
    setting: dict[str, Any],
    seed_suite: int,
    seed_global: int,
    resolved_config_hashes: dict[str, str],
    budgets: dict[str, Any],
    deterministic_mode: bool,
) -> None:
    eval_rows = list(setting.get("splits", {}).get("eval", []))
    ref_rows = list(setting.get("splits", {}).get("ref", []))
    if not eval_rows or not ref_rows:
        raise RunError("Synthetic setting missing eval/ref splits")

    baseline_vals = _synth_metric_values(eval_rows)
    s0 = float(sum(abs(v) for v in baseline_vals) / len(baseline_vals)) if baseline_vals else 0.0
    eps_abs_min = float(canonical["DIAGNOSTICS"]["EPSILON_ABS_MIN"])
    eps_rel = float(canonical["DIAGNOSTICS"]["EPSILON_REL_FRAC"])
    epsilon = max(eps_abs_min, eps_rel * s0)
    tau_rel = float(canonical["DIAGNOSTICS"]["TAU_REL_FRAC_NECESSITY"])
    tau_abs = max(eps_abs_min, tau_rel * s0)

    dataset_obj = {"setting": setting.get("setting_id"), "eval": eval_rows}
    ref_obj = {"setting": setting.get("setting_id"), "ref": ref_rows}
    dataset_fp = sha256_hex(canonical_json_bytes(dataset_obj, canonical))
    ref_fp = sha256_hex(canonical_json_bytes(ref_obj, canonical))

    refdist_id = str(setting.get("reference_distribution_id", canonical["REFERENCE"]["REFDIST_DEFAULT_BY_SUITE"][suite_id]))
    refassign_id = str(setting.get("reference_assignment_id", canonical["REFERENCE"]["REF_ASSIGNMENT_DEFAULT_BY_SUITE"][suite_id]))

    run_key = compute_run_key(
        canonical=canonical,
        suite_id=suite_id,
        experiment_id=experiment_id,
        model_id=model_id,
        model_revision=model_revision,
        component_granularity=component_granularity,
        intervention_family_id=intervention_family_id,
        candidate_generator_ids=[candidate_generator_id],
        reference_distribution_id=refdist_id,
        reference_assignment_id=refassign_id,
        resolved_config_hashes=resolved_config_hashes,
        dataset_fingerprints={"eval": dataset_fp, "ref": ref_fp},
        budgets=budgets,
        epsilon_abs=epsilon,
        tau_abs=tau_abs,
    )

    seeds = derive_seeds(canonical=canonical, seed_global=seed_global, run_key=run_key)
    seed_everything(seeds.seed_effective)

    run_id, run_dir = _make_run_dir(runs_root=runs_root, canonical=canonical)
    logger = make_logger(run_dir=run_dir, canonical=canonical, run_id=run_id)

    logger.log(
        canonical["ENUMS"]["LOG_EVENT_TYPES"]["EVENT_RUN_START"],
        {
            "suite_id": suite_id,
            "experiment_id": experiment_id,
            "intervention_family_id": intervention_family_id,
            "candidate_generator_id": candidate_generator_id,
            "component_granularity": component_granularity,
            "deterministic_mode": deterministic_mode,
            "seed_global": seed_global,
            "seed_suite": seed_suite,
            "reference_distribution_id": refdist_id,
            "reference_assignment_id": refassign_id,
            "model_id": model_id,
            "model_revision": model_revision or "unknown",
            str(get_canonical(canonical, "OUTPUT_NAMING.RUN_KEY_FIELD")): run_key,
            # Synthetic setting provenance.
            "setting_id": str(setting.get("setting_id")),
            "template_id": str(setting.get("template_id")),
            "redundancy_factor": int(setting.get("redundancy_factor")),
        },
    )

    logger.log(
        canonical["ENUMS"]["LOG_EVENT_TYPES"]["EVENT_DATASET_WRITTEN"],
        {
            "dataset_fingerprint": dataset_fp,
            "reference_dataset_fingerprint": ref_fp,
            "sizes": {"num_examples_eval": int(len(eval_rows)), "num_examples_ref": int(len(ref_rows))},
        },
    )

    generator = get_generator(candidate_generator_id)
    task_spec = {
        "suite_id": suite_id,
        "canonical": canonical,
        "component_granularity": component_granularity,
        "intervention_family_id": intervention_family_id,
        "synth_setting": setting,
        "eval_rows": eval_rows,
    }
    candset = generator(None, task_spec, intervention_family_id, budgets, seed=int(seeds.seed_effective))
    candidate_circuits = list(candset.get("candidate_circuits", []))
    if not candidate_circuits:
        raise RunError("Candidate generator returned empty candidate_circuits")

    candidate_records: list[dict[str, Any]] = []
    for circ in candidate_circuits:
        comps = sorted([str(x) for x in circ.get("components", [])])
        circuit_obj = {"components": comps}
        scrubbed_vals = _synth_scrubbed_metric_values(
            setting=setting,
            circuit_components=set(comps),
            baseline_vals=baseline_vals,
        )
        delta = _synth_delta(baseline_vals=baseline_vals, scrubbed_vals=scrubbed_vals)
        mdl = compute_mdl(circuit_obj, canonical)
        faithful = bool(delta <= epsilon)
        candidate_records.append({"components": comps, "delta": float(delta), "mdl": float(mdl), "faithful": faithful})

    rr_res = compute_rr(candidate_records, canonical)
    s_near = list(rr_res.get("s_near", []))

    # SSS replicate discovery: rerun generator with derived replicate seeds.
    R = int(canonical["DIAGNOSTICS"]["SSS_NUM_REPLICATES"])
    replicate_circuits: list[set[str]] = []
    for r in range(R):
        rep_seed = int(seeds.seed_replicate(int(r)))
        rep_candset = generator(None, task_spec, intervention_family_id, budgets, seed=rep_seed)
        rep_circs = list(rep_candset.get("candidate_circuits", []))
        rep_records: list[dict[str, Any]] = []
        for circ in rep_circs:
            comps = sorted([str(x) for x in circ.get("components", [])])
            delta = _synth_delta(
                baseline_vals=baseline_vals,
                scrubbed_vals=_synth_scrubbed_metric_values(
                    setting=setting, circuit_components=set(comps), baseline_vals=baseline_vals
                ),
            )
            mdl = compute_mdl({"components": comps}, canonical)
            rep_records.append({"components": comps, "delta": float(delta), "mdl": float(mdl), "faithful": bool(delta <= epsilon)})
        best = _select_best_faithful(rep_records)
        replicate_circuits.append(set(best["components"]) if best is not None else set())

    sss_res = compute_sss(replicate_circuits, canonical)

    # CC necessity labeling over the near-optimal set.
    for rec in s_near:
        comps = [str(x) for x in rec.get("components", [])]
        base_delta = float(rec.get("delta", 0.0))
        necessity: dict[str, bool] = {}
        for v in comps:
            comps2 = sorted([c for c in comps if c != v])
            delta2 = _synth_delta(
                baseline_vals=baseline_vals,
                scrubbed_vals=_synth_scrubbed_metric_values(
                    setting=setting, circuit_components=set(comps2), baseline_vals=baseline_vals
                ),
            )
            necessity[v] = bool(float(delta2) - float(base_delta) >= float(tau_abs))
        rec["necessity"] = dict(sorted(necessity.items()))

    cc_res = compute_cc(s_near, canonical)

    best = _select_best_faithful(candidate_records)
    best_components = sorted([str(x) for x in (best.get("components", []) if best is not None else [])])

    best_path = run_dir / str(get_canonical(canonical, "FILES.BEST_CIRCUIT_FILENAME"))
    _write_json(best_path, {"components": best_components})

    certificate_path: Path | None = None
    certificate_obj: dict[str, Any] | None = None
    if (
        rr_res["RR_verdict"] == canonical["ENUMS"]["VERDICTS"]["VERDICT_FAIL"]
        or sss_res["SSS_verdict"] == canonical["ENUMS"]["VERDICTS"]["VERDICT_FAIL"]
        or cc_res["CC_verdict"] == canonical["ENUMS"]["VERDICTS"]["VERDICT_FAIL"]
    ):
        reason_codes = _audit_certificate_reason_codes(
            canonical=canonical,
            rr_verdict=str(rr_res["RR_verdict"]),
            sss_verdict=str(sss_res["SSS_verdict"]),
            cc_verdict=str(cc_res["CC_verdict"]),
        )
        certificate_obj = build_certificate(
            {
                "reason_codes": reason_codes,
                "suite_id": suite_id,
                "experiment_id": experiment_id,
                "intervention_family_id": intervention_family_id,
                "candidate_generator_id": candidate_generator_id,
                "component_granularity": component_granularity,
                "model_id": model_id,
                "model_revision": model_revision or "unknown",
                "dataset_fingerprint": dataset_fp,
                "reference_dataset_fingerprint": ref_fp,
                "baseline_score_s0": float(s0),
                "epsilon": float(epsilon),
                "tau": float(tau_abs),
                "reference_distribution_id": refdist_id,
                "reference_assignment_id": refassign_id,
                "RR": float(rr_res["RR"]),
                "RR_verdict": rr_res["RR_verdict"],
                "SSS": float(sss_res["SSS"]),
                "SSS_verdict": sss_res["SSS_verdict"],
                "CC": float(cc_res["CC"]),
                "CC_verdict": cc_res["CC_verdict"],
                "s_near": [
                    {
                        "components": sorted([str(x) for x in r.get("components", [])]),
                        "delta": float(r.get("delta", 0.0)),
                        "mdl": float(r.get("mdl", 0.0)),
                        "necessity": dict(r.get("necessity", {})),
                    }
                    for r in s_near
                ],
                "replicate_circuits": [sorted(list(c)) for c in replicate_circuits],
            },
            canonical,
        )
        certificate_path = run_dir / str(get_canonical(canonical, "FILES.CERTIFICATE_FILENAME"))
        _write_json(certificate_path, certificate_obj)
        logger.log(
            canonical["ENUMS"]["LOG_EVENT_TYPES"]["EVENT_CERTIFICATE_WRITTEN"],
            {"certificate_file": certificate_path.name},
        )

    diagnostics = {
        "suite_id": suite_id,
        "experiment_id": experiment_id,
        "intervention_family_id": intervention_family_id,
        "candidate_generator_id": candidate_generator_id,
        "component_granularity": component_granularity,
        "setting_id": str(setting.get("setting_id")),
        "template_id": str(setting.get("template_id")),
        "redundancy_factor": int(setting.get("redundancy_factor")),
        "baseline_score_s0": float(s0),
        "epsilon": float(epsilon),
        "tau": float(tau_abs),
        "best_circuit": {"components": best_components, "mdl": float(best.get("mdl", 0.0)) if best is not None else 0.0, "delta": float(best.get("delta", 0.0)) if best is not None else 0.0},
        "RR": float(rr_res["RR"]),
        "RR_verdict": rr_res["RR_verdict"],
        "SSS": float(sss_res["SSS"]),
        "SSS_verdict": sss_res["SSS_verdict"],
        "CC": float(cc_res["CC"]),
        "CC_verdict": cc_res["CC_verdict"],
        "candidate_records": candidate_records,
        "s_near": [
            {"components": sorted([str(x) for x in r.get("components", [])]), "delta": float(r.get("delta", 0.0)), "mdl": float(r.get("mdl", 0.0)), "necessity": dict(r.get("necessity", {}))}
            for r in s_near
        ],
        "certificate_file": certificate_path.name if certificate_path is not None else None,
    }
    diag_path = run_dir / str(get_canonical(canonical, "FILES.DIAGNOSTICS_FILENAME"))
    _write_json(diag_path, diagnostics)

    logger.log(
        canonical["ENUMS"]["LOG_EVENT_TYPES"]["EVENT_CANDIDATES_WRITTEN"],
        {
            "num_candidates": int(len(candidate_circuits)),
            "topk_summary": [
                {"mdl": float(r.get("mdl", 0.0)), "size": int(len(r.get("components", []))), "delta": float(r.get("delta", 0.0))}
                for r in sorted(candidate_records, key=_candidate_key)[:5]
            ],
        },
    )
    logger.log(
        canonical["ENUMS"]["LOG_EVENT_TYPES"]["EVENT_DIAGNOSTICS_WRITTEN"],
        {
            "RR": float(rr_res["RR"]),
            "SSS": float(sss_res["SSS"]),
            "CC": float(cc_res["CC"]),
            "RR_verdict": rr_res["RR_verdict"],
            "SSS_verdict": sss_res["SSS_verdict"],
            "CC_verdict": cc_res["CC_verdict"],
            "certificate_emitted": bool(certificate_path is not None),
            "diagnostics_fingerprint": sha256_hex(diag_path.read_bytes()),
        },
    )

    verdicts = [rr_res["RR_verdict"], sss_res["SSS_verdict"], cc_res["CC_verdict"]]
    if canonical["ENUMS"]["VERDICTS"]["VERDICT_HARD_FAIL"] in verdicts:
        overall = canonical["ENUMS"]["VERDICTS"]["VERDICT_HARD_FAIL"]
    elif canonical["ENUMS"]["VERDICTS"]["VERDICT_FAIL"] in verdicts:
        overall = canonical["ENUMS"]["VERDICTS"]["VERDICT_FAIL"]
    elif canonical["ENUMS"]["VERDICTS"]["VERDICT_WARN"] in verdicts:
        overall = canonical["ENUMS"]["VERDICTS"]["VERDICT_WARN"]
    else:
        overall = canonical["ENUMS"]["VERDICTS"]["VERDICT_PASS"]

    run_record = {
        "schema_version": int(get_canonical(canonical, "SCHEMAS.RUN_RECORD_SCHEMA_VERSION")),
        "project_id": str(get_canonical(canonical, "PROJECT_ID")),
        "project_version": str(get_canonical(canonical, "PROJECT_VERSION")),
        str(get_canonical(canonical, "OUTPUT_NAMING.RUN_KEY_FIELD")): run_key,
        "run_id": run_id,
        "suite_id": suite_id,
        "experiment_id": experiment_id,
        "model_id": model_id,
        "model_revision": model_revision or "unknown",
        "intervention_family_id": intervention_family_id,
        "candidate_generator_id": candidate_generator_id,
        "component_granularity": component_granularity,
        "reference_distribution_id": refdist_id,
        "reference_assignment_id": refassign_id,
        "timestamp_utc": datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z"),
        "git_commit": _git_commit_hex(canonical=canonical, repo_root=repo_root),
        "python_version": environment_fingerprint()["python_version"],
        "platform": environment_fingerprint()["platform"],
        "deterministic_mode": bool(deterministic_mode),
        "seed_global": int(seed_global),
        "seed_suite": int(seed_suite),
        "seed_reference_pairing": int(seeds.seed_reference_pairing),
        "dataset_fingerprint": dataset_fp,
        "reference_dataset_fingerprint": ref_fp,
        "baseline_score_s0": float(s0),
        "epsilon": float(epsilon),
        "best_circuit_mdl": float(best.get("mdl", 0.0)) if best is not None else 0.0,
        "best_circuit_size": int(len(best_components)),
        "faithfulness_delta": float(best.get("delta", 0.0)) if best is not None else 0.0,
        "RR": float(rr_res["RR"]),
        "RR_verdict": rr_res["RR_verdict"],
        "SSS": float(sss_res["SSS"]),
        "SSS_verdict": sss_res["SSS_verdict"],
        "CC": float(cc_res["CC"]),
        "CC_verdict": cc_res["CC_verdict"],
        "overall_verdict": overall,
        "quality_gates_passed": bool(overall == canonical["ENUMS"]["VERDICTS"]["VERDICT_PASS"]),
        "paths": {
            "logs_dir": ".",
            "results_dir": ".",
            "diagnostics_file": diag_path.name,
            **({"certificate_file": certificate_path.name} if certificate_path is not None else {}),
        },
        # Synthetic setting provenance for report aggregation.
        "setting_id": str(setting.get("setting_id")),
        "template_id": str(setting.get("template_id")),
        "redundancy_factor": int(setting.get("redundancy_factor")),
    }

    run_record[str(get_canonical(canonical, "OUTPUT_NAMING.RUN_RECORD_HASH_FIELD"))] = compute_run_record_hash(
        run_record, canonical
    )
    write_run_record(str(run_dir / str(get_canonical(canonical, "FILES.RUN_RECORD_FILENAME"))), run_record)

    logger.log(
        canonical["ENUMS"]["LOG_EVENT_TYPES"]["EVENT_RUN_END"],
        {
            "overall_verdict": overall,
            "run_record": str(get_canonical(canonical, "FILES.RUN_RECORD_FILENAME")),
            "diagnostics_file": str(get_canonical(canonical, "FILES.DIAGNOSTICS_FILENAME")),
        },
    )


def _run_synth(
    args: Any,
    *,
    experiment_filter: set[str] | None,
) -> int:
    common = _common_args(args)
    repo_root = Path.cwd()
    canonical = load_canonical(str(repo_root))

    exp_cfg = load_config(common.config_path, canonical)
    gen_cfg = load_config(str(repo_root / str(get_canonical(canonical, "FILES.CONFIG_GENERATORS"))), canonical)
    int_cfg = load_config(str(repo_root / str(get_canonical(canonical, "FILES.CONFIG_INTERVENTIONS"))), canonical)
    det_cfg = load_config(str(repo_root / str(get_canonical(canonical, "FILES.CONFIG_DETERMINISM"))), canonical)
    budgets_cfg = load_config(str(repo_root / str(get_canonical(canonical, "FILES.CONFIG_BUDGETS"))), canonical)

    deterministic_mode = bool(common.deterministic or bool(det_cfg.get("deterministic_mode_default_key", False)))
    configure_determinism(deterministic_mode=deterministic_mode, canonical=canonical)

    suite_id = str(args.suite_id)
    if suite_id != "SUITE_SYNTH_V1":
        raise RunError(f"synth commands only support SUITE_SYNTH_V1, got {suite_id}")

    suites = list(exp_cfg.get("suites", []))
    suite_entry = next((s for s in suites if str(s.get("suite_id")) == suite_id), None)
    if suite_entry is None:
        raise RunError(f"suite_id not found in config: {suite_id}")

    experiment_ids = [str(x) for x in suite_entry.get("experiment_ids", [])]
    if experiment_filter is not None:
        experiment_ids = [x for x in experiment_ids if x in experiment_filter]

    if not experiment_ids:
        raise RunError("No synthetic experiment_ids selected from configs/experiments.yaml")

    seed_suite = int(args.seed) if getattr(args, "seed", None) is not None else int(suite_entry.get("seed_key"))
    seed_global = int(canonical["SEEDS"]["SEED_GLOBAL"])

    component_granularity = str(get_canonical(canonical, "COMPONENT_GRANULARITY.GRANULARITY_SYNTH_NODE"))
    if component_granularity not in canonical["COMPONENT_GRANULARITY"]["GRANULARITY_OPTIONS"]:
        raise RunError(f"Invalid synthetic component_granularity: {component_granularity}")

    # Enabled generators and intervention families.
    generators: list[str] = []
    for g in gen_cfg.get("generators", []):
        if bool(g.get("enabled_key", False)):
            generators.append(str(g.get("id")))
    if not generators:
        raise RunError("No enabled candidate generators in configs/generators.yaml")

    interventions = [str(x.get("id")) for x in int_cfg.get("intervention_families", [])]
    if not interventions:
        raise RunError("No intervention families listed in configs/interventions.yaml")
    _enforce_paper_repro_interventions(interventions=interventions, canonical=canonical)

    budgets_block = budgets_cfg.get("budgets", {})
    budget_synth = budgets_block.get("synth_candidate_budget_key") or budgets_block.get("default_budget_key")
    if not isinstance(budget_synth, dict):
        raise RunError("Resolved synth_candidate_budget_key budget is not a mapping")

    # Config hashes for run_key.
    resolved_config_hashes = {
        "experiments": _sha256_file(repo_root / common.config_path),
        "generators": _sha256_file(repo_root / str(get_canonical(canonical, "FILES.CONFIG_GENERATORS"))),
        "interventions": _sha256_file(repo_root / str(get_canonical(canonical, "FILES.CONFIG_INTERVENTIONS"))),
        "determinism": _sha256_file(repo_root / str(get_canonical(canonical, "FILES.CONFIG_DETERMINISM"))),
        "budgets": _sha256_file(repo_root / str(get_canonical(canonical, "FILES.CONFIG_BUDGETS"))),
        "logging_schema": _sha256_file(repo_root / str(get_canonical(canonical, "FILES.CONFIG_LOGGING_SCHEMA"))),
    }

    output_root = _resolve_output_root(repo_root, canonical, common.output_root)
    runs_root = output_root / Path(str(get_canonical(canonical, "PATHS.PATH_RUNS_ROOT"))).name
    runs_root.mkdir(parents=True, exist_ok=True)

    # Build the suite (includes all settings); filter per experiment.
    suite_builder = get_suite(suite_id)
    suite = suite_builder(canonical=canonical, seed=seed_suite, model_id=suite_id)

    for experiment_id in experiment_ids:
        settings = _synth_settings_for_experiment(suite, experiment_id=experiment_id, canonical=canonical)
        for setting in settings:
            # Propagate reference ids for run_key computation.
            setting = {
                **setting,
                "reference_distribution_id": suite.get("reference_distribution_id"),
                "reference_assignment_id": suite.get("reference_assignment_id"),
            }
            for intervention_family_id in interventions:
                for generator_id in generators:
                    _run_one_synth(
                        repo_root=repo_root,
                        canonical=canonical,
                        runs_root=runs_root,
                        suite_id=suite_id,
                        experiment_id=experiment_id,
                        intervention_family_id=intervention_family_id,
                        candidate_generator_id=generator_id,
                        component_granularity=component_granularity,
                        model_id=suite_id,
                        model_revision="unknown",
                        setting=setting,
                        seed_suite=seed_suite,
                        seed_global=seed_global,
                        resolved_config_hashes=resolved_config_hashes,
                        budgets=budget_synth,
                        deterministic_mode=deterministic_mode,
                    )

    return 0


def run_synth_generate_only(args: Any) -> int:
    return _run_synth(args, experiment_filter={"EXP_SYNTH_SANITY_V1"})


def run_synth_generate_with_candidates(args: Any) -> int:
    return _run_synth(args, experiment_filter={"EXP_SYNTH_REDUNDANCY_SWEEP_V1"})


def run_synth_diagnostics(args: Any) -> int:
    # v1.0.3 baseline: diagnostics runs the same full matrix (sweep + sanity) for convenience.
    return _run_synth(args, experiment_filter=None)


def run_real_matrix(args: Any) -> int:
    common = _common_args(args)
    repo_root = Path.cwd()
    canonical = load_canonical(str(repo_root))

    exp_cfg = load_config(common.config_path, canonical)
    gen_cfg = load_config(str(repo_root / str(get_canonical(canonical, "FILES.CONFIG_GENERATORS"))), canonical)
    int_cfg = load_config(str(repo_root / str(get_canonical(canonical, "FILES.CONFIG_INTERVENTIONS"))), canonical)
    det_cfg = load_config(str(repo_root / str(get_canonical(canonical, "FILES.CONFIG_DETERMINISM"))), canonical)
    budgets_cfg = load_config(str(repo_root / str(get_canonical(canonical, "FILES.CONFIG_BUDGETS"))), canonical)

    deterministic_mode = bool(common.deterministic or bool(det_cfg.get("deterministic_mode_default_key", False)))
    configure_determinism(deterministic_mode=deterministic_mode, canonical=canonical)

    suite_id = str(args.suite_id)
    experiment_id = str(args.experiment_id)

    suites = list(exp_cfg.get("suites", []))
    suite_entry = next((s for s in suites if str(s.get("suite_id")) == suite_id), None)
    if suite_entry is None:
        raise RunError(f"suite_id not found in config: {suite_id}")

    seed_suite = int(args.seed) if getattr(args, "seed", None) is not None else int(suite_entry.get("seed_key"))

    default_model_id = str(exp_cfg.get("defaults", {}).get("model_id_key"))
    model_id = str(args.model_id) if getattr(args, "model_id", None) else default_model_id

    component_granularity = str(exp_cfg.get("defaults", {}).get("component_granularity_key"))
    if component_granularity not in canonical["COMPONENT_GRANULARITY"]["GRANULARITY_OPTIONS"]:
        raise RunError(f"Invalid component_granularity: {component_granularity}")

    # Enabled generators and intervention families.
    generators = []
    for g in gen_cfg.get("generators", []):
        if bool(g.get("enabled_key", False)):
            generators.append(str(g.get("id")))
    if not generators:
        raise RunError("No enabled candidate generators in configs/generators.yaml")

    interventions = [str(x.get("id")) for x in int_cfg.get("intervention_families", [])]
    if not interventions:
        raise RunError("No intervention families listed in configs/interventions.yaml")
    _enforce_paper_repro_interventions(interventions=interventions, canonical=canonical)

    # Budgets
    budgets_block = budgets_cfg.get("budgets", {})
    budget_real = budgets_block.get("real_candidate_budget_key") or budgets_block.get("default_budget_key")
    if not isinstance(budget_real, dict):
        raise RunError("Resolved real_candidate_budget_key budget is not a mapping")

    # Load model once.
    device = common.device or ("cuda" if torch.cuda.is_available() else "cpu")
    dtype = "bfloat16" if device.startswith("cuda") else "float32"
    model, resolved_model_id, model_revision, model_local_path = _load_model(
        canonical=canonical, model_id=model_id, device=device, dtype=dtype
    )
    model.eval()
    hooks = make_hooks(model, canonical)

    # Build suite dataset once.
    suite_builder = get_suite(suite_id)
    suite = suite_builder(canonical=canonical, seed=seed_suite, model_id=resolved_model_id)
    ood_experiment_ids = {
        str(get_canonical(canonical, "IDS.EXPERIMENT_IDS.EXP_REAL_IOI_OOD_V1")),
        str(get_canonical(canonical, "IDS.EXPERIMENT_IDS.EXP_REAL_GREATERTHAN_OOD_V1")),
        str(get_canonical(canonical, "IDS.EXPERIMENT_IDS.EXP_REAL_GREATERTHAN_YN_OOD_V1")),
        str(get_canonical(canonical, "IDS.EXPERIMENT_IDS.EXP_REAL_INDUCTION_OOD_V1")),
    }
    split_id = "ood" if experiment_id in ood_experiment_ids else "id"
    eval_rows = list(suite["splits"].get(split_id, []))
    if not eval_rows:
        raise RunError(f"Empty eval split for suite={suite_id} split={split_id}")

    eval_batch, ref_tokens, dataset_fp, ref_fp = _build_eval_batch(
        suite_id=suite_id, model=model, eval_rows=eval_rows, canonical=canonical
    )

    # Baseline metric scale for epsilon and tau.
    eval_batch_size = int(budget_real["EVAL_BATCH_SIZE"])
    baseline_vals = _compute_metric_batched(suite_id=suite_id, model=model, batch=eval_batch, batch_size=eval_batch_size)
    s0 = float(sum(abs(v) for v in baseline_vals) / len(baseline_vals)) if baseline_vals else 0.0
    eps_abs_min = float(canonical["DIAGNOSTICS"]["EPSILON_ABS_MIN"])
    eps_rel = float(canonical["DIAGNOSTICS"]["EPSILON_REL_FRAC"])
    epsilon = max(eps_abs_min, eps_rel * s0)
    tau_rel = float(canonical["DIAGNOSTICS"]["TAU_REL_FRAC_NECESSITY"])
    tau_abs = max(eps_abs_min, tau_rel * s0)

    # Config hashes for run_key.
    resolved_config_hashes = {
        "experiments": _sha256_file(repo_root / common.config_path),
        "generators": _sha256_file(repo_root / str(get_canonical(canonical, "FILES.CONFIG_GENERATORS"))),
        "interventions": _sha256_file(repo_root / str(get_canonical(canonical, "FILES.CONFIG_INTERVENTIONS"))),
        "determinism": _sha256_file(repo_root / str(get_canonical(canonical, "FILES.CONFIG_DETERMINISM"))),
        "budgets": _sha256_file(repo_root / str(get_canonical(canonical, "FILES.CONFIG_BUDGETS"))),
        "logging_schema": _sha256_file(repo_root / str(get_canonical(canonical, "FILES.CONFIG_LOGGING_SCHEMA"))),
    }

    dataset_fps = {"eval": dataset_fp, "ref": ref_fp}

    seed_global = int(canonical["SEEDS"]["SEED_GLOBAL"])
    output_root = _resolve_output_root(repo_root, canonical, common.output_root)
    runs_rel = Path(str(get_canonical(canonical, "PATHS.PATH_RUNS_ROOT")))
    out_rel = Path(str(get_canonical(canonical, "PATHS.PATH_OUTPUT_ROOT")))
    try:
        runs_sub = runs_rel.relative_to(out_rel)
    except ValueError:
        runs_sub = Path("runs")
    runs_root = output_root / runs_sub
    runs_root.mkdir(parents=True, exist_ok=True)

    # Run the matrix: each (intervention, generator) writes an immutable run directory.
    for intervention_family_id in interventions:
        for generator_id in generators:
            _run_one(
                repo_root=repo_root,
                canonical=canonical,
                runs_root=runs_root,
                suite_id=suite_id,
                experiment_id=experiment_id,
                intervention_family_id=intervention_family_id,
                candidate_generator_id=generator_id,
                component_granularity=component_granularity,
                model_id=resolved_model_id,
                model_revision=model_revision,
                model_local_path=model_local_path,
                model=model,
                hooks=hooks,
                eval_batch=eval_batch,
                ref_tokens=ref_tokens,
                suite_reference_ids=(str(suite["reference_distribution_id"]), str(suite["reference_assignment_id"])),
                seed_suite=seed_suite,
                seed_global=seed_global,
                # Per-run derived from run_key inside _run_one.
                seed_reference_pairing=None,
                resolved_config_hashes=resolved_config_hashes,
                dataset_fps=dataset_fps,
                budgets=budget_real,
                epsilon=epsilon,
                tau_abs=tau_abs,
                deterministic_mode=deterministic_mode,
            )

    return 0


def aggregate_results(args: Any) -> int:
    common = _common_args(args)
    repo_root = Path.cwd()
    canonical = load_canonical(str(repo_root))

    det_cfg = load_config(str(repo_root / str(get_canonical(canonical, "FILES.CONFIG_DETERMINISM"))), canonical)
    deterministic_mode = bool(common.deterministic or bool(det_cfg.get("deterministic_mode_default_key", False)))
    configure_determinism(deterministic_mode=deterministic_mode, canonical=canonical)

    output_root = _resolve_output_root(repo_root, canonical, common.output_root)
    runs_root = _runs_root_for_output_root(output_root=output_root, canonical=canonical)
    reports_root = _reports_root_for_output_root(output_root=output_root, canonical=canonical)
    reports_root.mkdir(parents=True, exist_ok=True)

    _, report_dir = _make_report_dir(reports_root=reports_root, canonical=canonical)

    run_records = _load_run_records(runs_root=runs_root, canonical=canonical)
    if not run_records:
        raise RunError(f"No run records found under runs root: {runs_root}")

    _write_report_tables(report_dir=report_dir, canonical=canonical, run_records=run_records)
    return 0


def build_paper_artifacts(args: Any) -> int:
    common = _common_args(args)
    repo_root = Path.cwd()
    canonical = load_canonical(str(repo_root))

    det_cfg = load_config(str(repo_root / str(get_canonical(canonical, "FILES.CONFIG_DETERMINISM"))), canonical)
    deterministic_mode = bool(common.deterministic or bool(det_cfg.get("deterministic_mode_default_key", False)))
    configure_determinism(deterministic_mode=deterministic_mode, canonical=canonical)

    output_root = _resolve_output_root(repo_root, canonical, common.output_root)
    runs_root = _runs_root_for_output_root(output_root=output_root, canonical=canonical)
    reports_root = _reports_root_for_output_root(output_root=output_root, canonical=canonical)
    reports_root.mkdir(parents=True, exist_ok=True)

    _, report_dir = _make_report_dir(reports_root=reports_root, canonical=canonical)

    run_records = _load_run_records(runs_root=runs_root, canonical=canonical)
    if not run_records:
        raise RunError(f"No run records found under runs root: {runs_root}")

    # Paper artifacts are scope-locked to activation patching (I_ACTPATCH).
    actpatch_id = str(get_canonical(canonical, "IDS.INTERVENTION_FAMILY_IDS.I_ACTPATCH"))
    run_records_paper = [r for r in run_records if str(r.get("intervention_family_id")) == actpatch_id]
    if not run_records_paper:
        raise RunError(f"No I_ACTPATCH run records found under runs root: {runs_root}")

    paper_suite_id_keys = canonical.get("PAPER", {}).get("PAPER_SUITE_ID_KEYS")
    if not isinstance(paper_suite_id_keys, list) or not paper_suite_id_keys:
        raise RunError("Missing or invalid CANONICAL.PAPER.PAPER_SUITE_ID_KEYS")
    paper_suite_ids = {str(get_canonical(canonical, str(k))) for k in paper_suite_id_keys}
    run_records_paper = [r for r in run_records_paper if str(r.get("suite_id")) in paper_suite_ids]
    if not run_records_paper:
        raise RunError(f"No paper-eligible runs found under runs root after suite filter: {runs_root}")

    tables = _write_report_tables(report_dir=report_dir, canonical=canonical, run_records=run_records_paper)
    _write_fig_synth(canonical=canonical, report_dir=report_dir, table_t1=tables["T1"])
    return 0


def validate_paper_manifest(args: Any) -> int:
    common = _common_args(args)
    repo_root = Path.cwd()
    canonical = load_canonical(str(repo_root))

    # Ensure configs resolve (SDR + canonical references), even though validation is read-only.
    _ = load_config(common.config_path, canonical)

    manifest_rel = str(get_canonical(canonical, "FILES.PAPER_RESULTS_MANIFEST"))
    manifest_path = repo_root / manifest_rel
    if not manifest_path.exists():
        raise RunError(f"Missing paper results manifest: {manifest_path}")

    manifest_obj = json.loads(manifest_path.read_text(encoding="utf-8"))
    if not isinstance(manifest_obj, dict):
        raise RunError(f"{manifest_rel} did not parse as a JSON mapping")

    expected_schema = int(get_canonical(canonical, "SCHEMAS.PAPER_RESULTS_MANIFEST_SCHEMA_VERSION"))
    schema_version = manifest_obj.get("schema_version")
    if int(schema_version) != expected_schema:
        raise RunError(
            f"{manifest_rel} schema_version mismatch: got {schema_version}, expected {expected_schema}"
        )

    if str(manifest_obj.get("project_id")) != str(canonical["PROJECT_ID"]):
        raise RunError(
            f"{manifest_rel} project_id mismatch: got {manifest_obj.get('project_id')}, expected {canonical['PROJECT_ID']}"
        )
    if str(manifest_obj.get("project_version")) != str(canonical["PROJECT_VERSION"]):
        raise RunError(
            f"{manifest_rel} project_version mismatch: got {manifest_obj.get('project_version')}, expected {canonical['PROJECT_VERSION']}"
        )

    def _require_relpath(path_s: str) -> Path:
        if Path(path_s).is_absolute() or ":" in path_s:
            raise RunError(f"{manifest_rel} contains non-relative path: {path_s}")
        p = Path(path_s)
        if ".." in p.parts:
            raise RunError(f"{manifest_rel} contains forbidden '..' path segment: {path_s}")
        return p

    def _validate_file_hash(*, relpath: str, expected_sha256: str) -> Path:
        p = _require_relpath(relpath)
        abs_path = repo_root / p
        if not abs_path.exists():
            raise RunError(f"Missing artifact referenced by {manifest_rel}: {relpath}")
        actual = _sha256_file(abs_path).lower()
        expected = expected_sha256.lower()
        if actual != expected:
            raise RunError(f"sha256 mismatch for {relpath}: got {actual}, expected {expected}")
        return abs_path

    artifacts = manifest_obj.get("artifacts")
    if not isinstance(artifacts, list):
        raise RunError(f"{manifest_rel} missing artifacts list")
    artifacts_by_id: dict[str, dict[str, Any]] = {}
    for a in artifacts:
        if not isinstance(a, dict):
            raise RunError(f"{manifest_rel} artifact entry is not a mapping: {a!r}")
        artifact_id = a.get("artifact_id")
        path_s = a.get("path")
        sha = a.get("sha256")
        if not isinstance(artifact_id, str) or not artifact_id:
            raise RunError(f"{manifest_rel} artifact_id missing/invalid: {a!r}")
        if artifact_id in artifacts_by_id:
            raise RunError(f"{manifest_rel} duplicate artifact_id: {artifact_id}")
        if not isinstance(path_s, str) or not path_s:
            raise RunError(f"{manifest_rel} artifact path missing/invalid for {artifact_id}")
        if not isinstance(sha, str) or not sha:
            raise RunError(f"{manifest_rel} artifact sha256 missing/invalid for {artifact_id}")
        _validate_file_hash(relpath=path_s, expected_sha256=sha)
        artifacts_by_id[artifact_id] = a

    runs = manifest_obj.get("runs")
    if not isinstance(runs, list):
        raise RunError(f"{manifest_rel} missing runs list")
    runs_by_id: dict[str, dict[str, Any]] = {}
    required_run_files = {
        str(get_canonical(canonical, "FILES.RUN_RECORD_FILENAME")),
        str(get_canonical(canonical, "FILES.LOG_JSONL_FILENAME")),
        str(get_canonical(canonical, "FILES.DIAGNOSTICS_FILENAME")),
        str(get_canonical(canonical, "FILES.BEST_CIRCUIT_FILENAME")),
    }
    for r in runs:
        if not isinstance(r, dict):
            raise RunError(f"{manifest_rel} run entry is not a mapping: {r!r}")
        run_id = r.get("run_id")
        dir_path = r.get("dir_path")
        files = r.get("files")
        if not isinstance(run_id, str) or not run_id:
            raise RunError(f"{manifest_rel} run_id missing/invalid: {r!r}")
        if run_id in runs_by_id:
            raise RunError(f"{manifest_rel} duplicate run_id: {run_id}")
        if not isinstance(dir_path, str) or not dir_path:
            raise RunError(f"{manifest_rel} dir_path missing/invalid for {run_id}")
        if not isinstance(files, list) or not files:
            raise RunError(f"{manifest_rel} files missing/invalid for {run_id}")

        abs_dir = repo_root / _require_relpath(dir_path)
        if not abs_dir.exists():
            raise RunError(f"{manifest_rel} run directory not found for {run_id}: {dir_path}")

        seen_names: set[str] = set()
        for f in files:
            if not isinstance(f, dict):
                raise RunError(f"{manifest_rel} run file entry is not a mapping: {f!r}")
            fp = f.get("path")
            sha = f.get("sha256")
            if not isinstance(fp, str) or not fp:
                raise RunError(f"{manifest_rel} run file path missing/invalid for {run_id}")
            if not isinstance(sha, str) or not sha:
                raise RunError(f"{manifest_rel} run file sha256 missing/invalid for {run_id}")
            abs_path = _validate_file_hash(relpath=fp, expected_sha256=sha)
            if abs_dir not in abs_path.parents:
                raise RunError(f"{manifest_rel} run file is not under its run dir: {fp} (run {run_id})")
            seen_names.add(abs_path.name)

        missing = sorted(required_run_files - seen_names)
        if missing:
            raise RunError(f"{manifest_rel} run {run_id} missing required files: {missing}")
        runs_by_id[run_id] = r

    # Ensure the manifest covers all claims defined in spec/19_PAPER_WRITING_PLAN.md.
    spec_dir = Path(str(get_canonical(canonical, "PATHS.PATH_SPEC_DIR")))
    plan_path = repo_root / spec_dir / "19_PAPER_WRITING_PLAN.md"
    claim_pat = re.compile(r"^###\s+Claim\s+(C\d+):")
    expected_claim_ids: list[str] = []
    for line in plan_path.read_text(encoding="utf-8").splitlines():
        m = claim_pat.match(line.strip())
        if m:
            expected_claim_ids.append(m.group(1))
    if not expected_claim_ids:
        raise RunError(f"Could not extract any claim IDs from {plan_path}")

    claims = manifest_obj.get("claims")
    if not isinstance(claims, list):
        raise RunError(f"{manifest_rel} missing claims list")
    claims_by_id: dict[str, dict[str, Any]] = {}
    for c in claims:
        if not isinstance(c, dict):
            raise RunError(f"{manifest_rel} claim entry is not a mapping: {c!r}")
        claim_id = c.get("claim_id")
        evidence = c.get("evidence")
        if not isinstance(claim_id, str) or not claim_id:
            raise RunError(f"{manifest_rel} claim_id missing/invalid: {c!r}")
        if claim_id in claims_by_id:
            raise RunError(f"{manifest_rel} duplicate claim_id: {claim_id}")
        if not isinstance(evidence, list) or not evidence:
            raise RunError(f"{manifest_rel} claim {claim_id} missing evidence list")
        claims_by_id[claim_id] = c

    missing_claims = sorted(set(expected_claim_ids) - set(claims_by_id))
    if missing_claims:
        raise RunError(f"{manifest_rel} missing claims: {missing_claims}")

    def _validate_csv_row_cells(*, artifact_path: Path, where: dict[str, Any], expect: dict[str, Any]) -> None:
        with artifact_path.open("r", encoding="utf-8", newline="") as f:
            reader = csv.DictReader(f)
            rows = list(reader)
        matches: list[dict[str, str]] = []
        for row in rows:
            ok = True
            for k, v in where.items():
                if k not in row:
                    raise RunError(f"CSV missing where-column {k} in {artifact_path}")
                if row[k] != str(v):
                    ok = False
                    break
            if ok:
                matches.append(row)
        if len(matches) != 1:
            raise RunError(f"CSV selector matched {len(matches)} rows in {artifact_path} (expected 1)")
        row = matches[0]
        for k, v in expect.items():
            if k not in row:
                raise RunError(f"CSV missing expect-column {k} in {artifact_path}")
            if row[k] != str(v):
                raise RunError(
                    f"CSV value mismatch in {artifact_path} for {k}: got {row[k]!r}, expected {str(v)!r}"
                )

    def _validate_json_fields(*, artifact_path: Path, expect: dict[str, Any]) -> None:
        obj = json.loads(artifact_path.read_text(encoding="utf-8"))
        if not isinstance(obj, dict):
            raise RunError(f"JSON artifact did not parse as mapping: {artifact_path}")
        for k, v in expect.items():
            if k.endswith("_len"):
                base = k.removesuffix("_len")
                if base not in obj or not isinstance(obj[base], list):
                    raise RunError(f"JSON missing list field for length check: {base} in {artifact_path}")
                if len(obj[base]) != int(v):
                    raise RunError(
                        f"JSON length mismatch in {artifact_path} for {base}: got {len(obj[base])}, expected {int(v)}"
                    )
                continue
            if k not in obj:
                raise RunError(f"JSON missing field {k} in {artifact_path}")
            if obj[k] != v:
                raise RunError(f"JSON value mismatch in {artifact_path} for {k}: got {obj[k]!r}, expected {v!r}")

    for claim_id, claim in claims_by_id.items():
        evidence_list = claim.get("evidence")
        if not isinstance(evidence_list, list):
            raise RunError(f"{manifest_rel} claim {claim_id} evidence is not a list")
        for ev in evidence_list:
            if not isinstance(ev, dict):
                raise RunError(f"{manifest_rel} evidence entry is not a mapping: {ev!r}")
            evidence_type = ev.get("evidence_type")
            artifact_id = ev.get("artifact_id")
            run_ids = ev.get("run_ids", [])
            if not isinstance(evidence_type, str) or not evidence_type:
                raise RunError(f"{manifest_rel} evidence_type missing/invalid in claim {claim_id}")
            if not isinstance(artifact_id, str) or not artifact_id:
                raise RunError(f"{manifest_rel} artifact_id missing/invalid in claim {claim_id}")
            if artifact_id not in artifacts_by_id:
                raise RunError(f"{manifest_rel} unknown artifact_id {artifact_id} in claim {claim_id}")
            if run_ids:
                if not isinstance(run_ids, list) or not all(isinstance(x, str) and x for x in run_ids):
                    raise RunError(f"{manifest_rel} run_ids must be a list of strings in claim {claim_id}")
                missing_runs = sorted(set(run_ids) - set(runs_by_id))
                if missing_runs:
                    raise RunError(f"{manifest_rel} unknown run_ids {missing_runs} in claim {claim_id}")

            art_path_s = artifacts_by_id[artifact_id].get("path")
            if not isinstance(art_path_s, str) or not art_path_s:
                raise RunError(f"{manifest_rel} artifact {artifact_id} missing path")
            artifact_path = _validate_file_hash(
                relpath=art_path_s,
                expected_sha256=str(artifacts_by_id[artifact_id].get("sha256")),
            )

            if evidence_type == "artifact_hash":
                continue
            if evidence_type == "csv_row_cells":
                where = ev.get("where")
                expect = ev.get("expect")
                if not isinstance(where, dict) or not isinstance(expect, dict):
                    raise RunError(f"{manifest_rel} csv_row_cells requires where+expect in claim {claim_id}")
                _validate_csv_row_cells(artifact_path=artifact_path, where=where, expect=expect)
                continue
            if evidence_type == "json_fields":
                expect = ev.get("expect")
                if not isinstance(expect, dict):
                    raise RunError(f"{manifest_rel} json_fields requires expect in claim {claim_id}")
                _validate_json_fields(artifact_path=artifact_path, expect=expect)
                continue
            raise RunError(f"{manifest_rel} unsupported evidence_type: {evidence_type}")

    return 0


def determinism_smoke_test(args: Any) -> int:
    from types import SimpleNamespace

    common = _common_args(args)
    repo_root = Path.cwd()
    canonical = load_canonical(str(repo_root))

    det_cfg = load_config(str(repo_root / str(get_canonical(canonical, "FILES.CONFIG_DETERMINISM"))), canonical)
    deterministic_mode = bool(common.deterministic or bool(det_cfg.get("deterministic_mode_default_key", False)))
    if not deterministic_mode:
        raise RunError("determinism_smoke requires deterministic mode (--deterministic)")
    configure_determinism(deterministic_mode=True, canonical=canonical)

    output_root = _resolve_output_root(repo_root, canonical, common.output_root)
    reports_root = _reports_root_for_output_root(output_root=output_root, canonical=canonical)
    reports_root.mkdir(parents=True, exist_ok=True)

    # Use fresh report directories as isolated output roots for the two runs.
    _, out_a = _make_report_dir(reports_root=reports_root, canonical=canonical)
    _, out_b = _make_report_dir(reports_root=reports_root, canonical=canonical)
    _, report_dir = _make_report_dir(reports_root=reports_root, canonical=canonical)

    suite_id = str(get_canonical(canonical, "IDS.SUITE_IDS.SUITE_SYNTH_V1"))
    seed_suite = int(get_canonical(canonical, "SEEDS.SEED_SYNTH_SUITE"))

    synth_args_a = SimpleNamespace(
        config_path=common.config_path,
        deterministic=True,
        device=common.device,
        output_root=str(out_a),
        suite_id=suite_id,
        seed=seed_suite,
    )
    synth_args_b = SimpleNamespace(
        config_path=common.config_path,
        deterministic=True,
        device=common.device,
        output_root=str(out_b),
        suite_id=suite_id,
        seed=seed_suite,
    )

    # Run the synthetic redundancy sweep twice.
    _run_synth(synth_args_a, experiment_filter={str(get_canonical(canonical, "IDS.EXPERIMENT_IDS.EXP_SYNTH_REDUNDANCY_SWEEP_V1"))})
    _run_synth(synth_args_b, experiment_filter={str(get_canonical(canonical, "IDS.EXPERIMENT_IDS.EXP_SYNTH_REDUNDANCY_SWEEP_V1"))})

    rr_name = str(get_canonical(canonical, "FILES.RUN_RECORD_FILENAME"))
    diag_name = str(get_canonical(canonical, "FILES.DIAGNOSTICS_FILENAME"))
    best_name = str(get_canonical(canonical, "FILES.BEST_CIRCUIT_FILENAME"))
    cert_name = str(get_canonical(canonical, "FILES.CERTIFICATE_FILENAME"))
    run_key_field = str(get_canonical(canonical, "OUTPUT_NAMING.RUN_KEY_FIELD"))
    rrh_field = str(get_canonical(canonical, "OUTPUT_NAMING.RUN_RECORD_HASH_FIELD"))

    def runs_root_for(out_root: Path) -> Path:
        return _runs_root_for_output_root(output_root=out_root, canonical=canonical)

    def map_run_key_to_dir(runs_root: Path) -> dict[str, Path]:
        mapping: dict[str, Path] = {}
        prefix = str(get_canonical(canonical, "OUTPUT_NAMING.RUN_DIR_PREFIX"))
        for run_dir in sorted([p for p in runs_root.iterdir() if p.is_dir() and p.name.startswith(prefix)]):
            rr_path = run_dir / rr_name
            if not rr_path.exists():
                continue
            rec = json.loads(rr_path.read_text(encoding="utf-8"))
            rk = rec.get(run_key_field)
            if not isinstance(rk, str) or not rk:
                continue
            if rk in mapping:
                raise RunError(f"Duplicate run_key in runs_root: {rk}")
            mapping[rk] = run_dir
        return mapping

    runs_root_a = runs_root_for(out_a)
    runs_root_b = runs_root_for(out_b)
    map_a = map_run_key_to_dir(runs_root_a)
    map_b = map_run_key_to_dir(runs_root_b)

    if set(map_a.keys()) != set(map_b.keys()):
        only_a = sorted(set(map_a.keys()) - set(map_b.keys()))
        only_b = sorted(set(map_b.keys()) - set(map_a.keys()))
        raise RunError(f"Run key mismatch between runs. only_a={only_a[:3]} only_b={only_b[:3]}")

    # Compare per-run deterministic artifacts.
    for rk in sorted(map_a.keys()):
        run_dir_a = map_a[rk]
        run_dir_b = map_b[rk]
        rec_a = json.loads((run_dir_a / rr_name).read_text(encoding="utf-8"))
        rec_b = json.loads((run_dir_b / rr_name).read_text(encoding="utf-8"))
        if str(rec_a.get(rrh_field, "")) != str(rec_b.get(rrh_field, "")):
            raise RunError(f"run_record_hash mismatch for run_key={rk}")

        diag_a = (run_dir_a / diag_name).read_bytes()
        diag_b = (run_dir_b / diag_name).read_bytes()
        if diag_a != diag_b:
            raise RunError(f"diagnostics.json mismatch for run_key={rk}")

        best_a_path = run_dir_a / best_name
        best_b_path = run_dir_b / best_name
        if best_a_path.exists() != best_b_path.exists():
            raise RunError(f"best_circuit.json presence mismatch for run_key={rk}")
        if best_a_path.exists() and best_a_path.read_bytes() != best_b_path.read_bytes():
            raise RunError(f"best_circuit.json mismatch for run_key={rk}")

        cert_a_path = run_dir_a / cert_name
        cert_b_path = run_dir_b / cert_name
        if cert_a_path.exists() != cert_b_path.exists():
            raise RunError(f"certificate.json presence mismatch for run_key={rk}")
        if cert_a_path.exists() and cert_a_path.read_bytes() != cert_b_path.read_bytes():
            raise RunError(f"certificate.json mismatch for run_key={rk}")

    # Regenerate canonical tables from logs.jsonl only and compare bytes.
    def synth_rows_from_logs(runs_root: Path) -> list[dict[str, Any]]:
        prefix = str(get_canonical(canonical, "OUTPUT_NAMING.RUN_DIR_PREFIX"))
        logs_name = str(get_canonical(canonical, "FILES.LOG_JSONL_FILENAME"))
        ev_run_start = canonical["ENUMS"]["LOG_EVENT_TYPES"]["EVENT_RUN_START"]
        ev_diag = canonical["ENUMS"]["LOG_EVENT_TYPES"]["EVENT_DIAGNOSTICS_WRITTEN"]
        ev_run_end = canonical["ENUMS"]["LOG_EVENT_TYPES"]["EVENT_RUN_END"]

        rows: list[dict[str, Any]] = []
        for run_dir in sorted([p for p in runs_root.iterdir() if p.is_dir() and p.name.startswith(prefix)]):
            logs_path = run_dir / logs_name
            if not logs_path.exists():
                continue
            run_start: dict[str, Any] | None = None
            diag: dict[str, Any] | None = None
            run_end: dict[str, Any] | None = None
            for line in logs_path.read_text(encoding="utf-8").splitlines():
                if not line.strip():
                    continue
                obj = json.loads(line)
                if obj.get("event_type") == ev_run_start:
                    run_start = obj
                elif obj.get("event_type") == ev_diag:
                    diag = obj
                elif obj.get("event_type") == ev_run_end:
                    run_end = obj
            if run_start is None or diag is None or run_end is None:
                continue
            rk = run_start.get(run_key_field)
            if not isinstance(rk, str) or not rk:
                continue
            rows.append(
                {
                    "suite_id": str(run_start.get("suite_id")),
                    "experiment_id": str(run_start.get("experiment_id")),
                    "setting_id": str(run_start.get("setting_id", "")),
                    "template_id": str(run_start.get("template_id", "")),
                    "redundancy_factor": int(run_start.get("redundancy_factor", 0)),
                    "RR": float(diag.get("RR", 0.0)),
                    "SSS": float(diag.get("SSS", 0.0)),
                    "CC": float(diag.get("CC", 0.0)),
                    "certificate_emitted": bool(diag.get("certificate_emitted", False)),
                }
            )
        return rows

    def tables_from_logs(rows: list[dict[str, Any]]) -> dict[str, bytes]:
        sweep_id = str(get_canonical(canonical, "IDS.EXPERIMENT_IDS.EXP_SYNTH_REDUNDANCY_SWEEP_V1"))
        synth_id = str(get_canonical(canonical, "IDS.SUITE_IDS.SUITE_SYNTH_V1"))

        seed_global = int(canonical["SEEDS"]["SEED_GLOBAL"])
        salt = str(get_canonical(canonical, "DETERMINISM.SEED_DERIVATION.SALT_BOOTSTRAP"))
        resamples = int(get_canonical(canonical, "STATS.BOOTSTRAP_RESAMPLES"))
        ci_lo = float(get_canonical(canonical, "STATS.CI_LO"))
        ci_hi = float(get_canonical(canonical, "STATS.CI_HI"))

        grouped: dict[str, list[dict[str, Any]]] = {}
        for r in rows:
            if str(r.get("suite_id")) != synth_id:
                continue
            if str(r.get("experiment_id")) != sweep_id:
                continue
            sid = str(r.get("setting_id", ""))
            if not sid:
                continue
            grouped.setdefault(sid, []).append(r)

        from scrubid.hashing import sha256_uint32

        t1_rows: list[dict[str, Any]] = []
        for setting_id in sorted(grouped.keys()):
            recs = grouped[setting_id]
            redundancy_factor = int(recs[0].get("redundancy_factor", 0))
            rr_vals = [float(x.get("RR", 0.0)) for x in recs]
            sss_vals = [float(x.get("SSS", 0.0)) for x in recs]
            cc_vals = [float(x.get("CC", 0.0)) for x in recs]
            certs = [1.0 if bool(x.get("certificate_emitted", False)) else 0.0 for x in recs]

            rr_mean, rr_lo, rr_hi = _bootstrap_mean_ci(
                values=rr_vals,
                seed=sha256_uint32(f"{seed_global}|{salt}|{setting_id}|RR"),
                resamples=resamples,
                ci_lo=ci_lo,
                ci_hi=ci_hi,
            )
            sss_mean, sss_lo, sss_hi = _bootstrap_mean_ci(
                values=sss_vals,
                seed=sha256_uint32(f"{seed_global}|{salt}|{setting_id}|SSS"),
                resamples=resamples,
                ci_lo=ci_lo,
                ci_hi=ci_hi,
            )
            cc_mean, cc_lo, cc_hi = _bootstrap_mean_ci(
                values=cc_vals,
                seed=sha256_uint32(f"{seed_global}|{salt}|{setting_id}|CC"),
                resamples=resamples,
                ci_lo=ci_lo,
                ci_hi=ci_hi,
            )
            non_id_rate = float(sum(certs) / len(certs)) if certs else 0.0

            t1_rows.append(
                {
                    "setting_id": setting_id,
                    "planted_redundancy_factor": int(redundancy_factor),
                    "RR_mean": decimal_str(rr_mean),
                    "RR_ci": _format_ci(rr_lo, rr_hi),
                    "SSS_mean": decimal_str(sss_mean),
                    "SSS_ci": _format_ci(sss_lo, sss_hi),
                    "CC_mean": decimal_str(cc_mean),
                    "CC_ci": _format_ci(cc_lo, cc_hi),
                    "non_identifiability_rate": decimal_str(non_id_rate),
                }
            )

        t1_rows.sort(key=lambda r: (int(r["planted_redundancy_factor"]), str(r["setting_id"])))

        def csv_bytes(fieldnames: list[str], rows_: list[dict[str, Any]]) -> bytes:
            import csv
            import io

            buf = io.StringIO(newline="")
            w = csv.DictWriter(buf, fieldnames=fieldnames, extrasaction="ignore", lineterminator="\n")
            w.writeheader()
            for row in rows_:
                w.writerow({k: row.get(k, "") for k in fieldnames})
            return buf.getvalue().encode("utf-8")

        t1_bytes = csv_bytes(
            [
                "setting_id",
                "planted_redundancy_factor",
                "RR_mean",
                "RR_ci",
                "SSS_mean",
                "SSS_ci",
                "CC_mean",
                "CC_ci",
                "non_identifiability_rate",
            ],
            t1_rows,
        )
        t2_bytes = csv_bytes(
            [
                "suite_id",
                "intervention_family_id",
                "candidate_generator_id",
                "best_circuit_size",
                "faithfulness_delta",
                "RR",
                "SSS",
                "CC",
                "overall_verdict",
            ],
            [],
        )
        t3_bytes = csv_bytes(["variant_id", "Δ_size", "Δ_faithfulness", "Δ_RR", "Δ_SSS", "Δ_CC"], [])

        return {
            str(get_canonical(canonical, "FILES.REPORT_TABLE_T1_FILENAME")): t1_bytes,
            str(get_canonical(canonical, "FILES.REPORT_TABLE_T2_FILENAME")): t2_bytes,
            str(get_canonical(canonical, "FILES.REPORT_TABLE_T3_FILENAME")): t3_bytes,
        }

    tables_a = tables_from_logs(synth_rows_from_logs(runs_root_a))
    tables_b = tables_from_logs(synth_rows_from_logs(runs_root_b))
    if tables_a.keys() != tables_b.keys():
        raise RunError("Smoke tables key mismatch")
    for name in sorted(tables_a.keys()):
        if tables_a[name] != tables_b[name]:
            raise RunError(f"Smoke table mismatch for {name}")

    # Write the canonical tables as the smoke report.
    for name, b in tables_a.items():
        (report_dir / name).write_bytes(b)

    return 0


def reproduce_paper(args: Any) -> int:
    """
    Fresh-bundle reproduction of the main paper artifacts.

    This command:
      - allocates a new output_root (unless provided) and fails if it exists,
      - runs the canonical synth + real pipelines into that output_root,
      - builds paper artifacts,
      - verifies generated artifact hashes against paper_results_manifest.json.
    """
    from types import SimpleNamespace

    common = _common_args(args)
    repo_root = Path.cwd()
    canonical = load_canonical(str(repo_root))

    det_cfg = load_config(str(repo_root / str(get_canonical(canonical, "FILES.CONFIG_DETERMINISM"))), canonical)
    deterministic_mode = bool(common.deterministic or bool(det_cfg.get("deterministic_mode_default_key", False)))
    if not deterministic_mode:
        raise RunError("repro paper requires deterministic mode (--deterministic)")
    configure_determinism(deterministic_mode=True, canonical=canonical)

    # Allocate a fresh output_root (clean bundle root).
    if common.output_root:
        output_root = Path(common.output_root)
        if output_root.exists():
            raise RunError(f"repro paper output_root already exists: {output_root}")
        output_root.mkdir(parents=True, exist_ok=False)
    else:
        base_output_root = repo_root / str(get_canonical(canonical, "PATHS.PATH_OUTPUT_ROOT"))
        base_output_root.mkdir(parents=True, exist_ok=True)
        output_root = _make_repro_output_root(base_output_root=base_output_root, canonical=canonical)

    # 1) Synthetic suite (full matrix).
    synth_args = SimpleNamespace(
        config_path=common.config_path,
        deterministic=True,
        device=common.device,
        output_root=str(output_root),
        suite_id=str(get_canonical(canonical, "IDS.SUITE_IDS.SUITE_SYNTH_V1")),
        seed=int(get_canonical(canonical, "SEEDS.SEED_SYNTH_SUITE")),
    )
    run_synth_diagnostics(synth_args)

    # 2) Real suites (paper scope): ID + OOD for each paper suite.
    real_runs: list[tuple[str, str]] = [
        (
            str(get_canonical(canonical, "IDS.SUITE_IDS.SUITE_REAL_IOI_V1")),
            str(get_canonical(canonical, "IDS.EXPERIMENT_IDS.EXP_REAL_IOI_V1")),
        ),
        (
            str(get_canonical(canonical, "IDS.SUITE_IDS.SUITE_REAL_IOI_V1")),
            str(get_canonical(canonical, "IDS.EXPERIMENT_IDS.EXP_REAL_IOI_OOD_V1")),
        ),
        (
            str(get_canonical(canonical, "IDS.SUITE_IDS.SUITE_REAL_GREATERTHAN_YN_V1")),
            str(get_canonical(canonical, "IDS.EXPERIMENT_IDS.EXP_REAL_GREATERTHAN_YN_V1")),
        ),
        (
            str(get_canonical(canonical, "IDS.SUITE_IDS.SUITE_REAL_GREATERTHAN_YN_V1")),
            str(get_canonical(canonical, "IDS.EXPERIMENT_IDS.EXP_REAL_GREATERTHAN_YN_OOD_V1")),
        ),
        (
            str(get_canonical(canonical, "IDS.SUITE_IDS.SUITE_REAL_INDUCTION_V1")),
            str(get_canonical(canonical, "IDS.EXPERIMENT_IDS.EXP_REAL_INDUCTION_V1")),
        ),
        (
            str(get_canonical(canonical, "IDS.SUITE_IDS.SUITE_REAL_INDUCTION_V1")),
            str(get_canonical(canonical, "IDS.EXPERIMENT_IDS.EXP_REAL_INDUCTION_OOD_V1")),
        ),
    ]
    base_seed_real = int(get_canonical(canonical, "SEEDS.SEED_REAL_SUITE"))
    seed_offsets = list(get_canonical(canonical, "SEEDS.PAPER_REAL_SEED_OFFSETS"))
    for suite_id, experiment_id in real_runs:
        for offset in seed_offsets:
            real_args = SimpleNamespace(
                config_path=common.config_path,
                deterministic=True,
                device=common.device,
                output_root=str(output_root),
                suite_id=suite_id,
                experiment_id=experiment_id,
                model_id=None,
                seed=int(base_seed_real + int(offset)),
            )
            run_real_matrix(real_args)

    # 3) Build paper artifacts (tables + figure) into a new report dir under this output_root.
    paper_args = SimpleNamespace(
        config_path=common.config_path,
        deterministic=True,
        device=common.device,
        output_root=str(output_root),
    )
    build_paper_artifacts(paper_args)

    # Find the (single) generated report directory.
    reports_root = _reports_root_for_output_root(output_root=output_root, canonical=canonical)
    report_prefix = str(get_canonical(canonical, "OUTPUT_NAMING.REPORT_DIR_PREFIX"))
    report_dirs = sorted([p for p in reports_root.iterdir() if p.is_dir() and p.name.startswith(report_prefix)])
    if len(report_dirs) != 1:
        raise RunError(f"Expected exactly 1 report dir under {reports_root}, found {len(report_dirs)}")
    report_dir = report_dirs[0]

    # 4) Verify hashes against the shipped manifest.
    manifest_rel = str(get_canonical(canonical, "FILES.PAPER_RESULTS_MANIFEST"))
    manifest_path = repo_root / manifest_rel
    if not manifest_path.exists():
        raise RunError(f"Missing paper results manifest: {manifest_path}")
    manifest_obj = json.loads(manifest_path.read_text(encoding="utf-8"))
    if not isinstance(manifest_obj, dict):
        raise RunError(f"{manifest_rel} did not parse as a JSON mapping")
    artifacts = manifest_obj.get("artifacts")
    if not isinstance(artifacts, list):
        raise RunError(f"{manifest_rel} missing artifacts list")
    expected_by_id: dict[str, str] = {}
    for a in artifacts:
        if not isinstance(a, dict):
            continue
        aid = a.get("artifact_id")
        sha = a.get("sha256")
        if isinstance(aid, str) and isinstance(sha, str) and aid and sha:
            expected_by_id[aid] = sha.lower()

    required = ["T1", "T2", "T3", "T4", "FIG_SYNTH"]
    missing_ids = [x for x in required if x not in expected_by_id]
    if missing_ids:
        raise RunError(f"{manifest_rel} missing required artifact ids: {missing_ids}")

    produced_paths = {
        "T1": report_dir / str(get_canonical(canonical, "FILES.REPORT_TABLE_T1_FILENAME")),
        "T2": report_dir / str(get_canonical(canonical, "FILES.REPORT_TABLE_T2_FILENAME")),
        "T3": report_dir / str(get_canonical(canonical, "FILES.REPORT_TABLE_T3_FILENAME")),
        "T4": report_dir / str(get_canonical(canonical, "FILES.REPORT_TABLE_T4_FILENAME")),
        "FIG_SYNTH": report_dir / str(get_canonical(canonical, "FILES.REPORT_FIG_SYNTH_FILENAME")),
    }

    for aid, p in produced_paths.items():
        if not p.exists():
            raise RunError(f"repro paper missing produced artifact: {p}")
        actual = _sha256_file(p).lower()
        expected = expected_by_id[aid]
        if actual != expected:
            raise RunError(f"repro paper sha256 mismatch for {aid}: got {actual}, expected {expected}")

    print(f"REPRO_OK output_root={output_root} report_dir={report_dir}")
    return 0


def run_experiment(config: dict[str, Any], canonical: dict[str, Any]) -> dict[str, Any]:
    """
    Orchestrates end-to-end runs and enforces immutability.

    This is implemented in later phases; this stub exists to satisfy the required interface.
    """
    raise RunError("run_experiment not implemented yet")


def _single_token_id(tokenizer: Any, text: str) -> int | None:
    ids = tokenizer.encode(text, add_special_tokens=False)
    if len(ids) != 1:
        return None
    return int(ids[0])


def _build_eval_batch(
    *,
    suite_id: str,
    model: HookedTransformer,
    eval_rows: Sequence[dict[str, Any]],
    canonical: dict[str, Any],
) -> tuple[dict[str, Any], torch.Tensor, str, str]:
    """
    Build (eval_batch, ref_tokens, dataset_fp, ref_fp) for the given suite on the ID split.
    """
    tokenizer = model.tokenizer
    prepend_bos = getattr(model.cfg, "default_prepend_bos", None)

    clean_prompts: list[str] = []
    ref_prompts: list[str] = []

    meta: dict[str, Any] = {}

    if suite_id == "SUITE_REAL_IOI_V1":
        token_correct: list[int] = []
        token_incorrect: list[int] = []
        for r in eval_rows:
            t_corr = _single_token_id(tokenizer, " " + str(r["name_correct"]))
            t_inc = _single_token_id(tokenizer, " " + str(r["name_incorrect"]))
            if t_corr is None or t_inc is None:
                continue
            clean_prompts.append(str(r["prompt_clean"]))
            ref_prompts.append(str(r["prompt_ref"]))
            token_correct.append(t_corr)
            token_incorrect.append(t_inc)
        meta["token_correct"] = token_correct
        meta["token_incorrect"] = token_incorrect

    elif suite_id == "SUITE_REAL_GREATERTHAN_V1":
        # Precompute all single-token two-digit completions.
        zz_ids: list[tuple[int, int]] = []
        for zz in range(100):
            s = f"{zz:02d}"
            tid = _single_token_id(tokenizer, s)
            if tid is None:
                continue
            zz_ids.append((zz, tid))

        good_token_ids: list[list[int]] = []
        bad_token_ids: list[list[int]] = []
        for r in eval_rows:
            yy = int(r["yy"])
            g = [tid for zz, tid in zz_ids if zz > yy]
            b = [tid for zz, tid in zz_ids if zz <= yy]
            if not g or not b:
                continue
            clean_prompts.append(str(r["prompt_clean"]))
            ref_prompts.append(str(r["prompt_ref"]))
            good_token_ids.append(g)
            bad_token_ids.append(b)
        meta["good_token_ids"] = good_token_ids
        meta["bad_token_ids"] = bad_token_ids

    elif suite_id == "SUITE_REAL_GREATERTHAN_YN_V1":
        # Tokenizer-agnostic: predict yes/no (single-token) instead of two-digit completion tokens.
        yes_text = str(canonical["TOKEN_LISTS"]["GREATERTHAN_ANSWER_YES"])
        no_text = str(canonical["TOKEN_LISTS"]["GREATERTHAN_ANSWER_NO"])
        t_yes = _single_token_id(tokenizer, yes_text)
        t_no = _single_token_id(tokenizer, no_text)
        if t_yes is None or t_no is None:
            raise RunError("Greater-than Y/N answer tokens are not single-token for this tokenizer")

        token_correct: list[int] = []
        token_incorrect: list[int] = []
        for r in eval_rows:
            is_greater = bool(r["is_greater"])
            clean_prompts.append(str(r["prompt_clean"]))
            ref_prompts.append(str(r["prompt_ref"]))
            token_correct.append(int(t_yes) if is_greater else int(t_no))
            token_incorrect.append(int(t_no) if is_greater else int(t_yes))
        meta["token_correct"] = token_correct
        meta["token_incorrect"] = token_incorrect

    elif suite_id == "SUITE_REAL_INDUCTION_V1":
        token_correct: list[int] = []
        token_distract: list[int] = []
        for r in eval_rows:
            t_corr = _single_token_id(tokenizer, " " + str(r["b"]))
            t_dis = _single_token_id(tokenizer, " " + str(r["distractor"]))
            if t_corr is None or t_dis is None:
                continue
            clean_prompts.append(str(r["prompt_clean"]))
            ref_prompts.append(str(r["prompt_ref"]))
            token_correct.append(t_corr)
            token_distract.append(t_dis)
        meta["token_correct"] = token_correct
        meta["token_distract"] = token_distract
    else:
        raise RunError(f"Unknown suite_id for eval batch: {suite_id}")

    if not clean_prompts:
        raise RunError(f"All rows filtered out by single-token constraints for suite {suite_id}")

    tokens = model.to_tokens(clean_prompts, prepend_bos=prepend_bos)
    ref_tokens = model.to_tokens(ref_prompts, prepend_bos=prepend_bos)

    # Fingerprints hash prompts + labels deterministically.
    dataset_obj = {"prompts": clean_prompts, "meta": meta}
    ref_obj = {"prompts": ref_prompts}
    dataset_fp = sha256_hex(canonical_json_bytes(dataset_obj, canonical))
    ref_fp = sha256_hex(canonical_json_bytes(ref_obj, canonical))

    eval_batch = {"tokens": tokens, **meta}
    return eval_batch, ref_tokens, dataset_fp, ref_fp


def _slice_batch(batch: dict[str, Any], idxs: slice) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for k, v in batch.items():
        if k == "tokens":
            out[k] = v[idxs]
        elif isinstance(v, list):
            out[k] = v[idxs]
        else:
            out[k] = v
    return out


def _compute_metric_batched(
    *,
    suite_id: str,
    model: HookedTransformer,
    batch: dict[str, Any],
    batch_size: int,
) -> list[float]:
    tokens: torch.Tensor = batch["tokens"]
    n = int(tokens.shape[0])
    out: list[float] = []
    for start in range(0, n, int(batch_size)):
        sl = slice(start, min(start + int(batch_size), n))
        out.extend(compute_metric(suite_id, model, _slice_batch(batch, sl)))
    return out


def _run_one(
    *,
    repo_root: Path,
    canonical: dict[str, Any],
    runs_root: Path,
    suite_id: str,
    experiment_id: str,
    intervention_family_id: str,
    candidate_generator_id: str,
    component_granularity: str,
    model_id: str,
    model_revision: str,
    model_local_path: str | None,
    model: HookedTransformer,
    hooks: dict[str, Any],
    eval_batch: dict[str, Any],
    ref_tokens: torch.Tensor,
    suite_reference_ids: tuple[str, str],
    seed_suite: int,
    seed_global: int,
    seed_reference_pairing: int | None,
    resolved_config_hashes: dict[str, str],
    dataset_fps: dict[str, str],
    budgets: dict[str, Any],
    epsilon: float,
    tau_abs: float,
    deterministic_mode: bool,
) -> None:
    # Compute per-run run_key and seed bundle.
    run_key = compute_run_key(
        canonical=canonical,
        suite_id=suite_id,
        experiment_id=experiment_id,
        model_id=model_id,
        model_revision=model_revision,
        component_granularity=component_granularity,
        intervention_family_id=intervention_family_id,
        candidate_generator_ids=[candidate_generator_id],
        reference_distribution_id=suite_reference_ids[0],
        reference_assignment_id=suite_reference_ids[1],
        resolved_config_hashes=resolved_config_hashes,
        dataset_fingerprints=dataset_fps,
        budgets=budgets,
        epsilon_abs=epsilon,
        tau_abs=tau_abs,
    )
    seeds = derive_seeds(canonical=canonical, seed_global=seed_global, run_key=run_key)
    seed_everything(seeds.seed_effective)
    seed_reference_pairing = int(seed_reference_pairing) if seed_reference_pairing is not None else int(seeds.seed_reference_pairing)

    run_id, run_dir = _make_run_dir(runs_root=runs_root, canonical=canonical)
    logger = make_logger(run_dir=run_dir, canonical=canonical, run_id=run_id)

    refdist_id, refassign_id = suite_reference_ids
    logger.log(
        canonical["ENUMS"]["LOG_EVENT_TYPES"]["EVENT_RUN_START"],
        {
            "suite_id": suite_id,
            "experiment_id": experiment_id,
            "intervention_family_id": intervention_family_id,
            "candidate_generator_id": candidate_generator_id,
            "component_granularity": component_granularity,
            "deterministic_mode": deterministic_mode,
            "seed_global": seed_global,
            "reference_distribution_id": refdist_id,
            "reference_assignment_id": refassign_id,
            "model_id": model_id,
            "model_revision": model_revision or "unknown",
            **({"model_local_path": model_local_path} if model_local_path else {}),
            str(get_canonical(canonical, "OUTPUT_NAMING.RUN_KEY_FIELD")): run_key,
        },
    )

    logger.log(
        canonical["ENUMS"]["LOG_EVENT_TYPES"]["EVENT_DATASET_WRITTEN"],
        {
            "dataset_fingerprint": dataset_fps["eval"],
            "reference_dataset_fingerprint": dataset_fps["ref"],
            "sizes": {"num_examples_eval": int(eval_batch["tokens"].shape[0]), "num_examples_ref": int(ref_tokens.shape[0])},
        },
    )

    generator = get_generator(candidate_generator_id)
    task_spec = {
        "suite_id": suite_id,
        "canonical": canonical,
        "component_granularity": component_granularity,
        "hooks": hooks,
        "eval_batch": eval_batch,
        "ref_tokens": ref_tokens,
    }
    candset = generator(model, task_spec, intervention_family_id, budgets, seed=int(seeds.seed_effective))
    candidate_circuits = list(candset.get("candidate_circuits", []))
    if not candidate_circuits:
        raise RunError("Candidate generator returned empty candidate_circuits")

    def canonical_edges(edges: Any) -> list[list[str]]:
        if edges is None:
            return []
        if not isinstance(edges, list):
            raise RunError("circuit['edges'] must be a list")
        out: set[tuple[str, str]] = set()
        for e in edges:
            if isinstance(e, dict) and "src" in e and "dst" in e:
                out.add((str(e["src"]), str(e["dst"])))
                continue
            if isinstance(e, (list, tuple)) and len(e) == 2:
                out.add((str(e[0]), str(e[1])))
                continue
            raise RunError("Each edge must be [src, dst] (or {'src':..., 'dst':...})")
        return [[a, b] for (a, b) in sorted(out, key=lambda x: (x[0], x[1]))]

    def circuit_key(components: list[str], edges: list[list[str]]) -> tuple[tuple[str, ...], tuple[tuple[str, str], ...]]:
        comps_key = tuple(sorted([str(x) for x in components]))
        edges_key = tuple(sorted([(str(e[0]), str(e[1])) for e in edges], key=lambda x: (x[0], x[1])))
        return (comps_key, edges_key)

    # Normalize candidate circuit objects for deterministic ordering.
    normed_candidates: list[dict[str, Any]] = []
    for circ in candidate_circuits:
        if not isinstance(circ, dict):
            raise RunError("candidate_circuits must be a list of objects")
        comps = sorted([str(x) for x in circ.get("components", [])])
        edges = canonical_edges(circ.get("edges", []))
        normed_candidates.append({**circ, "components": comps, "edges": edges})
    candidate_circuits = normed_candidates

    # Evaluate all candidates on the full eval set in batch-major order to reuse reference caches.
    hook_names = hooks["head_hooknames"] + hooks["mlp_hooknames"]
    if not isinstance(budgets, dict):
        raise RunError("Resolved budget is not a mapping")
    batch_size = int(budgets["EVAL_BATCH_SIZE"])
    baseline_vals = _compute_metric_batched(suite_id=suite_id, model=model, batch=eval_batch, batch_size=batch_size)
    n = len(baseline_vals)
    if n == 0:
        raise RunError("No baseline metric values computed")
    baseline_score_s0 = float(sum(abs(v) for v in baseline_vals) / n)
    sum_abs: list[float] = [0.0 for _ in candidate_circuits]
    count = 0
    with torch.no_grad():
        for start in range(0, n, batch_size):
            sl = slice(start, min(start + batch_size, n))
            clean_bt = eval_batch["tokens"][sl]
            ref_bt = ref_tokens[sl]
            _, cache = model.run_with_cache(ref_bt, names_filter=lambda name: name in hook_names)
            reference_cache = {name: cache[name] for name in hook_names if name in cache}
            if len(reference_cache) != len(hook_names):
                missing = [h for h in hook_names if h not in reference_cache]
                raise RunError(f"Missing reference activations for hooks: {missing}")

            base_vals_bt = baseline_vals[sl]
            batch_meta = _slice_batch(eval_batch, sl)
            for i, circ in enumerate(candidate_circuits):
                circ_components = list(circ.get("components", []))
                circ_edges = list(circ.get("edges", []))
                circuit_obj = {"components": circ_components, "edges": circ_edges}
                if intervention_family_id == "I_ACTPATCH":
                    scrubbed_model = apply_actpatch(model, circuit_obj, hooks, reference_cache)
                elif intervention_family_id == "I_PATHPATCH":
                    scrubbed_model = apply_pathpatch(model, circuit_obj, hooks, reference_cache)
                elif intervention_family_id == "I_CAUSAL_SCRUB":
                    scrubbed_model = apply_causal_scrub(model, circuit_obj, hooks, reference_cache)
                else:
                    raise RunError(f"Unknown intervention_family_id: {intervention_family_id}")

                scrubbed_vals_bt = compute_metric(suite_id, scrubbed_model, batch_meta)
                if len(scrubbed_vals_bt) != len(base_vals_bt):
                    raise RunError("Scrubbed metric batch length mismatch")
                for a, b in zip(base_vals_bt, scrubbed_vals_bt, strict=True):
                    sum_abs[i] += abs(a - b)
            count += len(base_vals_bt)

    candidate_records: list[dict[str, Any]] = []
    for i, circ in enumerate(candidate_circuits):
        components = sorted([str(x) for x in circ.get("components", [])])
        edges = canonical_edges(circ.get("edges", []))
        rec = {
            "components": components,
            "edges": edges,
            "delta": float(sum_abs[i] / count) if count else 0.0,
            "mdl": compute_mdl({"components": components, "edges": edges}, canonical),
        }
        rec["faithful"] = bool(rec["delta"] <= float(epsilon))
        candidate_records.append(rec)

    delta_cache: dict[tuple[tuple[str, ...], tuple[tuple[str, str], ...]], float] = {}
    for r in candidate_records:
        k = circuit_key(list(r.get("components", [])), canonical_edges(r.get("edges", [])))
        delta_cache[k] = float(r.get("delta", 0.0))

    def deltas_for(circuits: list[dict[str, Any]]) -> list[float]:
        keys: list[tuple[tuple[str, ...], tuple[tuple[str, str], ...]]] = []
        normed: list[dict[str, Any]] = []
        for c in circuits:
            comps = sorted([str(x) for x in c.get("components", [])])
            eds = canonical_edges(c.get("edges", []))
            keys.append(circuit_key(comps, eds))
            normed.append({"components": comps, "edges": eds})

        missing_keys: list[tuple[tuple[str, ...], tuple[tuple[str, str], ...]]] = []
        missing_circuits: list[dict[str, Any]] = []
        seen_missing: set[tuple[tuple[str, ...], tuple[tuple[str, str], ...]]] = set()
        for k, c in zip(keys, normed, strict=True):
            if k in delta_cache or k in seen_missing:
                continue
            seen_missing.add(k)
            missing_keys.append(k)
            missing_circuits.append(c)

        if missing_circuits:
            sum_abs2: list[float] = [0.0 for _ in missing_circuits]
            count2 = 0
            with torch.no_grad():
                for start in range(0, n, batch_size):
                    sl = slice(start, min(start + batch_size, n))
                    clean_bt = eval_batch["tokens"][sl]
                    ref_bt = ref_tokens[sl]
                    _, cache = model.run_with_cache(ref_bt, names_filter=lambda name: name in hook_names)
                    reference_cache = {name: cache[name] for name in hook_names if name in cache}
                    if len(reference_cache) != len(hook_names):
                        missing = [h for h in hook_names if h not in reference_cache]
                        raise RunError(f"Missing reference activations for hooks: {missing}")

                    base_vals_bt = baseline_vals[sl]
                    batch_meta = _slice_batch(eval_batch, sl)
                    for i2, circ in enumerate(missing_circuits):
                        circuit_obj = {"components": circ["components"], "edges": circ["edges"]}
                        if intervention_family_id == "I_ACTPATCH":
                            scrubbed_model = apply_actpatch(model, circuit_obj, hooks, reference_cache)
                        elif intervention_family_id == "I_PATHPATCH":
                            scrubbed_model = apply_pathpatch(model, circuit_obj, hooks, reference_cache)
                        elif intervention_family_id == "I_CAUSAL_SCRUB":
                            scrubbed_model = apply_causal_scrub(model, circuit_obj, hooks, reference_cache)
                        else:
                            raise RunError(f"Unknown intervention_family_id: {intervention_family_id}")

                        scrubbed_vals_bt = compute_metric(suite_id, scrubbed_model, batch_meta)
                        if len(scrubbed_vals_bt) != len(base_vals_bt):
                            raise RunError("Scrubbed metric batch length mismatch")
                        for a, b in zip(base_vals_bt, scrubbed_vals_bt, strict=True):
                            sum_abs2[i2] += abs(a - b)
                    count2 += len(base_vals_bt)

            for k, s in zip(missing_keys, sum_abs2, strict=True):
                delta_cache[k] = float(s / count2) if count2 else 0.0

        return [float(delta_cache[k]) for k in keys]

    # Deterministic ordering key.
    def key(rec: dict[str, Any]):
        comps = tuple(sorted(rec["components"]))
        return (float(rec["mdl"]), int(len(comps)), comps)

    faithful = [r for r in candidate_records if bool(r["faithful"])]
    faithful_sorted = sorted(faithful, key=key)
    best = faithful_sorted[0] if faithful_sorted else {"components": [], "mdl": float("inf"), "delta": float("inf")}

    rr_res = compute_rr(candidate_records, canonical)
    s_near = list(rr_res.get("s_near", []))

    # SSS replicate discovery runs (spec/06 + spec/11).
    num_reps = int(canonical["DIAGNOSTICS"]["SSS_NUM_REPLICATES"])
    replicate_circuits: list[set[str]] = []
    for r in range(num_reps):
        rep_seed = int(seeds.seed_replicate(int(r)))
        rep_candset = generator(model, task_spec, intervention_family_id, budgets, seed=rep_seed)
        rep_circuit_objs: list[dict[str, Any]] = []
        for c in rep_candset.get("candidate_circuits", []):
            if not isinstance(c, dict):
                continue
            rep_circuit_objs.append(
                {
                    "components": sorted([str(x) for x in c.get("components", [])]),
                    "edges": canonical_edges(c.get("edges", [])),
                }
            )
        if not rep_circuit_objs:
            replicate_circuits.append(set())
            continue
        rep_deltas = deltas_for(rep_circuit_objs)
        rep_records: list[dict[str, Any]] = []
        for circ, d in zip(rep_circuit_objs, rep_deltas, strict=True):
            comps = list(circ.get("components", []))
            eds = canonical_edges(circ.get("edges", []))
            rec = {
                "components": comps,
                "edges": eds,
                "delta": float(d),
                "mdl": compute_mdl({"components": comps, "edges": eds}, canonical),
            }
            rec["faithful"] = bool(rec["delta"] <= float(epsilon))
            rep_records.append(rec)
        rep_faithful = [x for x in rep_records if bool(x.get("faithful", False))]
        rep_best = sorted(rep_faithful, key=key)[0] if rep_faithful else None
        replicate_circuits.append(set(rep_best["components"]) if rep_best is not None else set())

    sss_res = compute_sss(replicate_circuits, canonical)

    # CC necessity labeling (spec/06). Optimize: if |S_near| < 2, CC is deterministically 0.
    if len(s_near) < 2:
        for r in s_near:
            r["necessity"] = {}
        cc_res = compute_cc(s_near, canonical)
    else:
        for r in s_near:
            comps = sorted([str(x) for x in r.get("components", [])])
            edges = canonical_edges(r.get("edges", []))
            base_delta = float(r.get("delta", float("inf")))
            removed_circuits: list[dict[str, Any]] = []
            for v in comps:
                comps2 = [c for c in comps if c != v]
                edges2 = [e for e in edges if str(e[0]) != str(v) and str(e[1]) != str(v)]
                removed_circuits.append({"components": comps2, "edges": edges2})
            removed_deltas = deltas_for(removed_circuits) if removed_circuits else []
            necessity: dict[str, bool] = {}
            for v, d2 in zip(comps, removed_deltas, strict=True):
                necessity[v] = bool(float(d2) - float(base_delta) >= float(tau_abs))
            r["necessity"] = dict(sorted(necessity.items()))
        cc_res = compute_cc(s_near, canonical)

    # Certificate trigger.
    best_path = run_dir / str(get_canonical(canonical, "FILES.BEST_CIRCUIT_FILENAME"))
    _write_json(best_path, {"components": sorted([str(x) for x in best.get("components", [])])})

    certificate_path: Path | None = None
    if (
        rr_res["RR_verdict"] == canonical["ENUMS"]["VERDICTS"]["VERDICT_FAIL"]
        or sss_res["SSS_verdict"] == canonical["ENUMS"]["VERDICTS"]["VERDICT_FAIL"]
        or cc_res["CC_verdict"] == canonical["ENUMS"]["VERDICTS"]["VERDICT_FAIL"]
    ):
        reason_codes = _audit_certificate_reason_codes(
            canonical=canonical,
            rr_verdict=str(rr_res["RR_verdict"]),
            sss_verdict=str(sss_res["SSS_verdict"]),
            cc_verdict=str(cc_res["CC_verdict"]),
        )
        certificate_obj = build_certificate(
            {
                "reason_codes": reason_codes,
                "suite_id": suite_id,
                "experiment_id": experiment_id,
                "intervention_family_id": intervention_family_id,
                "candidate_generator_id": candidate_generator_id,
                "component_granularity": component_granularity,
                "model_id": model_id,
                "model_revision": model_revision or "unknown",
                "dataset_fingerprint": dataset_fps["eval"],
                "reference_dataset_fingerprint": dataset_fps["ref"],
                "baseline_score_s0": float(baseline_score_s0),
                "epsilon": float(epsilon),
                "tau": float(tau_abs),
                "reference_distribution_id": refdist_id,
                "reference_assignment_id": refassign_id,
                "RR": float(rr_res["RR"]),
                "RR_verdict": rr_res["RR_verdict"],
                "SSS": float(sss_res["SSS"]),
                "SSS_verdict": sss_res["SSS_verdict"],
                "CC": float(cc_res["CC"]),
                "CC_verdict": cc_res["CC_verdict"],
                "s_near": [
                    {
                        "components": sorted([str(x) for x in rec.get("components", [])]),
                        "delta": float(rec.get("delta", 0.0)),
                        "mdl": float(rec.get("mdl", 0.0)),
                        "necessity": dict(rec.get("necessity", {})),
                    }
                    for rec in s_near
                ],
                "replicate_circuits": [sorted(list(c)) for c in replicate_circuits],
            },
            canonical,
        )
        certificate_path = run_dir / str(get_canonical(canonical, "FILES.CERTIFICATE_FILENAME"))
        _write_json(certificate_path, certificate_obj)
        logger.log(
            canonical["ENUMS"]["LOG_EVENT_TYPES"]["EVENT_CERTIFICATE_WRITTEN"],
            {"certificate_file": certificate_path.name},
        )

    diagnostics = {
        "suite_id": suite_id,
        "experiment_id": experiment_id,
        "intervention_family_id": intervention_family_id,
        "candidate_generator_id": candidate_generator_id,
        "baseline_score_s0": float(baseline_score_s0),
        "epsilon": float(epsilon),
        "tau": float(tau_abs),
        "best_circuit": best,
        "RR": rr_res["RR"],
        "RR_verdict": rr_res["RR_verdict"],
        "SSS": sss_res["SSS"],
        "SSS_verdict": sss_res["SSS_verdict"],
        "CC": cc_res["CC"],
        "CC_verdict": cc_res["CC_verdict"],
        "candidate_records": candidate_records,
        "certificate_file": certificate_path.name if certificate_path is not None else None,
    }

    diag_path = run_dir / str(get_canonical(canonical, "FILES.DIAGNOSTICS_FILENAME"))
    _write_json(diag_path, diagnostics)

    logger.log(
        canonical["ENUMS"]["LOG_EVENT_TYPES"]["EVENT_CANDIDATES_WRITTEN"],
        {
            "num_candidates": int(len(candidate_circuits)),
            "topk_summary": [
                {"mdl": float(r["mdl"]), "size": int(len(r["components"])), "delta": float(r["delta"])}
                for r in sorted(candidate_records, key=key)[:5]
            ],
        },
    )
    logger.log(
        canonical["ENUMS"]["LOG_EVENT_TYPES"]["EVENT_DIAGNOSTICS_WRITTEN"],
        {
            "RR": float(rr_res["RR"]),
            "SSS": float(sss_res["SSS"]),
            "CC": float(cc_res["CC"]),
            "RR_verdict": rr_res["RR_verdict"],
            "SSS_verdict": sss_res["SSS_verdict"],
            "CC_verdict": cc_res["CC_verdict"],
            "certificate_emitted": bool(certificate_path is not None),
            "diagnostics_fingerprint": sha256_hex(diag_path.read_bytes()),
        },
    )

    # Overall verdict: PASS if all diagnostics PASS, WARN if any WARN, else FAIL.
    verdicts = [rr_res["RR_verdict"], sss_res["SSS_verdict"], cc_res["CC_verdict"]]
    if canonical["ENUMS"]["VERDICTS"]["VERDICT_HARD_FAIL"] in verdicts:
        overall = canonical["ENUMS"]["VERDICTS"]["VERDICT_HARD_FAIL"]
    elif canonical["ENUMS"]["VERDICTS"]["VERDICT_FAIL"] in verdicts:
        overall = canonical["ENUMS"]["VERDICTS"]["VERDICT_FAIL"]
    elif canonical["ENUMS"]["VERDICTS"]["VERDICT_WARN"] in verdicts:
        overall = canonical["ENUMS"]["VERDICTS"]["VERDICT_WARN"]
    else:
        overall = canonical["ENUMS"]["VERDICTS"]["VERDICT_PASS"]

    rr0 = diagnostics["baseline_score_s0"]
    run_record = {
        "schema_version": int(get_canonical(canonical, "SCHEMAS.RUN_RECORD_SCHEMA_VERSION")),
        "project_id": str(get_canonical(canonical, "PROJECT_ID")),
        "project_version": str(get_canonical(canonical, "PROJECT_VERSION")),
        str(get_canonical(canonical, "OUTPUT_NAMING.RUN_KEY_FIELD")): run_key,
        "run_id": run_id,
        "suite_id": suite_id,
        "experiment_id": experiment_id,
        "model_id": model_id,
        "model_revision": model_revision or "unknown",
        **({"model_local_path": model_local_path} if model_local_path else {}),
        "intervention_family_id": intervention_family_id,
        "candidate_generator_id": candidate_generator_id,
        "component_granularity": component_granularity,
        "reference_distribution_id": refdist_id,
        "reference_assignment_id": refassign_id,
        "timestamp_utc": datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z"),
        "git_commit": _git_commit_hex(canonical=canonical, repo_root=repo_root),
        "python_version": environment_fingerprint()["python_version"],
        "platform": environment_fingerprint()["platform"],
        "deterministic_mode": bool(deterministic_mode),
        "seed_global": int(seed_global),
        "seed_suite": int(seed_suite),
        "seed_reference_pairing": int(seed_reference_pairing),
        "dataset_fingerprint": dataset_fps["eval"],
        "reference_dataset_fingerprint": dataset_fps["ref"],
        "baseline_score_s0": float(rr0),
        "epsilon": float(epsilon),
        "best_circuit_mdl": float(best.get("mdl", 0.0)) if faithful_sorted else 0.0,
        "best_circuit_size": int(len(best["components"])) if faithful_sorted else 0,
        "faithfulness_delta": float(best.get("delta", 0.0)) if faithful_sorted else 0.0,
        "RR": float(rr_res["RR"]),
        "RR_verdict": rr_res["RR_verdict"],
        "SSS": float(sss_res["SSS"]),
        "SSS_verdict": sss_res["SSS_verdict"],
        "CC": float(cc_res["CC"]),
        "CC_verdict": cc_res["CC_verdict"],
        "overall_verdict": overall,
        "quality_gates_passed": bool(overall == canonical["ENUMS"]["VERDICTS"]["VERDICT_PASS"]),
        "paths": {
            "logs_dir": ".",
            "results_dir": ".",
            "diagnostics_file": diag_path.name,
            **({"certificate_file": certificate_path.name} if certificate_path is not None else {}),
        },
    }
    run_record[str(get_canonical(canonical, "OUTPUT_NAMING.RUN_RECORD_HASH_FIELD"))] = compute_run_record_hash(
        run_record, canonical
    )

    write_run_record(str(run_dir / str(get_canonical(canonical, "FILES.RUN_RECORD_FILENAME"))), run_record)

    logger.log(
        canonical["ENUMS"]["LOG_EVENT_TYPES"]["EVENT_RUN_END"],
        {
            "overall_verdict": overall,
            "run_record": str(get_canonical(canonical, "FILES.RUN_RECORD_FILENAME")),
            "diagnostics_file": str(get_canonical(canonical, "FILES.DIAGNOSTICS_FILENAME")),
        },
    )
