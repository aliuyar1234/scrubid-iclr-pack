from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

from scrubid.canonical import load_canonical
from scrubid.experiments.runner import (
    aggregate_results,
    build_paper_artifacts,
    determinism_smoke_test,
    reproduce_paper,
    run_real_matrix,
    run_synth_diagnostics,
    run_synth_generate_only,
    run_synth_generate_with_candidates,
    validate_paper_manifest,
    validate_spec_pack,
)


def _add_common_args(parser: argparse.ArgumentParser, flags: dict[str, str]) -> None:
    parser.add_argument(flags["FLAG_CONFIG"], dest="config_path", required=True)
    parser.add_argument(flags["FLAG_DETERMINISTIC"], dest="deterministic", action="store_true")
    parser.add_argument(flags["FLAG_DEVICE"], dest="device", default=None)
    parser.add_argument(flags["FLAG_OUTPUT_ROOT"], dest="output_root", default=None)


def _add_suite_args(parser: argparse.ArgumentParser, flags: dict[str, str], *, require_experiment: bool) -> None:
    parser.add_argument(flags["FLAG_SUITE_ID"], dest="suite_id", required=True)
    if require_experiment:
        parser.add_argument(flags["FLAG_EXPERIMENT_ID"], dest="experiment_id", required=True)
    parser.add_argument(flags["FLAG_MODEL_ID"], dest="model_id", default=None)
    parser.add_argument(flags["FLAG_SEED"], dest="seed", type=int, default=None)


def _build_parser(canonical: dict[str, Any]) -> argparse.ArgumentParser:
    flags = canonical["CLI"]["CANONICAL_FLAGS"]

    p = argparse.ArgumentParser(prog="scrubid")
    sp = p.add_subparsers(dest="group", required=True)

    spec = sp.add_parser("spec")
    spec_sp = spec.add_subparsers(dest="cmd", required=True)
    spec_val = spec_sp.add_parser("validate")
    _add_common_args(spec_val, flags)
    spec_val.set_defaults(func=lambda a: validate_spec_pack(a.config_path))

    synth = sp.add_parser("synth")
    synth_sp = synth.add_subparsers(dest="cmd", required=True)
    synth_gen = synth_sp.add_parser("generate")
    _add_common_args(synth_gen, flags)
    synth_gen.add_argument(flags["FLAG_SUITE_ID"], dest="suite_id", required=True)
    synth_gen.add_argument(flags["FLAG_SEED"], dest="seed", type=int, default=None)
    synth_gen.set_defaults(func=run_synth_generate_only)

    synth_cand = synth_sp.add_parser("candidates")
    _add_common_args(synth_cand, flags)
    synth_cand.add_argument(flags["FLAG_SUITE_ID"], dest="suite_id", required=True)
    synth_cand.add_argument(flags["FLAG_SEED"], dest="seed", type=int, default=None)
    synth_cand.set_defaults(func=run_synth_generate_with_candidates)

    synth_diag = synth_sp.add_parser("diagnostics")
    _add_common_args(synth_diag, flags)
    synth_diag.add_argument(flags["FLAG_SUITE_ID"], dest="suite_id", required=True)
    synth_diag.add_argument(flags["FLAG_SEED"], dest="seed", type=int, default=None)
    synth_diag.set_defaults(func=run_synth_diagnostics)

    real = sp.add_parser("real")
    real_sp = real.add_subparsers(dest="cmd", required=True)
    real_run = real_sp.add_parser("run")
    _add_common_args(real_run, flags)
    _add_suite_args(real_run, flags, require_experiment=True)
    real_run.set_defaults(func=run_real_matrix)

    report = sp.add_parser("report")
    report_sp = report.add_subparsers(dest="cmd", required=True)
    rep_agg = report_sp.add_parser("aggregate")
    _add_common_args(rep_agg, flags)
    rep_agg.set_defaults(func=aggregate_results)

    rep_paper = report_sp.add_parser("paper_artifacts")
    _add_common_args(rep_paper, flags)
    rep_paper.set_defaults(func=build_paper_artifacts)

    rep_manifest = report_sp.add_parser("validate_paper_manifest")
    _add_common_args(rep_manifest, flags)
    rep_manifest.set_defaults(func=validate_paper_manifest)

    test = sp.add_parser("test")
    test_sp = test.add_subparsers(dest="cmd", required=True)
    det = test_sp.add_parser("determinism_smoke")
    _add_common_args(det, flags)
    det.set_defaults(func=determinism_smoke_test)

    repro = sp.add_parser("repro")
    repro_sp = repro.add_subparsers(dest="cmd", required=True)
    repro_paper = repro_sp.add_parser("paper")
    _add_common_args(repro_paper, flags)
    repro_paper.set_defaults(func=reproduce_paper)

    return p


def main() -> int:
    # CLI runs from the implementation repo root (where spec/ and configs/ live).
    repo_root = Path.cwd()
    canonical = load_canonical(str(repo_root))
    parser = _build_parser(canonical)
    args = parser.parse_args()

    try:
        return int(args.func(args))  # type: ignore[misc]
    except KeyboardInterrupt:
        return 130


if __name__ == "__main__":
    raise SystemExit(main())
