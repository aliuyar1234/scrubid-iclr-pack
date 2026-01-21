from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path
from typing import Any

from scrubid.canonical import get_canonical, load_canonical
from scrubid.config import load_config


def _runs_root_for_output_root(*, output_root: Path, canonical: dict[str, Any]) -> Path:
    runs_rel = Path(str(get_canonical(canonical, "PATHS.PATH_RUNS_ROOT")))
    out_rel = Path(str(get_canonical(canonical, "PATHS.PATH_OUTPUT_ROOT")))
    try:
        runs_sub = runs_rel.relative_to(out_rel)
    except ValueError:
        runs_sub = Path(runs_rel.name)
    return output_root / runs_sub


def _load_run_records(*, runs_root: Path, canonical: dict[str, Any]) -> list[dict[str, Any]]:
    prefix = str(get_canonical(canonical, "OUTPUT_NAMING.RUN_DIR_PREFIX"))
    rr_name = str(get_canonical(canonical, "FILES.RUN_RECORD_FILENAME"))
    records: list[dict[str, Any]] = []
    if not runs_root.exists():
        return records
    for run_dir in sorted([p for p in runs_root.iterdir() if p.is_dir() and p.name.startswith(prefix)]):
        rr_path = run_dir / rr_name
        if not rr_path.exists():
            continue
        try:
            obj = json.loads(rr_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            continue
        if isinstance(obj, dict):
            records.append(obj)
    return records


def _enabled_generators(*, repo_root: Path, canonical: dict[str, Any]) -> list[str]:
    gen_cfg_path = repo_root / str(get_canonical(canonical, "FILES.CONFIG_GENERATORS"))
    gen_cfg = load_config(str(gen_cfg_path), canonical)
    enabled: list[str] = []
    for g in gen_cfg.get("generators", []):
        if bool(g.get("enabled_key", False)):
            enabled.append(str(g.get("id")))
    if not enabled:
        raise RuntimeError("No enabled generators found in configs/generators.yaml")
    return enabled


def _paper_real_suite_experiments(canonical: dict[str, Any]) -> list[tuple[str, str]]:
    return [
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


def _paper_real_seeds(canonical: dict[str, Any]) -> list[int]:
    base_seed = int(get_canonical(canonical, "SEEDS.SEED_REAL_SUITE"))
    offsets_obj = get_canonical(canonical, "SEEDS.PAPER_REAL_SEED_OFFSETS")
    if not isinstance(offsets_obj, list) or not offsets_obj:
        raise RuntimeError("Missing or invalid SEEDS.PAPER_REAL_SEED_OFFSETS")
    seeds = [int(base_seed + int(o)) for o in offsets_obj]
    if not seeds:
        raise RuntimeError("Empty seed list computed from SEEDS.PAPER_REAL_SEED_OFFSETS")
    return seeds


def _missing_real_commands(
    *,
    run_records: list[dict[str, Any]],
    required_generators: list[str],
    required_suite_exps: list[tuple[str, str]],
    required_seeds: list[int],
) -> list[tuple[str, str, int]]:
    present: set[tuple[str, str, int, str]] = set()
    for r in run_records:
        suite_id = r.get("suite_id")
        experiment_id = r.get("experiment_id")
        seed_suite = r.get("seed_suite")
        generator_id = r.get("candidate_generator_id")
        if not isinstance(suite_id, str) or not suite_id:
            continue
        if not isinstance(experiment_id, str) or not experiment_id:
            continue
        if not isinstance(seed_suite, int):
            # JSON loader reads numbers as int already; tolerate string but ignore.
            continue
        if not isinstance(generator_id, str) or not generator_id:
            continue
        present.add((suite_id, experiment_id, int(seed_suite), generator_id))

    missing: list[tuple[str, str, int]] = []
    for suite_id, experiment_id in required_suite_exps:
        for seed_suite in required_seeds:
            have = {g for (_s, _e, _seed, g) in present if (_s, _e, _seed) == (suite_id, experiment_id, int(seed_suite))}
            if all(g in have for g in required_generators):
                continue
            missing.append((suite_id, experiment_id, int(seed_suite)))

    # Deterministic ordering.
    missing.sort(key=lambda x: (x[0], x[1], int(x[2])))
    return missing


def _run_cli(cmd: list[str], *, dry_run: bool) -> None:
    print("+ " + " ".join(cmd), flush=True)
    if dry_run:
        return
    subprocess.run(cmd, check=True)


def main() -> int:
    p = argparse.ArgumentParser(description="Resume missing ScrubID paper-scope real runs into an output_root.")
    p.add_argument("--output_root", required=True, help="Target output_root (created if missing).")
    p.add_argument("--config", default="configs/experiments.yaml", help="Path to experiments config (default: configs/experiments.yaml).")
    p.add_argument("--device", default="cuda", help="Device passed to scrubid CLI (default: cuda).")
    p.add_argument("--include_synth", action="store_true", help="Also run synth diagnostics once into output_root.")
    p.add_argument("--dry_run", action="store_true", help="Print commands without executing them.")
    args = p.parse_args()

    repo_root = Path.cwd()
    canonical = load_canonical(str(repo_root))

    output_root = Path(str(args.output_root))
    output_root.mkdir(parents=True, exist_ok=True)

    runs_root = _runs_root_for_output_root(output_root=output_root, canonical=canonical)
    run_records = _load_run_records(runs_root=runs_root, canonical=canonical)

    if args.include_synth:
        cmd = [
            sys.executable,
            "-m",
            "scrubid.cli",
            "synth",
            "diagnostics",
            "--suite_id",
            str(get_canonical(canonical, "IDS.SUITE_IDS.SUITE_SYNTH_V1")),
            "--config",
            str(args.config),
            "--deterministic",
            "--output_root",
            str(output_root),
        ]
        if args.device:
            cmd.extend(["--device", str(args.device)])
        _run_cli(cmd, dry_run=bool(args.dry_run))

        # Refresh records after synth, even though we only use real records below.
        run_records = _load_run_records(runs_root=runs_root, canonical=canonical)

    required_generators = _enabled_generators(repo_root=repo_root, canonical=canonical)
    required_suite_exps = _paper_real_suite_experiments(canonical)
    required_seeds = _paper_real_seeds(canonical)

    missing_cmds = _missing_real_commands(
        run_records=run_records,
        required_generators=required_generators,
        required_suite_exps=required_suite_exps,
        required_seeds=required_seeds,
    )

    print(f"output_root={output_root}")
    print(f"runs_root={runs_root}")
    print(f"enabled_generators={required_generators}")
    print(f"required_real_commands={len(required_suite_exps) * len(required_seeds)}")
    print(f"missing_real_commands={len(missing_cmds)}")

    for suite_id, experiment_id, seed_suite in missing_cmds:
        cmd = [
            sys.executable,
            "-m",
            "scrubid.cli",
            "real",
            "run",
            "--suite_id",
            suite_id,
            "--experiment_id",
            experiment_id,
            "--config",
            str(args.config),
            "--deterministic",
            "--output_root",
            str(output_root),
            "--seed",
            str(int(seed_suite)),
        ]
        if args.device:
            cmd.extend(["--device", str(args.device)])
        _run_cli(cmd, dry_run=bool(args.dry_run))

    print("OK", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
