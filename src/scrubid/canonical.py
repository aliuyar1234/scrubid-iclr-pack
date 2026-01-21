from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml


class CanonicalError(RuntimeError):
    pass


@dataclass(frozen=True)
class CanonicalLoadResult:
    canonical: dict[str, Any]
    canonical_path: Path


def _extract_yaml_block(md_text: str) -> str:
    fence = "```yaml"
    start = md_text.find(fence)
    if start == -1:
        raise CanonicalError("Could not find ```yaml block in spec/00_CANONICAL.md")
    start = md_text.find("\n", start)
    if start == -1:
        raise CanonicalError("Malformed yaml fence in spec/00_CANONICAL.md")
    end = md_text.find("```", start)
    if end == -1:
        raise CanonicalError("Could not find closing ``` fence for canonical yaml block")
    return md_text[start + 1 : end].strip() + "\n"


def load_canonical(spec_root: str) -> dict[str, Any]:
    """
    Load the canonical block from spec/00_CANONICAL.md.

    This is the SSOT for constants, IDs, paths, thresholds, enums, and CLI literals.
    """
    canonical_path = Path(spec_root) / "spec" / "00_CANONICAL.md"
    if not canonical_path.exists():
        raise CanonicalError(f"Missing canonical spec file: {canonical_path}")
    md_text = canonical_path.read_text(encoding="utf-8")
    yaml_text = _extract_yaml_block(md_text)
    canonical = yaml.safe_load(yaml_text)
    if not isinstance(canonical, dict):
        raise CanonicalError("Canonical yaml block did not parse as a mapping")

    required_top = ["PROJECT_ID", "PROJECT_VERSION", "PATHS", "FILES", "CLI", "IDS", "ENUMS", "HASHING"]
    missing = [k for k in required_top if k not in canonical]
    if missing:
        raise CanonicalError(f"Canonical block missing required keys: {missing}")
    return canonical


def get_canonical(canonical: dict[str, Any], dotted_path: str) -> Any:
    """
    Resolve dotted canonical paths like:
      - IDS.SUITE_IDS.SUITE_SYNTH_V1
      - CANONICAL.PATHS.PATH_OUTPUT_ROOT
    """
    if dotted_path.startswith("CANONICAL."):
        dotted_path = dotted_path.removeprefix("CANONICAL.")
    cur: Any = canonical
    for part in dotted_path.split("."):
        if not isinstance(cur, dict) or part not in cur:
            raise CanonicalError(f"Undefined canonical path: {dotted_path}")
        cur = cur[part]
    return cur

