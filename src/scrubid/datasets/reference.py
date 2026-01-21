from __future__ import annotations

from typing import Any, Callable

from scrubid.canonical import get_canonical


def build_reference(suite_id: str, canonical: dict[str, Any], seed: int) -> tuple[list[Any], Callable[[int], int]]:
    """
    Build D_ref and a deterministic pairing function for a given suite.

    For real suites in this pack, pairing is index-aligned and D_ref is provided
    by the suite builder.
    """
    # This function is used as an adapter for suite-specific builders.
    raise RuntimeError("build_reference is suite-specific; use suite builders that return D_ref and pairing")

