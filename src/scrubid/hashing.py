from __future__ import annotations

import hashlib
import json
from decimal import Decimal
from typing import Any

from scrubid.canonical import get_canonical


class HashingError(RuntimeError):
    pass


def sha256_hex(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_uint32(text: str) -> int:
    """
    sha256_uint32(x): compute SHA-256 over UTF-8 bytes of x, take the first 8 hex
    characters as a big-endian unsigned 32-bit integer.
    """
    d = sha256_hex(text.encode("utf-8"))
    first8 = d[:8]
    return int(first8, 16)


def _assert_no_floats(obj: Any, *, path: str = "$") -> None:
    if isinstance(obj, float):
        raise HashingError(f"Floating point value found in hashed object at {path}")
    if isinstance(obj, dict):
        for k, v in obj.items():
            _assert_no_floats(v, path=f"{path}.{k}")
    elif isinstance(obj, list):
        for i, v in enumerate(obj):
            _assert_no_floats(v, path=f"{path}[{i}]")


def decimal_str(x: float | int | Decimal) -> str:
    # Ensure non-exponent decimal string, stable across runs.
    if isinstance(x, Decimal):
        return format(x, "f")
    if isinstance(x, int):
        return str(x)
    return format(Decimal(str(x)), "f")


def canonical_json_bytes(obj: Any, canonical: dict[str, Any]) -> bytes:
    """
    Deterministic JSON byte representation per CANONICAL.HASHING.CANONICAL_JSON.
    """
    _assert_no_floats(obj)
    sort_keys = bool(get_canonical(canonical, "HASHING.CANONICAL_JSON.SORT_KEYS"))
    ensure_ascii = bool(get_canonical(canonical, "HASHING.CANONICAL_JSON.ENSURE_ASCII"))
    separators = tuple(get_canonical(canonical, "HASHING.CANONICAL_JSON.SEPARATORS"))
    allow_nan = bool(get_canonical(canonical, "HASHING.CANONICAL_JSON.ALLOW_NAN"))
    s = json.dumps(
        obj,
        sort_keys=sort_keys,
        ensure_ascii=ensure_ascii,
        separators=separators,
        allow_nan=allow_nan,
    )
    return s.encode("utf-8")

