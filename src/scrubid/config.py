from __future__ import annotations

import re
from pathlib import Path
from typing import Any

import yaml

from scrubid.canonical import get_canonical


class ConfigError(RuntimeError):
    pass


_CANONICAL_REF_RE = re.compile(r"^(CANONICAL\.)?[A-Z][A-Z0-9_]*(\.[A-Z][A-Z0-9_]*)+$")
_CANONICAL_TEMPLATE_RE = re.compile(r"\$\{CANONICAL\.([A-Za-z0-9_.]+)\}")


def _resolve_value(value: Any, canonical: dict[str, Any]) -> Any:
    if isinstance(value, str):
        # Template replacement: "${CANONICAL.PATHS.PATH_OUTPUT_ROOT}"
        def repl(match: re.Match[str]) -> str:
            dotted = match.group(1)
            resolved = get_canonical(canonical, dotted)
            return str(resolved)

        value = _CANONICAL_TEMPLATE_RE.sub(repl, value)

        if _CANONICAL_REF_RE.match(value):
            return get_canonical(canonical, value)
        return value
    if isinstance(value, list):
        return [_resolve_value(v, canonical) for v in value]
    if isinstance(value, dict):
        return {k: _resolve_value(v, canonical) for k, v in value.items()}
    return value


def load_config(config_path: str, canonical: dict[str, Any]) -> dict[str, Any]:
    path = Path(config_path)
    if not path.exists():
        raise ConfigError(f"Config not found: {config_path}")
    obj = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(obj, dict):
        raise ConfigError(f"Config {config_path} did not parse as a mapping")

    resolved = _resolve_value(obj, canonical)

    # Basic schema version sanity: if present, must match canonical schema version.
    schema_key = resolved.get("schema_version_key")
    if schema_key is not None:
        expected = int(get_canonical(canonical, "SCHEMAS.CONFIG_SCHEMA_VERSION"))
        if int(schema_key) != expected:
            raise ConfigError(f"Config schema_version_key mismatch: got {schema_key}, expected {expected}")
    return resolved

