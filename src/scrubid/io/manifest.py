from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from scrubid.hashing import sha256_hex


@dataclass(frozen=True)
class ManifestEntry:
    sha256: str
    relpath: str


def compute_manifest(root: Path, *, exclude_paths: set[str] | None = None) -> list[ManifestEntry]:
    exclude_paths = exclude_paths or set()
    entries: list[ManifestEntry] = []
    for p in sorted(root.rglob("*")):
        if p.is_dir():
            continue
        rel = p.relative_to(root).as_posix()
        if rel in exclude_paths:
            continue
        data = p.read_bytes()
        entries.append(ManifestEntry(sha256=sha256_hex(data), relpath=rel))
    return entries


def write_manifest(path: Path, entries: list[ManifestEntry]) -> None:
    lines = [f"{e.sha256}  {e.relpath}" for e in entries]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")

