from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from scrubid.canonical import get_canonical


def now_utc_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


@dataclass
class JsonlLogger:
    path: Path
    run_id: str
    schema_version: int

    def log(self, event_type: str, payload: dict[str, Any]) -> None:
        obj = {
            "event_type": event_type,
            "timestamp_utc": now_utc_iso(),
            "run_id": self.run_id,
            "schema_version": self.schema_version,
            **payload,
        }
        with self.path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(obj, ensure_ascii=False) + "\n")


def make_logger(*, run_dir: Path, canonical: dict[str, Any], run_id: str) -> JsonlLogger:
    logs_name = str(get_canonical(canonical, "FILES.LOG_JSONL_FILENAME"))
    schema_version = int(get_canonical(canonical, "SCHEMAS.LOG_SCHEMA_VERSION"))
    logs_path = run_dir / logs_name
    return JsonlLogger(path=logs_path, run_id=run_id, schema_version=schema_version)

