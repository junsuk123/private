from __future__ import annotations

import json
from dataclasses import asdict, is_dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

from app.schemas.domain import RawSourceRecord


class RawArchive:
    def __init__(self, root: Path) -> None:
        self.root = root
        self.root.mkdir(parents=True, exist_ok=True)

    def write(self, record: RawSourceRecord) -> Path:
        source_id = record.source.source_id or "source"
        safe_id = "".join(char if char.isalnum() or char in ("-", "_") else "_" for char in source_id)
        path = self.root / f"{safe_id}.json"
        path.write_text(json.dumps(_to_jsonable(record), ensure_ascii=False, indent=2), encoding="utf-8")
        return path


def _to_jsonable(value: Any) -> Any:
    if is_dataclass(value):
        return _to_jsonable(asdict(value))
    if isinstance(value, dict):
        return {str(key): _to_jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_to_jsonable(item) for item in value]
    if isinstance(value, datetime):
        return value.isoformat()
    return value
