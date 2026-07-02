from __future__ import annotations

import json
from dataclasses import asdict, is_dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


class DecisionLogger:
    def __init__(self, path: str | Path = "logs/decision-log.jsonl") -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def record(self, event_type: str, payload: dict[str, Any]) -> None:
        data = {
            "event_type": event_type,
            "recorded_at": datetime.now(timezone.utc).isoformat(),
            **_jsonable(payload),
        }
        with self.path.open("a", encoding="utf-8") as file:
            file.write(json.dumps(data, ensure_ascii=True, sort_keys=True) + "\n")


def _jsonable(value: Any) -> Any:
    if is_dataclass(value):
        return _jsonable(asdict(value))
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_jsonable(item) for item in value]
    return value
