from __future__ import annotations

import json
from dataclasses import asdict, is_dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from app.runtime import DataMode, default_environment


class ModelArtifactStore:
    def __init__(self, root: Path | None = None, mode: DataMode | None = None) -> None:
        if root is None:
            environment = default_environment()
            self.root = environment.model_dir
            self.mode = environment.mode
        else:
            self.root = root
            self.mode = mode or "custom"
        self.root.mkdir(parents=True, exist_ok=True)

    def save_json(
        self,
        model_name: str,
        artifact: Any,
        *,
        simulated: bool = False,
        model_family: str | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> Path:
        if simulated:
            raise ValueError(f"Refusing to save simulated model artifact in realtime-only model store: {model_name}")
        family = _safe_name(model_family or model_name.split(":", 1)[0] or "default")
        safe_name = "".join(char if char.isalnum() or char in ("-", "_") else "_" for char in model_name)
        model_dir = self.root / family
        model_dir.mkdir(parents=True, exist_ok=True)
        saved_at = datetime.now(timezone.utc)
        payload = {
            "model_name": model_name,
            "model_family": family,
            "simulated": simulated,
            "saved_at": saved_at.isoformat(),
            "metadata": metadata or {},
            "artifact": _to_jsonable(artifact),
        }
        path = model_dir / f"{safe_name}.{saved_at.strftime('%Y%m%dT%H%M%S%fZ')}.json"
        path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True), encoding="utf-8")
        latest_path = model_dir / f"{safe_name}.latest.json"
        latest_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True), encoding="utf-8")
        return path

    def load_json(self, model_name: str, *, model_family: str | None = None) -> dict[str, Any]:
        family = _safe_name(model_family or model_name.split(":", 1)[0] or "default")
        safe_name = "".join(char if char.isalnum() or char in ("-", "_") else "_" for char in model_name)
        latest_path = self.root / family / f"{safe_name}.latest.json"
        legacy_path = self.root / f"{safe_name}.json"
        path = latest_path if latest_path.exists() else legacy_path
        return json.loads(path.read_text(encoding="utf-8"))


def _safe_name(value: str) -> str:
    return "".join(char if char.isalnum() or char in ("-", "_") else "_" for char in value)


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
