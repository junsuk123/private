from __future__ import annotations

import json
import os
import threading
from dataclasses import asdict, is_dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

SENSITIVE_KEYS = {
    "app_key",
    "app_secret",
    "authorization",
    "access_token",
    "refresh_token",
    "token",
    "password",
    "secret",
    "account_no",
    "account_number",
    "cano",
    "acnt_prdt_cd",
}
REDACTED = "***REDACTED***"

#: Default directory for the operator's journals.
DEFAULT_LOG_DIR = "logs"


def log_path(name: str) -> Path:
    """Resolve a journal filename under the active log directory.

    Every default journal path goes through here so that ``OBAITS_LOG_DIR`` can move
    the whole set at once. That exists because the paths used to be hardcoded as
    default ARGUMENTS -- ``LiveOrderJournal(path="logs/live-orders.jsonl")``,
    ``RiskManager``'s fallback ``AuditLogger(Path("logs/principal-protection.jsonl"))``
    -- so any caller that constructed one without naming a path wrote to the
    operator's real audit trail. A test suite is exactly such a caller, and the
    result was measurable: 6,162 ``LAB`` order records in ``logs/live-orders.jsonl``
    and 665 demo-issuer (900001/900002) entries in ``logs/audit.jsonl``, sitting
    indistinguishable from real ones in the journal a funded account is reconciled
    against.

    Resolved on CALL, never at import, so redirecting it does not depend on which
    modules an interpreter happened to import first.
    """
    root = (os.getenv("OBAITS_LOG_DIR") or "").strip() or DEFAULT_LOG_DIR
    return Path(root) / name


class AuditLogger:
    def __init__(
        self,
        path: Path,
        *,
        max_bytes: int | None = None,
        backup_count: int = 3,
    ) -> None:
        self.path = path
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.max_bytes = max(
            1,
            int(
                max_bytes
                if max_bytes is not None
                else os.getenv("AUDIT_LOG_MAX_BYTES", str(128 * 1024 * 1024))
            ),
        )
        self.backup_count = max(1, int(backup_count))
        self._write_lock = threading.Lock()

    def _rotate_if_needed(self, incoming_bytes: int) -> None:
        try:
            current_bytes = self.path.stat().st_size
        except FileNotFoundError:
            return
        if current_bytes + incoming_bytes <= self.max_bytes:
            return
        oldest = self.path.with_name(f"{self.path.name}.{self.backup_count}")
        oldest.unlink(missing_ok=True)
        for index in range(self.backup_count - 1, 0, -1):
            source = self.path.with_name(f"{self.path.name}.{index}")
            if source.exists():
                source.replace(self.path.with_name(f"{self.path.name}.{index + 1}"))
        self.path.replace(self.path.with_name(f"{self.path.name}.1"))

    def record(self, event_type: str, payload: Any) -> None:
        event = {
            "event_type": event_type,
            "recorded_at": datetime.now(timezone.utc).isoformat(),
            "payload": _redact_sensitive(_to_jsonable(payload)),
        }
        line = json.dumps(event, ensure_ascii=True, sort_keys=True) + "\n"
        with self._write_lock:
            self._rotate_if_needed(len(line.encode("utf-8")))
            with self.path.open("a", encoding="utf-8") as file:
                file.write(line)


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


def _redact_sensitive(value: Any) -> Any:
    if isinstance(value, dict):
        redacted: dict[str, Any] = {}
        for key, item in value.items():
            key_text = str(key)
            if _is_sensitive_key(key_text):
                redacted[key_text] = REDACTED
            else:
                redacted[key_text] = _redact_sensitive(item)
        return redacted
    if isinstance(value, list):
        return [_redact_sensitive(item) for item in value]
    return value


def _is_sensitive_key(key: str) -> bool:
    lowered = key.lower()
    return lowered in SENSITIVE_KEYS or any(token in lowered for token in ("secret", "password", "token"))
