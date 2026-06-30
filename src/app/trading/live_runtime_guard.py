from __future__ import annotations

import json
import os
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path


ARMING_FILE = Path("config/secrets/live_trading_armed.json")
SECRETS_ENV_FILE = Path("config/secrets/kis_api_keys.env")


@dataclass(frozen=True)
class LiveRuntimeGateResult:
    ok: bool
    failures: tuple[str, ...]


def env_bool(name: str, default: bool) -> bool:
    value = os.getenv(name)
    if value is None:
        value = _env_from_secrets_file(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


def _env_from_secrets_file(name: str) -> str | None:
    if not SECRETS_ENV_FILE.exists():
        return None
    try:
        for raw_line in SECRETS_ENV_FILE.read_text(encoding="utf-8").splitlines():
            line = raw_line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, value = line.split("=", 1)
            if key.strip() == name:
                return value.strip()
    except OSError:
        return None
    return None


def live_flag_failures() -> list[str]:
    failures: list[str] = []
    expected = {
        "LIVE_TRADING_ENABLED": True,
        "KIS_LIVE_ENABLED": True,
        "KIS_PAPER_TRADING": False,
        "LIVE_ORDER_SUBMIT_ENABLED": True,
        "KILL_SWITCH_ENABLED": False,
    }
    for name, required in expected.items():
        if env_bool(name, not required) != required:
            failures.append(f"{name}_NOT_{str(required).upper()}")
    return failures


def create_arming_file(path: Path | None = None, *, ttl_seconds: int = 900) -> Path:
    path = path or ARMING_FILE
    path.parent.mkdir(parents=True, exist_ok=True)
    now = datetime.now(timezone.utc)
    payload = {
        "armed_at": now.isoformat(),
        "expires_at": (now + timedelta(seconds=ttl_seconds)).isoformat(),
    }
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return path


def disarm(path: Path | None = None) -> None:
    path = path or ARMING_FILE
    try:
        path.unlink()
    except FileNotFoundError:
        return


def arming_failures(path: Path | None = None, *, require_manual_arming: bool = True) -> list[str]:
    path = path or ARMING_FILE
    if not require_manual_arming:
        return []
    if not path.exists():
        return ["MANUAL_ARMING_FILE_MISSING"]
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
        expires_at = datetime.fromisoformat(str(payload["expires_at"]))
        if expires_at.tzinfo is None:
            expires_at = expires_at.replace(tzinfo=timezone.utc)
    except (KeyError, ValueError, json.JSONDecodeError, TypeError):
        return ["MANUAL_ARMING_FILE_INVALID"]
    if expires_at <= datetime.now(timezone.utc):
        return ["MANUAL_ARMING_EXPIRED"]
    return []


def evaluate_live_runtime_gates(*, require_manual_arming: bool = True) -> LiveRuntimeGateResult:
    failures = [*live_flag_failures(), *arming_failures(require_manual_arming=require_manual_arming)]
    return LiveRuntimeGateResult(ok=not failures, failures=tuple(failures))
