from __future__ import annotations

import os
import threading
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from app.execution.kis_errors import KisModeMismatchError, KisReadinessError
from app.execution.kis_real import (
    KIS_LIVE_BASE_URL,
    KisCredentials,
    KisDevelopersApiClient,
    load_kis_env_file,
)
from app.execution.kis_types import KisHealthCheck, KisMode


REQUIRED_KIS_KEYS = (
    "KIS_APP_KEY",
    "KIS_APP_SECRET",
    "KIS_ACCOUNT_NO",
    "KIS_ACCOUNT_PRODUCT_CODE",
)
OPTIONAL_KIS_KEYS = ("KIS_HTS_ID", "KIS_CUSTTYPE")


def env_bool(name: str, default: bool) -> bool:
    value = os.getenv(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


def load_kis_mode() -> KisMode:
    load_kis_env_file()
    base_url = os.getenv("KIS_BASE_URL_REAL") or os.getenv("KIS_BASE_URL") or KIS_LIVE_BASE_URL
    live_enabled = env_bool("KIS_LIVE_ENABLED", False)
    mode = KisMode(paper=False, live_enabled=live_enabled, base_url=base_url)
    validate_kis_mode(mode)
    return mode


def validate_kis_mode(mode: KisMode) -> None:
    base = mode.base_url.lower()
    if mode.paper:
        raise KisModeMismatchError("KIS paper trading mode has been removed; live KIS mode is required")
    if "openapivts" in base:
        raise KisModeMismatchError("KIS paper trading domain is not allowed; use the live KIS domain")


def validate_live_secret_file(path: str | Path = "config/secrets/kis_api_keys.env") -> dict[str, bool]:
    load_kis_env_file(path, override=True)
    secret_path = Path(path)
    results = {"file_exists": secret_path.exists()}
    for key in REQUIRED_KIS_KEYS:
        results[key] = bool(os.getenv(key, "").strip())
    for key in OPTIONAL_KIS_KEYS:
        value = os.getenv(key, "").strip()
        results[key] = bool(value) if key != "KIS_CUSTTYPE" else bool(value or "P")
    return results


_approval_key_cache: dict[str, tuple[str, float]] = {}
_approval_key_lock = threading.Lock()


def _approval_key_ttl_seconds() -> float:
    """How long a cached approval key stays reusable.

    KIS issues a fresh key per request, and each key opens its own realtime
    session against the account. Requesting a new key on every reconnect made
    the collector accumulate distinct sessions, so the account's realtime
    registration budget drained until only one symbol could be subscribed.
    Reusing one key across reconnects keeps that to a single session.
    """
    try:
        hours = float(os.getenv("KIS_APPROVAL_KEY_TTL_HOURS", "6"))
    except (TypeError, ValueError):
        hours = 6.0
    return max(60.0, hours * 3600.0)


def reset_websocket_approval_key_cache() -> None:
    with _approval_key_lock:
        _approval_key_cache.clear()


def issue_websocket_approval_key(
    client: KisDevelopersApiClient,
    *,
    force_refresh: bool = False,
) -> str:
    client.credentials.validate()
    cache_key = str(getattr(client.credentials, "app_key", "") or "")
    now = time.monotonic()
    if cache_key and not force_refresh:
        with _approval_key_lock:
            cached = _approval_key_cache.get(cache_key)
        if cached and now - cached[1] < _approval_key_ttl_seconds():
            return cached[0]
    response = client.transport.request(
        "POST",
        client._url("/oauth2/Approval"),  # KIS approval-key endpoint for WebSocket access.
        headers={"Content-Type": "application/json; charset=utf-8"},
        body={
            "grant_type": "client_credentials",
            "appkey": client.credentials.app_key,
            "secretkey": client.credentials.app_secret,
        },
        timeout=client.timeout,
    )
    key = str(response.get("approval_key") or response.get("approvalKey") or "")
    if not key:
        raise RuntimeError("KIS WebSocket approval-key response did not include approval_key")
    if cache_key:
        with _approval_key_lock:
            _approval_key_cache[cache_key] = (key, now)
    return key


def build_kis_client(
    *,
    transport: Any | None = None,
    enabled: bool | None = None,
    token_cache_path: str | Path | None = None,
) -> KisDevelopersApiClient:
    mode = load_kis_mode()
    credentials = KisCredentials.from_env(False)
    return KisDevelopersApiClient(
        app_key=credentials.app_key,
        app_secret=credentials.app_secret,
        account_no=credentials.account_no,
        account_product_code=credentials.account_product_code,
        base_url=mode.base_url,
        paper=False,
        enabled=mode.live_enabled if enabled is None else enabled,
        transport=transport,
        token_cache_path=token_cache_path,
    )


def run_kis_health_check(
    client: KisDevelopersApiClient,
    *,
    include_account: bool = True,
    include_websocket: bool = True,
) -> KisHealthCheck:
    gates: dict[str, bool] = {}
    failures: dict[str, str] = {}

    def gate(name: str, func: Any) -> None:
        try:
            func()
            gates[name] = True
        except Exception as exc:  # noqa: BLE001 - convert every health failure to a gate result.
            gates[name] = False
            failures[name] = exc.__class__.__name__

    gate("credentials", client.credentials.validate)
    gate("mode", lambda: validate_kis_mode(KisMode(client.paper, client.enabled, client.endpoints.base_url)))
    gate("token", lambda: client.issue_access_token())
    if include_account:
        gate("account_read", client.get_portfolio)
    if include_websocket:
        gate("websocket_approval_key", lambda: issue_websocket_approval_key(client))

    return KisHealthCheck(
        ok=all(gates.values()),
        mode="paper" if client.paper else "live",
        checked_at=datetime.now(timezone.utc),
        gates=gates,
        failures=failures,
    )


def require_kis_health(client: KisDevelopersApiClient) -> KisHealthCheck:
    health = run_kis_health_check(client)
    if not health.ok:
        raise KisReadinessError(health.failures)
    return health
