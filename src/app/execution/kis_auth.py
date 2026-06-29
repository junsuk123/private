from __future__ import annotations

import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from app.execution.kis_errors import KisModeMismatchError, KisReadinessError
from app.execution.kis_real import (
    KIS_LIVE_BASE_URL,
    KIS_PAPER_BASE_URL,
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
    "KIS_HTS_ID",
    "KIS_CUSTTYPE",
)


def env_bool(name: str, default: bool) -> bool:
    value = os.getenv(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


def load_kis_mode() -> KisMode:
    load_kis_env_file()
    paper = env_bool("KIS_PAPER_TRADING", True)
    base_url = (
        os.getenv("KIS_BASE_URL_PAPER" if paper else "KIS_BASE_URL_REAL")
        or os.getenv("KIS_BASE_URL")
        or (KIS_PAPER_BASE_URL if paper else KIS_LIVE_BASE_URL)
    )
    live_enabled = env_bool("KIS_LIVE_ENABLED", False)
    mode = KisMode(paper=paper, live_enabled=live_enabled, base_url=base_url)
    validate_kis_mode(mode)
    return mode


def validate_kis_mode(mode: KisMode) -> None:
    base = mode.base_url.lower()
    if mode.paper and "openapivts" not in base:
        raise KisModeMismatchError("KIS_PAPER_TRADING=true but base URL is not the paper domain")
    if not mode.paper and "openapivts" in base:
        raise KisModeMismatchError("KIS_PAPER_TRADING=false but base URL is the paper domain")


def validate_live_secret_file(path: str | Path = "config/secrets/kis_api_keys.env") -> dict[str, bool]:
    load_kis_env_file(path)
    secret_path = Path(path)
    results = {"file_exists": secret_path.exists()}
    for key in REQUIRED_KIS_KEYS:
        results[key] = bool(os.getenv(key, "").strip())
    return results


def issue_websocket_approval_key(client: KisDevelopersApiClient) -> str:
    client.credentials.validate()
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
    return key


def build_kis_client(
    *,
    transport: Any | None = None,
    enabled: bool | None = None,
    token_cache_path: str | Path | None = None,
) -> KisDevelopersApiClient:
    mode = load_kis_mode()
    credentials = KisCredentials.from_env(mode.paper)
    return KisDevelopersApiClient(
        app_key=credentials.app_key,
        app_secret=credentials.app_secret,
        account_no=credentials.account_no,
        account_product_code=credentials.account_product_code,
        base_url=mode.base_url,
        paper=mode.paper,
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
