"""Opening the app to the network must not open its trading controls.

The app has no authentication and 22 POST endpoints that change real state,
including ``/api/system/graceful-shutdown`` and ``/api/live-trading/terminate``,
and the launcher starts it with live order submission enabled. These tests pin
the two properties that make an external bind survivable: the local workflow is
untouched, and an external client without the token gets nothing.
"""

from __future__ import annotations

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from app.web_access_guard import (
    ACCESS_COOKIE,
    AccessGuardMiddleware,
    ExternalAccessDenied,
    binds_externally,
    is_loopback,
    require_token_for_external_bind,
)


TOKEN = "s3cret-token-long-enough"


def _app(token: str | None = TOKEN) -> FastAPI:
    application = FastAPI()
    application.add_middleware(AccessGuardMiddleware, token_provider=lambda: token or "")

    @application.get("/account")
    def account() -> dict[str, str]:
        return {"page": "terminal"}

    @application.post("/api/system/graceful-shutdown")
    def shutdown() -> dict[str, bool]:
        return {"stopping": True}

    @application.get("/static/app.js")
    def static_asset() -> dict[str, str]:
        return {"asset": "js"}

    return application


def _client(app: FastAPI, host: str) -> TestClient:
    # ``client`` sets the peer address the guard reads, which is what decides
    # loopback vs external.
    return TestClient(app, client=(host, 51234))


# --------------------------------------------------------------------------- #
# Address classification
# --------------------------------------------------------------------------- #

@pytest.mark.parametrize("host", ["127.0.0.1", "localhost", "::1", "127.0.0.5"])
def test_loopback_addresses_are_recognised(host: str) -> None:
    assert is_loopback(host)
    assert not binds_externally(host)


@pytest.mark.parametrize("host", ["0.0.0.0", "::", "192.168.0.108", "10.0.0.4"])
def test_routable_addresses_count_as_external(host: str) -> None:
    assert not is_loopback(host)
    assert binds_externally(host)


def test_unknown_host_string_is_treated_as_external() -> None:
    """Guessing "safe" for something unparseable is the expensive direction."""
    assert binds_externally("some-host-name")


# --------------------------------------------------------------------------- #
# Startup refusal
# --------------------------------------------------------------------------- #

def test_external_bind_without_a_token_is_refused() -> None:
    with pytest.raises(ExternalAccessDenied) as excinfo:
        require_token_for_external_bind("0.0.0.0", token="")
    assert "APP_ACCESS_TOKEN" in str(excinfo.value)


def test_external_bind_with_a_short_token_is_refused() -> None:
    with pytest.raises(ExternalAccessDenied):
        require_token_for_external_bind("0.0.0.0", token="short")


def test_external_bind_with_a_real_token_is_allowed() -> None:
    require_token_for_external_bind("0.0.0.0", token=TOKEN)


def test_loopback_bind_never_needs_a_token() -> None:
    require_token_for_external_bind("127.0.0.1", token="")


# --------------------------------------------------------------------------- #
# Request-time behaviour
# --------------------------------------------------------------------------- #

def test_local_workflow_is_untouched_by_the_guard() -> None:
    """The browser on this machine must behave exactly as before."""
    client = _client(_app(), "127.0.0.1")
    assert client.get("/account").status_code == 200
    assert client.post("/api/system/graceful-shutdown").status_code == 200


def test_no_configured_token_leaves_the_guard_inert() -> None:
    """Default install: loopback-only, no token, no behaviour change anywhere."""
    client = _client(_app(token=""), "192.168.0.50")
    assert client.get("/account").status_code == 200


def test_external_client_without_a_token_is_refused() -> None:
    client = _client(_app(), "192.168.0.50")
    response = client.get("/account")
    assert response.status_code == 401
    assert response.json()["error"] == "ACCESS_TOKEN_REQUIRED"


def test_external_client_cannot_reach_trading_endpoints_without_a_token() -> None:
    client = _client(_app(), "192.168.0.50")
    assert client.post("/api/system/graceful-shutdown").status_code == 401


def test_wrong_token_is_refused() -> None:
    client = _client(_app(), "192.168.0.50")
    assert client.get("/account", headers={"X-App-Token": "nope"}).status_code == 401


@pytest.mark.parametrize(
    "headers",
    [{"Authorization": f"Bearer {TOKEN}"}, {"X-App-Token": TOKEN}],
)
def test_external_client_with_a_header_token_is_allowed(headers: dict[str, str]) -> None:
    client = _client(_app(), "192.168.0.50")
    assert client.get("/account", headers=headers).status_code == 200


def test_query_token_is_promoted_to_a_cookie() -> None:
    """A phone browser cannot set headers on the first URL it opens.

    Without the cookie the page would load and then every relative fetch it
    makes would 401, which looks like a broken dashboard rather than an auth
    problem.
    """
    client = _client(_app(), "192.168.0.50")
    response = client.get(f"/account?token={TOKEN}")
    assert response.status_code == 200
    assert response.cookies.get(ACCESS_COOKIE) == TOKEN
    # The page's own relative XHRs now authenticate with no token in the URL.
    assert client.get("/api/system/graceful-shutdown") is not None
    assert client.post("/api/system/graceful-shutdown").status_code == 200


def test_static_assets_stay_reachable_so_the_401_page_can_render() -> None:
    client = _client(_app(), "192.168.0.50")
    assert client.get("/static/app.js").status_code == 200
