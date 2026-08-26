"""The boundary the published proxy enforces.

This proxy reaches the live server over loopback, so the server's own access
guard does not challenge it, and that server can place real orders. These tests
pin the two properties that make publishing the port defensible at all. A
failure here is not a broken feature; it is an open door onto a funded account.
"""
from __future__ import annotations

import os

import pytest
from starlette.testclient import TestClient

from app import public_proxy

TOKEN = "0123456789abcdef-a-long-enough-secret"


class _StubUpstream:
    """Stands in for the live server, and records what reached it.

    Recording is the point: for the mutating verbs the assertion is not merely
    that the caller got an error, but that the request never left this process.
    """

    def __init__(self) -> None:
        self.seen: list[tuple[str, str]] = []
        self.params: list[list[tuple[str, str]]] = []
        self.headers: list[dict[str, str]] = []

    async def request(self, method, url, **kwargs):  # noqa: ANN001
        import httpx

        self.seen.append((method, str(url)))
        self.params.append(list(kwargs.get("params") or []))
        self.headers.append(dict(kwargs.get("headers") or {}))
        return httpx.Response(
            200,
            json={"ok": True, "path": str(url)},
            headers={"content-type": "application/json"},
        )

    async def aclose(self) -> None:
        return None


@pytest.fixture()
def client(monkeypatch):
    monkeypatch.setenv("PUBLIC_PROXY_TOKEN", TOKEN)
    app = public_proxy.create_app()
    upstream = _StubUpstream()

    async def _startup() -> None:
        app.state.client = upstream

    app.router.on_startup.clear()
    app.router.on_shutdown.clear()
    app.router.on_startup.append(_startup)
    with TestClient(app) as test_client:
        test_client.upstream = upstream  # type: ignore[attr-defined]
        yield test_client


# -- gate 1: the token ------------------------------------------------------- #

def test_no_token_is_refused_and_never_reaches_upstream(client):
    response = client.get("/api/status")
    assert response.status_code == 401
    assert client.upstream.seen == []


def test_wrong_token_is_refused(client):
    response = client.get("/api/status", headers={"X-App-Token": "wrong-but-long-enough"})
    assert response.status_code == 401
    assert client.upstream.seen == []


@pytest.mark.parametrize(
    "send",
    [
        lambda c: c.get("/api/status", headers={"X-App-Token": TOKEN}),
        lambda c: c.get("/api/status", headers={"Authorization": f"Bearer {TOKEN}"}),
        lambda c: c.get(f"/api/status?token={TOKEN}"),
    ],
)
def test_every_documented_token_channel_works(client, send):
    assert send(client).status_code == 200


def test_an_unset_secret_fails_closed(monkeypatch):
    """No secret must mean no access, never open access."""
    monkeypatch.delenv("PUBLIC_PROXY_TOKEN", raising=False)
    app = public_proxy.create_app()
    upstream = _StubUpstream()
    app.router.on_startup.clear()
    app.router.on_shutdown.clear()

    async def _startup() -> None:
        app.state.client = upstream

    app.router.on_startup.append(_startup)
    with TestClient(app) as client:
        assert client.get("/api/status").status_code == 401
    assert upstream.seen == []


def test_a_short_secret_fails_closed(monkeypatch):
    monkeypatch.setenv("PUBLIC_PROXY_TOKEN", "short")
    app = public_proxy.create_app()
    upstream = _StubUpstream()
    app.router.on_startup.clear()
    app.router.on_shutdown.clear()

    async def _startup() -> None:
        app.state.client = upstream

    app.router.on_startup.append(_startup)
    with TestClient(app) as client:
        assert client.get("/api/status", headers={"X-App-Token": "short"}).status_code == 401
    assert upstream.seen == []


# -- gate 2: the method ------------------------------------------------------ #

MUTATING_ROUTES = [
    "/api/live-trading/terminate",
    "/api/system/graceful-shutdown",
    "/api/live-flags/apply",
    "/api/operation-mode/start",
    "/api/operation-mode/stop-learning",
    "/api/start",
    "/api/paper-trading/start",
    "/api/research/refresh",
    "/api/investor-flow/refresh",
    "/api/kiosk/exit",
]


@pytest.mark.parametrize("path", MUTATING_ROUTES)
def test_mutating_routes_are_refused_even_with_a_valid_token(client, path):
    response = client.post(path, headers={"X-App-Token": TOKEN})
    assert response.status_code == 405
    assert client.upstream.seen == [], "a mutating request reached the live server"


@pytest.mark.parametrize("method", ["POST", "PUT", "PATCH", "DELETE"])
def test_every_non_read_verb_is_refused(client, method):
    response = client.request(method, "/api/status", headers={"X-App-Token": TOKEN})
    assert response.status_code == 405
    assert client.upstream.seen == []


# -- relaying ---------------------------------------------------------------- #

def test_a_read_is_relayed_to_the_upstream(client):
    response = client.get("/api/system-diagnostics", headers={"X-App-Token": TOKEN})
    assert response.status_code == 200
    assert client.upstream.seen == [("GET", "/api/system-diagnostics")]


def test_the_token_is_not_forwarded_to_the_upstream(client):
    """Our secret has no meaning there and would land in its logs."""
    client.get(f"/api/status?token={TOKEN}&symbol=INTC")
    forwarded = client.upstream.params[-1]
    assert ("symbol", "INTC") in forwarded, "a real query parameter was dropped"
    assert all(key != "token" for key, _ in forwarded), "the secret was forwarded upstream"
    assert all(
        key.lower() != "cookie" for key in client.upstream.headers[-1]
    ), "the access cookie was forwarded upstream"


def test_forwarded_headers_are_not_relayed(client):
    """Relaying them made every page through Funnel return 401.

    Uvicorn trusts X-Forwarded-For from a loopback client by default, so passing
    the public address through made the upstream's guard treat this proxy's
    request as an external one and challenge a viewer who had already
    authenticated here.
    """
    client.get(
        "/api/status",
        headers={
            "X-App-Token": TOKEN,
            "X-Forwarded-For": "203.0.113.7",
            "X-Forwarded-Proto": "https",
            "X-Real-IP": "203.0.113.7",
            "Forwarded": "for=203.0.113.7",
        },
    )
    relayed = {key.lower() for key in client.upstream.headers[-1]}
    assert not (relayed & {
        "x-forwarded-for", "x-forwarded-proto", "x-real-ip", "forwarded",
    })


def test_a_query_token_is_promoted_to_a_cookie(client):
    response = client.get(f"/api/status?token={TOKEN}")
    assert response.cookies.get(public_proxy.ACCESS_COOKIE) == TOKEN


def test_denied_paths_are_withheld(client):
    response = client.get("/api/mock-kis/portfolio", headers={"X-App-Token": TOKEN})
    assert response.status_code == 404
    assert client.upstream.seen == []


def test_unknown_read_paths_are_relayed_not_guessed(client):
    """The published page must be the page the server serves.

    A curated allowlist silently 404s any panel the dashboard grows, which is the
    drift that made the published site stop matching the local one.
    """
    response = client.get("/api/account/summary", headers={"X-App-Token": TOKEN})
    assert response.status_code == 200
    assert client.upstream.seen == [("GET", "/api/account/summary")]


# -- what a person actually sees --------------------------------------------- #

BROWSER = {
    "accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
}


def test_browser_visit_without_a_token_gets_a_page_not_a_download(client):
    """A browser handed ``application/json`` does not render a page.

    Measured 2026-08-19 on the published Funnel URL: opening it returned 401 with
    ``content-type: application/json``, so Safari saved the body to disk and other
    browsers showed the raw JSON. The reported symptom was "a text file downloads and
    the GUI never appears". Since the token cookie is only set on a successful token
    request, a plain visit had no way in at all.
    """
    response = client.get("/", headers=BROWSER)

    assert response.status_code == 401
    assert response.headers["content-type"].startswith("text/html")
    assert "<!doctype html>" in response.text.lower()
    # It has to offer a way IN, not just explain the refusal.
    assert 'name="token"' in response.text
    assert response.headers["cache-control"] == "no-store"
    # Still fails closed: nothing reached the live server.
    assert client.upstream.seen == []


def test_programmatic_client_still_gets_json(client):
    """Content negotiation must not break the API shape for non-browsers."""
    response = client.get("/api/status", headers={"accept": "application/json"})

    assert response.status_code == 401
    assert response.headers["content-type"].startswith("application/json")
    assert response.json()["error"] == "ACCESS_TOKEN_REQUIRED"
    assert client.upstream.seen == []


def test_the_token_form_submits_straight_back_into_the_page(client):
    """The form is a GET carrying ``token``, which is the one shape the gate forwards."""
    response = client.get("/account", params={"token": TOKEN}, headers=BROWSER)

    assert response.status_code == 200
    assert client.upstream.seen == [("GET", "/account")]
    # And the session continues without the secret in any later URL.
    assert client.cookies.get(public_proxy.ACCESS_COOKIE) == TOKEN


def test_error_page_does_not_offer_a_form_when_no_token_is_configured(monkeypatch):
    """With no usable secret the proxy is closed entirely; a form would be a lie."""
    monkeypatch.setenv("PUBLIC_PROXY_TOKEN", "")
    app = public_proxy.create_app()
    app.router.on_startup.clear()
    app.router.on_shutdown.clear()

    with TestClient(app) as test_client:
        response = test_client.get("/", headers=BROWSER)

    assert response.status_code == 401
    assert response.headers["content-type"].startswith("text/html")
    assert 'name="token"' not in response.text


def test_upstream_date_and_server_headers_are_not_relayed(client, monkeypatch):
    """``date`` is a singleton field and ``server`` names a live trading stack.

    Relaying both put two ``date`` headers on every published response and advertised
    the upstream's server software past the proxy's own ``--no-server-header``.
    """
    import httpx

    async def _request(method, url, **kwargs):  # noqa: ANN001
        client.upstream.seen.append((method, str(url)))
        return httpx.Response(
            200,
            text="<html></html>",
            headers={
                "content-type": "text/html; charset=utf-8",
                "date": "Tue, 18 Aug 2026 00:00:00 GMT",
                "server": "uvicorn",
                "x-keep-me": "yes",
            },
        )

    monkeypatch.setattr(client.upstream, "request", _request)
    response = client.get("/", params={"token": TOKEN}, headers=BROWSER)

    assert response.status_code == 200
    assert response.headers.get("x-keep-me") == "yes"
    assert response.headers.get("server") != "uvicorn"
    assert response.headers.get("date") != "Tue, 18 Aug 2026 00:00:00 GMT"
    assert len(response.headers.get_list("date")) <= 1


def test_published_root_lands_on_the_operator_console(client):
    """``/`` is the general home page; the console this system is run from is ``/account``.

    The local launcher opens ``/account`` and this proxy's banner advertises
    ``/account?token=...``, so publishing the root unchanged landed visitors on a
    different page from the one the URL is meant to show.
    """
    response = client.get("/", params={"token": TOKEN}, follow_redirects=False)

    assert response.status_code == 302
    assert response.headers["location"] == "/account"
    # The redirect is the first response a ?token= visit gets, so it must carry the
    # cookie or the follow-up arrives unauthenticated and bounces to the token page.
    assert client.cookies.get(public_proxy.ACCESS_COOKIE) == TOKEN


def test_root_redirect_is_behind_the_token_gate(client):
    """An unauthenticated caller must not get the redirect either."""
    response = client.get("/", follow_redirects=False)

    assert response.status_code == 401
    assert client.upstream.seen == []


def test_landing_redirect_can_be_switched_off(client, monkeypatch):
    """Serving the root as-is stays available for anyone who wants it."""
    monkeypatch.setenv("PUBLIC_PROXY_LANDING_PATH", "")

    response = client.get("/", params={"token": TOKEN}, follow_redirects=False)

    assert response.status_code == 200
    assert client.upstream.seen == [("GET", "/")]


def test_non_root_paths_are_never_redirected(client):
    """Only the root is remapped; every other published path must proxy unchanged."""
    response = client.get("/api/status", params={"token": TOKEN}, follow_redirects=False)

    assert response.status_code == 200
    assert client.upstream.seen == [("GET", "/api/status")]


def test_idempotent_read_retries_one_stale_upstream_connection(client, monkeypatch):
    """A proxy that outlives a server restart must discard one dead pooled socket."""
    import httpx

    attempts = 0

    async def _request(method, url, **kwargs):  # noqa: ANN001
        nonlocal attempts
        attempts += 1
        if attempts == 1:
            raise httpx.RemoteProtocolError("peer closed pooled connection")
        return httpx.Response(200, json={"ok": True})

    monkeypatch.setattr(client.upstream, "request", _request)

    response = client.get("/api/status", headers={"X-App-Token": TOKEN})

    assert response.status_code == 200
    assert response.json() == {"ok": True}
    assert attempts == 2


def test_persistent_upstream_failure_remains_fail_closed(client, monkeypatch):
    import httpx

    attempts = 0

    async def _request(method, url, **kwargs):  # noqa: ANN001
        nonlocal attempts
        attempts += 1
        raise httpx.ConnectError("loopback unavailable")

    monkeypatch.setattr(client.upstream, "request", _request)

    response = client.get("/api/status", headers={"X-App-Token": TOKEN})

    assert response.status_code == 502
    assert response.json()["error"] == "UPSTREAM_UNAVAILABLE"
    assert attempts == 2


# -- anonymous mode: gate 1 off, gate 2 must still hold ----------------------- #

@pytest.fixture()
def open_client(monkeypatch):
    """The proxy with the token gate switched off by operator decision."""
    monkeypatch.setenv("PUBLIC_PROXY_TOKEN", TOKEN)
    monkeypatch.setenv("PUBLIC_PROXY_ALLOW_ANONYMOUS", "true")
    app = public_proxy.create_app()
    upstream = _StubUpstream()

    async def _startup() -> None:
        app.state.client = upstream

    app.router.on_startup.clear()
    app.router.on_shutdown.clear()
    app.router.on_startup.append(_startup)
    with TestClient(app) as test_client:
        test_client.upstream = upstream  # type: ignore[attr-defined]
        yield test_client


def test_anonymous_mode_serves_reads_without_a_token(open_client):
    assert open_client.get("/api/account/summary").status_code == 200
    assert open_client.upstream.seen == [("GET", "/api/account/summary")]


def test_anonymous_mode_lands_a_bare_visit_on_the_console(open_client):
    response = open_client.get("/", follow_redirects=False)

    assert response.status_code == 302
    assert response.headers["location"] == "/account"


@pytest.mark.parametrize("path", MUTATING_ROUTES)
def test_anonymous_mode_still_refuses_every_mutating_route(open_client, path):
    """Gate 2 is what keeps this a read exposure rather than account control.

    Dropping the token must not make a single order-placing, arming or shutdown
    route reachable, so the assertion is that it never left this process.
    """
    response = open_client.post(path)

    assert response.status_code == 405
    assert open_client.upstream.seen == []


@pytest.mark.parametrize("method", ["POST", "PUT", "PATCH", "DELETE"])
def test_anonymous_mode_still_refuses_every_write_verb(open_client, method):
    response = open_client.request(method, "/api/status")

    assert response.status_code == 405
    assert open_client.upstream.seen == []


def test_anonymous_mode_still_withholds_denied_paths(open_client):
    response = open_client.get("/api/mock-kis/portfolio")

    assert response.status_code == 404
    assert open_client.upstream.seen == []


@pytest.mark.parametrize("value", ["", "false", "0", "no", "off", "maybe"])
def test_the_gate_stays_on_unless_anonymous_is_set_deliberately(monkeypatch, value):
    """A malformed or absent value must never be read as "open"."""
    monkeypatch.setenv("PUBLIC_PROXY_TOKEN", TOKEN)
    monkeypatch.setenv("PUBLIC_PROXY_ALLOW_ANONYMOUS", value)
    app = public_proxy.create_app()
    upstream = _StubUpstream()

    async def _startup() -> None:
        app.state.client = upstream

    app.router.on_startup.clear()
    app.router.on_shutdown.clear()
    app.router.on_startup.append(_startup)
    with TestClient(app) as client:
        assert client.get("/api/status").status_code == 401
    assert upstream.seen == []
