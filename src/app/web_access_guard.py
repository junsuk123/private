"""Access control for the web app once it is reachable off this machine.

The app has no authentication of any kind, and it exposes 22 POST endpoints that
change real state: ``/api/system/graceful-shutdown``, ``/api/live-trading/terminate``,
``/api/operation-mode/start``, ``/api/live-flags/apply``. The launcher also starts
it with ``LIVE_ORDER_SUBMIT_ENABLED=true`` and ``REQUIRE_MANUAL_ARMING=false``, so
those endpoints reach a broker that fills real orders.

While the server binds to ``127.0.0.1`` that is fine: the only client is the
person at this keyboard. The moment it binds to a routable address, "no auth"
means every device on the network — and anything that has compromised one of
them — can flatten positions, change live flags, or stop the server. So binding
off-loopback and having a token are ONE decision, not two:

- Loopback requests are never challenged. The local browser keeps working exactly
  as before, and no existing workflow changes.
- Non-loopback requests must present the token. Missing or wrong token is 401,
  and the body says how to supply one rather than leaking anything about state.

The token travels three ways, in this order: an ``Authorization: Bearer`` header,
an ``X-App-Token`` header, or a ``?token=`` query parameter. The query form
exists because the first thing an external client does is type a URL into a
phone browser, which cannot set headers. On a successful query-token request the
guard sets an ``HttpOnly`` cookie so the page's own relative XHRs authenticate
without every fetch having to carry the token.

Deliberately NOT implemented here: user accounts, roles, TLS. This is a guard
against casual reachability on a home LAN, not an authorization system, and
pretending otherwise would be worse than stating the limit.
"""

from __future__ import annotations

import hmac
import ipaddress
import os
from typing import Iterable

from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request
from starlette.responses import JSONResponse, Response

#: Env var holding the shared token. Empty/unset means "no token configured".
ACCESS_TOKEN_ENV = "APP_ACCESS_TOKEN"

#: Cookie the guard sets after a successful query-parameter handshake.
ACCESS_COOKIE = "app_access"

#: Paths reachable without a token even from outside. Only the favicon and the
#: static assets the login-less pages need to render; nothing that reads account
#: state and nothing that mutates.
_PUBLIC_PREFIXES: tuple[str, ...] = ("/static/", "/favicon.ico")


class ExternalAccessDenied(RuntimeError):
    """Raised at startup when the configuration would expose the app unguarded."""


def is_loopback(host: str | None) -> bool:
    """True when ``host`` names this machine only."""
    if not host:
        return False
    text = host.strip()
    if not text:
        return False
    if text.lower() in {"localhost", "::1"}:
        return True
    try:
        return ipaddress.ip_address(text).is_loopback
    except ValueError:
        return False


def binds_externally(host: str | None) -> bool:
    """Would binding to ``host`` make the app reachable from other machines?

    ``0.0.0.0`` and ``::`` are the wildcards that matter in practice; anything
    that is not a loopback literal is treated as external, because guessing
    wrong in that direction is the expensive mistake.
    """
    if not host:
        return False
    return not is_loopback(host)


def configured_token() -> str:
    return (os.getenv(ACCESS_TOKEN_ENV) or "").strip()


def require_token_for_external_bind(host: str | None, token: str | None = None) -> None:
    """Refuse to start an unguarded server on a routable address.

    Called before uvicorn binds. A hard failure is the point: the alternative is
    a server that silently came up with trading controls open to the LAN, and an
    operator who finds out later.
    """
    if not binds_externally(host):
        return
    resolved = configured_token() if token is None else token.strip()
    if len(resolved) < 16:
        raise ExternalAccessDenied(
            f"--host {host} exposes this server beyond this machine, and it has "
            f"no authentication of its own while accepting live-trading and "
            f"shutdown requests. Set {ACCESS_TOKEN_ENV} to a secret of at least "
            f"16 characters before binding off-loopback, or keep --host 127.0.0.1."
        )


def _client_host(request: Request) -> str | None:
    client = request.client
    return client.host if client else None


def _presented_tokens(request: Request) -> Iterable[str]:
    authorization = request.headers.get("authorization") or ""
    if authorization.lower().startswith("bearer "):
        yield authorization[7:].strip()
    header = request.headers.get("x-app-token")
    if header:
        yield header.strip()
    query = request.query_params.get("token")
    if query:
        yield query.strip()
    cookie = request.cookies.get(ACCESS_COOKIE)
    if cookie:
        yield cookie.strip()


class AccessGuardMiddleware(BaseHTTPMiddleware):
    """Challenge non-loopback requests when a token is configured."""

    def __init__(self, app, *, token_provider=configured_token) -> None:
        super().__init__(app)
        self._token_provider = token_provider

    async def dispatch(self, request: Request, call_next) -> Response:
        token = (self._token_provider() or "").strip()
        # No token configured means the operator is running the default
        # loopback-only setup; the guard stays out of the way entirely.
        if not token:
            return await call_next(request)
        if is_loopback(_client_host(request)):
            return await call_next(request)
        path = request.url.path
        if path.startswith(_PUBLIC_PREFIXES):
            return await call_next(request)

        matched = any(
            hmac.compare_digest(presented, token)
            for presented in _presented_tokens(request)
        )
        if not matched:
            return JSONResponse(
                {
                    "error": "ACCESS_TOKEN_REQUIRED",
                    "detail": (
                        "This server is reachable from other machines and "
                        "requires a token. Append ?token=... once, or send it as "
                        "an Authorization: Bearer header."
                    ),
                },
                status_code=401,
            )

        response = await call_next(request)
        # Promote a query-parameter token to a cookie so the page's own relative
        # fetches carry it without the token sitting in every URL (and in the
        # browser history of every subsequent request).
        if request.query_params.get("token") and request.cookies.get(ACCESS_COOKIE) != token:
            response.set_cookie(
                ACCESS_COOKIE,
                token,
                httponly=True,
                samesite="lax",
                max_age=60 * 60 * 24 * 7,
            )
        return response
