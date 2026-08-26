"""Read-only reverse proxy that publishes the LIVE dashboard, unchanged.

Why a proxy and not a second instance
-------------------------------------
The published site used to be a second full application process in read-only
mode. It read the same SQLite stores, so account and market data matched, but
everything the live process holds in memory -- the trading engine's cycle count,
which workers are alive, the operation mode -- it could only guess at, and it
guessed about itself. The published dashboard therefore reported a stopped
engine and a degraded readiness score while the engine was mid-session a port
away. Every fix for that is a mechanism for copying one process's memory into
another's view of it.

A proxy removes the question instead of answering it. There is one application
process, one set of workers, one owner of the stores, and the external site is
byte-for-byte the page the local server serves, because it IS that page.

THE SECURITY PROPERTY, STATED PLAINLY
-------------------------------------
This proxy connects to the upstream over loopback. ``AccessGuardMiddleware``
decides whether to demand a token from ``request.client.host`` alone, so to the
live server every request arriving through here looks like the operator at the
keyboard and is waved through without a token. The upstream also runs with
``LIVE_ORDER_SUBMIT_ENABLED=true`` and ``REQUIRE_MANUAL_ARMING=false``.

So this process is the ONLY thing standing between the public internet and a
server that will place real orders. Two independent gates, both fail-closed:

1. **Token.** Every request must carry the shared secret. No token, no upstream
   call -- the request is refused here and never reaches 8010. This gate can be
   switched off; see ANONYMOUS MODE below.
2. **Method gate.** Only GET and HEAD are forwarded. Every mutating route in the
   application is POST, PUT or DELETE without exception -- the route table holds
   78 GET routes that read and 25 non-GET routes, and everything that starts,
   stops, applies, arms, terminates or submits is among the latter -- so refusing
   the other verbs here makes all of them unreachable through this port even for
   a caller holding the token. This gate is NOT optional.

ANONYMOUS MODE
--------------
``PUBLIC_PROXY_ALLOW_ANONYMOUS=true`` drops gate 1 by operator decision. State the
consequence exactly rather than softly:

* Every published GET becomes world-readable. That is the whole dashboard --
  balance, cash, holdings, realised and unrealised PnL, the strategy session, the
  decision journal and every ``/api`` read behind it.
* "Nobody knows the URL" is not a control. Funnel serves this on a Let's Encrypt
  certificate for the tailnet hostname, and issued certificates are published to
  Certificate Transparency logs, so the name is discoverable by anyone watching
  them rather than guessable.
* What still holds: gate 2 means no visitor can place, amend or cancel an order,
  arm trading, change flags or shut anything down, because those routes are all
  non-GET and are refused here without ever reaching 8010. Checked at the time of
  writing: no read path returns broker credentials or the access token, and no
  served page embeds one.

So the exposure is READ of account data, not control of the account. That is a
real cost and it is the operator's to accept; the default stays off.

   The gate is on the METHOD rather than on a curated list of paths because the
   published page has to BE the page the server serves. A hand-maintained path
   allowlist silently 404s a panel the moment the dashboard grows one, and that
   drift is the whole reason this port stopped matching the local site.

Neither gate depends on the upstream's own posture, because the upstream's
posture is "live". Do not add a passthrough for convenience.
"""

from __future__ import annotations

import hmac
import os
import re
from typing import Iterable

import httpx
from starlette.applications import Starlette
from starlette.requests import Request
from starlette.responses import JSONResponse, RedirectResponse, Response
from starlette.routing import Route

#: Hop-by-hop headers that must not be relayed (RFC 7230 6.1), plus the ones
#: whose value belongs to this connection rather than the proxied one.
_HOP_BY_HOP = frozenset(
    {
        "connection", "keep-alive", "proxy-authenticate", "proxy-authorization",
        "te", "trailers", "transfer-encoding", "upgrade",
        "content-encoding", "content-length", "host",
    }
)

#: Headers describing the hop BEFORE this proxy. They must not be relayed, and
#: the reason is specific rather than stylistic.
#:
#: Uvicorn runs with ``proxy_headers=True`` and ``forwarded_allow_ips="127.0.0.1"``
#: by default, so it trusts these headers from a loopback client -- which is
#: exactly what this proxy is. Relaying Funnel's ``X-Forwarded-For: <public ip>``
#: therefore made uvicorn rewrite ``request.client.host`` to that public address,
#: AccessGuardMiddleware saw a non-loopback client and demanded a token the
#: viewer had already given to this proxy, and every page through Funnel returned
#: 401 while the same request straight to this port returned 200.
#:
#: Stripping them makes the upstream see this proxy as the local client it
#: actually is. Note which way that cuts: it means the upstream's guard is NOT a
#: second line of defence behind this one, and the token check and method gate
#: here are the whole boundary. They are written to be exactly that.
_FORWARDED = frozenset(
    {
        "forwarded",
        "x-forwarded-for", "x-forwarded-host", "x-forwarded-proto",
        "x-forwarded-port", "x-forwarded-server", "x-real-ip",
    }
)

#: Read paths withheld anyway. Deliberately short, and NOT the security boundary
#: -- the method gate is. This is where a GET route goes if it should not be
#: published even to a viewer holding the token.
_DENIED: tuple[re.Pattern[str], ...] = (
    # Test-harness surface. Not part of the dashboard and not worth publishing.
    re.compile(r"^/api/mock-kis/.*$"),
)

ACCESS_COOKIE = "app_access"

#: Response headers the UPSTREAM sets that this hop must own instead of relaying.
#:
#: ``date`` is a singleton field (RFC 9110 5.5.1) and uvicorn emits its own for this
#: connection, so relaying the upstream's too put TWO ``date`` headers on every
#: published response -- malformed, and left to whatever each intermediary decides to
#: do about it. ``server`` is worse than untidy: it advertised the exact stack of a
#: server that places real orders to the public internet, defeating the
#: ``--no-server-header`` this process is launched with.
_RESPONSE_OWNED_BY_THIS_HOP = frozenset({"date", "server"})


def _wants_html(request: Request) -> bool:
    """Is this a browser navigation rather than a programmatic fetch?

    Errors used to be JSON unconditionally, which is unreadable in the one case that
    matters most: a person opening the published URL. A browser handed
    ``application/json`` does not render a page -- Safari saves it to disk, others show
    the raw body -- so the site appeared to "download a file instead of loading", and
    since the token cookie is only set on a SUCCESSFUL token request there was no way
    in from a plain visit at all.
    """
    return "text/html" in (request.headers.get("accept") or "")


def _error(
    request: Request,
    *,
    status_code: int,
    error: str,
    detail: str,
    heading: str,
    prompt_for_token: bool = False,
) -> Response:
    """One error shape, rendered for whichever kind of client asked."""
    if not _wants_html(request):
        return JSONResponse({"error": error, "detail": detail}, status_code=status_code)
    # A GET form, because the method gate here forwards GET and HEAD only -- and it
    # needs no action attribute, so submitting keeps whichever path was requested and
    # simply attaches the token to it. The successful request then sets the cookie and
    # the rest of the session needs no token in any URL.
    form = (
        """
    <form method="get" class="row">
      <input type="password" name="token" placeholder="Access token" aria-label="Access token"
             autocomplete="current-password" autofocus>
      <button type="submit">Open</button>
    </form>"""
        if prompt_for_token
        else ""
    )
    body = f"""<!doctype html>
<html lang="ko">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>{heading}</title>
<style>
  :root {{ color-scheme: light dark; }}
  body {{ margin:0; min-height:100vh; display:grid; place-items:center;
         font:16px/1.55 system-ui,-apple-system,"Segoe UI",sans-serif;
         background:#f6f7f9; color:#1a1d21; padding:24px; }}
  main {{ width:100%; max-width:26rem; background:#fff; border:1px solid #e3e6ea;
          border-radius:12px; padding:28px; }}
  h1 {{ margin:0 0 .5rem; font-size:1.15rem; }}
  p {{ margin:0 0 1.25rem; color:#5b6472; }}
  .row {{ display:flex; gap:8px; }}
  input {{ flex:1 1 auto; min-width:0; padding:9px 11px; font-size:1rem;
           border:1px solid #ccd2da; border-radius:8px; background:#fff; color:inherit; }}
  button {{ padding:9px 16px; font-size:1rem; font-weight:600; cursor:pointer;
            border:0; border-radius:8px; background:#1f6feb; color:#fff; }}
  code {{ background:#eef0f3; padding:.1em .4em; border-radius:4px; font-size:.9em; }}
  @media (prefers-color-scheme: dark) {{
    body {{ background:#0f1216; color:#e6e9ee; }}
    main {{ background:#161a20; border-color:#2a3039; }}
    p {{ color:#9aa4b2; }}
    input {{ background:#0f1216; border-color:#39414c; }}
    code {{ background:#232a33; }}
  }}
</style>
</head>
<body>
  <main>
    <h1>{heading}</h1>
    <p>{detail}</p>{form}
  </main>
</body>
</html>
"""
    return Response(
        content=body,
        status_code=status_code,
        media_type="text/html; charset=utf-8",
        # An error page must never be cached, least of all one behind a shared URL.
        headers={"cache-control": "no-store"},
    )


def _landing_path() -> str:
    """Where a bare visit to the published root should land.

    The site's root is not the page this system is operated from. ``/`` serves the
    general home page ("개인 투자 분석 시스템"), while the operator's console is
    ``/account`` -- which is exactly what the local launcher opens
    (``http://127.0.0.1:8010/account``) and what this proxy's own start-up banner
    advertises. Publishing the root unchanged therefore landed visitors on a
    different page from the one the URL is meant to show.

    Set ``PUBLIC_PROXY_LANDING_PATH`` to empty to serve the root as-is.
    """
    value = os.getenv("PUBLIC_PROXY_LANDING_PATH")
    return ("/account" if value is None else value).strip()


def _with_token_cookie(request: Request, response: Response) -> Response:
    """Promote a query token to a cookie.

    So the page's own relative fetches carry it without the secret sitting in every
    URL and in browser history. Applied to the redirect below as well as to proxied
    pages: the redirect is the first response a ``?token=`` visit receives, and if it
    did not carry the cookie the follow-up request would arrive unauthenticated.
    """
    token = _configured_token()
    if request.query_params.get("token") and request.cookies.get(ACCESS_COOKIE) != token:
        response.set_cookie(
            ACCESS_COOKIE,
            token,
            httponly=True,
            samesite="lax",
            max_age=60 * 60 * 24 * 7,
        )
    return response


def _configured_token() -> str:
    return (os.getenv("PUBLIC_PROXY_TOKEN") or "").strip()


def _upstream() -> str:
    return (os.getenv("PUBLIC_PROXY_UPSTREAM") or "http://127.0.0.1:8010").rstrip("/")


def _presented(request: Request) -> Iterable[str]:
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


def _anonymous_reads_allowed() -> bool:
    """Is the token gate switched off, leaving the method gate as the only one?

    Operator decision, off by default, and it must be set deliberately -- an absent
    or malformed value keeps the gate on. Read what it costs in the module docstring
    under ANONYMOUS MODE before enabling it.
    """
    return (os.getenv("PUBLIC_PROXY_ALLOW_ANONYMOUS") or "").strip().lower() in {
        "1",
        "true",
        "yes",
        "on",
    }


def _authorized(request: Request) -> bool:
    if _anonymous_reads_allowed():
        # The method gate and the denied-path list still apply; this only drops the
        # token requirement. Nothing here can reach a mutating route.
        return True
    token = _configured_token()
    if len(token) < 16:
        # No usable secret means no access at all. Failing closed matters more
        # here than anywhere else in this file: the alternative is an open proxy
        # onto a live trading server.
        return False
    return any(hmac.compare_digest(candidate, token) for candidate in _presented(request))


def _path_allowed(path: str) -> bool:
    return not any(pattern.match(path) for pattern in _DENIED)


async def _proxy(request: Request) -> Response:
    if request.method not in {"GET", "HEAD"}:
        return _error(
            request,
            status_code=405,
            error="READ_ONLY_ENDPOINT",
            detail="This published view forwards read requests only.",
            heading="읽기 전용 페이지입니다",
        )
    if not _authorized(request):
        return _error(
            request,
            status_code=401,
            error="ACCESS_TOKEN_REQUIRED",
            detail=(
                "이 페이지는 액세스 토큰이 필요합니다. 아래에 입력하면 이후 접속은 "
                "쿠키로 유지됩니다."
            ),
            heading="토큰이 필요합니다",
            prompt_for_token=_configured_token() != "",
        )
    path = request.url.path
    if not _path_allowed(path):
        return _error(
            request,
            status_code=404,
            error="NOT_PUBLISHED",
            detail="This path is not published.",
            heading="공개되지 않은 경로입니다",
        )
    # Deliberately AFTER the token gate, so the redirect is never a hop an
    # unauthenticated caller can use to probe which paths exist.
    landing = _landing_path()
    if path == "/" and landing and landing != "/":
        return _with_token_cookie(
            request,
            RedirectResponse(landing, status_code=302),
        )

    # The token is ours, not the upstream's; forwarding it would put the secret
    # in the live server's logs for no benefit.
    query = [(k, v) for k, v in request.query_params.multi_items() if k != "token"]
    headers = {
        key: value
        for key, value in request.headers.items()
        if key.lower() not in _HOP_BY_HOP
        and key.lower() not in _FORWARDED
        and key.lower() != "cookie"
    }
    client: httpx.AsyncClient = request.app.state.client
    try:
        upstream = await client.request(
            request.method, path, params=query, headers=headers
        )
    except httpx.TimeoutException:
        return JSONResponse(
            {"error": "UPSTREAM_TIMEOUT", "detail": "The local server did not respond in time."},
            status_code=504,
        )
    except httpx.HTTPError as first_exc:
        # This process deliberately outlives the live server.  After a live-server
        # restart, a pooled loopback keep-alive socket can be half-closed: the first
        # read gets RemoteProtocolError/ReadError even though a new connection would
        # succeed immediately.  Every method reaching this point is GET or HEAD, so
        # one transport retry is idempotent and cannot duplicate an order or mutation.
        try:
            upstream = await client.request(
                request.method, path, params=query, headers=headers
            )
        except httpx.TimeoutException:
            return JSONResponse(
                {
                    "error": "UPSTREAM_TIMEOUT",
                    "detail": "The local server did not respond in time after reconnecting.",
                },
                status_code=504,
            )
        except httpx.HTTPError as retry_exc:
            return JSONResponse(
                {
                    "error": "UPSTREAM_UNAVAILABLE",
                    "detail": str(retry_exc),
                    "first_error": str(first_exc),
                },
                status_code=502,
            )

    relayed = {
        key: value
        for key, value in upstream.headers.items()
        if key.lower() not in _HOP_BY_HOP
        and key.lower() not in _RESPONSE_OWNED_BY_THIS_HOP
    }
    response = Response(
        content=upstream.content,
        status_code=upstream.status_code,
        headers=relayed,
        media_type=upstream.headers.get("content-type"),
    )
    return _with_token_cookie(request, response)


def create_app() -> Starlette:
    async def _startup() -> None:
        app.state.client = httpx.AsyncClient(
            base_url=_upstream(),
            timeout=httpx.Timeout(30.0, connect=5.0),
            follow_redirects=False,
        )

    async def _shutdown() -> None:
        await app.state.client.aclose()

    app = Starlette(
        routes=[
            Route("/{path:path}", _proxy, methods=["GET", "HEAD", "POST", "PUT", "PATCH", "DELETE"]),
        ],
        on_startup=[_startup],
        on_shutdown=[_shutdown],
    )
    return app


app = create_app()
