"""HTTP and WebSocket surface for the context -> decision -> gate chain.

Read-only by construction. There is no route here that opens, sizes or cancels anything:
every endpoint reports what the pipeline already decided. A dashboard that could place an
order would make the gate reachable over HTTP, which is the one thing the whole risk
layer exists to prevent.

Availability is explicit
------------------------
Every endpoint answers ``{"available": false, "reason": ...}`` rather than 404 or an empty
object when the pipeline has not produced a cycle yet. "No data" and "the ticker is not a
candidate" are different answers and the UI renders them differently; collapsing both
into an empty body is how a dashboard ends up showing zeros that look like readings.

WebSocket channels
------------------
``context``, ``candidates``, ``orders``, ``health`` — one connection per channel,
push-on-interval, and each frame carries its own ``as_of`` so a client can tell a repeat
from a refresh. Disconnects are normal and are not logged as errors.
"""

from __future__ import annotations

import asyncio
import contextlib
from datetime import datetime, timezone
from typing import Any, Callable

from fastapi import APIRouter, Query, WebSocket, WebSocketDisconnect
from fastapi.responses import JSONResponse

__all__ = ["REALTIME_CHANNELS", "create_context_router"]

REALTIME_CHANNELS: tuple[str, ...] = ("context", "candidates", "orders", "health")

#: Push interval per channel, in seconds. Context and candidates move with the trading
#: cycle; health is cheap and worth seeing sooner; orders change only on a transition, so
#: a shorter interval buys nothing.
_CHANNEL_INTERVAL_SECONDS: dict[str, float] = {
    "context": 5.0,
    "candidates": 5.0,
    "orders": 3.0,
    "health": 2.0,
}


def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


def create_context_router(
    *,
    runtime_provider: Callable[[], Any],
) -> APIRouter:
    """Build the router.

    ``runtime_provider`` is a callable rather than an instance so the router can be
    mounted before the runtime exists, and so a test can swap it without rebuilding the
    application.
    """
    router = APIRouter()

    def runtime() -> Any:
        return runtime_provider()

    def _guarded(action: Callable[[Any], dict[str, Any]]) -> JSONResponse:
        """Run a view, turning an unexpected failure into an explicit unavailability.

        A dashboard route that 500s takes the operator's only view of the system with it,
        exactly when something is already wrong. The failure is reported in the body
        instead, where it is visible rather than hidden behind a stack trace.
        """
        try:
            service = runtime()
        except Exception as exc:  # noqa: BLE001
            return JSONResponse(
                {"available": False, "reason": "RUNTIME_UNAVAILABLE", "error": str(exc)},
                status_code=503,
            )
        if service is None:
            return JSONResponse(
                {"available": False, "reason": "RUNTIME_NOT_STARTED"}, status_code=503
            )
        try:
            return JSONResponse(action(service))
        except Exception as exc:  # noqa: BLE001
            return JSONResponse(
                {
                    "available": False,
                    "reason": "VIEW_FAILED",
                    "error": f"{type(exc).__name__}: {exc}",
                },
                status_code=500,
            )

    # ------------------------------------------------------------------ #
    # session and contexts
    # ------------------------------------------------------------------ #
    @router.get("/api/session/current")
    def session_current() -> JSONResponse:
        return _guarded(lambda service: service.session_view())

    @router.get("/api/context/global")
    def context_global() -> JSONResponse:
        return _guarded(lambda service: service.global_view())

    @router.get("/api/context/domestic")
    def context_domestic() -> JSONResponse:
        return _guarded(lambda service: service.domestic_view())

    @router.get("/api/context/sector/{sector}")
    def context_sector(sector: str) -> JSONResponse:
        return _guarded(lambda service: service.sector_view(sector))

    @router.get("/api/context/stock/{ticker}")
    def context_stock(ticker: str) -> JSONResponse:
        return _guarded(lambda service: service.stock_view(ticker))

    # ------------------------------------------------------------------ #
    # decisions
    # ------------------------------------------------------------------ #
    @router.get("/api/regime/current")
    def regime_current() -> JSONResponse:
        return _guarded(lambda service: service.regime_view())

    @router.get("/api/candidates")
    def candidates(limit: int = Query(default=50, ge=1, le=500)) -> JSONResponse:
        return _guarded(lambda service: service.candidates_view(limit=limit))

    @router.get("/api/decision/{ticker}")
    def decision(ticker: str) -> JSONResponse:
        return _guarded(lambda service: service.decision_view(ticker))

    @router.get("/api/gate/{ticker}")
    def gate(ticker: str) -> JSONResponse:
        return _guarded(lambda service: service.gate_view(ticker))

    # ------------------------------------------------------------------ #
    # health
    # ------------------------------------------------------------------ #
    @router.get("/api/model/health")
    def model_health() -> JSONResponse:
        return _guarded(lambda service: service.model_health_view())

    @router.get("/api/data/health")
    def data_health() -> JSONResponse:
        return _guarded(lambda service: service.data_health_view())

    @router.get("/api/context/dashboard")
    def context_dashboard() -> JSONResponse:
        return _guarded(lambda service: service.dashboard_view())

    @router.get("/api/execution/authority-path")
    def execution_authority_path() -> JSONResponse:
        """The actual chain, stage by stage, as the code runs it.

        Exists because the dashboards used to describe a path the system no longer
        takes — "final approval rests with RiskManager and ProfitabilityGate" — which is
        now false: those decide BEFORE election and have no vote afterwards. A screen
        that names the wrong authority is worse than one that names none, because an
        operator will look for a veto that cannot happen.
        """
        return _guarded(lambda service: service.authority_path_view())

    @router.get("/api/execution/latency")
    def execution_latency() -> JSONResponse:
        return _guarded(lambda service: service.latency_view())

    @router.get("/api/orders/open")
    def open_orders() -> JSONResponse:
        return _guarded(
            lambda service: {
                "available": True,
                "as_of": _utcnow().isoformat(),
                "summary": service.state_machine.summary(),
                "open": [
                    record.as_dict() for record in service.state_machine.open_intents()
                ],
            }
        )

    # ------------------------------------------------------------------ #
    # realtime channels
    # ------------------------------------------------------------------ #
    def _frame(service: Any, channel: str) -> dict[str, Any]:
        if channel == "context":
            payload = {
                "session": service.session_view(),
                "global": service.global_view(),
                "domestic": service.domestic_view(),
                "regime": service.regime_view(),
                "dashboard": service.dashboard_view(),
            }
        elif channel == "candidates":
            payload = service.candidates_view()
        elif channel == "orders":
            payload = {
                "summary": service.state_machine.summary(),
                "open": [
                    record.as_dict() for record in service.state_machine.open_intents()
                ],
            }
        else:
            payload = {
                "model": service.model_health_view(),
                "data": service.data_health_view(),
                "authority_path": service.authority_path_view(),
                "latency": service.latency_view(),
            }
        return {"channel": channel, "as_of": _utcnow().isoformat(), "payload": payload}

    @router.websocket("/ws/{channel}")
    async def channel_socket(websocket: WebSocket, channel: str) -> None:
        if channel not in REALTIME_CHANNELS:
            await websocket.close(code=4404)
            return
        await websocket.accept()
        interval = _CHANNEL_INTERVAL_SECONDS.get(channel, 5.0)
        try:
            while True:
                try:
                    service = runtime()
                except Exception as exc:  # noqa: BLE001
                    await websocket.send_json(
                        {
                            "channel": channel,
                            "as_of": _utcnow().isoformat(),
                            "error": f"{type(exc).__name__}: {exc}",
                        }
                    )
                else:
                    if service is None:
                        await websocket.send_json(
                            {
                                "channel": channel,
                                "as_of": _utcnow().isoformat(),
                                "payload": {
                                    "available": False,
                                    "reason": "RUNTIME_NOT_STARTED",
                                },
                            }
                        )
                    else:
                        # Views read cached state and touch SQLite; run them off the
                        # event loop so a slow read cannot stall every other socket.
                        frame = await asyncio.to_thread(_frame, service, channel)
                        await websocket.send_json(frame)
                await asyncio.sleep(interval)
        except WebSocketDisconnect:
            return
        except Exception:  # noqa: BLE001 - a dropped client is not a server error.
            with contextlib.suppress(Exception):
                await websocket.close()

    return router
