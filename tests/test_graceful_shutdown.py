"""Replacing a running server must not abandon a managed position.

``run.ps1`` used to force-kill whatever was listening. That terminates the realtime
trading engine mid-cycle and, worse, discards the in-memory exit state of an open
trade -- the armed stop, target, trailing high-watermark and holding clock. The
broker keeps the position; nothing is left watching it. These tests pin the checks
that now stand between a relaunch and that outcome.
"""

from __future__ import annotations

import json
from types import SimpleNamespace

import pytest

import app.web as web


def _payload(response) -> dict:
    """Parse a JSONResponse body. Substring-matching the raw body is brittle --
    the serializer emits no space after the colon, which is not the contract."""
    return json.loads(response.body.decode("utf-8"))


class _Holding:
    def __init__(self, ticker: str) -> None:
        self.ticker = ticker


def _account(*tickers: str) -> SimpleNamespace:
    return SimpleNamespace(holdings=tuple(_Holding(t) for t in tickers))


@pytest.fixture(autouse=True)
def _no_real_engine(monkeypatch):
    """Default: flat account, no engine. Individual tests override."""
    monkeypatch.setattr(web, "_realtime_engine_account_snapshot", lambda: _account())
    monkeypatch.setattr(web, "_realtime_trading_engine", None, raising=False)
    yield


def test_flat_and_idle_is_safe() -> None:
    report = web._restart_safety_report()
    assert report["safe"] is True
    assert report["reasons"] == []
    assert report["holdings_count"] == 0


def test_open_position_is_unsafe(monkeypatch) -> None:
    monkeypatch.setattr(
        web, "_realtime_engine_account_snapshot", lambda: _account("005930")
    )
    report = web._restart_safety_report()
    assert report["safe"] is False
    assert any(reason.startswith("OPEN_POSITIONS") for reason in report["reasons"])
    assert report["holdings_count"] == 1
    assert report["positions"] == ["005930"]


def test_unreadable_account_fails_closed(monkeypatch) -> None:
    """An unreadable account is exactly when you least want to assume "no position"."""

    def _boom():
        raise RuntimeError("broker unreachable")

    monkeypatch.setattr(web, "_realtime_engine_account_snapshot", _boom)
    report = web._restart_safety_report()
    assert report["safe"] is False
    assert any("ACCOUNT_SNAPSHOT_UNAVAILABLE" in r for r in report["reasons"])


def test_missing_account_snapshot_fails_closed(monkeypatch) -> None:
    monkeypatch.setattr(web, "_realtime_engine_account_snapshot", lambda: None)
    report = web._restart_safety_report()
    assert report["safe"] is False


@pytest.mark.parametrize("phase", ["ARMED", "ENTERING", "EXITING"])
def test_in_flight_session_phase_is_unsafe(monkeypatch, phase: str) -> None:
    """An order in flight or an exit being worked must not be interrupted."""
    engine = SimpleNamespace(
        get_status=lambda: {"strategy_session": {"phase": phase}}
    )
    monkeypatch.setattr(web, "_realtime_trading_engine", engine, raising=False)
    report = web._restart_safety_report()
    assert report["safe"] is False
    assert any("SESSION_PHASE_IN_FLIGHT" in r for r in report["reasons"])
    assert report["session_phase"] == phase


def test_scanning_session_is_safe(monkeypatch) -> None:
    engine = SimpleNamespace(
        get_status=lambda: {"strategy_session": {"phase": "SCANNING"}}
    )
    monkeypatch.setattr(web, "_realtime_trading_engine", engine, raising=False)
    report = web._restart_safety_report()
    assert report["safe"] is True
    assert report["engine_running"] is True


def test_endpoint_refuses_unsafe_shutdown(monkeypatch) -> None:
    monkeypatch.setattr(
        web, "_realtime_engine_account_snapshot", lambda: _account("005930")
    )
    exits: list[float] = []
    monkeypatch.setattr(
        web, "_schedule_app_process_shutdown", lambda *a, **k: exits.append(1.0)
    )
    payload = _payload(web.system_graceful_shutdown(force=False))
    assert payload["ok"] is False
    assert payload["status"] == "refused"
    assert payload["positions"] == ["005930"]
    assert exits == [], "a refused shutdown must not schedule an exit"


def test_endpoint_proceeds_when_forced(monkeypatch) -> None:
    monkeypatch.setattr(
        web, "_realtime_engine_account_snapshot", lambda: _account("005930")
    )
    exits: list[float] = []
    monkeypatch.setattr(
        web, "_schedule_app_process_shutdown", lambda *a, **k: exits.append(1.0)
    )
    payload = _payload(web.system_graceful_shutdown(force=True))
    assert payload["ok"] is True
    assert payload["status"] == "shutting_down"
    assert payload["forced"] is True
    assert exits == [1.0], "an explicitly forced shutdown must proceed"


def test_endpoint_proceeds_when_safe(monkeypatch) -> None:
    exits: list[float] = []
    monkeypatch.setattr(
        web, "_schedule_app_process_shutdown", lambda *a, **k: exits.append(1.0)
    )
    payload = _payload(web.system_graceful_shutdown(force=False))
    assert payload["ok"] is True
    assert payload["status"] == "shutting_down"
    assert exits == [1.0]


# --------------------------------------------------------------------------- #
# Teardown ordering                                                            #
# --------------------------------------------------------------------------- #
#: Every stopper ``_graceful_teardown`` invokes, in its order. Kept as one list so adding
#: a worker updates both tests together — and so a worker added to the teardown but not
#: to this list fails the completeness check below rather than passing unnoticed.
_TEARDOWN_STOPPERS: tuple[str, ...] = (
    "_stop_realtime_trading_engine",
    "_stop_context_refresher",
    "_stop_auto_reliability_controller",
    "_stop_live_training_worker",
    "_stop_temporal_gnn_training_worker",
    "_stop_krx_feature_frame_worker",
    "_stop_kis_overseas_realtime_collector",
    "_stop_kis_realtime_collector",
    "_stop_asset_history_sampler",
    "_stop_live_worker",
)


def test_teardown_list_matches_the_stoppers_under_test(monkeypatch) -> None:
    """A worker added to the teardown must be added to ``_TEARDOWN_STOPPERS`` too."""
    called: list[str] = []
    for name in _TEARDOWN_STOPPERS:
        monkeypatch.setattr(web, name, (lambda n: lambda: called.append(n))(name))
    stopped = web._graceful_teardown()
    assert len(stopped) == len(called), (
        "a stopper ran that this test does not patch; add it to _TEARDOWN_STOPPERS"
    )


def test_teardown_stops_trading_before_market_data(monkeypatch) -> None:
    """An engine still evaluating while its price feed disappears is the exact
    stale-data condition every gate in this codebase exists to prevent."""
    order: list[str] = []
    for name in _TEARDOWN_STOPPERS:
        monkeypatch.setattr(web, name, (lambda n: lambda: order.append(n))(name))

    stopped = web._graceful_teardown()
    assert order.index("_stop_realtime_trading_engine") == 0
    assert order.index("_stop_realtime_trading_engine") < order.index(
        "_stop_kis_realtime_collector"
    )
    assert "realtime_trading_engine" in stopped
    assert len(stopped) == len(_TEARDOWN_STOPPERS)


def test_one_stuck_worker_does_not_block_the_rest(monkeypatch) -> None:
    order: list[str] = []
    for name in _TEARDOWN_STOPPERS:
        if name == "_stop_realtime_trading_engine":
            continue
        monkeypatch.setattr(web, name, (lambda n: lambda: order.append(n))(name))

    def _stuck():
        raise RuntimeError("worker will not stop")

    monkeypatch.setattr(web, "_stop_realtime_trading_engine", _stuck)

    stopped = web._graceful_teardown()
    assert "realtime_trading_engine" not in stopped
    assert len(stopped) == len(_TEARDOWN_STOPPERS) - 1, (
        "every other worker must still be stopped"
    )
