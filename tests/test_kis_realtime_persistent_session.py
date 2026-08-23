"""The KIS realtime collector must hold ONE session and re-diff in place.

Regression context: the collector used to disconnect on every resubscribe and
reconnect with a freshly issued approval key. KIS bills realtime registrations
per session, so those abandoned sessions drained the account's budget until only
a single symbol could be subscribed and candidate discovery starved.
"""

from __future__ import annotations

import asyncio
import json
from types import SimpleNamespace

import pytest

from app.data import kis_realtime
from app.data.kis_realtime import (
    kis_realtime_subscription_message,
    kis_realtime_unsubscribe_message,
    run_kis_realtime_websocket_collector,
)


class FakeWebSocket:
    """Records control frames and hands back queued inbound messages."""

    def __init__(self) -> None:
        self.sent: list[dict] = []
        self.closed = False

    async def send(self, raw: str) -> None:
        self.sent.append(json.loads(raw))

    async def recv(self) -> str:
        await asyncio.sleep(0)
        raise TimeoutError

    async def __aenter__(self):
        return self

    async def __aexit__(self, *exc):
        self.closed = True
        return False

    # -- assertions helpers -------------------------------------------- #
    def frames(self, tr_type: str) -> list[tuple[str, str]]:
        return [
            (f["body"]["input"]["tr_key"], f["body"]["input"]["tr_id"])
            for f in self.sent
            if f["header"]["tr_type"] == tr_type
        ]


class _StopAfter:
    """stop_event that reports 'set' only after N checks."""

    def __init__(self, checks: int) -> None:
        self.remaining = checks

    def is_set(self) -> bool:
        self.remaining -= 1
        return self.remaining <= 0


class _OneShotEvent:
    """resubscribe_event that fires once and can be cleared."""

    def __init__(self) -> None:
        self._set = True
        self.clear_calls = 0

    def is_set(self) -> bool:
        return self._set

    def clear(self) -> None:
        self._set = False
        self.clear_calls += 1


@pytest.fixture()
def patched(monkeypatch):
    socket = FakeWebSocket()
    monkeypatch.setattr(kis_realtime, "_load_websockets", lambda: SimpleNamespace(connect=lambda *a, **k: socket))
    monkeypatch.setattr(kis_realtime, "build_kis_client", lambda **k: object())
    monkeypatch.setattr(kis_realtime, "issue_websocket_approval_key", lambda client: "KEY")
    monkeypatch.setattr(kis_realtime, "LiveFeatureFrameBuilder", lambda store: None)
    monkeypatch.setenv("KIS_REALTIME_SUBSCRIBE_DELAY_SEC", "0")
    monkeypatch.setenv("KIS_REALTIME_POST_SUBSCRIBE_DRAIN_SEC", "0")
    return socket


def _run(**kwargs):
    return asyncio.run(
        run_kis_realtime_websocket_collector(
            store=SimpleNamespace(),
            subscription_tr_ids=("H0STCNT0", "H0STASP0"),
            **kwargs,
        )
    )


class TestFrames:
    def test_subscribe_and_unsubscribe_target_the_same_registration(self):
        sub = json.loads(kis_realtime_subscription_message("K", "H0STCNT0", "005930"))
        uns = json.loads(kis_realtime_unsubscribe_message("K", "H0STCNT0", "005930"))
        assert sub["header"]["tr_type"] == "1"
        assert uns["header"]["tr_type"] == "2"
        assert sub["body"] == uns["body"]


class TestPersistentSession:
    def test_registrations_are_released_on_normal_exit(self, patched):
        counts = _run(symbols=["005930"], stop_event=_StopAfter(1))
        assert patched.frames("1") == [("005930", "H0STCNT0"), ("005930", "H0STASP0")]
        # Every held registration handed back, so the next session starts clean.
        assert set(patched.frames("2")) == {("005930", "H0STCNT0"), ("005930", "H0STASP0")}
        assert counts["subscriptions_released"] == 2

    def test_resubscribe_diffs_in_place_without_reconnecting(self, patched, monkeypatch):
        issued = {"n": 0}

        def _key(client):
            issued["n"] += 1
            return "KEY"

        monkeypatch.setattr(kis_realtime, "issue_websocket_approval_key", _key)
        event = _OneShotEvent()
        counts = _run(
            symbols=["005930"],
            symbols_provider=lambda: ["000660"],
            resubscribe_event=event,
            stop_event=_StopAfter(3),
        )
        # One session, one approval key — not one per resubscribe.
        assert issued["n"] == 1
        assert counts["in_place_resubscribes"] == 1
        assert event.clear_calls == 1
        # The dropped symbol is unsubscribed and the new one subscribed.
        subs = patched.frames("1")
        assert ("005930", "H0STCNT0") in subs and ("000660", "H0STCNT0") in subs
        unsubs = patched.frames("2")
        assert ("005930", "H0STCNT0") in unsubs and ("005930", "H0STASP0") in unsubs

    def test_unchanged_symbol_set_keeps_existing_registrations(self, patched):
        event = _OneShotEvent()
        _run(
            symbols=["005930"],
            symbols_provider=lambda: ["005930"],
            resubscribe_event=event,
            stop_event=_StopAfter(3),
        )
        # Only the final teardown unsubscribes; no churn mid-session.
        assert len(patched.frames("1")) == 2
        assert len(patched.frames("2")) == 2

    def test_provider_absent_still_breaks_for_the_legacy_caller(self, patched):
        event = _OneShotEvent()
        counts = _run(symbols=["005930"], resubscribe_event=event, stop_event=_StopAfter(3))
        assert counts["resubscribe_requested"] == 1
        assert "in_place_resubscribes" not in counts

    def test_session_owner_change_releases_registrations(self, patched):
        active = iter((True, False))
        counts = _run(
            symbols=["005930"],
            session_active_provider=lambda: next(active),
            stop_event=_StopAfter(5),
        )
        assert counts["session_relinquished"] == 1
        assert set(patched.frames("2")) == {
            ("005930", "H0STCNT0"),
            ("005930", "H0STASP0"),
        }


class _SlowSilentWebSocket(FakeWebSocket):
    """Connected, subscribed, and permanently silent — the observed failure.

    On 2026-08-21 the overseas socket stayed ESTABLISHED with the kernel receive
    queue growing past 3MB while every subscribed symbol stopped at the same
    instant. Nothing raised, nothing reconnected, and US market data was gone for
    15+ minutes with the collector still reporting healthy.
    """

    def __init__(self, tick_seconds: float = 0.01) -> None:
        super().__init__()
        self.tick_seconds = tick_seconds
        self.recv_calls = 0

    async def recv(self) -> str:
        self.recv_calls += 1
        await asyncio.sleep(self.tick_seconds)
        raise TimeoutError


class _KeepaliveWebSocket(FakeWebSocket):
    """Silent tape but a live session: KIS keeps sending PINGPONG frames."""

    def __init__(self, tick_seconds: float = 0.01) -> None:
        super().__init__()
        self.tick_seconds = tick_seconds

    async def recv(self) -> str:
        await asyncio.sleep(self.tick_seconds)
        return json.dumps({"header": {"tr_id": "PINGPONG", "datetime": "20260821"}})


def _patch_socket(monkeypatch, socket):
    monkeypatch.setattr(
        kis_realtime, "_load_websockets", lambda: SimpleNamespace(connect=lambda *a, **k: socket)
    )
    monkeypatch.setattr(kis_realtime, "build_kis_client", lambda **k: object())
    monkeypatch.setattr(kis_realtime, "issue_websocket_approval_key", lambda client: "KEY")
    monkeypatch.setattr(kis_realtime, "LiveFeatureFrameBuilder", lambda store: None)
    monkeypatch.setenv("KIS_REALTIME_SUBSCRIBE_DELAY_SEC", "0")
    monkeypatch.setenv("KIS_REALTIME_POST_SUBSCRIBE_DRAIN_SEC", "0")
    return socket


class TestStreamStallWatchdog:
    def test_a_silent_socket_is_torn_down_instead_of_looped_on_forever(
        self, monkeypatch
    ):
        socket = _patch_socket(monkeypatch, _SlowSilentWebSocket())
        monkeypatch.setenv("KIS_REALTIME_WS_STALL_SEC", "0.05")
        monkeypatch.setenv("KIS_REALTIME_WS_MAX_SESSION_SEC", "0")

        counts = _run(
            symbols=["AAPL"],
            symbols_provider=lambda: ["AAPL"],
            stop_event=_StopAfter(5_000),
        )

        assert counts.get("stream_stalled") == 1
        assert counts.get("stream_stalled_seconds") >= 0.05
        # Broke out of the loop rather than exhausting the stop budget.
        assert socket.recv_calls < 5_000
        # Registrations handed back, so the reconnect starts with a full budget.
        assert counts["subscriptions_released"] == 2

    def test_a_quiet_tape_with_keepalives_is_not_mistaken_for_a_dead_stream(
        self, monkeypatch
    ):
        # No trade for the whole session, but the connection keeps emitting
        # frames. Gating the watchdog on ticks instead of frames would tear down
        # a perfectly healthy socket during every quiet auction.
        _patch_socket(monkeypatch, _KeepaliveWebSocket())
        monkeypatch.setenv("KIS_REALTIME_WS_STALL_SEC", "0.05")
        monkeypatch.setenv("KIS_REALTIME_WS_MAX_SESSION_SEC", "0")

        counts = _run(
            symbols=["AAPL"],
            symbols_provider=lambda: ["AAPL"],
            stop_event=_StopAfter(40),
        )

        assert not counts.get("stream_stalled")
        assert counts.get("pingpongs", 0) > 0

    def test_the_persistent_session_is_recycled_on_its_absolute_cap(
        self, monkeypatch
    ):
        # With ``symbols_provider`` the session is deliberately long-lived, which
        # left it with no upper bound at all.
        _patch_socket(monkeypatch, _KeepaliveWebSocket())
        monkeypatch.setenv("KIS_REALTIME_WS_STALL_SEC", "0")
        monkeypatch.setenv("KIS_REALTIME_WS_MAX_SESSION_SEC", "0.05")

        counts = _run(
            symbols=["AAPL"],
            symbols_provider=lambda: ["AAPL"],
            stop_event=_StopAfter(5_000),
        )

        assert counts.get("session_recycled") == 1
        assert counts["subscriptions_released"] == 2


class _TickFeedWebSocket(FakeWebSocket):
    """Streams one trade frame per recv, then keeps the session alive."""

    def __init__(self, frames: list[str]) -> None:
        super().__init__()
        self._frames = list(frames)

    async def recv(self) -> str:
        await asyncio.sleep(0)
        if self._frames:
            return self._frames.pop(0)
        raise TimeoutError


class _BlockingStore:
    """A store whose writes take real wall-clock time, like a contended SQLite."""

    def __init__(self, delay: float = 0.05) -> None:
        self.delay = delay
        self.calls = 0
        self.thread_names: set[str] = set()

    def _work(self) -> int:
        import threading
        import time as _time

        self.calls += 1
        self.thread_names.add(threading.current_thread().name)
        _time.sleep(self.delay)
        return 0

    def save_ticks(self, ticks) -> int:
        return self._work()

    def save_orderbooks(self, books) -> int:
        return self._work()

    def build_latest_minute_bar(self, symbol, *, now=None, stream_id=None):
        self._work()
        return None


class TestEventLoopIsNotBlockedByStoreWork:
    def test_minute_bar_rebuild_runs_off_the_websocket_event_loop(self, monkeypatch):
        """Store work in the message path must not stop the socket being drained.

        ``_build_minute_bars_throttled`` re-reads the current minute and upserts a
        bar. Throttled to once per symbol per 2s, six subscribed symbols made that
        up to three synchronous SQLite round trips per second on the websocket
        event loop, against an 8GB store with a 30s ``busy_timeout``. One
        contended statement stopped the loop reaching ``recv()``; the websockets
        library paused the transport and the kernel receive queue grew while KIS
        kept sending. The US feed died ~40s after every reconnect with no error.
        """
        import asyncio as _asyncio
        import threading

        store = _BlockingStore(delay=0.05)
        loop_thread: dict[str, str] = {}

        async def _scenario() -> None:
            loop_thread["name"] = threading.current_thread().name
            await kis_realtime._process_kis_realtime_raw(
                raw="0|HDFSCNT0|001|" + "^".join(["RBAQAAPL"] + ["1"] * 30),
                websocket=FakeWebSocket(),
                symbols={"AAPL"},
                store=store,
                feature_builder=None,
                counts={"ticks": 0, "orderbooks": 0, "messages": 0},
                event_sink=None,
            )

        _asyncio.run(_scenario())

        # Whatever the parser made of that frame, every store call it triggered
        # must have happened on a worker thread, never on the loop thread.
        assert store.calls > 0
        assert store.thread_names
        assert loop_thread["name"] not in store.thread_names

    def test_progress_callback_never_blocks_the_message_loop(self, monkeypatch):
        """Telemetry is published at most once per interval, not per message."""
        _patch_socket(
            monkeypatch,
            _TickFeedWebSocket(
                [json.dumps({"header": {"tr_id": "PINGPONG"}}) for _ in range(30)]
            ),
        )
        monkeypatch.setenv("KIS_REALTIME_WS_STALL_SEC", "0")
        monkeypatch.setenv("KIS_REALTIME_WS_MAX_SESSION_SEC", "0")
        monkeypatch.setenv("KIS_REALTIME_PROGRESS_INTERVAL_SEC", "60")
        published: list[int] = []

        counts = _run(
            symbols=["AAPL"],
            symbols_provider=lambda: ["AAPL"],
            stop_event=_StopAfter(40),
            progress_callback=lambda counts: published.append(
                int(counts.get("messages") or 0)
            ),
        )

        # Every frame was consumed, so the throttle is what limited publishing.
        assert counts["pingpongs"] == 30
        # 30 inbound frames, but the per-message publish is throttled: only the
        # state-transition publishes (subscribe/teardown) get through.
        assert 0 < len(published) <= 5, published
        assert counts["subscriptions_released"] == 2  # tore down cleanly
