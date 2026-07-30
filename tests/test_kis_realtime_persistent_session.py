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
