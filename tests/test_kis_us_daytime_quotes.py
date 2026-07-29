"""US daytime (주간거래) realtime quotes.

Per the KIS HDFSCNT0/HDFSASP0 documentation the daytime session is served by the
same TRs but a different subscription key family:

    night / regular   D + NAS|NYS|AMS + ticker      e.g. DNASAAPL
    daytime (주간거래) R + BAQ|BAY|BAA + ticker      e.g. RBAQAAPL

Subscribing with the wrong family is silent — KIS accepts it and simply never
delivers data for that session, which is why the daytime window produced no
ticks at all.
"""

from __future__ import annotations

import asyncio
import json
from datetime import datetime, timezone
from types import SimpleNamespace
from zoneinfo import ZoneInfo

import pytest

from app.data import kis_realtime
from app.data.kis_realtime import (
    US_DAYTIME_EXCHANGE_CODES,
    is_us_daytime_quote_session,
    overseas_realtime_subscription_key,
    parse_kis_realtime_message,
    run_kis_overseas_realtime_websocket_collector,
)

_SEOUL = ZoneInfo("Asia/Seoul")


def kst(hour: int, minute: int = 0, *, day: int = 30) -> datetime:
    """A weekday at the given KST wall-clock time, as UTC."""
    return datetime(2026, 7, day, hour, minute, tzinfo=_SEOUL).astimezone(timezone.utc)


@pytest.fixture(autouse=True)
def _no_env_override(monkeypatch):
    monkeypatch.delenv("KIS_FORCE_US_DAYTIME_QUOTES", raising=False)


class TestDaytimeWindow:
    @pytest.mark.parametrize(
        "hour,expected",
        [(9, False), (10, True), (13, True), (15, True), (16, False), (22, False)],
    )
    def test_window_is_10_to_16_kst(self, hour, expected):
        assert is_us_daytime_quote_session(kst(hour, 30 if hour != 16 else 0)) is expected

    def test_weekend_is_never_daytime(self):
        saturday = datetime(2026, 8, 1, 13, 0, tzinfo=_SEOUL).astimezone(timezone.utc)
        assert is_us_daytime_quote_session(saturday) is False

    def test_env_override_forces_and_disables(self, monkeypatch):
        monkeypatch.setenv("KIS_FORCE_US_DAYTIME_QUOTES", "true")
        assert is_us_daytime_quote_session(kst(22)) is True
        monkeypatch.setenv("KIS_FORCE_US_DAYTIME_QUOTES", "false")
        assert is_us_daytime_quote_session(kst(13)) is False


class TestSubscriptionKey:
    @pytest.mark.parametrize(
        "hint,night,day",
        [
            ("NASDAQ", "DNASAAPL", "RBAQAAPL"),
            ("NYSE", "DNYSAAPL", "RBAYAAPL"),
            ("AMEX", "DAMSAAPL", "RBAAAAPL"),
        ],
    )
    def test_key_family_follows_the_session(self, hint, night, day):
        assert overseas_realtime_subscription_key("AAPL", hint, now=kst(22)) == night
        assert overseas_realtime_subscription_key("AAPL", hint, now=kst(13)) == day

    def test_exchange_code_map_matches_the_documented_families(self):
        assert US_DAYTIME_EXCHANGE_CODES == {"NAS": "BAQ", "NYS": "BAY", "AMS": "BAA"}

    def test_domestic_symbol_is_rejected(self):
        with pytest.raises(ValueError):
            overseas_realtime_subscription_key("005930", "KRX")


class TestResponseParsing:
    """Both key families must map back to the plain ticker."""

    @staticmethod
    def _frame(tr_key: str) -> str:
        row = [""] * kis_realtime.KIS_OVERSEAS_TRADE_FIELDS_PER_RECORD
        row[0] = tr_key
        row[1] = "AAPL"
        row[4] = "20260730"
        row[5] = "103000"
        row[8] = "212.34"
        row[9] = "10"
        return f"0|HDFSCNT0|001|{'^'.join(row)}"

    @pytest.mark.parametrize("tr_key", ["DNASAAPL", "RBAQAAPL", "RBAYAAPL", "RBAAAAPL"])
    def test_daytime_and_night_keys_both_yield_the_ticker(self, tr_key):
        parsed = parse_kis_realtime_message(
            self._frame(tr_key), received_at=datetime(2026, 7, 30, 4, 0, tzinfo=timezone.utc)
        )
        assert parsed.event_type == "overseas_trade"
        assert parsed.ticks[0].symbol == "AAPL"


class _FakeSocket:
    def __init__(self):
        self.sent = []

    async def send(self, raw):
        self.sent.append(json.loads(raw))

    async def recv(self):
        await asyncio.sleep(0)
        raise TimeoutError

    async def __aenter__(self):
        return self

    async def __aexit__(self, *exc):
        return False

    def keys(self, tr_type):
        return [
            f["body"]["input"]["tr_key"]
            for f in self.sent
            if f["header"]["tr_type"] == tr_type and f["body"]["input"]["tr_id"] == "HDFSCNT0"
        ]


class _StopAfter:
    def __init__(self, n):
        self.n = n

    def is_set(self):
        self.n -= 1
        return self.n <= 0


class TestCollectorUsesDaytimeKeys:
    @pytest.fixture()
    def socket(self, monkeypatch):
        sock = _FakeSocket()
        monkeypatch.setattr(
            kis_realtime, "_load_websockets", lambda: SimpleNamespace(connect=lambda *a, **k: sock)
        )
        monkeypatch.setattr(kis_realtime, "build_kis_client", lambda **k: object())
        monkeypatch.setattr(kis_realtime, "issue_websocket_approval_key", lambda c: "KEY")
        monkeypatch.setattr(kis_realtime, "LiveFeatureFrameBuilder", lambda store: None)
        monkeypatch.setenv("KIS_REALTIME_SUBSCRIBE_DELAY_SEC", "0")
        monkeypatch.setenv("KIS_REALTIME_POST_SUBSCRIBE_DRAIN_SEC", "0")
        return sock

    def test_daytime_session_subscribes_the_r_family(self, socket, monkeypatch):
        monkeypatch.setenv("KIS_FORCE_US_DAYTIME_QUOTES", "true")
        asyncio.run(
            run_kis_overseas_realtime_websocket_collector(
                symbols=["AAPL"],
                store=SimpleNamespace(),
                stop_event=_StopAfter(1),
            )
        )
        assert socket.keys("1") == ["RBAQAAPL"]

    def test_night_session_subscribes_the_d_family(self, socket, monkeypatch):
        monkeypatch.setenv("KIS_FORCE_US_DAYTIME_QUOTES", "false")
        asyncio.run(
            run_kis_overseas_realtime_websocket_collector(
                symbols=["AAPL"],
                store=SimpleNamespace(),
                stop_event=_StopAfter(1),
            )
        )
        assert socket.keys("1") == ["DNASAAPL"]
        # And the registration is handed back rather than orphaned.
        assert socket.keys("2") == ["DNASAAPL"]
