"""통합 (KRX+NXT) realtime feed: parsing, TR selection, extended-hours windows.

Field counts and positional layouts below were verified against the official KIS
workbook (research_notes/한국투자증권_오픈API_전체문서*.xlsx):

    trade      H0STCNT0 / H0UNCNT0 / H0NXCNT0 = 46 fields, same order
               H0STOUP0 (시간외)               = 43 fields, same head layout
    orderbook  H0STASP0                        = 59 fields
               H0UNASP0 / H0NXASP0             = 65 (first 59 identical, then
                                                 KMID_*/NMID_* mid-price fields)
"""

from __future__ import annotations

from datetime import datetime, timezone
from zoneinfo import ZoneInfo

import pytest

from app.data.kis_realtime import (
    ORDERBOOK_FIELDS_BY_TR_ID,
    ORDERBOOK_TR_IDS,
    TRADE_FIELDS_BY_TR_ID,
    TRADE_TR_IDS,
    _domestic_subscription_tr_ids,
    parse_kis_realtime_message,
)
from app.data.market_session import MarketPhase, market_phase, streaming_phase

_SEOUL = ZoneInfo("Asia/Seoul")

RECEIVED = datetime(2026, 7, 30, 1, 0, tzinfo=timezone.utc)


def trade_record(symbol: str, price: str, volume: str, *, fields: int) -> str:
    """Build one positional trade record of the given width."""
    row = [""] * fields
    row[0] = symbol
    row[1] = "100000"
    row[2] = price
    row[12] = volume          # CNTG_VOL
    row[13] = "1000"          # ACML_VOL
    row[14] = "2000"          # ACML_TR_PBMN
    row[21] = "1"             # 체결구분 (CCLD_DVSN / CNTG_CLS_CODE) -> BUY
    return "^".join(row)


def orderbook_record(symbol: str, *, fields: int) -> str:
    row = [""] * fields
    row[0] = symbol
    row[1] = "100000"
    row[2] = "0"
    for i in range(3, 13):     # ASKP1..ASKP10
        row[i] = str(70000 + i)
    for i in range(13, 23):    # BIDP1..BIDP10
        row[i] = str(69000 + i)
    for i in range(23, 43):    # remaining quantities
        row[i] = "10"
    return "^".join(row)


class TestFeedSelection:
    def test_defaults_to_unified_so_nxt_hours_are_covered(self, monkeypatch):
        monkeypatch.delenv("KIS_REALTIME_FEED", raising=False)
        assert _domestic_subscription_tr_ids() == ("H0UNCNT0", "H0UNASP0")

    def test_krx_only_feed_is_still_selectable(self, monkeypatch):
        monkeypatch.setenv("KIS_REALTIME_FEED", "krx")
        assert _domestic_subscription_tr_ids() == ("H0STCNT0", "H0STASP0")

    def test_nxt_only_feed_is_selectable(self, monkeypatch):
        monkeypatch.setenv("KIS_REALTIME_FEED", "nxt")
        assert _domestic_subscription_tr_ids() == ("H0NXCNT0", "H0NXASP0")

    def test_every_declared_tr_has_a_field_count(self):
        # A TR without a schema entry would be split with the wrong width and
        # silently mis-parse every multi-record frame.
        assert set(TRADE_FIELDS_BY_TR_ID) >= TRADE_TR_IDS
        assert set(ORDERBOOK_FIELDS_BY_TR_ID) >= ORDERBOOK_TR_IDS


class TestTradeParsing:
    @pytest.mark.parametrize(
        "tr_id,width",
        [("H0STCNT0", 46), ("H0UNCNT0", 46), ("H0NXCNT0", 46), ("H0STOUP0", 43)],
    )
    def test_single_record_parses_on_every_trade_tr(self, tr_id, width):
        raw = f"0|{tr_id}|001|{trade_record('005930', '71000', '12', fields=width)}"
        parsed = parse_kis_realtime_message(raw, received_at=RECEIVED)
        assert parsed.event_type == "trade"
        assert len(parsed.ticks) == 1
        tick = parsed.ticks[0]
        assert tick.symbol == "005930"
        assert tick.price == 71000
        assert tick.volume == 12
        assert tick.trade_direction == "BUY"

    @pytest.mark.parametrize(
        "tr_id,width",
        [("H0UNCNT0", 46), ("H0NXCNT0", 46), ("H0STOUP0", 43)],
    )
    def test_multi_record_frames_split_at_the_right_width(self, tr_id, width):
        payload = "^".join(
            (
                trade_record("005930", "71000", "10", fields=width),
                trade_record("000660", "180000", "20", fields=width),
            )
        )
        parsed = parse_kis_realtime_message(f"0|{tr_id}|002|{payload}", received_at=RECEIVED)
        assert [t.symbol for t in parsed.ticks] == ["005930", "000660"]
        assert [t.price for t in parsed.ticks] == [71000, 180000]
        assert [t.volume for t in parsed.ticks] == [10, 20]


class TestOrderbookParsing:
    @pytest.mark.parametrize(
        "tr_id,width", [("H0STASP0", 59), ("H0UNASP0", 65), ("H0NXASP0", 65)]
    )
    def test_single_record_parses_on_every_orderbook_tr(self, tr_id, width):
        raw = f"0|{tr_id}|001|{orderbook_record('005930', fields=width)}"
        parsed = parse_kis_realtime_message(raw, received_at=RECEIVED)
        assert parsed.event_type == "orderbook"
        assert len(parsed.orderbooks) == 1
        assert parsed.orderbooks[0].symbol == "005930"

    def test_unified_multi_record_uses_65_not_59(self):
        """With the KRX width the second record would start mid-way and corrupt."""
        payload = "^".join(
            (orderbook_record("005930", fields=65), orderbook_record("000660", fields=65))
        )
        parsed = parse_kis_realtime_message(f"0|H0UNASP0|002|{payload}", received_at=RECEIVED)
        assert [b.symbol for b in parsed.orderbooks] == ["005930", "000660"]


class TestExtendedHoursWindow:
    """NXT quotes 08:00-20:00 KST; KRX order gating must stay narrow."""

    @staticmethod
    def _kst(hour: int, minute: int = 0) -> datetime:
        """A Thursday at the given KST wall-clock time, expressed in UTC."""
        return datetime(2026, 7, 30, hour, minute, tzinfo=_SEOUL).astimezone(timezone.utc)

    @pytest.mark.parametrize(
        "hour,minute,expected",
        [
            (8, 10, MarketPhase.PRE),       # NXT pre-market, KRX still shut
            (9, 30, MarketPhase.REGULAR),
            (16, 0, MarketPhase.AFTER),
            (19, 30, MarketPhase.AFTER),    # NXT after-market
            (20, 30, MarketPhase.CLOSED),
        ],
    )
    def test_streaming_window_follows_nxt(self, hour, minute, expected):
        assert streaming_phase("KRX", self._kst(hour, minute)) is expected

    def test_order_gating_phase_stays_on_krx_hours(self):
        # 19:30 has NXT data but routing an order to KRX would be rejected.
        assert streaming_phase("KRX", self._kst(19, 30)) is MarketPhase.AFTER
        assert market_phase("KRX", self._kst(19, 30)) is MarketPhase.CLOSED

    def test_include_nxt_false_falls_back_to_krx_hours(self):
        assert streaming_phase("KRX", self._kst(8, 10), include_nxt=False) is MarketPhase.CLOSED

    def test_us_group_is_unaffected(self):
        moment = datetime(2026, 7, 30, 17, 0, tzinfo=timezone.utc)  # 13:00 ET
        assert streaming_phase("US", moment) is market_phase("US", moment)

    def test_weekend_is_closed(self):
        saturday = datetime(2026, 8, 1, 2, 0, tzinfo=timezone.utc)  # 11:00 KST Sat
        assert streaming_phase("KRX", saturday) is MarketPhase.CLOSED
