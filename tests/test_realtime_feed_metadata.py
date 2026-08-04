"""파서 단계에서 붙는 출처 metadata 검증 (Phase 4).

metadata 가 없으면 저장소가 KRX 체결과 NXT 체결을 구분할 수 없고, 통합 피드와 venue
피드가 같은 분 bar 행을 다툰다. 그래서 ``parse_kis_realtime_message`` 가 이벤트마다
venue/session/feed_scope/TR/subscription_key 를 부착해야 한다.
"""

from __future__ import annotations

from datetime import datetime, timezone
from zoneinfo import ZoneInfo

import pytest

from app.data.kis_realtime import (
    feed_metadata_for_tr,
    is_us_daytime_quote_session,
    parse_kis_realtime_message,
)
from app.data.market_capabilities import FeedScope, MarketGroup, SessionId, Venue

SEOUL = ZoneInfo("Asia/Seoul")
NEW_YORK = ZoneInfo("America/New_York")

#: 2026-08-05 (수) 11:00 KST — KRX 정규장 + NXT 정규장.
KRX_REGULAR_AT = datetime(2026, 8, 5, 11, 0, tzinfo=SEOUL).astimezone(timezone.utc)
#: 2026-08-05 (수) 16:30 KST — KRX 시간외 단일가 + NXT_POST.
KRX_AFTER_AT = datetime(2026, 8, 5, 16, 30, tzinfo=SEOUL).astimezone(timezone.utc)
#: 2026-08-05 (수) 11:00 ET — 미국 정규장.
US_REGULAR_AT = datetime(2026, 8, 5, 11, 0, tzinfo=NEW_YORK).astimezone(timezone.utc)
#: 2026-08-05 (수) 12:00 KST — 미국 주간거래 (시세창 안).
US_DAYTIME_AT = datetime(2026, 8, 5, 12, 0, tzinfo=SEOUL).astimezone(timezone.utc)


def trade_record(symbol: str, price: str, volume: str, *, fields: int) -> str:
    row = [""] * fields
    row[0] = symbol
    row[1] = "100000"
    row[2] = price
    row[12] = volume
    row[13] = "1000"
    row[14] = "2000"
    row[21] = "1"
    return "^".join(row)


def orderbook_record(symbol: str, *, fields: int) -> str:
    row = [""] * fields
    row[0] = symbol
    row[1] = "100000"
    row[2] = "0"
    for index in range(3, 13):
        row[index] = str(70000 + index)
    for index in range(13, 23):
        row[index] = str(69000 + index)
    for index in range(23, 43):
        row[index] = "10"
    return "^".join(row)


def domestic_message(tr_id: str, record: str) -> str:
    return f"0|{tr_id}|001|{record}"


# --------------------------------------------------------------------------- #
# TR → 피드 identity
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize(
    ("tr_id", "venue", "scope", "exchange", "consolidated"),
    [
        ("H0STCNT0", Venue.KRX, FeedScope.VENUE_SPECIFIC, "KRX", False),
        ("H0STASP0", Venue.KRX, FeedScope.VENUE_SPECIFIC, "KRX", False),
        ("H0STOUP0", Venue.KRX, FeedScope.VENUE_SPECIFIC, "KRX", False),
        ("H0NXCNT0", Venue.NXT, FeedScope.VENUE_SPECIFIC, "NXT", False),
        ("H0NXASP0", Venue.NXT, FeedScope.VENUE_SPECIFIC, "NXT", False),
        ("H0UNCNT0", Venue.KRX_NXT_UNIFIED, FeedScope.UNIFIED, "KRX", True),
        ("H0UNASP0", Venue.KRX_NXT_UNIFIED, FeedScope.UNIFIED, "KRX", True),
    ],
)
def test_domestic_tr_identity(tr_id, venue, scope, exchange, consolidated):
    meta = feed_metadata_for_tr(tr_id, subscription_key="005930", now=KRX_REGULAR_AT)
    assert meta.market_group is MarketGroup.KR
    assert meta.venue is venue
    assert meta.feed_scope is scope
    assert meta.exchange == exchange
    assert meta.is_consolidated is consolidated
    assert meta.currency == "KRW"
    assert meta.tr_id == tr_id
    assert meta.metadata_inferred is False


def test_krx_and_nxt_get_their_own_sessions():
    krx = feed_metadata_for_tr("H0STCNT0", subscription_key="005930", now=KRX_REGULAR_AT)
    nxt = feed_metadata_for_tr("H0NXCNT0", subscription_key="005930", now=KRX_REGULAR_AT)
    assert krx.session is SessionId.KRX_REGULAR
    assert nxt.session is SessionId.NXT_REGULAR
    assert krx.stream_id != nxt.stream_id


def test_overtime_tr_maps_to_the_overtime_session():
    meta = feed_metadata_for_tr("H0STOUP0", subscription_key="005930", now=KRX_AFTER_AT)
    assert meta.session is SessionId.KRX_AFTER_SINGLE_PRICE
    assert meta.venue is Venue.KRX


def test_unified_feed_is_never_tradeable():
    """통합 피드는 시세 전용이다. ``EXCG_ID_DVSN_CD`` 에 통합값이 없다."""
    meta = feed_metadata_for_tr("H0UNCNT0", subscription_key="005930", now=KRX_REGULAR_AT)
    assert meta.is_tradeable is False
    ok, reasons = meta.is_live_buy_eligible()
    assert ok is False
    assert "NON_TRADEABLE_FEED" in reasons


def test_krx_regular_feed_is_live_buy_eligible():
    meta = feed_metadata_for_tr("H0STCNT0", subscription_key="005930", now=KRX_REGULAR_AT)
    ok, reasons = meta.is_live_buy_eligible()
    assert ok is True, reasons


def test_unknown_tr_yields_unknown_metadata():
    meta = feed_metadata_for_tr("H0ZZZZZ9", subscription_key="005930", now=KRX_REGULAR_AT)
    assert meta.venue is Venue.UNKNOWN
    assert meta.session is SessionId.UNKNOWN
    assert meta.feed_scope is FeedScope.UNKNOWN
    assert meta.is_live_buy_eligible()[0] is False


# --------------------------------------------------------------------------- #
# 미국 subscription key → venue / 주간·야간
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize(
    ("key", "venue", "exchange"),
    [
        ("DNASAAPL", Venue.NASDAQ, "NASD"),
        ("DNYSKO", Venue.NYSE, "NYSE"),
        ("DAMSUSO", Venue.AMEX, "AMEX"),
    ],
)
def test_us_night_keys_resolve_venue(key, venue, exchange):
    meta = feed_metadata_for_tr("HDFSCNT0", subscription_key=key, now=US_REGULAR_AT)
    assert meta.market_group is MarketGroup.US
    assert meta.venue is venue
    assert meta.exchange == exchange
    assert meta.currency == "USD"
    assert meta.session is SessionId.US_REGULAR
    assert meta.feed_scope is FeedScope.FREE_REALTIME
    # 미국 무료 호가는 나스닥 마켓센터 단일 시장이므로 consolidated 가 아니다.
    assert meta.is_consolidated is False


def test_us_daytime_key_maps_to_daytime_session():
    meta = feed_metadata_for_tr("HDFSCNT0", subscription_key="RBAQAAPL", now=US_DAYTIME_AT)
    assert meta.session is SessionId.US_DAYTIME
    assert meta.venue is Venue.US_DAYTIME_VENUE


def test_us_daytime_and_night_streams_are_distinct():
    day = feed_metadata_for_tr("HDFSCNT0", subscription_key="RBAQAAPL", now=US_DAYTIME_AT)
    night = feed_metadata_for_tr("HDFSCNT0", subscription_key="DNASAAPL", now=US_REGULAR_AT)
    assert day.stream_id != night.stream_id
    assert day.subscription_key == "RBAQAAPL"
    assert night.subscription_key == "DNASAAPL"


# --------------------------------------------------------------------------- #
# 파서가 실제로 metadata 를 붙이는지
# --------------------------------------------------------------------------- #
def test_parser_attaches_metadata_to_domestic_trade():
    message = domestic_message("H0UNCNT0", trade_record("005930", "70000", "12", fields=46))
    parsed = parse_kis_realtime_message(message, received_at=KRX_REGULAR_AT)
    assert parsed.event_type == "trade"
    assert len(parsed.ticks) == 1
    tick = parsed.ticks[0]
    assert tick.meta.tr_id == "H0UNCNT0"
    assert tick.meta.venue is Venue.KRX_NXT_UNIFIED
    assert tick.meta.feed_scope is FeedScope.UNIFIED
    assert tick.meta.stream_id == "KR:KRX_NXT_UNIFIED:UNIFIED:H0UNCNT0"


def test_parser_attaches_metadata_to_domestic_orderbook():
    message = domestic_message("H0NXASP0", orderbook_record("005930", fields=65))
    parsed = parse_kis_realtime_message(message, received_at=KRX_REGULAR_AT)
    assert parsed.event_type == "orderbook"
    book = parsed.orderbooks[0]
    assert book.meta.venue is Venue.NXT
    assert book.meta.tr_id == "H0NXASP0"
    assert book.meta.session is SessionId.NXT_REGULAR


def test_same_trade_on_krx_and_nxt_gets_distinct_record_ids():
    record = trade_record("005930", "70000", "12", fields=46)
    krx = parse_kis_realtime_message(
        domestic_message("H0STCNT0", record), received_at=KRX_REGULAR_AT
    ).ticks[0]
    nxt = parse_kis_realtime_message(
        domestic_message("H0NXCNT0", record), received_at=KRX_REGULAR_AT
    ).ticks[0]
    assert krx.record_id != nxt.record_id
    # 물리적으로 같은 체결이라면 dedup_key 는 같다 (교차 스트림 중복 관측용).
    assert krx.dedup_key == nxt.dedup_key


# --------------------------------------------------------------------------- #
# 미국 주간 시세창 (10:00-16:00 KST) vs 주문창 (10:00-18:00 KST)
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize(
    ("hour", "minute", "expected"),
    [
        (9, 59, False),
        (10, 0, True),
        (15, 59, True),
        (16, 0, False),   # 시세창 종료 — 주문창은 아직 열려 있다
        (17, 59, False),
        (18, 0, False),
    ],
)
def test_us_daytime_quote_window_is_ten_to_sixteen_kst(hour, minute, expected, monkeypatch):
    monkeypatch.delenv("KIS_FORCE_US_DAYTIME_QUOTES", raising=False)
    moment = datetime(2026, 8, 5, hour, minute, tzinfo=SEOUL).astimezone(timezone.utc)
    assert is_us_daytime_quote_session(moment) is expected


def test_us_daytime_quote_window_closed_on_weekend(monkeypatch):
    monkeypatch.delenv("KIS_FORCE_US_DAYTIME_QUOTES", raising=False)
    saturday = datetime(2026, 8, 8, 12, 0, tzinfo=SEOUL).astimezone(timezone.utc)
    assert is_us_daytime_quote_session(saturday) is False


def test_us_daytime_quote_override_still_works(monkeypatch):
    monkeypatch.setenv("KIS_FORCE_US_DAYTIME_QUOTES", "true")
    midnight = datetime(2026, 8, 5, 3, 0, tzinfo=SEOUL).astimezone(timezone.utc)
    assert is_us_daytime_quote_session(midnight) is True
    monkeypatch.setenv("KIS_FORCE_US_DAYTIME_QUOTES", "false")
    noon = datetime(2026, 8, 5, 12, 0, tzinfo=SEOUL).astimezone(timezone.utc)
    assert is_us_daytime_quote_session(noon) is False
