"""세션·capability 도메인 모델의 결정론적 테스트.

경계값은 ``docs/kis_market_session_capability_matrix.md`` 의 공식 인용에서 유도한 것이며,
이 테스트가 그 문서와 코드의 계약을 고정한다.
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from zoneinfo import ZoneInfo

import pytest

from app.data.market_capabilities import (
    MarketGroup,
    MarketSessionService,
    OrderRouteFamily,
    ReasonCode,
    SessionId,
    Venue,
    VERIFIED_ORDER_DIVISIONS,
)

SEOUL = ZoneInfo("Asia/Seoul")
NEW_YORK = ZoneInfo("America/New_York")


@pytest.fixture()
def service() -> MarketSessionService:
    return MarketSessionService()


def kst(year: int, month: int, day: int, hour: int, minute: int = 0, second: int = 0) -> datetime:
    return datetime(year, month, day, hour, minute, second, tzinfo=SEOUL).astimezone(timezone.utc)


def et(year: int, month: int, day: int, hour: int, minute: int = 0, second: int = 0) -> datetime:
    return datetime(year, month, day, hour, minute, second, tzinfo=NEW_YORK).astimezone(timezone.utc)


def sessions(service: MarketSessionService, group: str, moment: datetime) -> set[str]:
    return {item.session.value for item in service.active_capabilities(group, moment)}


# 2026-08-05 (수), 2026-11-04 (수, EST 전환 후), 2026-08-08 (토)
WEDNESDAY_SUMMER = (2026, 8, 5)
WEDNESDAY_WINTER = (2026, 11, 4)


# --------------------------------------------------------------------------- #
# 국내 세션 경계 — 1초 전 / 정각 / 1초 후
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize(
    ("hour", "minute", "expected"),
    [
        (7, 59, set()),
        (8, 0, {"NXT_PRE"}),
        (8, 29, {"NXT_PRE"}),
        (8, 30, {"KRX_PREOPEN", "NXT_PRE"}),
        (8, 39, {"KRX_PREOPEN", "NXT_PRE"}),
        (8, 40, {"KRX_OPENING_AUCTION", "NXT_PRE"}),
        (8, 50, {"KRX_OPENING_AUCTION"}),
        (9, 0, {"KRX_REGULAR", "NXT_REGULAR"}),
        (15, 19, {"KRX_REGULAR", "NXT_REGULAR"}),
        (15, 20, {"KRX_CLOSING_AUCTION"}),
        (15, 29, {"KRX_CLOSING_AUCTION"}),
        (15, 30, {"NXT_POST"}),
        (15, 39, {"NXT_POST"}),
        (15, 40, {"KRX_AFTER_CLOSE", "NXT_POST"}),
        (15, 59, {"KRX_AFTER_CLOSE", "NXT_POST"}),
        (16, 0, {"KRX_AFTER_SINGLE_PRICE", "NXT_POST"}),
        (17, 59, {"KRX_AFTER_SINGLE_PRICE", "NXT_POST"}),
        (18, 0, {"NXT_POST"}),
        (19, 59, {"NXT_POST"}),
        (20, 0, set()),
    ],
)
def test_domestic_session_boundaries(service, hour, minute, expected):
    assert sessions(service, "KR", kst(*WEDNESDAY_SUMMER, hour, minute)) == expected


def test_domestic_boundary_one_second_either_side(service):
    """경계 정각은 새 세션에 속하고, 1초 전은 이전 세션에 속한다."""
    just_before = kst(*WEDNESDAY_SUMMER, 8, 59, 59)
    exactly = kst(*WEDNESDAY_SUMMER, 9, 0, 0)
    just_after = kst(*WEDNESDAY_SUMMER, 9, 0, 1)
    assert "KRX_OPENING_AUCTION" in sessions(service, "KR", just_before)
    assert "KRX_REGULAR" not in sessions(service, "KR", just_before)
    assert "KRX_REGULAR" in sessions(service, "KR", exactly)
    assert "KRX_REGULAR" in sessions(service, "KR", just_after)


def test_krx_regular_ends_at_1520_not_1530(service):
    """15:20-15:30 은 종가 단일가다. 정규장 연속체결 세션이 아니다."""
    assert "KRX_REGULAR" in sessions(service, "KR", kst(*WEDNESDAY_SUMMER, 15, 19, 59))
    assert "KRX_REGULAR" not in sessions(service, "KR", kst(*WEDNESDAY_SUMMER, 15, 20))


# --------------------------------------------------------------------------- #
# 미국 세션 경계 + DST
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize(
    ("hour", "minute", "expected_contains", "expected_absent"),
    [
        (4, 0, {"US_PREMARKET"}, set()),
        (9, 29, {"US_PREMARKET"}, {"US_REGULAR"}),
        (9, 30, {"US_REGULAR"}, {"US_PREMARKET"}),
        (15, 59, {"US_REGULAR"}, set()),
        (16, 0, {"US_AFTERMARKET"}, {"US_REGULAR"}),
    ],
)
def test_us_session_boundaries(service, hour, minute, expected_contains, expected_absent):
    active = sessions(service, "US", et(*WEDNESDAY_SUMMER, hour, minute))
    assert expected_contains <= active
    assert not (expected_absent & active)


def test_us_aftermarket_ends_1800_et_in_dst(service):
    """KIS 공문: 애프터마켓 Summer Time 05:00~07:00 KST → ET 16:00-18:00.

    리팩터링 이전 코드는 20:00 ET 까지 열려 있다고 판정했고, 18:00-20:00 ET 주문은
    브로커에서 거부됐다.
    """
    assert "US_AFTERMARKET" in sessions(service, "US", et(*WEDNESDAY_SUMMER, 17, 59))
    assert sessions(service, "US", et(*WEDNESDAY_SUMMER, 18, 0)) == set()
    assert sessions(service, "US", et(*WEDNESDAY_SUMMER, 19, 0)) == set()


def test_us_aftermarket_ends_1700_et_in_standard_time(service):
    """KIS 공문: 애프터마켓 06:00~07:00 KST (겨울) → ET 16:00-17:00."""
    moment = et(*WEDNESDAY_WINTER, 16, 30)
    assert moment.astimezone(NEW_YORK).dst() == timedelta(0), "EST 여야 한다"
    assert "US_AFTERMARKET" in sessions(service, "US", moment)
    assert "US_AFTERMARKET" not in sessions(service, "US", et(*WEDNESDAY_WINTER, 17, 0))


def test_us_daytime_is_a_korea_time_window(service):
    """주간거래는 한국시간 10:00-18:00 고정창 (Summer Time 동일)."""
    for month, day in ((8, 5), (11, 4)):
        assert "US_DAYTIME" in sessions(service, "US", kst(2026, month, day, 10, 0))
        assert "US_DAYTIME" in sessions(service, "US", kst(2026, month, day, 17, 59))
        assert "US_DAYTIME" not in sessions(service, "US", kst(2026, month, day, 9, 59))
        assert "US_DAYTIME" not in sessions(service, "US", kst(2026, month, day, 18, 0))


def test_us_daytime_quote_window_narrower_than_order_window(service):
    """공식 시세창은 10:00-16:00 KST, 주문창은 10:00-18:00 KST.

    16:00-18:00 KST 는 "주문은 되지만 공식 실시간 시세 근거가 없는" 구간이므로
    data_available=False 로 표시되고 신규 진입은 불가하다.
    """
    inside = service.capability("US", SessionId.US_DAYTIME, kst(*WEDNESDAY_SUMMER, 12, 0))
    outside = service.capability("US", SessionId.US_DAYTIME, kst(*WEDNESDAY_SUMMER, 17, 0))
    assert inside.data_available is True
    assert outside.data_available is False
    assert ReasonCode.DAYTIME_QUOTE_WINDOW_ENDED.value in outside.unavailable_reason
    assert outside.new_entry_allowed is False
    # 청산 route 는 여전히 살아 있어야 한다.
    assert outside.exit_allowed is True


def test_us_summer_daytime_premarket_overlap_blocks_new_entry(service):
    """썸머타임 17:00-18:00 KST 는 주간거래·프리마켓이 겹친다 → 신규 진입 fail-closed."""
    moment = kst(*WEDNESDAY_SUMMER, 17, 30)
    active = sessions(service, "US", moment)
    assert {"US_DAYTIME", "US_PREMARKET"} <= active
    route = service.resolve_order_route(market="US", side_is_buy=True, intent="entry", now_utc=moment)
    assert route.allowed is False
    assert ReasonCode.SESSION_ROUTE_AMBIGUOUS.value in route.reason_codes
    # 청산은 결정론적 우선순위로 허용된다.
    exit_route = service.resolve_order_route(
        market="US", side_is_buy=False, intent="exit", now_utc=moment
    )
    assert exit_route.allowed is True
    assert exit_route.route_family is OrderRouteFamily.OVERSEAS_REGULAR


def test_us_winter_has_no_daytime_premarket_overlap(service):
    """겨울에는 주간거래(→18:00)와 프리마켓(18:00→)이 접하기만 하고 겹치지 않는다."""
    active = sessions(service, "US", kst(*WEDNESDAY_WINTER, 17, 30))
    assert active == {"US_DAYTIME"}


# --------------------------------------------------------------------------- #
# 주말 / 휴장일 / 조기폐장 / 캘린더
# --------------------------------------------------------------------------- #
def test_weekend_has_no_sessions(service):
    assert sessions(service, "KR", kst(2026, 8, 8, 11, 0)) == set()   # 토
    assert sessions(service, "KR", kst(2026, 8, 9, 11, 0)) == set()   # 일
    assert sessions(service, "US", et(2026, 8, 8, 11, 0)) == set()


def test_us_holiday_has_no_sessions(service):
    assert service.is_trading_day(MarketGroup.US, et(2026, 12, 25, 11, 0)) is False
    assert sessions(service, "US", et(2026, 12, 25, 11, 0)) == set()


def test_kr_fixed_date_holiday_has_no_sessions(service):
    assert service.is_trading_day(MarketGroup.KR, kst(2026, 8, 17, 11, 0)) is True
    assert service.is_trading_day(MarketGroup.KR, kst(2026, 3, 2, 11, 0)) is True
    # 2026-08-15 는 토요일이므로 광복절 판정 대신 2026-10-09 (금, 한글날) 로 확인.
    assert service.is_trading_day(MarketGroup.KR, kst(2026, 10, 9, 11, 0)) is False
    assert sessions(service, "KR", kst(2026, 10, 9, 11, 0)) == set()


def test_us_early_close_shortens_regular_session(service):
    """조기폐장일(13:00 ET)에는 정규장 종료시각이 당겨진다."""
    capability = service.capability("US", SessionId.US_REGULAR, et(2026, 11, 27, 11, 0))
    assert capability.session_end is not None
    assert capability.session_end.astimezone(NEW_YORK).hour == 13


def test_us_early_close_aftermarket_window_is_marked_unverified(service):
    """조기폐장일 애프터마켓 창은 공문 근거가 없어 미검증으로 표기된다."""
    capability = service.capability("US", SessionId.US_AFTERMARKET, et(2026, 11, 27, 14, 0))
    assert (
        ReasonCode.EARLY_CLOSE_AFTERMARKET_WINDOW_UNVERIFIED.value
        in capability.unavailable_reason
    )


def test_date_outside_calendar_coverage_fails_closed_for_entry(service):
    """일정 정보를 가져오지 못하면 신규 진입 fail-closed, 청산 경로는 별도 평가."""
    beyond = kst(2029, 6, 6, 11, 0)  # coverage_end 2027-12-31 이후
    assert (
        ReasonCode.SESSION_CALENDAR_STALE.value
        in service.blocking_calendar_reasons(MarketGroup.KR, beyond)
    )
    capability = service.capability("KR", SessionId.KRX_REGULAR, beyond)
    assert capability.new_entry_allowed is False
    assert capability.exit_allowed is True


def test_incomplete_calendar_does_not_block_all_entry(service):
    """KR 스냅샷은 음력 휴장일을 포함하지 않지만, 그것이 국내 거래를 영구 차단해서는 안 된다.

    누락 휴장일은 freshness 게이트(피드 증거 없음)가 잡는다.
    """
    reasons = service.calendar_state(MarketGroup.KR, kst(*WEDNESDAY_SUMMER, 11, 0))
    assert ReasonCode.SESSION_CALENDAR_SUSPECT.value in reasons
    assert service.blocking_calendar_reasons(MarketGroup.KR, kst(*WEDNESDAY_SUMMER, 11, 0)) == ()
    assert service.new_entry_allowed("KR", kst(*WEDNESDAY_SUMMER, 11, 0)) is True


# --------------------------------------------------------------------------- #
# 데이터 가용 / 주문 가용 / 진입 / 청산의 분리
# --------------------------------------------------------------------------- #
def test_four_capabilities_are_reported_independently(service):
    moment = kst(*WEDNESDAY_SUMMER, 16, 30)  # KRX 시간외 단일가 + NXT_POST
    assert service.data_available("KR", moment) is True
    assert service.trade_available("KR", moment) is True
    assert service.new_entry_allowed("KR", moment) is False
    assert service.exit_allowed("KR", moment) is True


def test_entry_blocked_session_still_allows_exit_route(service):
    moment = kst(*WEDNESDAY_SUMMER, 16, 30)
    entry = service.resolve_order_route(
        market="KR", side_is_buy=True, intent="entry", now_utc=moment
    )
    exit_route = service.resolve_order_route(
        market="KR", side_is_buy=False, intent="exit", now_utc=moment
    )
    assert entry.allowed is False
    assert exit_route.allowed is True
    assert exit_route.venue is Venue.KRX
    assert exit_route.order_division == "07"  # 시간외 단일가


def test_closed_market_reports_rest_snapshot_only(service):
    capability = service.primary_capability("KR", kst(*WEDNESDAY_SUMMER, 3, 0))
    assert capability.session is SessionId.KR_CLOSED
    assert capability.data_available is False
    assert capability.trade_available is False
    assert ReasonCode.REST_SNAPSHOT_ONLY.value in capability.unavailable_reason


# --------------------------------------------------------------------------- #
# 검증된 주문 route / ORD_DVSN
# --------------------------------------------------------------------------- #
def test_domestic_order_divisions_match_official_tables():
    """공식 문서의 거래소별 ORD_DVSN 표. 시간외 3종은 KRX 전용이다."""
    for code in ("05", "06", "07"):
        assert code in VERIFIED_ORDER_DIVISIONS["KRX"]
        assert code not in VERIFIED_ORDER_DIVISIONS["NXT"]
        assert code not in VERIFIED_ORDER_DIVISIONS["SOR"]
    assert "01" in VERIFIED_ORDER_DIVISIONS["KRX"]      # 시장가
    assert "01" not in VERIFIED_ORDER_DIVISIONS["NXT"]  # NXT 표에 없음
    assert "01" in VERIFIED_ORDER_DIVISIONS["SOR"]
    assert "21" not in VERIFIED_ORDER_DIVISIONS["SOR"]  # 중간가는 SOR 표에 없음


@pytest.mark.parametrize(
    ("session", "expected_division"),
    [
        (SessionId.KRX_PREOPEN, "05"),
        (SessionId.KRX_REGULAR, "00"),
        (SessionId.KRX_AFTER_CLOSE, "06"),
        (SessionId.KRX_AFTER_SINGLE_PRICE, "07"),
        (SessionId.NXT_REGULAR, "00"),
        (SessionId.US_REGULAR, "00"),
        (SessionId.US_DAYTIME, "00"),
    ],
)
def test_session_limit_order_divisions(service, session, expected_division):
    capability = service.capability(
        "KR" if session.value.startswith(("KRX", "NXT")) else "US",
        session,
        kst(*WEDNESDAY_SUMMER, 11, 0),
    )
    assert capability.limit_order_division() == expected_division


def test_us_daytime_route_uses_daytime_endpoint(service):
    route = service.resolve_order_route(
        market="US", side_is_buy=True, intent="exit", now_utc=kst(*WEDNESDAY_SUMMER, 12, 0)
    )
    assert route.allowed is True
    assert route.route_family is OrderRouteFamily.OVERSEAS_DAYTIME
    assert route.endpoint == "/uapi/overseas-stock/v1/trading/daytime-order"
    assert route.tr_id == "TTTS6036U"
    assert route.revise_cancel_endpoint == (
        "/uapi/overseas-stock/v1/trading/daytime-order-rvsecncl"
    )
    assert route.revise_cancel_tr_id == "TTTS6038U"


def test_us_night_route_uses_regular_endpoint(service):
    route = service.resolve_order_route(
        market="US", side_is_buy=False, intent="entry", now_utc=et(*WEDNESDAY_SUMMER, 11, 0)
    )
    assert route.allowed is True
    assert route.route_family is OrderRouteFamily.OVERSEAS_REGULAR
    assert route.endpoint == "/uapi/overseas-stock/v1/trading/order"
    assert route.tr_id == "TTTT1006U"


def test_revise_cancel_keeps_original_route_family(service):
    """세션이 바뀌어도 정정·취소는 원주문 family 를 따른다.

    이전 구현은 정정 시점의 시각으로 daytime/regular 를 재판정해, 주간거래 주문을
    일반 order-rvsecncl 로 보내는 경로가 존재했다.
    """
    after_daytime = kst(*WEDNESDAY_SUMMER, 23, 0)  # 미국 정규장 시간대
    route = service.resolve_revise_cancel_route(
        market="US",
        original_route_family=OrderRouteFamily.OVERSEAS_DAYTIME,
        original_session=SessionId.US_DAYTIME,
        now_utc=after_daytime,
    )
    assert route.allowed is True
    assert route.route_family is OrderRouteFamily.OVERSEAS_DAYTIME
    assert route.revise_cancel_tr_id == "TTTS6038U"
    assert route.revise_cancel_endpoint == (
        "/uapi/overseas-stock/v1/trading/daytime-order-rvsecncl"
    )


def test_revise_cancel_domestic_keeps_original_venue(service):
    route = service.resolve_revise_cancel_route(
        market="KR",
        original_route_family=OrderRouteFamily.DOMESTIC_CASH,
        original_venue=Venue.NXT,
        original_session=SessionId.NXT_REGULAR,
        now_utc=kst(*WEDNESDAY_SUMMER, 11, 0),
    )
    assert route.exchange_id_code == "NXT"
    assert route.revise_cancel_tr_id == "TTTC0013U"


# --------------------------------------------------------------------------- #
# 통합 피드는 DATA_ONLY
# --------------------------------------------------------------------------- #
def test_unified_feed_is_data_only(service):
    capability = service.unified_feed_capability(kst(*WEDNESDAY_SUMMER, 11, 0))
    assert capability.venue is Venue.KRX_NXT_UNIFIED
    assert capability.source_scope.value == "UNIFIED"
    assert capability.is_consolidated is True
    assert capability.data_available is True
    assert capability.trade_available is False
    assert capability.new_entry_allowed is False
    assert ReasonCode.EXCHANGE_CODE_UNRESOLVED.value in capability.unavailable_reason


def test_unified_venue_hint_cannot_be_an_order_route(service):
    route = service.resolve_order_route(
        market="KR",
        side_is_buy=True,
        venue_hint=Venue.KRX_NXT_UNIFIED,
        now_utc=kst(*WEDNESDAY_SUMMER, 11, 0),
    )
    assert route.allowed is False
    assert ReasonCode.EXCHANGE_CODE_UNRESOLVED.value in route.reason_codes


# --------------------------------------------------------------------------- #
# subscription key — 세션 경계에서 D ↔ R 전환
# --------------------------------------------------------------------------- #
def test_us_subscription_key_switches_between_night_and_daytime(service):
    night = service.subscription_key(
        "AAPL", market_group=MarketGroup.US, session=SessionId.US_REGULAR, venue=Venue.NASDAQ
    )
    daytime = service.subscription_key(
        "AAPL", market_group=MarketGroup.US, session=SessionId.US_DAYTIME, venue=Venue.NASDAQ
    )
    assert night == "DNASAAPL"
    assert daytime == "RBAQAAPL"


@pytest.mark.parametrize(
    ("venue", "night", "daytime"),
    [
        (Venue.NASDAQ, "DNASTSLA", "RBAQTSLA"),
        (Venue.NYSE, "DNYSTSLA", "RBAYTSLA"),
        (Venue.AMEX, "DAMSTSLA", "RBAATSLA"),
    ],
)
def test_us_subscription_keys_per_venue(service, venue, night, daytime):
    assert (
        service.subscription_key(
            "tsla", market_group=MarketGroup.US, session=SessionId.US_PREMARKET, venue=venue
        )
        == night
    )
    assert (
        service.subscription_key(
            "tsla", market_group=MarketGroup.US, session=SessionId.US_DAYTIME, venue=venue
        )
        == daytime
    )


def test_domestic_subscription_key_is_the_symbol(service):
    assert (
        service.subscription_key(
            "005930", market_group=MarketGroup.KR, session=SessionId.KRX_REGULAR
        )
        == "005930"
    )


# --------------------------------------------------------------------------- #
# WebSocket TR 선택
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize(
    ("session", "trade_tr", "book_tr"),
    [
        (SessionId.KRX_REGULAR, "H0STCNT0", "H0STASP0"),
        (SessionId.KRX_AFTER_SINGLE_PRICE, "H0STOUP0", "H0STOAA0"),
        (SessionId.KRX_PREOPEN, "H0STOUP0", "H0STOAA0"),
        (SessionId.NXT_REGULAR, "H0NXCNT0", "H0NXASP0"),
        (SessionId.US_REGULAR, "HDFSCNT0", "HDFSASP0"),
        (SessionId.US_DAYTIME, "HDFSCNT0", "HDFSASP0"),
    ],
)
def test_websocket_tr_ids_per_session(service, session, trade_tr, book_tr):
    group = "KR" if session.value.startswith(("KRX", "NXT")) else "US"
    capability = service.capability(group, session, kst(*WEDNESDAY_SUMMER, 11, 0))
    assert capability.trade_ws_tr_id == trade_tr
    assert capability.orderbook_ws_tr_id == book_tr


def test_us_depth_is_ten_levels_and_not_consolidated(service):
    """공식 문서: 미국 무료 시세는 매수/매도 각 10호가. 단 나스닥 마켓센터 단일 시장."""
    capability = service.capability("US", SessionId.US_REGULAR, et(*WEDNESDAY_SUMMER, 11, 0))
    assert capability.depth_level_count == 10
    assert capability.is_consolidated is False
    assert ReasonCode.SINGLE_MARKET_CENTER_DEPTH.value in capability.unavailable_reason


# --------------------------------------------------------------------------- #
# 모의투자 제약
# --------------------------------------------------------------------------- #
def test_paper_mode_blocks_nxt_and_daytime(monkeypatch):
    """공식: "모의투자는 KRX만 가능", daytime-order 는 "모의투자 미지원"."""
    monkeypatch.setenv("KIS_ENV", "paper")
    paper = MarketSessionService()
    nxt = paper.capability("KR", SessionId.NXT_REGULAR, kst(*WEDNESDAY_SUMMER, 11, 0))
    assert nxt.trade_available is False
    assert ReasonCode.PAPER_VENUE_UNSUPPORTED.value in nxt.unavailable_reason

    daytime = paper.capability("US", SessionId.US_DAYTIME, kst(*WEDNESDAY_SUMMER, 12, 0))
    assert daytime.trade_available is False
    assert ReasonCode.PAPER_DAYTIME_UNSUPPORTED.value in daytime.unavailable_reason

    krx = paper.capability("KR", SessionId.KRX_REGULAR, kst(*WEDNESDAY_SUMMER, 11, 0))
    assert krx.trade_available is True
    assert krx.buy_tr_id == "VTTC0012U"
    assert krx.sell_tr_id == "VTTC0011U"


# --------------------------------------------------------------------------- #
# 세션별 진입 환경변수
# --------------------------------------------------------------------------- #
def test_per_session_entry_env_opens_the_session_gate_only(monkeypatch):
    """세션 게이트와 실주문 승인은 별개다.

    세션별 플래그는 세션 게이트(``new_entry_allowed``)를 연다. 그러나 실주문은
    ``live_order_authorized`` 가 추가로 필요하고, 그것은 YAML 정책의 일이다.
    두 값을 하나로 합치면 SHADOW 운영(의도는 만들되 주문은 보내지 않음)을 표현할 수 없다.
    """
    monkeypatch.setenv("TRADING_ALLOW_ENTRY_KRX_AFTER_SINGLE_PRICE", "true")
    service = MarketSessionService()
    moment = kst(*WEDNESDAY_SUMMER, 16, 30)
    capability = service.capability("KR", SessionId.KRX_AFTER_SINGLE_PRICE, moment)
    assert capability.new_entry_allowed is True
    assert capability.live_order_authorized is False
    assert (
        ReasonCode.EXTENDED_LIVE_ORDER_NOT_AUTHORIZED.value in capability.unavailable_reason
    )
    # 실주문 경로는 여전히 fail-closed 다 — 라우터가 두 값을 함께 요구한다.
    route = service.resolve_order_route(
        market="KR", side_is_buy=True, intent="entry", now_utc=moment
    )
    assert route.allowed is False
    assert ReasonCode.EXTENDED_LIVE_ORDER_NOT_AUTHORIZED.value in route.reason_codes


def test_legacy_global_extended_hours_flag_still_recognised(monkeypatch):
    """기존 환경변수는 backward-compatible alias 로 계속 인식된다 (장외 세션에만)."""
    monkeypatch.setenv("TRADING_ALLOW_EXTENDED_HOURS_ENTRY", "true")
    service = MarketSessionService()
    moment = kst(*WEDNESDAY_SUMMER, 16, 30)
    capability = service.capability("KR", SessionId.KRX_AFTER_SINGLE_PRICE, moment)
    assert ReasonCode.EXTENDED_ENTRY_DISABLED.value not in capability.unavailable_reason
    assert capability.new_entry_allowed is True
    # 실주문은 세션별 승인이 없으면 나가지 않는다.
    assert capability.live_order_authorized is False
    assert (
        service.resolve_order_route(
            market="KR", side_is_buy=True, intent="entry", now_utc=moment
        ).allowed
        is False
    )


def test_extended_flag_does_not_open_a_closed_session(monkeypatch):
    """전역 플래그로 완전 마감된 시장을 열 수는 없다."""
    monkeypatch.setenv("TRADING_ALLOW_EXTENDED_HOURS_ENTRY", "true")
    service = MarketSessionService()
    assert service.new_entry_allowed("KR", kst(*WEDNESDAY_SUMMER, 3, 0)) is False
    assert service.active_capabilities("KR", kst(*WEDNESDAY_SUMMER, 3, 0)) == ()


def test_regular_session_entry_not_affected_by_extended_flag(monkeypatch):
    monkeypatch.delenv("TRADING_ALLOW_EXTENDED_HOURS_ENTRY", raising=False)
    service = MarketSessionService()
    assert service.new_entry_allowed("KR", kst(*WEDNESDAY_SUMMER, 11, 0)) is True
    assert service.new_entry_allowed("US", et(*WEDNESDAY_SUMMER, 11, 0)) is True


# --------------------------------------------------------------------------- #
# 보고 payload
# --------------------------------------------------------------------------- #
def test_session_report_separates_data_and_trade(service):
    report = service.session_report(now_utc=kst(*WEDNESDAY_SUMMER, 16, 30))
    kr = report["groups"]["KR"]
    assert kr["data_available"] is True
    assert kr["new_entry_allowed"] is False
    assert kr["exit_allowed"] is True
    assert kr["new_entry_block_reasons"]
    assert kr["unified_feed"]["is_data_only"] is True
    assert report["calendar_version"]
    assert report["verification_source"] == "KIS_OPENAPI_WORKBOOK_20260625"


def test_capability_matrix_lists_unsupported_routes(service):
    matrix = service.capability_matrix(kst(*WEDNESDAY_SUMMER, 11, 0))
    sessions_in_matrix = {row["session"] for row in matrix["capabilities"]}
    assert {"KRX_REGULAR", "NXT_POST", "US_DAYTIME", "US_AFTERMARKET"} <= sessions_in_matrix
    unsupported = {row["session"] for row in matrix["unsupported_routes"]}
    # 통합 피드는 언제나 주문 route 가 없다.
    assert any(row["venue"] == "KRX_NXT_UNIFIED" for row in matrix["unsupported_routes"])
    assert isinstance(unsupported, set)


def test_unknown_market_group_fails_closed(service):
    route = service.resolve_order_route(market="MARS", side_is_buy=True)
    assert route.allowed is False
    assert ReasonCode.MARKET_SESSION_UNKNOWN.value in route.reason_codes
    assert service.active_capabilities("MARS") == ()
