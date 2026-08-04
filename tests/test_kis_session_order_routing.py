"""세션 인식 주문 라우팅 — fake transport 만 사용하며 실주문은 절대 보내지 않는다.

여기서 고정하는 계약 (``docs/realtime_session_gap_analysis.md`` §4 의 D1·D3·D4·D5·D6):

* D1 미국 주간거래 주문창은 10:00-18:00 KST 다 (09:00-16:50 이 아니다).
* D3 정정·취소는 **원주문 endpoint family** 를 유지한다.
* D4 ``EXCG_ID_DVSN_CD`` 는 KRX/NXT/SOR 를 실을 수 있다.
* D5 ``ORD_DVSN`` 은 거래소별 공식 허용 집합을 벗어나지 않는다.
* D6 시간외 종가매매는 종가가 아닌 임의 지정가를 거부한다.
"""

from __future__ import annotations

from datetime import datetime, timezone
from zoneinfo import ZoneInfo

import pytest

from app.data.market_capabilities import (
    MarketGroup,
    OrderRouteFamily,
    ReasonCode,
    SessionId,
    Venue,
)
from app.execution.kis_real import (
    _domestic_order_division_code,
    _is_us_daytime_order_session,
)
from app.execution.kis_session_order_router import (
    KisSessionOrderRouter,
    OrderRouteBlocked,
    order_error_reason_codes,
)
from app.schemas.domain import FinalOrder, OrderSide, OrderType

SEOUL = ZoneInfo("Asia/Seoul")
NEW_YORK = ZoneInfo("America/New_York")


def kst(hour: int, minute: int = 0, *, day: int = 5, month: int = 8) -> datetime:
    return datetime(2026, month, day, hour, minute, tzinfo=SEOUL).astimezone(timezone.utc)


def et(hour: int, minute: int = 0, *, day: int = 5, month: int = 8) -> datetime:
    return datetime(2026, month, day, hour, minute, tzinfo=NEW_YORK).astimezone(timezone.utc)


def order(
    *,
    ticker: str = "005930",
    market: str = "KRX",
    side: OrderSide = OrderSide.BUY,
    price: float = 70000.0,
    effect: str = "OPEN",
    venue: str = "",
    exchange_code: str = "",
    session: str = "",
) -> FinalOrder:
    return FinalOrder(
        ticker=ticker,
        market=market,
        order_type=OrderType.LIMIT,
        side=side,
        quantity=1,
        limit_price=price,
        position_effect=effect,
        execution_venue=venue,
        exchange_code=exchange_code,
        market_session=session,
    )


@pytest.fixture()
def router() -> KisSessionOrderRouter:
    return KisSessionOrderRouter()


# --------------------------------------------------------------------------- #
# D1 — 미국 주간거래 주문창
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize(
    ("hour", "minute", "expected"),
    [
        (9, 0, False),    # 이전 구현은 여기서 True 였다 → daytime-order 거부
        (9, 59, False),
        (10, 0, True),
        (16, 50, True),   # 이전 구현은 여기서 False 였다 → 잘못된 endpoint
        (17, 59, True),
        (18, 0, False),
    ],
)
def test_us_daytime_order_window_is_ten_to_eighteen_kst(hour, minute, expected, monkeypatch):
    monkeypatch.delenv("KIS_FORCE_OVERSEAS_DAYTIME_ORDER", raising=False)
    assert _is_us_daytime_order_session("US", kst(hour, minute)) is expected


def test_us_daytime_order_window_closed_on_weekend(monkeypatch):
    monkeypatch.delenv("KIS_FORCE_OVERSEAS_DAYTIME_ORDER", raising=False)
    assert _is_us_daytime_order_session("US", kst(12, 0, day=8)) is False  # 토요일


def test_domestic_market_is_never_a_daytime_session():
    assert _is_us_daytime_order_session("KRX", kst(12, 0)) is False


# --------------------------------------------------------------------------- #
# D5 — 거래소별 ORD_DVSN
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize(
    ("hour", "minute", "expected"),
    [
        (8, 35, "05"),   # 장전 시간외
        (11, 0, "00"),   # 정규장 지정가
        (15, 45, "06"),  # 장후 시간외
        (16, 30, "07"),  # 시간외 단일가
    ],
)
def test_krx_order_division_follows_session(hour, minute, expected, monkeypatch):
    monkeypatch.delenv("KIS_DOMESTIC_ORD_DVSN", raising=False)
    assert (
        _domestic_order_division_code(kst(hour, minute), exchange_id_code="KRX")
        == expected
    )


@pytest.mark.parametrize("hour,minute", [(8, 35), (11, 0), (15, 45), (16, 30), (19, 0)])
def test_nxt_never_uses_overtime_order_divisions(hour, minute, monkeypatch):
    """NXT ORD_DVSN 표에 05/06/07 이 없다. 보내면 조용히 거부된다."""
    monkeypatch.delenv("KIS_DOMESTIC_ORD_DVSN", raising=False)
    division = _domestic_order_division_code(kst(hour, minute), exchange_id_code="NXT")
    assert division not in {"05", "06", "07"}
    assert division == "00"


def test_forced_order_division_outside_official_set_is_rejected(monkeypatch):
    """운영자가 지정한 값도 공식 허용 집합을 벗어나면 쓰지 않는다."""
    monkeypatch.setenv("KIS_DOMESTIC_ORD_DVSN", "07")
    assert _domestic_order_division_code(kst(16, 30), exchange_id_code="KRX") == "07"
    # 07 은 NXT 표에 없다 → 안전한 지정가로 되돌린다.
    assert _domestic_order_division_code(kst(16, 30), exchange_id_code="NXT") == "00"

    monkeypatch.setenv("KIS_DOMESTIC_ORD_DVSN", "99")
    assert _domestic_order_division_code(kst(11, 0), exchange_id_code="KRX") == "00"


# --------------------------------------------------------------------------- #
# 라우터: 신규 진입 vs 청산
# --------------------------------------------------------------------------- #
def test_domestic_regular_entry_routes_to_order_cash(router):
    resolved = router.resolve_new_order(order(), now=kst(11, 0))
    assert resolved.route_family is OrderRouteFamily.DOMESTIC_CASH
    assert resolved.endpoint == "/uapi/domestic-stock/v1/trading/order-cash"
    assert resolved.tr_id == "TTTC0012U"
    assert resolved.revise_cancel_endpoint == (
        "/uapi/domestic-stock/v1/trading/order-rvsecncl"
    )
    assert resolved.revise_cancel_tr_id == "TTTC0013U"
    assert resolved.exchange_id_code == "KRX"
    assert resolved.order_division == "00"
    assert resolved.venue is Venue.KRX


def test_domestic_sell_uses_the_sell_tr_id(router):
    resolved = router.resolve_new_order(
        order(side=OrderSide.SELL, effect="CLOSE"), now=kst(11, 0)
    )
    assert resolved.tr_id == "TTTC0011U"


def test_entry_blocked_outside_regular_but_exit_still_routes(router):
    """신규 진입이 금지된 세션에서도 공식 route 가 있으면 청산은 가능해야 한다."""
    with pytest.raises(OrderRouteBlocked) as blocked:
        router.resolve_new_order(order(effect="OPEN"), now=kst(16, 30))
    assert blocked.value.reason_codes

    exit_route = router.resolve_new_order(
        order(side=OrderSide.SELL, effect="CLOSE"), now=kst(16, 30)
    )
    assert exit_route.session is SessionId.KRX_AFTER_SINGLE_PRICE
    assert exit_route.order_division == "07"
    assert exit_route.tr_id == "TTTC0011U"


def test_closed_market_blocks_both_directions(router):
    for effect, side in (("OPEN", OrderSide.BUY), ("CLOSE", OrderSide.SELL)):
        with pytest.raises(OrderRouteBlocked) as blocked:
            router.resolve_new_order(order(side=side, effect=effect), now=kst(3, 0))
        assert ReasonCode.SESSION_CLOSED.value in blocked.value.reason_codes


# --------------------------------------------------------------------------- #
# D4 — NXT / SOR 라우팅
# --------------------------------------------------------------------------- #
def test_nxt_venue_hint_produces_nxt_exchange_code(router):
    resolved = router.resolve_new_order(
        order(side=OrderSide.SELL, effect="CLOSE", venue="NXT"), now=kst(19, 0)
    )
    assert resolved.venue is Venue.NXT
    assert resolved.exchange_id_code == "NXT"
    assert resolved.session is SessionId.NXT_POST
    assert resolved.order_division == "00"


def test_unified_venue_cannot_be_an_order_route(router):
    """통합 피드는 시세 전용 — EXCG_ID_DVSN_CD 에 통합값이 없다."""
    with pytest.raises(OrderRouteBlocked) as blocked:
        router.resolve_new_order(
            order(side=OrderSide.SELL, effect="CLOSE", venue="KRX_NXT_UNIFIED"),
            now=kst(11, 0),
        )
    assert ReasonCode.EXCHANGE_CODE_UNRESOLVED.value in blocked.value.reason_codes


# --------------------------------------------------------------------------- #
# 미국 endpoint 선택
# --------------------------------------------------------------------------- #
def test_us_daytime_selects_daytime_endpoint(router):
    resolved = router.resolve_new_order(
        order(ticker="AAPL", market="NASDAQ", side=OrderSide.SELL, effect="CLOSE", price=190.0),
        now=kst(12, 0),
    )
    assert resolved.route_family is OrderRouteFamily.OVERSEAS_DAYTIME
    assert resolved.endpoint == "/uapi/overseas-stock/v1/trading/daytime-order"
    assert resolved.tr_id == "TTTS6037U"
    assert resolved.overseas_exchange_code == "NASD"
    assert resolved.order_division == "00"


def test_us_regular_selects_general_overseas_endpoint(router):
    resolved = router.resolve_new_order(
        order(ticker="AAPL", market="NASDAQ", price=190.0), now=et(11, 0)
    )
    assert resolved.route_family is OrderRouteFamily.OVERSEAS_REGULAR
    assert resolved.endpoint == "/uapi/overseas-stock/v1/trading/order"
    assert resolved.tr_id == "TTTT1002U"
    assert resolved.session is SessionId.US_REGULAR


def test_us_premarket_and_aftermarket_use_the_general_endpoint(router):
    """공식: 프리마켓·애프터마켓 시간대에도 일반 주문 endpoint 로 주문 가능."""
    for moment, session in ((et(5, 0), SessionId.US_PREMARKET), (et(16, 30), SessionId.US_AFTERMARKET)):
        resolved = router.resolve_new_order(
            order(ticker="AAPL", market="NASDAQ", side=OrderSide.SELL, effect="CLOSE", price=190.0),
            now=moment,
        )
        assert resolved.session is session
        assert resolved.endpoint == "/uapi/overseas-stock/v1/trading/order"
        assert resolved.tr_id == "TTTT1006U"


@pytest.mark.parametrize(
    ("market", "expected"),
    [("NASDAQ", "NASD"), ("NYSE", "NYSE"), ("AMEX", "AMEX")],
)
def test_overseas_exchange_code_resolution(router, market, expected):
    resolved = router.resolve_new_order(
        order(ticker="XYZ", market=market, price=10.0), now=et(11, 0)
    )
    assert resolved.overseas_exchange_code == expected


def test_unresolvable_overseas_exchange_fails_closed(router):
    with pytest.raises(OrderRouteBlocked) as blocked:
        router.resolve_new_order(
            order(ticker="AAPL", market="US-SOMEWHERE-ELSE", price=190.0), now=et(11, 0)
        )
    assert ReasonCode.EXCHANGE_CODE_UNRESOLVED.value in blocked.value.reason_codes


# --------------------------------------------------------------------------- #
# D3 — 정정·취소는 원주문 family 유지
# --------------------------------------------------------------------------- #
def test_revise_keeps_daytime_family_after_session_change(router):
    placed = router.resolve_new_order(
        order(ticker="AAPL", market="NASDAQ", side=OrderSide.SELL, effect="CLOSE", price=190.0),
        now=kst(12, 0),
    )
    journal = placed.journal_payload()
    assert journal["route_family"] == "OVERSEAS_DAYTIME"

    # 세션이 미국 정규장으로 넘어간 뒤 정정.
    revised = router.resolve_revise_cancel(
        order(ticker="AAPL", market="NASDAQ", side=OrderSide.SELL, effect="CLOSE", price=191.0),
        original_route=journal,
        now=et(11, 0),
    )
    assert revised.route_family is OrderRouteFamily.OVERSEAS_DAYTIME
    assert revised.revise_cancel_endpoint == (
        "/uapi/overseas-stock/v1/trading/daytime-order-rvsecncl"
    )
    assert revised.revise_cancel_tr_id == "TTTS6038U"


def test_revise_keeps_regular_family_during_daytime_window(router):
    placed = router.resolve_new_order(
        order(ticker="AAPL", market="NASDAQ", price=190.0), now=et(11, 0)
    )
    revised = router.resolve_revise_cancel(
        order(ticker="AAPL", market="NASDAQ", price=191.0),
        original_route=placed.journal_payload(),
        now=kst(12, 0),
    )
    assert revised.route_family is OrderRouteFamily.OVERSEAS_REGULAR
    assert revised.revise_cancel_tr_id == "TTTT1004U"


def test_domestic_revise_preserves_original_order_division(router):
    placed = router.resolve_new_order(
        order(side=OrderSide.SELL, effect="CLOSE"), now=kst(16, 30)
    )
    assert placed.order_division == "07"
    revised = router.resolve_revise_cancel(
        order(side=OrderSide.SELL, effect="CLOSE"),
        original_route=placed.journal_payload(),
        now=kst(11, 0, day=6),  # 다음 날 정규장
    )
    assert revised.order_division == "07", "원주문 주문구분을 유지해야 한다"
    assert revised.exchange_id_code == "KRX"


def test_revise_without_recorded_route_fails_closed(router):
    """원주문이 어느 엔드포인트로 갔는지 모르면 추정하지 않고 차단한다."""
    with pytest.raises(OrderRouteBlocked) as blocked:
        router.resolve_revise_cancel(
            order(ticker="AAPL", market="NASDAQ", price=190.0),
            original_route=None,
            now=et(11, 0),
        )
    assert ReasonCode.RECONCILIATION_REQUIRED.value in blocked.value.reason_codes


# --------------------------------------------------------------------------- #
# D6 — 시간외 종가매매의 가격 조건
# --------------------------------------------------------------------------- #
def test_closing_price_session_rejects_arbitrary_limit_price(router):
    resolved = router.resolve_new_order(
        order(side=OrderSide.SELL, effect="CLOSE", price=71000.0), now=kst(15, 45)
    )
    assert resolved.order_division == "06"
    assert resolved.requires_closing_price is True

    # 참조 종가를 모르면 차단.
    assert router.validate_limit_price(
        resolved, order(price=71000.0), reference_close=None
    ) == (ReasonCode.CLOSING_PRICE_ORDER_TYPE.value,)
    # 종가와 다른 지정가도 차단.
    assert router.validate_limit_price(
        resolved, order(price=71000.0), reference_close=70500.0
    ) == (ReasonCode.CLOSING_PRICE_ORDER_TYPE.value,)
    # 종가와 같으면 통과.
    assert router.validate_limit_price(
        resolved, order(price=70500.0), reference_close=70500.0
    ) == ()


def test_regular_session_has_no_closing_price_constraint(router):
    resolved = router.resolve_new_order(order(), now=kst(11, 0))
    assert resolved.requires_closing_price is False
    assert router.validate_limit_price(resolved, order(), reference_close=None) == ()


# --------------------------------------------------------------------------- #
# 모의투자 제약
# --------------------------------------------------------------------------- #
def test_paper_mode_blocks_daytime_orders(monkeypatch):
    """공식: daytime-order 는 "모의투자 미지원"."""
    monkeypatch.setenv("KIS_ENV", "paper")
    paper_router = KisSessionOrderRouter()
    with pytest.raises(OrderRouteBlocked) as blocked:
        paper_router.resolve_new_order(
            order(ticker="AAPL", market="NASDAQ", side=OrderSide.SELL, effect="CLOSE", price=190.0),
            now=kst(12, 0),
        )
    assert ReasonCode.PAPER_DAYTIME_UNSUPPORTED.value in blocked.value.reason_codes


def test_paper_mode_uses_paper_tr_ids(monkeypatch):
    monkeypatch.setenv("KIS_ENV", "paper")
    paper_router = KisSessionOrderRouter()
    resolved = paper_router.resolve_new_order(order(), now=kst(11, 0))
    assert resolved.tr_id == "VTTC0012U"
    assert resolved.revise_cancel_tr_id == "VTTC0013U"


# --------------------------------------------------------------------------- #
# 감사 이벤트 / 오류 사유코드
# --------------------------------------------------------------------------- #
def test_audit_event_has_no_secrets(router):
    resolved = router.resolve_new_order(order(), now=kst(11, 0))
    event = resolved.audit_event(order())
    serialized = repr(event)
    for secret_field in ("CANO", "ACNT_PRDT_CD", "appkey", "appsecret", "authorization"):
        assert secret_field not in serialized
    assert event["route"]["tr_id"] == "TTTC0012U"
    assert event["event"] == "kis_order_route_selected"


@pytest.mark.parametrize(
    ("msg_cd", "msg1", "expected"),
    [
        ("APBK2995", "장운영시간이 아닙니다", ReasonCode.SESSION_CLOSED.value),
        ("APBK0988", "매매가능한 수량이 없습니다", ReasonCode.RECONCILIATION_REQUIRED.value),
        ("", "주문가능금액을 초과했습니다", ReasonCode.BROKER_ACCOUNT_STALE.value),
        ("ZZZZ9999", "알 수 없는 오류", ReasonCode.BROKER_ROUTE_REJECTED.value),
    ],
)
def test_order_error_reason_codes(msg_cd, msg1, expected):
    assert expected in order_error_reason_codes(msg_cd, msg1)


def test_order_error_always_returns_at_least_one_code():
    assert order_error_reason_codes(None, None) == (
        ReasonCode.BROKER_ROUTE_REJECTED.value,
    )
