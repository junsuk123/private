"""세션·거래소 인식 KIS 주문 라우터 — 주문 route 결정의 유일한 지점.

리팩터링 이전에는 route 가 세 곳에서 따로 결정됐다:

* ``kis_real.py::_is_us_daytime_order_session`` 이 KST 09:00-16:50 로 주간거래를 판정
  (공식은 10:00-18:00) → 09:00-10:00 에 ``daytime-order`` 를 호출해 거부당하고,
  16:50-18:00 에는 주간거래 대신 일반 ``order`` 로 보내 다시 거부당했다.
* 정정·취소가 **정정 시점의 시각으로** daytime/regular 를 재판정 → 주간거래로 접수한
  주문을 일반 ``order-rvsecncl`` 로 보내는 경로가 존재했다.
* ``EXCG_ID_DVSN_CD`` 가 ``"KRX"`` 하드코딩 → NXT/SOR 주문 자체가 불가능했고,
  ``ORD_DVSN`` 은 거래소별 허용값 차이를 무시했다.

이 모듈은 그 세 결정을 하나로 모으고, 모든 값을 공식 문서에서 검증된 테이블
(:mod:`app.data.market_capabilities`) 에서만 가져온다.
근거 정리: ``docs/kis_market_session_capability_matrix.md``

이 라우터는 기존 안전 계층을 대체하지 않는다. RiskManager / FinalTradeGate /
PrincipalProtectionEngine / LiveExecutionCoordinator / idempotency 는 그대로 앞단에
있고, 여기서는 "그 주문을 어느 엔드포인트로 어떤 필드로 보낼지"만 정한다.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any

from app.data.market_capabilities import (
    MarketGroup,
    MarketSessionService,
    OrderRouteFamily,
    ReasonCode,
    SessionId,
    US_ORDER_EXCHANGE_CODES,
    Venue,
    default_service,
    normalize_market_group,
)
from app.schemas.domain import FinalOrder, OrderSide

#: 장전·장후 시간외 종가매매는 "종가로 체결되는" 주문유형이다. 임의 지정가를 실어 보내면
#: 거부되므로, 라우터는 이 조건을 만나면 호출자가 종가를 넘겼는지 확인해야 한다고 알린다.
CLOSING_PRICE_CONDITIONS = frozenset({"CLOSING_PRICE_ONLY"})
#: 단일가 경매 세션. 지정가는 접수되지만 즉시 체결되지 않는다.
AUCTION_CONDITIONS = frozenset({"AUCTION", "SINGLE_PRICE_AUCTION"})


class OrderRouteBlocked(RuntimeError):
    """공식 route 가 없거나 모호해서 주문을 만들 수 없다 (fail-closed)."""

    def __init__(self, reason_codes: tuple[str, ...]) -> None:
        self.reason_codes = tuple(reason_codes)
        super().__init__(":".join(self.reason_codes) or "ORDER_ROUTE_BLOCKED")


@dataclass(frozen=True)
class ResolvedOrderRoute:
    """확정된 주문 경로. 그대로 HTTP 호출에 쓸 수 있다."""

    market_group: MarketGroup
    venue: Venue
    session: SessionId
    route_family: OrderRouteFamily
    endpoint: str
    tr_id: str
    revise_cancel_endpoint: str
    revise_cancel_tr_id: str
    order_division: str
    #: 국내 전용 (``KRX`` / ``NXT`` / ``SOR``). 해외는 ``None``.
    exchange_id_code: str | None
    #: 해외 전용 (``NASD`` / ``NYSE`` / ``AMEX``). 국내는 ``None``.
    overseas_exchange_code: str | None
    order_condition: str
    reason_codes: tuple[str, ...] = ()

    @property
    def is_domestic(self) -> bool:
        return self.route_family is OrderRouteFamily.DOMESTIC_CASH

    @property
    def requires_closing_price(self) -> bool:
        return self.order_condition in CLOSING_PRICE_CONDITIONS

    def journal_payload(self) -> dict[str, Any]:
        """원주문 route 를 저널에 남기기 위한 표현.

        정정·취소는 이 값을 읽어 **원주문과 같은 family** 로 라우팅한다. 세션이 바뀌었다고
        다른 venue 로 자동 정정하지 않는다.
        """
        return {
            "market_group": self.market_group.value,
            "venue": self.venue.value,
            "session": self.session.value,
            "route_family": self.route_family.value,
            "endpoint": self.endpoint,
            "tr_id": self.tr_id,
            "revise_cancel_endpoint": self.revise_cancel_endpoint,
            "revise_cancel_tr_id": self.revise_cancel_tr_id,
            "order_division": self.order_division,
            "exchange_id_code": self.exchange_id_code,
            "overseas_exchange_code": self.overseas_exchange_code,
            "order_condition": self.order_condition,
        }

    def audit_event(self, order: FinalOrder) -> dict[str, Any]:
        """비밀정보가 제거된 감사 이벤트.

        계좌번호·앱키·토큰은 포함하지 않는다 (호출자가 body 에 채우는 값이며 여기서는
        route 선택 결과만 기록한다).
        """
        return {
            "event": "kis_order_route_selected",
            "ticker": order.ticker,
            "side": order.side.value if hasattr(order.side, "value") else str(order.side),
            "quantity": int(order.quantity),
            "position_effect": order.resolved_position_effect,
            "route": self.journal_payload(),
            "reason_codes": list(self.reason_codes),
        }


@dataclass
class KisSessionOrderRouter:
    """주문 route 를 결정한다. paper/live 판정은 capability service 가 이미 반영한다."""

    service: MarketSessionService = field(default_factory=default_service)

    # ------------------------------------------------------------------ #
    # 신규 주문
    # ------------------------------------------------------------------ #
    def resolve_new_order(
        self,
        order: FinalOrder,
        *,
        now: datetime | None = None,
        symbol_exchange: str | None = None,
    ) -> ResolvedOrderRoute:
        """신규/청산 주문의 route. 모호하거나 미검증이면 :class:`OrderRouteBlocked`.

        ``intent`` 는 ``order.resolved_position_effect`` 에서 유도한다: ``OPEN`` 은 신규
        진입(엄격), ``CLOSE`` 는 청산(세션 정책상 진입이 막혀도 허용).
        """
        current = _as_utc(now)
        intent = "entry" if order.resolved_position_effect == "OPEN" else "exit"
        group = normalize_market_group(order.market) or _group_from_symbol(order.ticker)
        if group is None:
            raise OrderRouteBlocked((ReasonCode.MARKET_SESSION_UNKNOWN.value,))

        venue_hint = _venue_hint(order, group, symbol_exchange)
        session_hint = _session_hint(order)

        route = self.service.resolve_order_route(
            market=group.value,
            side_is_buy=order.side == OrderSide.BUY,
            intent=intent,
            venue_hint=venue_hint,
            session_hint=session_hint,
            now_utc=current,
        )
        if not route.allowed:
            raise OrderRouteBlocked(route.reason_codes)

        overseas_code: str | None = None
        if group is MarketGroup.US:
            overseas_code = _overseas_exchange_code(order, symbol_exchange)
            if overseas_code is None:
                raise OrderRouteBlocked((ReasonCode.EXCHANGE_CODE_UNRESOLVED.value,))

        assert route.endpoint and route.tr_id  # resolve_order_route 가 보장
        return ResolvedOrderRoute(
            market_group=group,
            venue=route.venue,
            session=route.session,
            route_family=route.route_family,
            endpoint=route.endpoint,
            tr_id=route.tr_id,
            revise_cancel_endpoint=route.revise_cancel_endpoint or "",
            revise_cancel_tr_id=route.revise_cancel_tr_id or "",
            order_division=route.order_division or "00",
            exchange_id_code=route.exchange_id_code,
            overseas_exchange_code=overseas_code,
            order_condition=route.order_condition or "",
            reason_codes=route.reason_codes,
        )

    # ------------------------------------------------------------------ #
    # 정정 · 취소
    # ------------------------------------------------------------------ #
    def resolve_revise_cancel(
        self,
        order: FinalOrder,
        *,
        original_route: dict[str, Any] | None,
        now: datetime | None = None,
        symbol_exchange: str | None = None,
    ) -> ResolvedOrderRoute:
        """정정·취소 route. **원주문 route family 를 그대로 유지한다.**

        ``original_route`` 는 신규 주문 시 저널에 남긴 :meth:`ResolvedOrderRoute.journal_payload`.
        없으면 원주문이 어느 엔드포인트로 갔는지 알 수 없으므로 fail-closed 한다 —
        현재 시각으로 재추정하는 것이 정확히 이전 구현의 결함이었다.
        """
        current = _as_utc(now)
        if not original_route:
            raise OrderRouteBlocked((ReasonCode.RECONCILIATION_REQUIRED.value,))
        group = normalize_market_group(
            str(original_route.get("market_group") or order.market)
        )
        if group is None:
            raise OrderRouteBlocked((ReasonCode.MARKET_SESSION_UNKNOWN.value,))

        route = self.service.resolve_revise_cancel_route(
            market=group.value,
            original_route_family=str(original_route.get("route_family") or ""),
            original_venue=original_route.get("venue"),
            original_session=original_route.get("session"),
            now_utc=current,
        )
        if not route.allowed:
            raise OrderRouteBlocked(route.reason_codes)

        overseas_code: str | None = None
        if group is MarketGroup.US:
            overseas_code = str(
                original_route.get("overseas_exchange_code") or ""
            ) or _overseas_exchange_code(order, symbol_exchange)
            if not overseas_code:
                raise OrderRouteBlocked((ReasonCode.EXCHANGE_CODE_UNRESOLVED.value,))

        # 국내 정정은 원주문의 ORD_DVSN 을 유지한다. 시간외 단일가로 접수한 주문을
        # 지정가로 정정하면 거래소가 거부한다.
        division = str(original_route.get("order_division") or route.order_division or "00")
        return ResolvedOrderRoute(
            market_group=group,
            venue=route.venue,
            session=route.session,
            route_family=route.route_family,
            endpoint=str(original_route.get("endpoint") or route.endpoint or ""),
            tr_id=route.revise_cancel_tr_id or "",
            revise_cancel_endpoint=route.revise_cancel_endpoint or "",
            revise_cancel_tr_id=route.revise_cancel_tr_id or "",
            order_division=division,
            exchange_id_code=(
                str(original_route.get("exchange_id_code") or "") or route.exchange_id_code
            )
            if group is MarketGroup.KR
            else None,
            overseas_exchange_code=overseas_code,
            order_condition=str(original_route.get("order_condition") or ""),
            reason_codes=route.reason_codes,
        )

    # ------------------------------------------------------------------ #
    # 사전 검증
    # ------------------------------------------------------------------ #
    def validate_limit_price(
        self, route: ResolvedOrderRoute, order: FinalOrder, *, reference_close: float | None
    ) -> tuple[str, ...]:
        """세션별 가격 조건 검증. 위반 사유코드를 돌려준다 (빈 tuple = 통과).

        장전·장후 시간외 종가매매(``ORD_DVSN`` 05/06)는 전일/당일 종가로만 체결된다.
        임의 지정가를 실어 보내면 거부되므로, 참조 종가를 모르거나 어긋나면 차단한다.
        """
        if not route.requires_closing_price:
            return ()
        if reference_close is None or reference_close <= 0:
            return (ReasonCode.CLOSING_PRICE_ORDER_TYPE.value,)
        if abs(float(order.limit_price) - float(reference_close)) > 1e-9:
            return (ReasonCode.CLOSING_PRICE_ORDER_TYPE.value,)
        return ()


# --------------------------------------------------------------------------- #
# KIS 오류 메시지 → 구조화된 사유코드
# --------------------------------------------------------------------------- #
#: 확인된 KIS ``msg_cd`` 매핑. 알 수 없는 코드는 ``BROKER_ROUTE_REJECTED`` 로 떨어진다.
_MSG_CODE_REASONS: dict[str, str] = {
    "APBK2995": ReasonCode.SESSION_CLOSED.value,        # 장운영시간이 아닙니다
    "APBK0988": ReasonCode.RECONCILIATION_REQUIRED.value,  # 매매가능한 수량이 없습니다
    "APTR0057": ReasonCode.BROKER_ROUTE_REJECTED.value,    # 해외 호가단위 위반
    "OPSP8996": ReasonCode.FEED_DISCONNECTED.value,        # 실시간 세션 중복
}

#: msg1 본문에서 찾는 한국어 단서 (msg_cd 가 비어 오는 경우가 있다).
_MSG_TEXT_REASONS: tuple[tuple[str, str], ...] = (
    ("장운영시간이 아닙니다", ReasonCode.SESSION_CLOSED.value),
    ("매매가능한 수량이 없습니다", ReasonCode.RECONCILIATION_REQUIRED.value),
    ("주문가능금액", ReasonCode.BROKER_ACCOUNT_STALE.value),
    ("거래소", ReasonCode.BROKER_ROUTE_REJECTED.value),
    ("주문구분", ReasonCode.ORDER_DIVISION_UNVERIFIED.value),
)


def order_error_reason_codes(msg_cd: str | None, msg1: str | None) -> tuple[str, ...]:
    """브로커 거부 응답을 구조화된 사유코드로. 운영 화면과 저널이 같은 어휘를 쓰게 한다."""
    codes: list[str] = []
    code = str(msg_cd or "").strip().upper()
    if code in _MSG_CODE_REASONS:
        codes.append(_MSG_CODE_REASONS[code])
    text = str(msg1 or "")
    for needle, reason in _MSG_TEXT_REASONS:
        if needle in text and reason not in codes:
            codes.append(reason)
    if not codes:
        codes.append(ReasonCode.BROKER_ROUTE_REJECTED.value)
    return tuple(dict.fromkeys(codes))


# --------------------------------------------------------------------------- #
# 보조
# --------------------------------------------------------------------------- #
def _as_utc(now: datetime | None) -> datetime:
    current = now or datetime.now(timezone.utc)
    if current.tzinfo is None:
        current = current.replace(tzinfo=timezone.utc)
    return current


def _group_from_symbol(ticker: str) -> MarketGroup | None:
    symbol = str(ticker or "").strip().upper()
    if not symbol:
        return None
    if symbol.isdigit() and len(symbol) in (6, 7):
        return MarketGroup.KR
    if symbol.isalpha():
        return MarketGroup.US
    return None


def _venue_hint(
    order: FinalOrder, group: MarketGroup, symbol_exchange: str | None
) -> Venue | None:
    explicit = str(getattr(order, "execution_venue", "") or "").strip()
    if explicit:
        return explicit  # type: ignore[return-value]  # _coerce_venue 가 처리
    if group is MarketGroup.US and symbol_exchange:
        return symbol_exchange  # type: ignore[return-value]
    return None


def _session_hint(order: FinalOrder) -> SessionId | None:
    raw = str(getattr(order, "market_session", "") or "").strip().upper()
    if not raw:
        return None
    try:
        return SessionId(raw)
    except ValueError:
        return None


def _overseas_exchange_code(order: FinalOrder, symbol_exchange: str | None) -> str | None:
    """``OVRS_EXCG_CD``. 공식 허용값은 ``NASD`` / ``NYSE`` / ``AMEX`` (본 시스템 범위).

    추측하지 않는다 — 해석 불가면 ``None`` 을 돌려 호출자가 fail-closed 하게 한다.
    """
    for candidate in (
        getattr(order, "exchange_code", ""),
        symbol_exchange,
        getattr(order, "execution_venue", ""),
        order.market,
    ):
        code = _normalize_overseas_exchange(candidate)
        if code:
            return code
    return None


_OVERSEAS_ALIASES = {
    "NASD": "NASD", "NAS": "NASD", "NASDAQ": "NASD", "BAQ": "NASD",
    "NYSE": "NYSE", "NYS": "NYSE", "BAY": "NYSE",
    "AMEX": "AMEX", "AMS": "AMEX", "BAA": "AMEX",
}


def _normalize_overseas_exchange(value: Any) -> str | None:
    name = str(value or "").strip().upper()
    if not name:
        return None
    if name in _OVERSEAS_ALIASES:
        return _OVERSEAS_ALIASES[name]
    for alias, code in _OVERSEAS_ALIASES.items():
        if alias in name:
            return code
    return None


def venue_to_overseas_exchange(venue: Venue) -> str | None:
    return US_ORDER_EXCHANGE_CODES.get(venue)


__all__ = [
    "KisSessionOrderRouter",
    "OrderRouteBlocked",
    "ResolvedOrderRoute",
    "order_error_reason_codes",
    "venue_to_overseas_exchange",
]
