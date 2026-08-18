"""국내·미국 전 세션에 대한 단일 capability source of truth.

이 모듈이 존재하는 이유
-----------------------
리팩터링 이전에는 "지금 장이 열려 있는가"를 판정하는 코드가 최소 7곳에 독립적으로
존재했고 경계값이 서로 달랐다 (``docs/realtime_session_gap_analysis.md`` §3):

* ``market_session.py``            KRX 09:00-15:30 / US after → 20:00 ET
* ``web.py``                       KRX extended 09:00-16:50, 휴장일 집합 중복 정의
* ``realtime_trading_engine.py``   KRX 09:00-15:20  ← 세 번째 경계
* ``kis_real.py``                  미국 주간거래 09:00-16:50 KST ← 공식은 10:00-18:00

그 결과 (a) KIS 가 거부할 시각에 주문을 보내고, (b) 거부되지 않는 시각에는 주문을 보내지
않고, (c) 운영 화면은 세 값 중 아무거나 보여 주었다.

설계 원칙
---------
**시장 국가 / 거래소 venue / 거래 세션 / 데이터 가용성 / 주문 가능성을 서로 독립된 값으로
표현한다.** "닫혀 있지 않다"와 "매수해도 된다"는 전혀 다른 명제이고, "데이터가 온다"와
"주문이 접수된다"도 다르다. 특히 **신규 진입 가능성과 청산 가능성은 항상 분리**된다 —
신규 진입이 금지된 세션에서도 공식 주문 route 가 있으면 위험 축소는 가능해야 한다.

정보의 출처를 두 층으로 나눈다
------------------------------
1. **API 문서에서 검증된 것** (이 파일의 ``_VERIFIED_*`` 테이블): TR ID, 엔드포인트,
   ``EXCG_ID_DVSN_CD``, ``ORD_DVSN`` 허용값, subscription key 형식.
   근거: ``research_notes/한국투자증권_오픈API_전체문서_20260625_030000.xlsx``
   정리: ``docs/kis_market_session_capability_matrix.md``
   → 설정 파일로 덮어쓸 수 없다. 추측으로 채운 값은 없다.
2. **거래소 업무규정 / 우리 정책** (``config/market_sessions.yaml``): 세션 시각창,
   휴장일·조기폐장 스냅샷, 세션별 정책 임계값.
   → 버전이 기록되며, 커버리지를 벗어나면 신규 진입은 fail-closed 된다.
"""

from __future__ import annotations

import os
import threading
from collections.abc import Callable, Iterable, Mapping
from dataclasses import dataclass, field, replace
from datetime import date, datetime, time, timezone
from enum import Enum
from pathlib import Path
from zoneinfo import ZoneInfo

_SEOUL = ZoneInfo("Asia/Seoul")
_NEW_YORK = ZoneInfo("America/New_York")

DEFAULT_CONFIG_PATH = Path("config/market_sessions.yaml")

#: ``officially_verified_at`` / ``verification_source`` 에 기록되는 검증 스냅샷 식별자.
VERIFICATION_SOURCE = "KIS_OPENAPI_WORKBOOK_20260625"
VERIFICATION_DATE = "2026-06-25"


# --------------------------------------------------------------------------- #
# 도메인 값
# --------------------------------------------------------------------------- #
class MarketGroup(str, Enum):
    KR = "KR"
    US = "US"


class Venue(str, Enum):
    KRX = "KRX"
    NXT = "NXT"
    KRX_NXT_UNIFIED = "KRX_NXT_UNIFIED"
    NASDAQ = "NASDAQ"
    NYSE = "NYSE"
    AMEX = "AMEX"
    US_DAYTIME_VENUE = "US_DAYTIME_VENUE"
    UNKNOWN = "UNKNOWN"


class SessionId(str, Enum):
    # 국내
    KRX_PREOPEN = "KRX_PREOPEN"
    KRX_OPENING_AUCTION = "KRX_OPENING_AUCTION"
    KRX_REGULAR = "KRX_REGULAR"
    KRX_CLOSING_AUCTION = "KRX_CLOSING_AUCTION"
    KRX_AFTER_CLOSE = "KRX_AFTER_CLOSE"
    KRX_AFTER_SINGLE_PRICE = "KRX_AFTER_SINGLE_PRICE"
    NXT_PRE = "NXT_PRE"
    NXT_REGULAR = "NXT_REGULAR"
    NXT_POST = "NXT_POST"
    KR_CLOSED = "KR_CLOSED"
    # 미국
    US_DAYTIME = "US_DAYTIME"
    US_PREMARKET = "US_PREMARKET"
    US_REGULAR = "US_REGULAR"
    US_AFTERMARKET = "US_AFTERMARKET"
    US_CLOSED = "US_CLOSED"
    # 알 수 없음 (metadata 기본값). live-buy 적격을 절대 통과하지 못한다.
    UNKNOWN = "UNKNOWN"


class FeedScope(str, Enum):
    VENUE_SPECIFIC = "VENUE_SPECIFIC"
    UNIFIED = "UNIFIED"
    FREE_REALTIME = "FREE_REALTIME"
    REST_SNAPSHOT = "REST_SNAPSHOT"
    HISTORICAL = "HISTORICAL"
    UNKNOWN = "UNKNOWN"


class OrderRouteFamily(str, Enum):
    """엔드포인트 family. 정정·취소는 원주문과 **동일 family** 로만 라우팅된다."""

    DOMESTIC_CASH = "DOMESTIC_CASH"
    OVERSEAS_REGULAR = "OVERSEAS_REGULAR"
    OVERSEAS_DAYTIME = "OVERSEAS_DAYTIME"
    NONE = "NONE"


CLOSED_SESSIONS = {MarketGroup.KR: SessionId.KR_CLOSED, MarketGroup.US: SessionId.US_CLOSED}

_KR_GROUP_NAMES = {"KR", "KRX", "KOSPI", "KOSDAQ", "KONEX", "NXT", "DOMESTIC"}
_US_GROUP_NAMES = {
    "US", "USA", "NASDAQ", "NASD", "NAS", "NYSE", "NYS", "AMEX", "AMS",
    "OVERSEAS", "US-LISTED", "GLOBAL",
}


def normalize_market_group(group: str) -> MarketGroup | None:
    name = str(group or "").upper().strip()
    if name in _KR_GROUP_NAMES:
        return MarketGroup.KR
    if name in _US_GROUP_NAMES:
        return MarketGroup.US
    return None


# --------------------------------------------------------------------------- #
# 검증된 KIS route/피드 테이블 — 설정으로 변경 불가
# --------------------------------------------------------------------------- #
DOMESTIC_ORDER_ENDPOINT = "/uapi/domestic-stock/v1/trading/order-cash"
DOMESTIC_REVISE_CANCEL_ENDPOINT = "/uapi/domestic-stock/v1/trading/order-rvsecncl"
OVERSEAS_ORDER_ENDPOINT = "/uapi/overseas-stock/v1/trading/order"
OVERSEAS_REVISE_CANCEL_ENDPOINT = "/uapi/overseas-stock/v1/trading/order-rvsecncl"
OVERSEAS_DAYTIME_ORDER_ENDPOINT = "/uapi/overseas-stock/v1/trading/daytime-order"
OVERSEAS_DAYTIME_REVISE_CANCEL_ENDPOINT = (
    "/uapi/overseas-stock/v1/trading/daytime-order-rvsecncl"
)

#: (family, paper) -> (buy_tr, sell_tr, revise_cancel_tr). ``None`` = 공식 미지원.
_VERIFIED_TR_IDS: dict[tuple[OrderRouteFamily, bool], tuple[str, str, str] | None] = {
    (OrderRouteFamily.DOMESTIC_CASH, False): ("TTTC0012U", "TTTC0011U", "TTTC0013U"),
    (OrderRouteFamily.DOMESTIC_CASH, True): ("VTTC0012U", "VTTC0011U", "VTTC0013U"),
    (OrderRouteFamily.OVERSEAS_REGULAR, False): ("TTTT1002U", "TTTT1006U", "TTTT1004U"),
    (OrderRouteFamily.OVERSEAS_REGULAR, True): ("VTTT1002U", "VTTT1001U", "VTTT1004U"),
    (OrderRouteFamily.OVERSEAS_DAYTIME, False): ("TTTS6036U", "TTTS6037U", "TTTS6038U"),
    # daytime-order / daytime-order-rvsecncl 은 "모의투자 미지원" (공식 문서).
    (OrderRouteFamily.OVERSEAS_DAYTIME, True): None,
}

#: ``EXCG_ID_DVSN_CD`` 별 공식 ``ORD_DVSN`` 허용값. 거래소마다 표가 다르다.
#: 시간외 3종(05/06/07)은 KRX 전용 — NXT/SOR 표에 존재하지 않는다.
VERIFIED_ORDER_DIVISIONS: dict[str, frozenset[str]] = {
    "KRX": frozenset(
        {"00", "01", "02", "03", "04", "05", "06", "07",
         "11", "12", "13", "14", "15", "16", "21", "22", "23", "24"}
    ),
    "NXT": frozenset(
        {"00", "03", "04", "11", "12", "13", "14", "15", "16", "21", "22", "23", "24"}
    ),
    "SOR": frozenset({"00", "01", "03", "04", "11", "12", "13", "14", "15", "16"}),
}

#: 미국 주간거래는 지정가만 가능 (공식: "주간거래는 지정가만 가능").
VERIFIED_US_DAYTIME_ORDER_DIVISIONS = frozenset({"00"})
#: 미국 일반 주문 — 모의투자는 00 만. 실전은 LOO/LOC/TWAP/VWAP/MOO/MOC 도 있으나
#: 본 시스템은 limit-order-only 정책이라 00 만 사용한다.
VERIFIED_US_REGULAR_ORDER_DIVISIONS = frozenset({"00"})

#: 국내 WebSocket TR 쌍 (체결, 호가).
_KRX_CONTINUOUS_WS = ("H0STCNT0", "H0STASP0")
_KRX_OVERTIME_WS = ("H0STOUP0", "H0STOAA0")
_NXT_WS = ("H0NXCNT0", "H0NXASP0")
_UNIFIED_WS = ("H0UNCNT0", "H0UNASP0")
_US_WS = ("HDFSCNT0", "HDFSASP0")

#: ``OVRS_EXCG_CD`` (주문용). 일반·주간 공통.
US_ORDER_EXCHANGE_CODES = {Venue.NASDAQ: "NASD", Venue.NYSE: "NYSE", Venue.AMEX: "AMEX"}
#: 야간 subscription key 시장구분 (무료시세, ``D`` prefix).
US_NIGHT_FEED_CODES = {Venue.NASDAQ: "NAS", Venue.NYSE: "NYS", Venue.AMEX: "AMS"}
#: 주간거래 subscription key 시장구분 (``R`` prefix).
US_DAYTIME_FEED_CODES = {Venue.NASDAQ: "BAQ", Venue.NYSE: "BAY", Venue.AMEX: "BAA"}

#: 국내 호가는 10단, 미국 무료 호가도 매수/매도 각 10호가 (공식 문서).
#: 다만 미국은 나스닥 마켓센터 단일 시장 호가이므로 NBBO/통합호가가 아니다.
DOMESTIC_DEPTH_LEVELS = 10
US_DEPTH_LEVELS = 10


def domestic_subscription_key(symbol: str) -> str:
    """국내 tr_key 는 종목코드 그대로."""
    return str(symbol or "").strip().upper()


def us_night_subscription_key_factory(venue: Venue) -> Callable[[str], str]:
    """``D`` + 시장구분(3) + 종목  (예: ``DNASAAPL``)."""
    code = US_NIGHT_FEED_CODES[venue]

    def factory(symbol: str) -> str:
        return f"D{code}{str(symbol or '').strip().upper()}"

    return factory


def us_daytime_subscription_key_factory(venue: Venue) -> Callable[[str], str]:
    """``R`` + 주간 시장구분(3) + 종목  (예: ``RBAQAAPL``)."""
    code = US_DAYTIME_FEED_CODES[venue]

    def factory(symbol: str) -> str:
        return f"R{code}{str(symbol or '').strip().upper()}"

    return factory


# --------------------------------------------------------------------------- #
# fail-closed 사유코드
# --------------------------------------------------------------------------- #
class ReasonCode(str, Enum):
    MARKET_SESSION_UNKNOWN = "MARKET_SESSION_UNKNOWN"
    SESSION_CALENDAR_STALE = "SESSION_CALENDAR_STALE"
    SESSION_CALENDAR_SUSPECT = "SESSION_CALENDAR_SUSPECT"
    SESSION_CLOSED = "SESSION_CLOSED"
    SESSION_ORDER_ROUTE_UNVERIFIED = "SESSION_ORDER_ROUTE_UNVERIFIED"
    SESSION_ORDER_TYPE_UNVERIFIED = "SESSION_ORDER_TYPE_UNVERIFIED"
    NXT_ORDER_ROUTE_UNVERIFIED = "NXT_ORDER_ROUTE_UNVERIFIED"
    EXCHANGE_CODE_UNRESOLVED = "EXCHANGE_CODE_UNRESOLVED"
    ORDER_DIVISION_UNVERIFIED = "ORDER_DIVISION_UNVERIFIED"
    EXTENDED_ENTRY_DISABLED = "EXTENDED_ENTRY_DISABLED"
    EXTENDED_LIVE_ORDER_NOT_AUTHORIZED = "EXTENDED_LIVE_ORDER_NOT_AUTHORIZED"
    SESSION_MISMATCH = "SESSION_MISMATCH"
    SESSION_ROUTE_AMBIGUOUS = "SESSION_ROUTE_AMBIGUOUS"
    NON_TRADEABLE_FEED = "NON_TRADEABLE_FEED"
    STALE_TRADE = "STALE_TRADE"
    STALE_ORDERBOOK = "STALE_ORDERBOOK"
    PARTIAL_DEPTH_NOT_SUPPORTED_BY_STRATEGY = "PARTIAL_DEPTH_NOT_SUPPORTED_BY_STRATEGY"
    SINGLE_MARKET_CENTER_DEPTH = "SINGLE_MARKET_CENTER_DEPTH"
    REST_SNAPSHOT_ONLY = "REST_SNAPSHOT_ONLY"
    FEED_DISCONNECTED = "FEED_DISCONNECTED"
    SUBSCRIPTION_REJECTED = "SUBSCRIPTION_REJECTED"
    VENUE_UNKNOWN = "VENUE_UNKNOWN"
    BROKER_ACCOUNT_STALE = "BROKER_ACCOUNT_STALE"
    BROKER_ROUTE_REJECTED = "BROKER_ROUTE_REJECTED"
    RECONCILIATION_REQUIRED = "RECONCILIATION_REQUIRED"
    PAPER_VENUE_UNSUPPORTED = "PAPER_VENUE_UNSUPPORTED"
    PAPER_DAYTIME_UNSUPPORTED = "PAPER_DAYTIME_UNSUPPORTED"
    DAYTIME_QUOTE_WINDOW_ENDED = "DAYTIME_QUOTE_WINDOW_ENDED"
    EARLY_CLOSE_AFTERMARKET_WINDOW_UNVERIFIED = "EARLY_CLOSE_AFTERMARKET_WINDOW_UNVERIFIED"
    CLOSING_PRICE_ORDER_TYPE = "CLOSING_PRICE_ORDER_TYPE"


# --------------------------------------------------------------------------- #
# 세션 정책 / 시각창
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class SessionPolicy:
    """``config/market_sessions.yaml`` 의 세션별 운영 정책."""

    data_enabled: bool = True
    new_entry_enabled: bool = False
    exit_enabled: bool = True
    live_order_authorized: bool = False
    max_spread_bps: float = 100.0
    max_quote_age_ms: int = 60_000
    max_book_age_ms: int = 60_000
    minimum_trade_rate: float = 0.0
    maximum_position_weight: float = 0.05


@dataclass(frozen=True)
class SessionWindow:
    """세션 시각창. ``tz`` 는 창을 해석하는 timezone.

    미국 주간거래처럼 **한국시간 고정창**인 세션은 ``tz=Asia/Seoul`` 이다. ET 로 옮기면
    DST 마다 이동하는 창이 되어버리기 때문에, 창은 반드시 공문이 쓰인 timezone 으로
    보관한다.
    """

    tz: str
    start: time
    end: time
    #: DST 가 적용되는 동안 사용할 종료시각. 미국 애프터마켓이 유일한 사례
    #: (KIS 공문 애프터마켓 06:00~07:00 KST / ST 05:00~07:00 KST → ET 17:00 / 18:00).
    dst_end: time | None = None
    #: 공식 시세 제공창이 주문창보다 좁은 경우의 시세창 종료시각 (미국 주간거래).
    data_end: time | None = None

    @property
    def zone(self) -> ZoneInfo:
        return ZoneInfo(self.tz)

    def end_at(self, local: datetime) -> time:
        if self.dst_end is not None and bool(local.dst()):
            return self.dst_end
        return self.end

    def data_end_at(self, local: datetime) -> time:
        return self.data_end if self.data_end is not None else self.end_at(local)

    def contains(self, now_utc: datetime) -> bool:
        local = now_utc.astimezone(self.zone)
        return self.start <= local.time() < self.end_at(local)

    def data_contains(self, now_utc: datetime) -> bool:
        local = now_utc.astimezone(self.zone)
        return self.start <= local.time() < self.data_end_at(local)


# --------------------------------------------------------------------------- #
# capability
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class MarketCapability:
    """한 (market, venue, session) 조합의 완전한 능력 서술."""

    market_group: MarketGroup
    venue: Venue
    session: SessionId
    timezone: str
    session_start: datetime | None
    session_end: datetime | None

    data_available: bool
    trade_available: bool
    new_entry_allowed: bool
    exit_allowed: bool

    supported_order_types: tuple[str, ...]
    order_endpoint: str | None
    buy_tr_id: str | None
    sell_tr_id: str | None
    revise_cancel_endpoint: str | None
    revise_cancel_tr_id: str | None
    route_family: OrderRouteFamily

    trade_ws_tr_id: str | None
    orderbook_ws_tr_id: str | None
    subscription_key_factory: Callable[[str], str] | None
    exchange_id_code: str | None
    order_division_mapping: Mapping[str, str]
    depth_level_count: int
    is_consolidated: bool

    source_scope: FeedScope
    source_quality: float
    policy: SessionPolicy

    officially_verified_at: str
    verification_source: str
    unavailable_reason: tuple[str, ...] = ()

    @property
    def currency(self) -> str:
        return "KRW" if self.market_group is MarketGroup.KR else "USD"

    @property
    def is_data_only(self) -> bool:
        """시세는 오지만 주문 route 가 없는 상태 (예: KRX+NXT 통합 피드)."""
        return self.data_available and not self.trade_available

    @property
    def live_order_authorized(self) -> bool:
        """운영자가 이 세션에서 **실주문** 을 명시적으로 승인했는가.

        ``new_entry_allowed`` 와 의도적으로 분리된 값이다. 두 질문이 다르기 때문이다:

        * ``new_entry_allowed`` — 세션 정책상 신규 진입을 시도해도 되는가
          (route 검증 + 데이터 + 세션 정책 + 캘린더).
        * ``live_order_authorized`` — 그 시도를 **실제 주문으로 전송해도** 되는가.

        둘을 하나로 합치면 SHADOW 운영(의도는 생성하되 주문은 보내지 않음)을 표현할 수
        없고, 기존 ``TRADING_ALLOW_EXTENDED_HOURS_ENTRY`` 의 의미(세션 게이트 완화)도
        깨진다. 실주문은 두 값이 **모두** 참일 때만 나간다 — 라우터가 그것을 강제한다.
        """
        return bool(self.policy.live_order_authorized)

    def limit_order_division(self) -> str:
        """이 세션에서 지정가/그에 상응하는 주문에 쓸 ``ORD_DVSN``."""
        return self.order_division_mapping.get("limit", "00")

    def to_payload(self) -> dict[str, object]:
        return {
            "market_group": self.market_group.value,
            "venue": self.venue.value,
            "session": self.session.value,
            "timezone": self.timezone,
            "session_start": self.session_start.isoformat() if self.session_start else None,
            "session_end": self.session_end.isoformat() if self.session_end else None,
            "data_available": self.data_available,
            "trade_available": self.trade_available,
            "new_entry_allowed": self.new_entry_allowed,
            "exit_allowed": self.exit_allowed,
            "supported_order_types": list(self.supported_order_types),
            "order_endpoint": self.order_endpoint,
            "buy_tr_id": self.buy_tr_id,
            "sell_tr_id": self.sell_tr_id,
            "revise_cancel_endpoint": self.revise_cancel_endpoint,
            "revise_cancel_tr_id": self.revise_cancel_tr_id,
            "route_family": self.route_family.value,
            "trade_ws_tr_id": self.trade_ws_tr_id,
            "orderbook_ws_tr_id": self.orderbook_ws_tr_id,
            "exchange_id_code": self.exchange_id_code,
            "order_division_mapping": dict(self.order_division_mapping),
            "depth_level_count": self.depth_level_count,
            "is_consolidated": self.is_consolidated,
            "source_scope": self.source_scope.value,
            "source_quality": self.source_quality,
            "currency": self.currency,
            "is_data_only": self.is_data_only,
            "officially_verified_at": self.officially_verified_at,
            "verification_source": self.verification_source,
            "unavailable_reason": list(self.unavailable_reason),
            "policy": {
                "data_enabled": self.policy.data_enabled,
                "new_entry_enabled": self.policy.new_entry_enabled,
                "exit_enabled": self.policy.exit_enabled,
                "live_order_authorized": self.policy.live_order_authorized,
                "max_spread_bps": self.policy.max_spread_bps,
                "max_quote_age_ms": self.policy.max_quote_age_ms,
                "max_book_age_ms": self.policy.max_book_age_ms,
                "minimum_trade_rate": self.policy.minimum_trade_rate,
                "maximum_position_weight": self.policy.maximum_position_weight,
            },
        }


@dataclass(frozen=True)
class OrderRoute:
    """주문 라우팅 결정. ``allowed=False`` 면 ``reason_codes`` 가 사유를 설명한다."""

    allowed: bool
    market_group: MarketGroup | None = None
    venue: Venue = Venue.UNKNOWN
    session: SessionId = SessionId.UNKNOWN
    route_family: OrderRouteFamily = OrderRouteFamily.NONE
    endpoint: str | None = None
    tr_id: str | None = None
    revise_cancel_endpoint: str | None = None
    revise_cancel_tr_id: str | None = None
    exchange_id_code: str | None = None
    order_division: str | None = None
    order_condition: str | None = None
    reason_codes: tuple[str, ...] = ()

    def to_payload(self) -> dict[str, object]:
        return {
            "allowed": self.allowed,
            "market_group": self.market_group.value if self.market_group else None,
            "venue": self.venue.value,
            "session": self.session.value,
            "route_family": self.route_family.value,
            "endpoint": self.endpoint,
            "tr_id": self.tr_id,
            "revise_cancel_endpoint": self.revise_cancel_endpoint,
            "revise_cancel_tr_id": self.revise_cancel_tr_id,
            "exchange_id_code": self.exchange_id_code,
            "order_division": self.order_division,
            "order_condition": self.order_condition,
            "reason_codes": list(self.reason_codes),
        }


# --------------------------------------------------------------------------- #
# 캘린더
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class CalendarSnapshot:
    version: str
    source: str
    coverage_start: date
    coverage_end: date
    holidays: Mapping[str, frozenset[str]]
    early_close: Mapping[str, Mapping[str, time]]
    completeness: Mapping[str, str]
    provider: str = "local_snapshot"
    #: 설치되어 있으면 휴장일 권위. 없으면 스냅샷만 쓴다.
    external: "_ExternalCalendar | None" = None

    def covers(self, day: date) -> bool:
        return self.coverage_start <= day <= self.coverage_end

    def is_holiday(self, group: MarketGroup, day: date) -> bool:
        """외부 캘린더가 답할 수 있으면 그쪽이 권위, 아니면 스냅샷.

        스냅샷을 버리지 않고 fallback 으로 남기는 이유는 외부 캘린더의 수록
        범위가 유한하기 때문이다. 범위 밖에서는 고정일자 스냅샷이 불완전할지언정
        아무 답도 없는 것보다 낫다.
        """
        if self.external is not None:
            answer = self.external.is_holiday(group, day)
            if answer is not None:
                return answer
        return day.isoformat() in self.holidays.get(group.value, frozenset())

    def early_close_time(self, group: MarketGroup, day: date) -> time | None:
        return self.early_close.get(group.value, {}).get(day.isoformat())

    def is_complete(self, group: MarketGroup, day: date | None = None) -> bool:
        """이 그룹의 휴장일 목록을 완전하다고 주장할 수 있는가.

        ``day`` 를 주면 그 날짜에 대한 판단이다. 외부 캘린더가 그 날을 수록하고
        있으면 음력·대체공휴일까지 포함하므로 완전하다. 수록 범위를 벗어나면
        스냅샷으로 되돌아가므로 스냅샷 자신의 completeness 가 답이 된다 — KR 은
        ``fixed_date_only`` 라 거기서는 여전히 SESSION_CALENDAR_SUSPECT 가 뜬다.
        """
        if day is not None and self.external is not None and self.external.covers(group, day):
            return True
        return self.completeness.get(group.value, "unknown") == "full"


def _external_calendar_provider_available() -> bool:
    """검증된 외부 exchange calendar provider 가 설치되어 있는지."""
    try:  # pragma: no cover - 환경 의존
        import exchange_calendars  # noqa: F401
    except Exception:  # noqa: BLE001
        return False
    return True


#: 시장 그룹별 외부 캘린더 코드. KRX 는 XKRX, 미국 정규 거래소는 XNYS.
_EXTERNAL_CALENDAR_CODES: Mapping[str, str] = {
    MarketGroup.KR.value: "XKRX",
    MarketGroup.US.value: "XNYS",
}


class _ExternalCalendar:
    """``exchange_calendars`` 를 실제 휴장일 권위로 사용하는 어댑터.

    이것이 없던 동안 ``prefer_external_provider`` 는 텔레메트리 라벨만 바꾸고
    휴장일 조회는 전적으로 YAML 스냅샷이 했다. 스냅샷은 고정일자만 담을 수 있어서
    음력 휴일(설날·추석·부처님오신날)과 대체공휴일이 통째로 빠져 있었고,
    2026-08-17 (광복절 대체공휴일) 이 정규 거래일로 취급됐다.

    XKRX 가 2026 년에 대해 답하는 평일 휴장일은 15 일이고, 그중 여섯 —
    02-16/17/18 (설날), 05-25 (부처님오신날), 09-24/25 (추석) — 은 고정일자
    스냅샷이 원리적으로 표현할 수 없는 것들이다.

    답을 모르면 ``None`` 을 돌려준다
    ---------------------------------
    외부 캘린더의 수록 범위는 유한하다 (설치된 XKRX 는 2027-08-18 까지인데
    스냅샷의 coverage_end 는 2027-12-31 이다). 범위 밖 날짜를 "휴장일 아님" 으로
    답하면 조용히 거래일을 만들어내는 셈이라, 그런 날은 ``None`` 을 반환해
    호출자가 스냅샷으로 되돌아가게 한다.

    캘린더는 생성 시점에 전부 적재한다
    ----------------------------------
    첫 호출 때 지연 적재하고 실패를 기억하는 구조로 짰다가 되돌렸다. 이 서비스는
    여러 워커 스레드가 동시에 두드리는데, 그 첫 호출들이 겹치면 한쪽이
    ``import exchange_calendars`` 가 아직 초기화 중인 모듈을 만나 예외를 받는다.
    실측: 16 스레드 동시 첫 호출에서 결과가 True/False 로 갈리고 실패 캐시가
    오염됐다. 그 뒤로는 프로세스가 사는 내내 외부 캘린더가 꺼진 채로 돌아가면서도
    provider 라벨은 ``exchange_calendars`` 라고 말한다 — 조용히 틀린 상태다.

    그래서 import 와 캘린더 구축은 설정 로드 시점(단일 스레드)에 한 번 끝내고,
    실패하면 생성자가 예외를 던져 호출자가 아예 외부 권위를 주장하지 않게 한다.
    적재에 성공한 객체는 읽기 전용이라 이후 스레드 경합이 없다.
    """

    def __init__(self, groups: Sequence[MarketGroup] | None = None) -> None:
        import exchange_calendars as xc  # 실패는 호출자가 처리한다.

        wanted = tuple(groups or (MarketGroup.KR, MarketGroup.US))
        calendars: dict[str, Any] = {}
        bounds: dict[str, tuple[date, date]] = {}
        for group in wanted:
            code = _EXTERNAL_CALENDAR_CODES.get(group.value)
            if code is None:
                continue
            calendar = xc.get_calendar(code)
            calendars[group.value] = calendar
            bounds[group.value] = (
                calendar.first_session.date(),
                calendar.last_session.date(),
            )
        if not calendars:
            raise LookupError("외부 캘린더를 하나도 적재하지 못했다")
        self._calendars = calendars
        self._bounds = bounds

    @property
    def groups(self) -> tuple[str, ...]:
        return tuple(sorted(self._calendars))

    def covers(self, group: MarketGroup, day: date) -> bool:
        window = self._bounds.get(group.value)
        if window is None:
            return False
        first, last = window
        return first <= day <= last

    def is_holiday(self, group: MarketGroup, day: date) -> bool | None:
        """``True``/``False`` 는 확정 답, ``None`` 은 "모른다"."""
        if not self.covers(group, day):
            return None
        calendar = self._calendars.get(group.value)
        if calendar is None:
            return None
        try:
            return not bool(calendar.is_session(day.isoformat()))
        except Exception:  # noqa: BLE001 - 개별 조회 실패는 스냅샷으로 되돌린다.
            return None


# --------------------------------------------------------------------------- #
# 세션 정의 (venue/route 는 검증본, 시각창은 설정)
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class _SessionDefinition:
    session: SessionId
    market_group: MarketGroup
    venue: Venue
    route_family: OrderRouteFamily
    trade_ws_tr_id: str | None
    orderbook_ws_tr_id: str | None
    feed_scope: FeedScope
    depth_level_count: int
    is_consolidated: bool
    base_source_quality: float
    limit_order_division: str
    order_condition: str | None = None
    window: SessionWindow | None = None
    #: route 자체가 공식 미검증인 경우의 사유 (없으면 검증됨).
    route_unverified_reason: tuple[str, ...] = ()


def _t(text: str) -> time:
    hour, minute = str(text).split(":")
    return time(int(hour), int(minute))


#: 코드 기본 시각창 — YAML 이 없거나 항목이 빠졌을 때의 fallback.
#: 값은 ``config/market_sessions.yaml`` 과 동일하게 유지한다.
_DEFAULT_WINDOWS: dict[SessionId, SessionWindow] = {
    SessionId.KRX_PREOPEN: SessionWindow("Asia/Seoul", _t("08:30"), _t("08:40")),
    SessionId.KRX_OPENING_AUCTION: SessionWindow("Asia/Seoul", _t("08:40"), _t("09:00")),
    SessionId.KRX_REGULAR: SessionWindow("Asia/Seoul", _t("09:00"), _t("15:20")),
    SessionId.KRX_CLOSING_AUCTION: SessionWindow("Asia/Seoul", _t("15:20"), _t("15:30")),
    SessionId.KRX_AFTER_CLOSE: SessionWindow("Asia/Seoul", _t("15:40"), _t("16:00")),
    SessionId.KRX_AFTER_SINGLE_PRICE: SessionWindow("Asia/Seoul", _t("16:00"), _t("18:00")),
    SessionId.NXT_PRE: SessionWindow("Asia/Seoul", _t("08:00"), _t("08:50")),
    SessionId.NXT_REGULAR: SessionWindow("Asia/Seoul", _t("09:00"), _t("15:20")),
    SessionId.NXT_POST: SessionWindow("Asia/Seoul", _t("15:30"), _t("20:00")),
    SessionId.US_DAYTIME: SessionWindow(
        "Asia/Seoul", _t("10:00"), _t("18:00"), data_end=_t("16:00")
    ),
    SessionId.US_PREMARKET: SessionWindow("America/New_York", _t("04:00"), _t("09:30")),
    SessionId.US_REGULAR: SessionWindow("America/New_York", _t("09:30"), _t("16:00")),
    SessionId.US_AFTERMARKET: SessionWindow(
        "America/New_York", _t("16:00"), _t("17:00"), dst_end=_t("18:00")
    ),
}

_SESSION_DEFINITIONS: tuple[_SessionDefinition, ...] = (
    # ---- 국내 KRX ----
    _SessionDefinition(
        SessionId.KRX_PREOPEN, MarketGroup.KR, Venue.KRX, OrderRouteFamily.DOMESTIC_CASH,
        *_KRX_OVERTIME_WS, FeedScope.VENUE_SPECIFIC, DOMESTIC_DEPTH_LEVELS, False, 0.55,
        limit_order_division="05", order_condition="CLOSING_PRICE_ONLY",
    ),
    _SessionDefinition(
        SessionId.KRX_OPENING_AUCTION, MarketGroup.KR, Venue.KRX, OrderRouteFamily.DOMESTIC_CASH,
        *_KRX_CONTINUOUS_WS, FeedScope.VENUE_SPECIFIC, DOMESTIC_DEPTH_LEVELS, False, 0.6,
        limit_order_division="00", order_condition="AUCTION",
    ),
    _SessionDefinition(
        SessionId.KRX_REGULAR, MarketGroup.KR, Venue.KRX, OrderRouteFamily.DOMESTIC_CASH,
        *_KRX_CONTINUOUS_WS, FeedScope.VENUE_SPECIFIC, DOMESTIC_DEPTH_LEVELS, False, 1.0,
        limit_order_division="00",
    ),
    _SessionDefinition(
        SessionId.KRX_CLOSING_AUCTION, MarketGroup.KR, Venue.KRX, OrderRouteFamily.DOMESTIC_CASH,
        *_KRX_CONTINUOUS_WS, FeedScope.VENUE_SPECIFIC, DOMESTIC_DEPTH_LEVELS, False, 0.6,
        limit_order_division="00", order_condition="AUCTION",
    ),
    _SessionDefinition(
        SessionId.KRX_AFTER_CLOSE, MarketGroup.KR, Venue.KRX, OrderRouteFamily.DOMESTIC_CASH,
        *_KRX_OVERTIME_WS, FeedScope.VENUE_SPECIFIC, DOMESTIC_DEPTH_LEVELS, False, 0.55,
        limit_order_division="06", order_condition="CLOSING_PRICE_ONLY",
    ),
    _SessionDefinition(
        SessionId.KRX_AFTER_SINGLE_PRICE, MarketGroup.KR, Venue.KRX,
        OrderRouteFamily.DOMESTIC_CASH,
        *_KRX_OVERTIME_WS, FeedScope.VENUE_SPECIFIC, DOMESTIC_DEPTH_LEVELS, False, 0.5,
        limit_order_division="07", order_condition="SINGLE_PRICE_AUCTION",
    ),
    # ---- 국내 NXT ----
    _SessionDefinition(
        SessionId.NXT_PRE, MarketGroup.KR, Venue.NXT, OrderRouteFamily.DOMESTIC_CASH,
        *_NXT_WS, FeedScope.VENUE_SPECIFIC, DOMESTIC_DEPTH_LEVELS, False, 0.5,
        limit_order_division="00",
    ),
    _SessionDefinition(
        SessionId.NXT_REGULAR, MarketGroup.KR, Venue.NXT, OrderRouteFamily.DOMESTIC_CASH,
        *_NXT_WS, FeedScope.VENUE_SPECIFIC, DOMESTIC_DEPTH_LEVELS, False, 0.8,
        limit_order_division="00",
    ),
    _SessionDefinition(
        SessionId.NXT_POST, MarketGroup.KR, Venue.NXT, OrderRouteFamily.DOMESTIC_CASH,
        *_NXT_WS, FeedScope.VENUE_SPECIFIC, DOMESTIC_DEPTH_LEVELS, False, 0.5,
        limit_order_division="00",
    ),
    # ---- 미국 ----
    _SessionDefinition(
        SessionId.US_DAYTIME, MarketGroup.US, Venue.US_DAYTIME_VENUE,
        OrderRouteFamily.OVERSEAS_DAYTIME,
        *_US_WS, FeedScope.FREE_REALTIME, US_DEPTH_LEVELS, False, 0.55,
        limit_order_division="00",
    ),
    _SessionDefinition(
        SessionId.US_PREMARKET, MarketGroup.US, Venue.NASDAQ, OrderRouteFamily.OVERSEAS_REGULAR,
        *_US_WS, FeedScope.FREE_REALTIME, US_DEPTH_LEVELS, False, 0.6,
        limit_order_division="00",
    ),
    _SessionDefinition(
        SessionId.US_REGULAR, MarketGroup.US, Venue.NASDAQ, OrderRouteFamily.OVERSEAS_REGULAR,
        *_US_WS, FeedScope.FREE_REALTIME, US_DEPTH_LEVELS, False, 0.95,
        limit_order_division="00",
    ),
    _SessionDefinition(
        SessionId.US_AFTERMARKET, MarketGroup.US, Venue.NASDAQ, OrderRouteFamily.OVERSEAS_REGULAR,
        *_US_WS, FeedScope.FREE_REALTIME, US_DEPTH_LEVELS, False, 0.6,
        limit_order_division="00",
    ),
)

_DEFINITION_BY_SESSION = {item.session: item for item in _SESSION_DEFINITIONS}

#: 국내 통합(KRX+NXT) 피드는 **시세 전용**이다. ``EXCG_ID_DVSN_CD`` 에 "통합"에 해당하는
#: 값이 존재하지 않으므로, 통합 피드로 본 체결을 근거로 주문할 때는 KRX / NXT / SOR 중
#: 하나로 route 를 확정해야 한다.
UNIFIED_FEED_DEFINITION = _SessionDefinition(
    SessionId.UNKNOWN, MarketGroup.KR, Venue.KRX_NXT_UNIFIED, OrderRouteFamily.NONE,
    *_UNIFIED_WS, FeedScope.UNIFIED, DOMESTIC_DEPTH_LEVELS, True, 0.7,
    limit_order_division="00",
    route_unverified_reason=(ReasonCode.EXCHANGE_CODE_UNRESOLVED.value,),
)


def _env_flag(name: str, default: bool) -> bool:
    raw = os.getenv(name)
    if raw in (None, ""):
        return default
    return str(raw).strip().lower() in {"1", "true", "yes", "on"}


def _load_yaml(path: Path) -> dict:
    if not path.exists():
        return {}
    try:
        import yaml
    except ImportError:  # pragma: no cover - PyYAML 은 필수 의존성
        return {}
    try:
        loaded = yaml.safe_load(path.read_text(encoding="utf-8"))
    except Exception:  # noqa: BLE001 - 손상된 설정은 기본값으로 fallback
        return {}
    return loaded if isinstance(loaded, dict) else {}


def _policy_from_mapping(raw: Mapping[str, object] | None) -> SessionPolicy:
    data = dict(raw or {})

    def flag(key: str, default: bool) -> bool:
        value = data.get(key, default)
        if isinstance(value, bool):
            return value
        return str(value).strip().lower() in {"1", "true", "yes", "on"}

    def number(key: str, default: float) -> float:
        try:
            return float(data.get(key, default))
        except (TypeError, ValueError):
            return default

    return SessionPolicy(
        data_enabled=flag("data_enabled", True),
        new_entry_enabled=flag("new_entry_enabled", False),
        exit_enabled=flag("exit_enabled", True),
        live_order_authorized=flag("live_order_authorized", False),
        max_spread_bps=number("max_spread_bps", 100.0),
        max_quote_age_ms=int(number("max_quote_age_ms", 60_000)),
        max_book_age_ms=int(number("max_book_age_ms", 60_000)),
        minimum_trade_rate=number("minimum_trade_rate", 0.0),
        maximum_position_weight=number("maximum_position_weight", 0.05),
    )


def _window_from_mapping(
    session: SessionId, raw: Mapping[str, object] | None, data_end_raw: object | None
) -> SessionWindow:
    fallback = _DEFAULT_WINDOWS[session]
    data = dict(raw or {})
    if not data:
        return fallback
    try:
        window = SessionWindow(
            tz=str(data.get("tz") or fallback.tz),
            start=_t(str(data.get("start") or fallback.start.strftime("%H:%M"))),
            end=_t(str(data.get("end") or fallback.end.strftime("%H:%M"))),
            dst_end=_t(str(data["dst_end"])) if data.get("dst_end") else fallback.dst_end,
            data_end=_t(str(data_end_raw)) if data_end_raw else fallback.data_end,
        )
    except (TypeError, ValueError):
        return fallback
    return window


@dataclass(frozen=True)
class MarketSessionConfig:
    windows: Mapping[SessionId, SessionWindow]
    policies: Mapping[SessionId, SessionPolicy]
    calendar: CalendarSnapshot
    us_overlap_order_precedence: str = "night"

    @classmethod
    def load(cls, path: Path | str = DEFAULT_CONFIG_PATH) -> "MarketSessionConfig":
        raw = _load_yaml(Path(path))
        sessions_raw = raw.get("sessions") if isinstance(raw.get("sessions"), dict) else {}
        windows: dict[SessionId, SessionWindow] = {}
        policies: dict[SessionId, SessionPolicy] = {}
        for session in _DEFAULT_WINDOWS:
            entry = sessions_raw.get(session.value) if isinstance(sessions_raw, dict) else None
            entry = entry if isinstance(entry, dict) else {}
            windows[session] = _window_from_mapping(
                session, entry.get("window"), entry.get("data_window_end")
            )
            policies[session] = _policy_from_mapping(entry)
        return cls(
            windows=windows,
            policies=policies,
            calendar=_calendar_from_mapping(raw.get("calendar")),
            us_overlap_order_precedence=str(
                raw.get("us_overlap_order_precedence") or "night"
            ).strip().lower(),
        )


def _calendar_from_mapping(raw: object) -> CalendarSnapshot:
    data = dict(raw) if isinstance(raw, Mapping) else {}

    def parse_day(value: object, default: date) -> date:
        try:
            return date.fromisoformat(str(value))
        except (TypeError, ValueError):
            return default

    holidays: dict[str, frozenset[str]] = {}
    early: dict[str, dict[str, time]] = {}
    completeness: dict[str, str] = {}
    for group in (MarketGroup.KR, MarketGroup.US):
        section = data.get(group.value)
        section = dict(section) if isinstance(section, Mapping) else {}
        raw_holidays = section.get("holidays")
        holidays[group.value] = frozenset(
            str(item) for item in raw_holidays if item
        ) if isinstance(raw_holidays, Iterable) and not isinstance(raw_holidays, (str, bytes)) else frozenset()
        raw_early = section.get("early_close")
        parsed_early: dict[str, time] = {}
        if isinstance(raw_early, Mapping):
            for day, moment in raw_early.items():
                try:
                    parsed_early[str(day)] = _t(str(moment))
                except (TypeError, ValueError):
                    continue
        early[group.value] = parsed_early
        completeness[group.value] = str(section.get("completeness") or "unknown")

    provider = "local_snapshot"
    external: _ExternalCalendar | None = None
    if bool(data.get("prefer_external_provider", True)):
        # 라벨만 바꾸던 자리다. 이제 실제 조회 권위를 함께 세운다 — 라벨이
        # exchange_calendars 라고 말하면서 스냅샷으로 답하고 있으면, 어느
        # 캘린더로 학습·거래했는지 추적하겠다는 이 필드의 목적 자체가 무너진다.
        # 그래서 적재에 성공했을 때만 라벨이 바뀐다. 설치 여부만 보고 라벨을
        # 붙이면 적재 실패가 그대로 거짓 주장이 된다.
        try:
            external = _ExternalCalendar()
            provider = "exchange_calendars"
        except Exception:  # noqa: BLE001 - 미설치·적재 실패는 스냅샷으로 간다.
            external = None

    return CalendarSnapshot(
        version=str(data.get("version") or "unversioned"),
        source=str(data.get("source") or "unknown"),
        coverage_start=parse_day(data.get("coverage_start"), date(1970, 1, 1)),
        coverage_end=parse_day(data.get("coverage_end"), date(1970, 1, 1)),
        holidays=holidays,
        early_close=early,
        completeness=completeness,
        provider=provider,
        external=external,
    )


# --------------------------------------------------------------------------- #
# 서비스
# --------------------------------------------------------------------------- #
def _as_utc(now_utc: datetime | None) -> datetime:
    current = now_utc or datetime.now(timezone.utc)
    if current.tzinfo is None:
        current = current.replace(tzinfo=timezone.utc)
    return current


class MarketSessionService:
    """세션·capability·주문 route 판정의 유일한 권한. 순수 함수적이며 테스트 가능하다.

    다른 모듈은 자체 시각 조건을 갖지 않고 이 서비스를 호출한다.
    """

    def __init__(self, config: MarketSessionConfig | None = None) -> None:
        self._config = config or MarketSessionConfig.load()

    # -- 설정 -------------------------------------------------------------- #
    @property
    def config(self) -> MarketSessionConfig:
        return self._config

    @property
    def calendar(self) -> CalendarSnapshot:
        return self._config.calendar

    def policy(self, session: SessionId) -> SessionPolicy:
        return self._config.policies.get(session, SessionPolicy())

    def window(self, session: SessionId) -> SessionWindow | None:
        return self._config.windows.get(session)

    # -- 캘린더 ------------------------------------------------------------ #
    def calendar_state(self, group: MarketGroup, now_utc: datetime | None = None) -> tuple[str, ...]:
        """캘린더 신뢰도 사유코드. 빈 tuple 이면 완전히 신뢰 가능.

        두 사유코드의 강도가 다르다:

        ``SESSION_CALENDAR_STALE``
            요청 날짜가 스냅샷 커버리지를 벗어났다 = **일정 정보를 가져오지 못했다.**
            → 신규 진입 fail-closed (``blocking_calendar_reasons``).

        ``SESSION_CALENDAR_SUSPECT``
            스냅샷은 유효하지만 완전하지 않다 (예: KR 음력 휴장일·임시공휴일 미포함).
            → 신규 진입을 전면 차단하지 **않는다.** 그렇게 하면 국내 거래가 영구 정지된다.
            누락된 휴장일은 "세션은 열려 있는데 피드 증거가 없다"는 freshness 게이트
            (``STALE_TRADE`` / ``STALE_ORDERBOOK``) 가 실제로 잡아낸다. 이 코드는 운영자가
            캘린더를 보강해야 함을 알리는 관측용 신호다.
        """
        current = _as_utc(now_utc)
        reasons: list[str] = []
        zone = _SEOUL if group is MarketGroup.KR else _NEW_YORK
        local_day = current.astimezone(zone).date()
        if not self.calendar.covers(local_day):
            reasons.append(ReasonCode.SESSION_CALENDAR_STALE.value)
        if not self.calendar.is_complete(group, local_day):
            reasons.append(ReasonCode.SESSION_CALENDAR_SUSPECT.value)
        return tuple(reasons)

    def blocking_calendar_reasons(
        self, group: MarketGroup, now_utc: datetime | None = None
    ) -> tuple[str, ...]:
        """신규 진입을 실제로 차단해야 하는 캘린더 사유만."""
        return tuple(
            code
            for code in self.calendar_state(group, now_utc)
            if code == ReasonCode.SESSION_CALENDAR_STALE.value
        )

    def is_trading_day(self, group: MarketGroup, now_utc: datetime | None = None) -> bool:
        """시장의 거래소 현지 날짜 기준 영업일 여부."""
        current = _as_utc(now_utc)
        zone = _SEOUL if group is MarketGroup.KR else _NEW_YORK
        local = current.astimezone(zone)
        if local.weekday() >= 5:
            return False
        return not self.calendar.is_holiday(group, local.date())

    def _session_trading_day(
        self, definition: _SessionDefinition, window: SessionWindow, now_utc: datetime
    ) -> bool:
        """세션 단위 영업일 판정.

        일반적으로는 거래소 현지 날짜를 보면 되지만, **미국 주간거래는 한국시간 창**
        (10:00-18:00 KST) 이라서 그렇지 않다. 토요일 12:00 KST 는 뉴욕 기준 금요일
        23:00 이므로 뉴욕 날짜만 보면 "영업일"로 판정되어 토요일에 주간거래가 열려 있는
        것처럼 보인다. 창이 정의된 timezone 의 요일과 시장 캘린더의 휴장일을 **둘 다**
        만족해야 한다.
        """
        if not self.is_trading_day(definition.market_group, now_utc):
            return False
        window_local = now_utc.astimezone(window.zone)
        if window_local.weekday() >= 5:
            return False
        return True

    # -- 세션 판정 --------------------------------------------------------- #
    def active_capabilities(
        self, group: str | MarketGroup, now_utc: datetime | None = None
    ) -> tuple[MarketCapability, ...]:
        """현재 열려 있는 (venue, session) 전부. KRX 와 NXT 는 동시에 열릴 수 있다."""
        market = group if isinstance(group, MarketGroup) else normalize_market_group(str(group))
        if market is None:
            return ()
        current = _as_utc(now_utc)
        found: list[MarketCapability] = []
        for definition in _SESSION_DEFINITIONS:
            if definition.market_group is not market:
                continue
            window = self.window(definition.session)
            if window is None or not window.contains(current):
                continue
            if not self._session_trading_day(definition, window, current):
                continue
            found.append(self._build_capability(definition, current))
        return tuple(found)

    def capability(
        self,
        group: str | MarketGroup,
        session: SessionId,
        now_utc: datetime | None = None,
    ) -> MarketCapability:
        """특정 세션의 capability. 현재 열려 있지 않아도 서술은 반환한다."""
        market = group if isinstance(group, MarketGroup) else normalize_market_group(str(group))
        definition = _DEFINITION_BY_SESSION.get(session)
        current = _as_utc(now_utc)
        if definition is None or market is None:
            return self.closed_capability(market or MarketGroup.KR, current)
        return self._build_capability(definition, current)

    def unified_feed_capability(self, now_utc: datetime | None = None) -> MarketCapability:
        """국내 KRX+NXT 통합 피드 (DATA_ONLY)."""
        current = _as_utc(now_utc)
        active = self.active_capabilities(MarketGroup.KR, current)
        session = active[0].session if active else SessionId.KR_CLOSED
        definition = replace(UNIFIED_FEED_DEFINITION, session=session)
        capability = self._build_capability(definition, current)
        return replace(
            capability,
            data_available=bool(active),
            trade_available=False,
            new_entry_allowed=False,
            exit_allowed=False,
            unavailable_reason=(ReasonCode.EXCHANGE_CODE_UNRESOLVED.value,),
        )

    def closed_capability(
        self, group: MarketGroup, now_utc: datetime | None = None
    ) -> MarketCapability:
        current = _as_utc(now_utc)
        session = CLOSED_SESSIONS[group]
        reasons = (ReasonCode.SESSION_CLOSED.value, ReasonCode.REST_SNAPSHOT_ONLY.value)
        return MarketCapability(
            market_group=group,
            venue=Venue.UNKNOWN,
            session=session,
            timezone="Asia/Seoul" if group is MarketGroup.KR else "America/New_York",
            session_start=None,
            session_end=None,
            data_available=False,
            trade_available=False,
            new_entry_allowed=False,
            exit_allowed=False,
            supported_order_types=(),
            order_endpoint=None,
            buy_tr_id=None,
            sell_tr_id=None,
            revise_cancel_endpoint=None,
            revise_cancel_tr_id=None,
            route_family=OrderRouteFamily.NONE,
            trade_ws_tr_id=None,
            orderbook_ws_tr_id=None,
            subscription_key_factory=None,
            exchange_id_code=None,
            order_division_mapping={},
            depth_level_count=0,
            is_consolidated=False,
            source_scope=FeedScope.REST_SNAPSHOT,
            source_quality=0.0,
            policy=SessionPolicy(
                data_enabled=True, new_entry_enabled=False, exit_enabled=False
            ),
            officially_verified_at=VERIFICATION_DATE,
            verification_source=VERIFICATION_SOURCE,
            unavailable_reason=reasons + self.calendar_state(group, current),
        )

    def primary_capability(
        self, group: str | MarketGroup, now_utc: datetime | None = None
    ) -> MarketCapability:
        """신규 진입 판단에 쓰는 대표 세션.

        우선순위: 신규 진입 허용 세션 > source_quality 높은 세션. 없으면 closed.
        """
        market = group if isinstance(group, MarketGroup) else normalize_market_group(str(group))
        current = _as_utc(now_utc)
        if market is None:
            return self.closed_capability(MarketGroup.KR, current)
        active = self.active_capabilities(market, current)
        if not active:
            return self.closed_capability(market, current)
        return max(
            active, key=lambda item: (item.new_entry_allowed, item.source_quality)
        )

    # -- 개별 능력 질의 ---------------------------------------------------- #
    def data_available(self, group: str | MarketGroup, now_utc: datetime | None = None) -> bool:
        return any(item.data_available for item in self.active_capabilities(group, now_utc))

    def trade_available(self, group: str | MarketGroup, now_utc: datetime | None = None) -> bool:
        return any(item.trade_available for item in self.active_capabilities(group, now_utc))

    def new_entry_allowed(
        self, group: str | MarketGroup, now_utc: datetime | None = None
    ) -> bool:
        return any(item.new_entry_allowed for item in self.active_capabilities(group, now_utc))

    def exit_allowed(self, group: str | MarketGroup, now_utc: datetime | None = None) -> bool:
        return any(item.exit_allowed for item in self.active_capabilities(group, now_utc))

    def new_entry_block_reasons(
        self, group: str | MarketGroup, now_utc: datetime | None = None
    ) -> tuple[str, ...]:
        """왜 신규 진입이 막혔는지. 진입 가능하면 빈 tuple."""
        current = _as_utc(now_utc)
        active = self.active_capabilities(group, current)
        if not active:
            market = group if isinstance(group, MarketGroup) else normalize_market_group(str(group))
            if market is None:
                return (ReasonCode.MARKET_SESSION_UNKNOWN.value,)
            return self.closed_capability(market, current).unavailable_reason
        if any(item.new_entry_allowed for item in active):
            return ()
        reasons: list[str] = []
        for item in active:
            reasons.extend(item.unavailable_reason)
        if not reasons:
            reasons.append(ReasonCode.EXTENDED_ENTRY_DISABLED.value)
        return tuple(dict.fromkeys(reasons))

    # -- capability 조립 --------------------------------------------------- #
    def _build_capability(
        self, definition: _SessionDefinition, now_utc: datetime
    ) -> MarketCapability:
        window = self.window(definition.session) or _DEFAULT_WINDOWS.get(
            definition.session, SessionWindow("Asia/Seoul", _t("00:00"), _t("00:00"))
        )
        policy = self.policy(definition.session)
        market = definition.market_group
        calendar_reasons = list(self.calendar_state(market, now_utc))
        blocking_calendar = list(self.blocking_calendar_reasons(market, now_utc))
        reasons: list[str] = list(definition.route_unverified_reason)

        local = now_utc.astimezone(window.zone)
        start_local = local.replace(
            hour=window.start.hour, minute=window.start.minute, second=0, microsecond=0
        )
        end_time = window.end_at(local)
        end_local = local.replace(
            hour=end_time.hour, minute=end_time.minute, second=0, microsecond=0
        )

        # 조기폐장 반영. 미국 애프터마켓의 조기폐장일 창은 공문 근거가 없어 미검증 표기.
        early = self.calendar.early_close_time(market, now_utc.astimezone(
            _SEOUL if market is MarketGroup.KR else _NEW_YORK
        ).date())
        if early is not None and market is MarketGroup.US:
            if definition.session is SessionId.US_REGULAR:
                end_local = end_local.replace(hour=early.hour, minute=early.minute)
            elif definition.session is SessionId.US_AFTERMARKET:
                start_local = start_local.replace(hour=early.hour, minute=early.minute)
                reasons.append(
                    ReasonCode.EARLY_CLOSE_AFTERMARKET_WINDOW_UNVERIFIED.value
                )

        # 데이터 가용성: 정책 + (미국 주간거래) 공식 시세창.
        data_available = bool(policy.data_enabled)
        if definition.session is SessionId.US_DAYTIME and not window.data_contains(now_utc):
            data_available = False
            reasons.append(ReasonCode.DAYTIME_QUOTE_WINDOW_ENDED.value)

        # 주문 route.
        paper = _is_paper_mode()
        tr_ids = _VERIFIED_TR_IDS.get((definition.route_family, paper))
        route_available = definition.route_family is not OrderRouteFamily.NONE and tr_ids is not None
        if definition.route_family is OrderRouteFamily.NONE:
            reasons.append(ReasonCode.SESSION_ORDER_ROUTE_UNVERIFIED.value)
        elif tr_ids is None:
            reasons.append(
                ReasonCode.PAPER_DAYTIME_UNSUPPORTED.value
                if definition.route_family is OrderRouteFamily.OVERSEAS_DAYTIME
                else ReasonCode.SESSION_ORDER_ROUTE_UNVERIFIED.value
            )

        exchange_id_code = None
        if definition.route_family is OrderRouteFamily.DOMESTIC_CASH:
            exchange_id_code = "KRX" if definition.venue is Venue.KRX else "NXT"
            if paper and exchange_id_code != "KRX":
                # 공식: "모의투자는 KRX만 가능".
                route_available = False
                reasons.append(ReasonCode.PAPER_VENUE_UNSUPPORTED.value)

        division = definition.limit_order_division
        allowed_divisions = self._allowed_divisions(definition)
        if route_available and division not in allowed_divisions:
            route_available = False
            reasons.append(ReasonCode.SESSION_ORDER_TYPE_UNVERIFIED.value)

        # 신규 진입 / 청산.
        entry_enabled = _entry_enabled_for(definition.session, policy)
        # 세션 게이트. 실주문 승인(``live_order_authorized``)은 여기 포함하지 않는다 —
        # 별도 게이트이며 라우터가 둘을 함께 요구한다.
        new_entry_allowed = bool(
            route_available and data_available and entry_enabled and not blocking_calendar
        )
        exit_allowed = bool(route_available and policy.exit_enabled)

        if not entry_enabled:
            reasons.append(ReasonCode.EXTENDED_ENTRY_DISABLED.value)
        if not policy.live_order_authorized:
            reasons.append(ReasonCode.EXTENDED_LIVE_ORDER_NOT_AUTHORIZED.value)
        reasons.extend(calendar_reasons)
        if definition.market_group is MarketGroup.US:
            # 미국 무료 시세는 나스닥 마켓센터 단일 시장 호가 (NBBO 아님).
            reasons.append(ReasonCode.SINGLE_MARKET_CENTER_DEPTH.value)

        endpoint, revise_endpoint = _endpoints_for(definition.route_family)
        buy_tr, sell_tr, rvse_tr = tr_ids if tr_ids else (None, None, None)

        return MarketCapability(
            market_group=market,
            venue=definition.venue,
            session=definition.session,
            timezone=window.tz,
            session_start=start_local.astimezone(timezone.utc),
            session_end=end_local.astimezone(timezone.utc),
            data_available=data_available,
            trade_available=route_available,
            new_entry_allowed=new_entry_allowed,
            exit_allowed=exit_allowed,
            supported_order_types=tuple(sorted(allowed_divisions)),
            order_endpoint=endpoint if route_available else None,
            buy_tr_id=buy_tr if route_available else None,
            sell_tr_id=sell_tr if route_available else None,
            revise_cancel_endpoint=revise_endpoint if route_available else None,
            revise_cancel_tr_id=rvse_tr if route_available else None,
            route_family=definition.route_family if route_available else OrderRouteFamily.NONE,
            trade_ws_tr_id=definition.trade_ws_tr_id,
            orderbook_ws_tr_id=definition.orderbook_ws_tr_id,
            subscription_key_factory=self._subscription_key_factory(definition),
            exchange_id_code=exchange_id_code,
            order_division_mapping={"limit": division},
            depth_level_count=definition.depth_level_count,
            is_consolidated=definition.is_consolidated,
            source_scope=definition.feed_scope,
            source_quality=definition.base_source_quality,
            policy=policy,
            officially_verified_at=VERIFICATION_DATE,
            verification_source=VERIFICATION_SOURCE,
            unavailable_reason=tuple(dict.fromkeys(reasons)),
        )

    @staticmethod
    def _allowed_divisions(definition: _SessionDefinition) -> frozenset[str]:
        if definition.route_family is OrderRouteFamily.DOMESTIC_CASH:
            code = "KRX" if definition.venue is Venue.KRX else "NXT"
            return VERIFIED_ORDER_DIVISIONS[code]
        if definition.route_family is OrderRouteFamily.OVERSEAS_DAYTIME:
            return VERIFIED_US_DAYTIME_ORDER_DIVISIONS
        if definition.route_family is OrderRouteFamily.OVERSEAS_REGULAR:
            return VERIFIED_US_REGULAR_ORDER_DIVISIONS
        return frozenset()

    @staticmethod
    def _subscription_key_factory(
        definition: _SessionDefinition,
    ) -> Callable[[str], str] | None:
        if definition.market_group is MarketGroup.KR:
            return domestic_subscription_key
        if definition.session is SessionId.US_DAYTIME:
            return us_daytime_subscription_key_factory(Venue.NASDAQ)
        return us_night_subscription_key_factory(Venue.NASDAQ)

    def subscription_key(
        self,
        symbol: str,
        *,
        market_group: MarketGroup,
        session: SessionId,
        venue: Venue = Venue.NASDAQ,
    ) -> str:
        """세션·venue 에 맞는 tr_key. 미국은 세션 경계에서 D↔R 이 바뀐다."""
        if market_group is MarketGroup.KR:
            return domestic_subscription_key(symbol)
        resolved = venue if venue in US_NIGHT_FEED_CODES else Venue.NASDAQ
        if session is SessionId.US_DAYTIME:
            return us_daytime_subscription_key_factory(resolved)(symbol)
        return us_night_subscription_key_factory(resolved)(symbol)

    # -- 주문 라우팅 ------------------------------------------------------- #
    def resolve_order_route(
        self,
        *,
        market: str,
        side_is_buy: bool,
        intent: str = "entry",
        venue_hint: Venue | str | None = None,
        session_hint: SessionId | None = None,
        now_utc: datetime | None = None,
    ) -> OrderRoute:
        """주문 route 결정. 모호하거나 미검증이면 ``allowed=False`` 로 fail-closed.

        ``intent`` 는 ``"entry"`` (신규 진입) 또는 ``"exit"`` (청산/위험 축소).
        청산은 신규 진입이 금지된 세션에서도 공식 route 가 있으면 허용된다.
        """
        current = _as_utc(now_utc)
        group = normalize_market_group(market)
        if group is None:
            return OrderRoute(
                allowed=False, reason_codes=(ReasonCode.MARKET_SESSION_UNKNOWN.value,)
            )

        active = self.active_capabilities(group, current)
        candidates = [
            item for item in active if item.route_family is not OrderRouteFamily.NONE
        ]
        if not candidates and active:
            # 세션은 열려 있는데 공식 route 가 없다 (예: 모의투자의 주간거래·NXT).
            # 그 이유를 그대로 올려 준다 — "장이 닫혔다" 로 뭉개면 원인을 못 찾는다.
            reasons: list[str] = []
            for item in active:
                reasons.extend(item.unavailable_reason)
            return OrderRoute(
                allowed=False,
                market_group=group,
                session=active[0].session,
                venue=active[0].venue,
                reason_codes=tuple(dict.fromkeys(reasons))
                or (ReasonCode.SESSION_ORDER_ROUTE_UNVERIFIED.value,),
            )
        if session_hint is not None:
            candidates = [item for item in candidates if item.session is session_hint]
            if not candidates:
                return OrderRoute(
                    allowed=False,
                    market_group=group,
                    session=session_hint,
                    reason_codes=(ReasonCode.SESSION_MISMATCH.value,),
                )
        if not candidates:
            closed = self.closed_capability(group, current)
            return OrderRoute(
                allowed=False, market_group=group, session=closed.session,
                reason_codes=closed.unavailable_reason,
            )

        hint_venue = _coerce_venue(venue_hint)
        if hint_venue is not Venue.UNKNOWN:
            narrowed = [item for item in candidates if item.venue is hint_venue]
            if narrowed:
                candidates = narrowed
            elif group is MarketGroup.KR and hint_venue is Venue.KRX_NXT_UNIFIED:
                # 통합 피드는 주문 venue 가 될 수 없다.
                return OrderRoute(
                    allowed=False, market_group=group,
                    reason_codes=(ReasonCode.EXCHANGE_CODE_UNRESOLVED.value,),
                )

        chosen, ambiguity = self._choose_route_candidate(group, candidates, intent)
        reasons = list(ambiguity)

        if intent == "entry":
            # 실주문은 세션 게이트와 실주문 승인이 **모두** 참일 때만 나간다.
            if not chosen.new_entry_allowed or not chosen.live_order_authorized:
                reasons.extend(chosen.unavailable_reason)
                if chosen.new_entry_allowed and not chosen.live_order_authorized:
                    reasons.append(ReasonCode.EXTENDED_LIVE_ORDER_NOT_AUTHORIZED.value)
                return OrderRoute(
                    allowed=False, market_group=group, venue=chosen.venue,
                    session=chosen.session, route_family=chosen.route_family,
                    reason_codes=tuple(dict.fromkeys(reasons)),
                )
            if ambiguity:
                return OrderRoute(
                    allowed=False, market_group=group, venue=chosen.venue,
                    session=chosen.session, route_family=chosen.route_family,
                    reason_codes=tuple(dict.fromkeys(reasons)),
                )
        elif not chosen.exit_allowed:
            reasons.extend(chosen.unavailable_reason)
            return OrderRoute(
                allowed=False, market_group=group, venue=chosen.venue,
                session=chosen.session, route_family=chosen.route_family,
                reason_codes=tuple(dict.fromkeys(reasons)),
            )

        tr_id = chosen.buy_tr_id if side_is_buy else chosen.sell_tr_id
        if not tr_id or not chosen.order_endpoint:
            return OrderRoute(
                allowed=False, market_group=group, venue=chosen.venue,
                session=chosen.session, route_family=chosen.route_family,
                reason_codes=(ReasonCode.SESSION_ORDER_ROUTE_UNVERIFIED.value,),
            )

        definition = _DEFINITION_BY_SESSION.get(chosen.session)
        return OrderRoute(
            allowed=True,
            market_group=group,
            venue=chosen.venue,
            session=chosen.session,
            route_family=chosen.route_family,
            endpoint=chosen.order_endpoint,
            tr_id=tr_id,
            revise_cancel_endpoint=chosen.revise_cancel_endpoint,
            revise_cancel_tr_id=chosen.revise_cancel_tr_id,
            exchange_id_code=chosen.exchange_id_code,
            order_division=chosen.limit_order_division(),
            order_condition=definition.order_condition if definition else None,
            reason_codes=tuple(dict.fromkeys(reasons)),
        )

    def _choose_route_candidate(
        self, group: MarketGroup, candidates: list[MarketCapability], intent: str
    ) -> tuple[MarketCapability, tuple[str, ...]]:
        """결정론적 route 선택 + 모호성 사유.

        미국은 서머타임 구간에서 주간거래(10:00-18:00 KST)와 프리마켓(17:00-22:30 KST)이
        1시간 겹친다. 두 route 모두 공식 개방 상태라 자동 선택은 위험하므로 신규 진입은
        차단하고, 청산은 설정된 우선순위로 결정한다.
        """
        if len(candidates) == 1:
            return candidates[0], ()
        if group is MarketGroup.US:
            daytime = [i for i in candidates if i.session is SessionId.US_DAYTIME]
            night = [i for i in candidates if i.session is not SessionId.US_DAYTIME]
            if daytime and night:
                prefer_daytime = self._config.us_overlap_order_precedence == "daytime"
                chosen = (daytime if prefer_daytime else night)[0]
                return chosen, (ReasonCode.SESSION_ROUTE_AMBIGUOUS.value,)
            return candidates[0], ()
        # 국내: KRX 와 NXT 는 서로 다른 거래소이며 둘 다 명시적 route 가 있다.
        # 신규 진입 허용 세션을 먼저, 그 다음 데이터 품질 순.
        ranked = sorted(
            candidates,
            key=lambda item: (
                item.new_entry_allowed,
                item.venue is Venue.KRX,
                item.source_quality,
            ),
            reverse=True,
        )
        return ranked[0], ()

    def resolve_revise_cancel_route(
        self,
        *,
        market: str,
        original_route_family: OrderRouteFamily | str,
        original_venue: Venue | str | None = None,
        original_session: SessionId | str | None = None,
        now_utc: datetime | None = None,
    ) -> OrderRoute:
        """정정·취소 route. **원주문 family 를 그대로 따른다.**

        세션이 바뀌었다고 다른 venue/endpoint 로 자동 정정하지 않는다. 그런 전이는
        "원주문 취소 + 신규 주문" 이라는 명시적 상태 전이로만 허용된다.
        """
        current = _as_utc(now_utc)
        group = normalize_market_group(market)
        family = (
            original_route_family
            if isinstance(original_route_family, OrderRouteFamily)
            else _coerce_route_family(str(original_route_family))
        )
        if group is None or family is OrderRouteFamily.NONE:
            return OrderRoute(
                allowed=False, market_group=group,
                reason_codes=(ReasonCode.SESSION_ORDER_ROUTE_UNVERIFIED.value,),
            )
        tr_ids = _VERIFIED_TR_IDS.get((family, _is_paper_mode()))
        if tr_ids is None:
            return OrderRoute(
                allowed=False, market_group=group, route_family=family,
                reason_codes=(
                    ReasonCode.PAPER_DAYTIME_UNSUPPORTED.value
                    if family is OrderRouteFamily.OVERSEAS_DAYTIME
                    else ReasonCode.SESSION_ORDER_ROUTE_UNVERIFIED.value,
                ),
            )
        endpoint, revise_endpoint = _endpoints_for(family)
        venue = _coerce_venue(original_venue)
        session = _coerce_session(original_session)
        exchange_id_code = None
        if family is OrderRouteFamily.DOMESTIC_CASH:
            exchange_id_code = "NXT" if venue is Venue.NXT else "KRX"
        division = "00"
        definition = _DEFINITION_BY_SESSION.get(session)
        if definition is not None and family is OrderRouteFamily.DOMESTIC_CASH:
            division = definition.limit_order_division
        reasons: list[str] = []
        # 정정·취소 자체는 세션 종료 후에도 접수 가능한 경우가 있으므로 열림 여부로
        # 차단하지 않되, 세션이 닫혀 있으면 관측 가능하도록 사유를 남긴다.
        if not self.active_capabilities(group, current):
            reasons.append(ReasonCode.SESSION_CLOSED.value)
        return OrderRoute(
            allowed=True,
            market_group=group,
            venue=venue,
            session=session,
            route_family=family,
            endpoint=endpoint,
            tr_id=tr_ids[2],
            revise_cancel_endpoint=revise_endpoint,
            revise_cancel_tr_id=tr_ids[2],
            exchange_id_code=exchange_id_code,
            order_division=division,
            order_condition=definition.order_condition if definition else None,
            reason_codes=tuple(reasons),
        )

    # -- 보고 -------------------------------------------------------------- #
    def session_report(
        self,
        groups: tuple[str, ...] = ("KR", "US"),
        now_utc: datetime | None = None,
    ) -> dict[str, object]:
        """운영 화면용. **데이터 가용 / 주문 가용 / 신규 진입 / 청산을 분리**해 반환한다."""
        current = _as_utc(now_utc)
        payload: dict[str, object] = {}
        for name in groups:
            market = normalize_market_group(name)
            if market is None:
                continue
            active = self.active_capabilities(market, current)
            entry = {
                "market_group": market.value,
                "trading_day": self.is_trading_day(market, current),
                "calendar_reasons": list(self.calendar_state(market, current)),
                "data_available": any(item.data_available for item in active),
                "trade_available": any(item.trade_available for item in active),
                "new_entry_allowed": any(item.new_entry_allowed for item in active),
                "exit_allowed": any(item.exit_allowed for item in active),
                "new_entry_block_reasons": list(
                    self.new_entry_block_reasons(market, current)
                ),
                "primary_session": self.primary_capability(market, current).session.value,
                "sessions": [item.to_payload() for item in active],
            }
            if market is MarketGroup.KR:
                entry["unified_feed"] = self.unified_feed_capability(current).to_payload()
            payload[market.value] = entry
        return {
            "as_of": current.isoformat(),
            "calendar_version": self.calendar.version,
            "calendar_provider": self.calendar.provider,
            "verification_source": VERIFICATION_SOURCE,
            "groups": payload,
        }

    def capability_matrix(self, now_utc: datetime | None = None) -> dict[str, object]:
        """전 세션 capability + 미지원 route 목록 (진단 API 용)."""
        current = _as_utc(now_utc)
        rows = [
            self._build_capability(definition, current).to_payload()
            for definition in _SESSION_DEFINITIONS
        ]
        rows.append(self.unified_feed_capability(current).to_payload())
        unsupported = [
            {
                "session": row["session"],
                "venue": row["venue"],
                "reasons": row["unavailable_reason"],
            }
            for row in rows
            if not row["trade_available"]
        ]
        return {
            "as_of": current.isoformat(),
            "verification_source": VERIFICATION_SOURCE,
            "officially_verified_at": VERIFICATION_DATE,
            "calendar_version": self.calendar.version,
            "paper_mode": _is_paper_mode(),
            "capabilities": rows,
            "unsupported_routes": unsupported,
        }


# --------------------------------------------------------------------------- #
# 보조
# --------------------------------------------------------------------------- #
def _endpoints_for(family: OrderRouteFamily) -> tuple[str | None, str | None]:
    if family is OrderRouteFamily.DOMESTIC_CASH:
        return DOMESTIC_ORDER_ENDPOINT, DOMESTIC_REVISE_CANCEL_ENDPOINT
    if family is OrderRouteFamily.OVERSEAS_REGULAR:
        return OVERSEAS_ORDER_ENDPOINT, OVERSEAS_REVISE_CANCEL_ENDPOINT
    if family is OrderRouteFamily.OVERSEAS_DAYTIME:
        return OVERSEAS_DAYTIME_ORDER_ENDPOINT, OVERSEAS_DAYTIME_REVISE_CANCEL_ENDPOINT
    return None, None


def _coerce_venue(value: Venue | str | None) -> Venue:
    if isinstance(value, Venue):
        return value
    name = str(value or "").upper().strip()
    if not name:
        return Venue.UNKNOWN
    try:
        return Venue(name)
    except ValueError:
        return {
            "NASD": Venue.NASDAQ, "NAS": Venue.NASDAQ, "BAQ": Venue.NASDAQ,
            "NYS": Venue.NYSE, "BAY": Venue.NYSE,
            "AMS": Venue.AMEX, "BAA": Venue.AMEX,
            "UNIFIED": Venue.KRX_NXT_UNIFIED, "KRX+NXT": Venue.KRX_NXT_UNIFIED,
            "NEXTRADE": Venue.NXT,
        }.get(name, Venue.UNKNOWN)


def _coerce_session(value: SessionId | str | None) -> SessionId:
    if isinstance(value, SessionId):
        return value
    name = str(value or "").upper().strip()
    try:
        return SessionId(name)
    except ValueError:
        return SessionId.UNKNOWN


def _coerce_route_family(value: str) -> OrderRouteFamily:
    try:
        return OrderRouteFamily(str(value).upper().strip())
    except ValueError:
        return OrderRouteFamily.NONE


def _is_paper_mode() -> bool:
    """모의투자 모드. 모의는 NXT/SOR 및 주간거래를 지원하지 않는다."""
    raw = os.getenv("KIS_ENV", os.getenv("KIS_MODE", "")).strip().lower()
    if raw in {"paper", "vts", "virtual", "mock", "sim"}:
        return True
    return _env_flag("KIS_PAPER_TRADING", False)


def _entry_enabled_for(session: SessionId, policy: SessionPolicy) -> bool:
    """이 세션에서 신규 진입이 정책·환경변수상 허용되는지.

    우선순위:

    1. ``TRADING_ALLOW_ENTRY_<SESSION>`` — 세션별 명시 설정 (최우선).
    2. ``TRADING_ALLOW_EXTENDED_HOURS_ENTRY`` — 기존 전역 플래그.
       backward-compatible alias 로 유지되며 **장외 세션에만** 적용된다.
       정규장은 이 플래그와 무관하게 정책을 따른다 (원래도 "장외 허용" 플래그였다).
    3. ``config/market_sessions.yaml`` 의 ``new_entry_enabled``.

    정책이 이미 허용(1·3)이면 전역 플래그로 끌 수는 없다 — 끄는 것은 세션별 설정
    또는 YAML 의 일이다. 전역 플래그는 "추가로 열어 주는" 방향으로만 작동한다.
    """
    specific = os.getenv(f"TRADING_ALLOW_ENTRY_{session.value}")
    if specific not in (None, ""):
        return str(specific).strip().lower() in {"1", "true", "yes", "on"}
    if policy.new_entry_enabled:
        return True
    if session in _REGULAR_SESSIONS:
        return False
    return _env_flag("TRADING_ALLOW_EXTENDED_HOURS_ENTRY", False)


_REGULAR_SESSIONS = frozenset({SessionId.KRX_REGULAR, SessionId.US_REGULAR})


# --------------------------------------------------------------------------- #
# 기본 서비스 (프로세스 단일 인스턴스)
# --------------------------------------------------------------------------- #
_service_lock = threading.Lock()
_service: MarketSessionService | None = None


def default_service() -> MarketSessionService:
    global _service
    if _service is None:
        with _service_lock:
            if _service is None:
                _service = MarketSessionService()
    return _service


def reload_default_service(path: Path | str = DEFAULT_CONFIG_PATH) -> MarketSessionService:
    """설정을 다시 읽는다 (테스트 및 운영 중 정책 변경용)."""
    global _service
    with _service_lock:
        _service = MarketSessionService(MarketSessionConfig.load(path))
    return _service


__all__ = [
    "CalendarSnapshot",
    "DOMESTIC_DEPTH_LEVELS",
    "FeedScope",
    "MarketCapability",
    "MarketGroup",
    "MarketSessionConfig",
    "MarketSessionService",
    "OrderRoute",
    "OrderRouteFamily",
    "ReasonCode",
    "SessionId",
    "SessionPolicy",
    "SessionWindow",
    "US_DAYTIME_FEED_CODES",
    "US_DEPTH_LEVELS",
    "US_NIGHT_FEED_CODES",
    "US_ORDER_EXCHANGE_CODES",
    "VERIFICATION_DATE",
    "VERIFICATION_SOURCE",
    "VERIFIED_ORDER_DIVISIONS",
    "Venue",
    "default_service",
    "normalize_market_group",
    "reload_default_service",
]
