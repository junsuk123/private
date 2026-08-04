from __future__ import annotations

import hashlib
from dataclasses import dataclass, field, replace
from datetime import datetime, timezone
from typing import Any

from app.data.market_capabilities import FeedScope, MarketGroup, SessionId, Venue

KIS_REALTIME_SOURCE = "kis_realtime_websocket"
# REST snapshot fallback used when a market is fully closed (no pre/regular/after
# session). Deliberately DISTINCT from KIS_REALTIME_SOURCE so market-data health
# never marks a closed-market REST quote as live-buy eligible — it only keeps
# last-known prices fresh for valuation/display/history.
KIS_REST_SNAPSHOT_SOURCE = "kis_rest_snapshot"
# 국내 시간외 REST 보조 시세 (FHPST02300000 / FHPST02300400 / FHPST02310000).
# WebSocket 을 보조하거나 WebSocket 이 제공하지 않는 정보를 채우는 용도로만 쓰인다.
KIS_REST_OVERTIME_SOURCE = "kis_rest_overtime"

#: ``feed_scope`` 가 이 집합에 속하면 실시간 신규매수 적격 판정에 쓸 수 있다.
#: REST snapshot / HISTORICAL / UNKNOWN 은 절대 통과하지 못한다.
TRADEABLE_FEED_SCOPES = frozenset(
    {FeedScope.VENUE_SPECIFIC, FeedScope.UNIFIED, FeedScope.FREE_REALTIME}
)


@dataclass(frozen=True)
class FeedMetadata:
    """실시간 이벤트의 출처 식별 정보.

    왜 필요한가
    -----------
    이전에는 ``symbol`` 과 ``source`` 문자열만 있었다. 그래서

    * 같은 종목의 KRX 체결과 NXT 체결이 저장소에서 구분되지 않았고,
    * KRX+NXT 통합 피드와 venue 별 피드를 동시에 켜면 같은 체결이 이중 집계될 수 있었고,
    * 휴장 REST 스냅샷을 SQL 수준에서 배제할 방법이 없었고,
    * 어느 세션 데이터로 학습했는지 사후 확인이 불가능했다.

    기본값은 모두 ``UNKNOWN`` 이다. 기존 fixture 와 테스트는 그대로 동작하지만,
    ``UNKNOWN`` 은 :meth:`is_live_buy_eligible` 을 통과하지 못한다 (fail-closed).
    """

    market_group: MarketGroup | None = None
    #: KIS 거래소 코드 (국내는 ``KRX``/``NXT``, 미국은 ``NASD``/``NYSE``/``AMEX``).
    exchange: str = ""
    venue: Venue = Venue.UNKNOWN
    session: SessionId = SessionId.UNKNOWN
    currency: str = ""
    feed_scope: FeedScope = FeedScope.UNKNOWN
    #: 이 이벤트를 실어 온 TR ID (예: ``H0UNCNT0``, ``HDFSASP0``).
    tr_id: str = ""
    #: 구독에 사용한 ``tr_key``. 국내는 종목코드 그대로, 미국은 야간 ``D``+시장구분+종목 /
    #: 주간 ``R``+주간시장구분+종목 형식이다.
    subscription_key: str = ""
    #: 통합(consolidated) 피드에서 온 이벤트인지. 미국 무료 호가는 나스닥 마켓센터
    #: 단일 시장이므로 ``False`` 다 (NBBO 아님).
    is_consolidated: bool = False
    #: 이 피드로 관측한 가격에 근거해 주문 route 가 존재하는지.
    is_tradeable: bool = False
    #: 원본에 metadata 가 없어 symbol/timestamp 로 추정했는지.
    #: 추정된 행은 신규 진입과 high-trust 학습 표본에서 기본 제외된다.
    metadata_inferred: bool = False

    @property
    def is_known(self) -> bool:
        return (
            self.market_group is not None
            and self.venue is not Venue.UNKNOWN
            and self.session is not SessionId.UNKNOWN
            and self.feed_scope is not FeedScope.UNKNOWN
        )

    @property
    def stream_id(self) -> str:
        """이 이벤트가 속한 논리적 스트림의 안정적 식별자.

        분(minute) bar 의 identity 와 ``record_id`` 충돌 방지에 쓰인다. venue 별 bar 와
        통합 bar 는 서로 다른 ``stream_id`` 를 가지므로 **애초에 합산되지 않는다** —
        휴리스틱 dedup 없이 이중 집계를 구조적으로 막는다.
        """
        if not self.is_known and not self.tr_id:
            return ""
        group = self.market_group.value if self.market_group else "UNKNOWN"
        return f"{group}:{self.venue.value}:{self.feed_scope.value}:{self.tr_id or 'UNKNOWN'}"

    def is_live_buy_eligible(self) -> tuple[bool, tuple[str, ...]]:
        """실시간 신규매수 근거로 쓸 수 있는 출처인지 + 사유코드."""
        from app.data.market_capabilities import ReasonCode

        reasons: list[str] = []
        if self.market_group is None or self.venue is Venue.UNKNOWN:
            reasons.append(ReasonCode.VENUE_UNKNOWN.value)
        if self.session is SessionId.UNKNOWN:
            reasons.append(ReasonCode.MARKET_SESSION_UNKNOWN.value)
        if self.feed_scope is FeedScope.REST_SNAPSHOT:
            reasons.append(ReasonCode.REST_SNAPSHOT_ONLY.value)
        elif self.feed_scope not in TRADEABLE_FEED_SCOPES:
            reasons.append(ReasonCode.NON_TRADEABLE_FEED.value)
        if not self.is_tradeable:
            reasons.append(ReasonCode.NON_TRADEABLE_FEED.value)
        if self.metadata_inferred:
            reasons.append(ReasonCode.SESSION_MISMATCH.value)
        deduped = tuple(dict.fromkeys(reasons))
        return (not deduped, deduped)

    def to_row(self) -> dict[str, Any]:
        """저장소 컬럼 매핑."""
        return {
            "market_group": self.market_group.value if self.market_group else "",
            "exchange": self.exchange,
            "venue": self.venue.value,
            "session": self.session.value,
            "currency": self.currency,
            "feed_scope": self.feed_scope.value,
            "tr_id": self.tr_id,
            "subscription_key": self.subscription_key,
            "is_consolidated": int(bool(self.is_consolidated)),
            "is_tradeable": int(bool(self.is_tradeable)),
            "metadata_inferred": int(bool(self.metadata_inferred)),
            "stream_id": self.stream_id,
        }

    @classmethod
    def from_row(cls, row: Any) -> "FeedMetadata":
        """저장소 행에서 복원. 빠진 컬럼은 UNKNOWN 으로 둔다."""

        def get(key: str, default: Any = "") -> Any:
            try:
                value = row[key]
            except (KeyError, IndexError, TypeError):
                return default
            return default if value is None else value

        raw_group = str(get("market_group") or "")
        try:
            market_group = MarketGroup(raw_group) if raw_group else None
        except ValueError:
            market_group = None
        return cls(
            market_group=market_group,
            exchange=str(get("exchange") or ""),
            venue=_coerce_enum(Venue, get("venue"), Venue.UNKNOWN),
            session=_coerce_enum(SessionId, get("session"), SessionId.UNKNOWN),
            currency=str(get("currency") or ""),
            feed_scope=_coerce_enum(FeedScope, get("feed_scope"), FeedScope.UNKNOWN),
            tr_id=str(get("tr_id") or ""),
            subscription_key=str(get("subscription_key") or ""),
            is_consolidated=bool(int(get("is_consolidated", 0) or 0)),
            is_tradeable=bool(int(get("is_tradeable", 0) or 0)),
            metadata_inferred=bool(int(get("metadata_inferred", 0) or 0)),
        )


def _coerce_enum(enum_cls: Any, value: Any, default: Any) -> Any:
    try:
        return enum_cls(str(value))
    except ValueError:
        return default


UNKNOWN_FEED_METADATA = FeedMetadata()


def _stream_prefixed(meta: FeedMetadata, key: str) -> str:
    """``record_id`` 용 키에 스트림 identity 를 붙인다.

    metadata 가 없는(legacy/fixture) 이벤트는 **접두어를 붙이지 않는다.** 그래야 기존
    저장소 행의 ``record_id`` 와 정확히 같은 값이 유지되고, 마이그레이션 후 같은 체결이
    새 id 로 중복 삽입되지 않는다.
    """
    stream = meta.stream_id
    return f"{stream}|{key}" if stream else key


@dataclass(frozen=True)
class RealtimeTradeTick:
    """실시간 체결.

    ``exchange_timestamp`` 는 event time (거래소가 체결을 확정한 시각),
    ``received_at`` 은 ingestion time (우리 프로세스가 수신한 시각) 이다.
    point-in-time 피처는 항상 ``received_at`` 을 기준으로 잘라야 한다 — event time 이
    더 이르더라도 그 시점에 우리가 알 수는 없었기 때문이다.
    """

    symbol: str
    exchange_timestamp: datetime
    received_at: datetime
    source: str
    price: float
    volume: int
    trade_direction: str | None = None
    sequence_key: str | None = None
    raw_checksum: str | None = None
    latency_ms: float = 0.0
    meta: FeedMetadata = field(default_factory=FeedMetadata)

    @property
    def record_id(self) -> str:
        key = self.sequence_key or (
            f"{self.symbol}:{self.exchange_timestamp.isoformat()}:{self.price}:{self.volume}"
        )
        return hashlib.sha256(
            _stream_prefixed(self.meta, key).encode("utf-8")
        ).hexdigest()[:24]

    @property
    def dedup_key(self) -> str:
        """venue 를 제외한 물리적 체결 identity.

        통합 피드와 venue 별 피드가 같은 체결을 실어 올 때 두 이벤트는 같은 값을 갖는다.
        분 bar 는 ``stream_id`` 별로 따로 만들어지므로 집계에는 쓰이지 않고, 교차 스트림
        중복을 **관측·검증**하는 용도다.
        """
        return hashlib.sha256(
            f"{self.symbol}:{self.exchange_timestamp.isoformat()}:{self.price}:{self.volume}".encode()
        ).hexdigest()[:24]

    def with_meta(self, meta: FeedMetadata) -> "RealtimeTradeTick":
        return replace(self, meta=meta)


@dataclass(frozen=True)
class OrderbookLevel:
    bid_price: float
    bid_size: int
    ask_price: float
    ask_size: int


@dataclass(frozen=True)
class RealtimeOrderbookSnapshot:
    symbol: str
    exchange_timestamp: datetime
    received_at: datetime
    source: str
    levels: tuple[OrderbookLevel, ...]
    sequence_key: str | None = None
    raw_checksum: str | None = None
    latency_ms: float = 0.0
    meta: FeedMetadata = field(default_factory=FeedMetadata)

    @property
    def best_bid(self) -> float:
        return self.levels[0].bid_price if self.levels else 0.0

    @property
    def best_ask(self) -> float:
        return self.levels[0].ask_price if self.levels else 0.0

    @property
    def spread_bps(self) -> float:
        bid = self.best_bid
        ask = self.best_ask
        mid = (bid + ask) / 2 if bid > 0 and ask > 0 else 0.0
        return ((ask - bid) / mid) * 10_000 if mid > 0 and ask >= bid else 0.0

    @property
    def total_bid_volume(self) -> int:
        return sum(level.bid_size for level in self.levels)

    @property
    def total_ask_volume(self) -> int:
        return sum(level.ask_size for level in self.levels)

    @property
    def imbalance(self) -> float:
        total = self.total_bid_volume + self.total_ask_volume
        if total <= 0:
            return 0.0
        return (self.total_bid_volume - self.total_ask_volume) / total

    @property
    def depth_level_count(self) -> int:
        """실제로 채워진 호가 단계 수. 없는 단계를 만들어내지 않는다."""
        return sum(
            1
            for level in self.levels
            if level.bid_price > 0 or level.ask_price > 0
        )

    @property
    def record_id(self) -> str:
        key = self.sequence_key or (
            f"{self.symbol}:{self.exchange_timestamp.isoformat()}:{self.best_bid}:{self.best_ask}"
        )
        return hashlib.sha256(
            _stream_prefixed(self.meta, key).encode("utf-8")
        ).hexdigest()[:24]

    def with_meta(self, meta: FeedMetadata) -> "RealtimeOrderbookSnapshot":
        return replace(self, meta=meta)


@dataclass(frozen=True)
class RealtimeMinuteBar:
    """분 bar. **스트림별로** 만들어진다 (venue 별 bar 와 통합 bar 를 합산하지 않는다)."""

    symbol: str
    minute_start: datetime
    open: float
    high: float
    low: float
    close: float
    volume: int
    vwap: float
    trade_count: int
    spread_bps: float
    orderbook_imbalance: float
    liquidity_score: float
    volatility: float
    last_update_age_ms: float
    source_record_ids: tuple[str, ...] = ()
    meta: FeedMetadata = field(default_factory=FeedMetadata)

    @property
    def stream_id(self) -> str:
        return self.meta.stream_id


@dataclass(frozen=True)
class MarketDataHealth:
    """시장 데이터 건전성. symbol 뿐 아니라 market/venue/session/feed 별로 계산된다."""

    symbol: str
    checked_at: datetime
    quote_count: int
    orderbook_count: int
    latest_tick_at: datetime | None
    latest_orderbook_at: datetime | None
    max_quote_age_ms: int
    max_orderbook_age_ms: int
    source: str
    source_quality_score: float
    ok_for_live_buy: bool
    reason_codes: tuple[str, ...] = field(default_factory=tuple)
    market_group: str = ""
    venue: str = ""
    session: str = ""
    feed_scope: str = ""
    depth_level_count: int = 0
    is_consolidated: bool = False


def checksum(raw: str) -> str:
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


def aware_now() -> datetime:
    return datetime.now(timezone.utc)


def to_jsonable(value: Any) -> Any:
    if isinstance(value, (MarketGroup, Venue, SessionId, FeedScope)):
        return value.value
    if hasattr(value, "__dataclass_fields__"):
        return {key: to_jsonable(getattr(value, key)) for key in value.__dataclass_fields__}
    if isinstance(value, datetime):
        return value.isoformat()
    if isinstance(value, tuple):
        return [to_jsonable(item) for item in value]
    if isinstance(value, list):
        return [to_jsonable(item) for item in value]
    if isinstance(value, dict):
        return {str(key): to_jsonable(item) for key, item in value.items()}
    return value
