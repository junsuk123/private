"""Official KIS minute-history adapter used by demand-driven candidate warmup."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
import hashlib
from typing import Any, Mapping, Sequence
from zoneinfo import ZoneInfo

from app.data.market_capabilities import FeedScope, MarketGroup, SessionId, Venue, default_service
from app.data.minute_bar_warmup import MissingRange, ResolvedHistoryRequirement, retry_with_backoff
from app.data.realtime_types import FeedMetadata, RealtimeMinuteBar
from app.execution.kis_auth import build_kis_client


_SEOUL = ZoneInfo("Asia/Seoul")
_NEW_YORK = ZoneInfo("America/New_York")


class PersistentBarRepository:
    """Normalized access to the existing realtime SQLite store."""

    def __init__(self, store: Any) -> None:
        self.store = store

    def bars_for_requirement(
        self, requirement: ResolvedHistoryRequirement, *, as_of: datetime
    ) -> Sequence[RealtimeMinuteBar]:
        # Calendar minutes are not observations. A bounded multi-session horizon
        # gives the repository enough room without loading the whole DB.
        horizon = max(240, requirement.preferred_observations * 20)
        return self.store.reconciled_minute_bars(
            requirement.symbol,
            as_of - timedelta(minutes=horizon),
            limit=requirement.preferred_observations,
            market=requirement.market,
        )

    def merge_bars(self, bars: Sequence[RealtimeMinuteBar]) -> int:
        valid = tuple(bar for bar in bars if _valid_bar(bar))
        return int(self.store.save_minute_bars(valid)) if valid else 0


class KisMinuteHistoryProvider:
    """Fetch only requested KIS minute intervals; responses are filtered exactly."""

    def __init__(self) -> None:
        self.client = build_kis_client(enabled=True)

    def fetch(
        self,
        requirement: ResolvedHistoryRequirement,
        missing: tuple[MissingRange, ...],
    ) -> Sequence[RealtimeMinuteBar]:
        collected: dict[datetime, RealtimeMinuteBar] = {}
        for interval in missing:
            rows = retry_with_backoff(
                lambda interval=interval: self._fetch_interval(requirement, interval)
            )
            for bar in rows:
                if interval.start <= bar.minute_start < interval.end:
                    collected[bar.minute_start] = bar
        return tuple(collected[key] for key in sorted(collected))

    def _fetch_interval(
        self, requirement: ResolvedHistoryRequirement, interval: MissingRange
    ) -> Sequence[RealtimeMinuteBar]:
        return (
            self._domestic(requirement, interval)
            if requirement.market in {"KR", "KRX"}
            else self._overseas(requirement, interval)
        )

    def _domestic(
        self, requirement: ResolvedHistoryRequirement, interval: MissingRange
    ) -> Sequence[RealtimeMinuteBar]:
        local_end = interval.end.astimezone(_SEOUL)
        payload = self.client._get(
            "/uapi/domestic-stock/v1/quotations/inquire-time-itemchartprice",
            "FHKST03010200",
            {
                "FID_ETC_CLS_CODE": "",
                "FID_COND_MRKT_DIV_CODE": "J",
                "FID_INPUT_ISCD": requirement.symbol,
                "FID_INPUT_HOUR_1": local_end.strftime("%H%M%S"),
                "FID_PW_DATA_INCU_YN": "Y",
            },
        )
        _raise_api_error(payload)
        return _parse_rows(
            _response_rows(payload),
            symbol=requirement.symbol,
            market=MarketGroup.KR,
            exchange="KRX",
            venue=Venue.KRX,
            tr_id="FHKST03010200",
            default_day=local_end,
            zone=_SEOUL,
        )

    def _overseas(
        self, requirement: ResolvedHistoryRequirement, interval: MissingRange
    ) -> Sequence[RealtimeMinuteBar]:
        from app.trading.us_realtime_bridge import _exchange_code

        exchange = _exchange_code(requirement.symbol)
        venue = {"NAS": Venue.NASDAQ, "NYS": Venue.NYSE, "AMS": Venue.AMEX}.get(
            exchange, Venue.NASDAQ
        )
        payload = self.client._get(
            "/uapi/overseas-price/v1/quotations/inquire-time-itemchartprice",
            "HHDFS76950200",
            {
                "AUTH": "",
                "EXCD": exchange,
                "SYMB": requirement.symbol,
                "NMIN": str(requirement.timeframe_minutes),
                "PINC": "1",
                "NEXT": "",
                # KIS exposes at most 120 rows and has no end-time parameter on
                # this endpoint. Always request the bounded page, then filter to
                # the exact missing interval locally; a tiny NREC cannot reach a
                # prior-session gap even when that gap is within the provider page.
                "NREC": "120",
                "FILL": "",
                "KEYB": "",
            },
        )
        _raise_api_error(payload)
        return _parse_rows(
            _response_rows(payload),
            symbol=requirement.symbol,
            market=MarketGroup.US,
            exchange=exchange,
            venue=venue,
            tr_id="HHDFS76950200",
            default_day=interval.end.astimezone(_NEW_YORK),
            zone=_NEW_YORK,
        )


def expected_session_bar(stamp: datetime, market: str) -> bool:
    group = MarketGroup.KR if market.upper() in {"KR", "KRX"} else MarketGroup.US
    return default_service().data_available(group, stamp)


def _response_rows(payload: Mapping[str, Any]) -> tuple[Mapping[str, Any], ...]:
    rows: list[Mapping[str, Any]] = []
    for key in ("output2", "output1", "output"):
        value = payload.get(key)
        if isinstance(value, list):
            rows.extend(item for item in value if isinstance(item, Mapping))
    return tuple(rows)


def _raise_api_error(payload: Mapping[str, Any]) -> None:
    code = str(payload.get("rt_cd", "0"))
    if code not in {"", "0"}:
        raise RuntimeError(f"KIS minute history rt_cd={code}: {payload.get('msg1') or payload.get('msg_cd')}")


def _parse_rows(
    rows: Sequence[Mapping[str, Any]],
    *,
    symbol: str,
    market: MarketGroup,
    exchange: str,
    venue: Venue,
    tr_id: str,
    default_day: datetime,
    zone: ZoneInfo,
) -> tuple[RealtimeMinuteBar, ...]:
    parsed: dict[datetime, RealtimeMinuteBar] = {}
    for row in rows:
        stamp = _row_timestamp(row, default_day=default_day, zone=zone)
        if stamp is None:
            continue
        open_ = _number(row, "stck_oprc", "open", "optn")
        high = _number(row, "stck_hgpr", "high", "hgpr")
        low = _number(row, "stck_lwpr", "low", "lwpr")
        close = _number(row, "stck_prpr", "last", "close", "clos")
        volume = int(max(0.0, _number(row, "cntg_vol", "evol", "volume", "tvol")))
        if min(open_, high, low, close) <= 0 or high < max(open_, close) or low > min(open_, close):
            continue
        capability = default_service().primary_capability(market, stamp)
        session = capability.session
        if session in {SessionId.KR_CLOSED, SessionId.US_CLOSED}:
            session = SessionId.UNKNOWN
        meta = FeedMetadata(
            market_group=market,
            exchange=exchange,
            venue=venue,
            session=session,
            currency="KRW" if market is MarketGroup.KR else "USD",
            feed_scope=FeedScope.HISTORICAL,
            tr_id=tr_id,
            subscription_key=symbol,
            is_consolidated=False,
            is_tradeable=False,
            metadata_inferred=False,
        )
        identity = hashlib.sha256(f"{tr_id}:{symbol}:{stamp.isoformat()}".encode()).hexdigest()[:24]
        parsed[stamp] = RealtimeMinuteBar(
            symbol=symbol,
            minute_start=stamp,
            open=open_, high=high, low=low, close=close,
            volume=volume,
            vwap=_number(row, "acml_tr_pbmn", "vwap") or close,
            trade_count=int(max(0.0, _number(row, "cntg_cnt", "trade_count"))),
            spread_bps=0.0,
            orderbook_imbalance=0.0,
            liquidity_score=0.0,
            volatility=(high - low) / close if close else 0.0,
            last_update_age_ms=max(0.0, (datetime.now(timezone.utc) - stamp).total_seconds() * 1000.0),
            source_record_ids=(identity,),
            meta=meta,
        )
    return tuple(parsed[key] for key in sorted(parsed))


def _row_timestamp(row: Mapping[str, Any], *, default_day: datetime, zone: ZoneInfo) -> datetime | None:
    raw_day = str(row.get("stck_bsop_date") or row.get("xymd") or row.get("date") or default_day.strftime("%Y%m%d"))
    raw_time = str(row.get("stck_cntg_hour") or row.get("xhms") or row.get("time") or "").zfill(6)
    digits_day = "".join(ch for ch in raw_day if ch.isdigit())[:8]
    digits_time = "".join(ch for ch in raw_time if ch.isdigit())[:6]
    if len(digits_day) != 8 or len(digits_time) < 4:
        return None
    digits_time = digits_time.ljust(6, "0")
    try:
        local = datetime.strptime(digits_day + digits_time, "%Y%m%d%H%M%S").replace(tzinfo=zone)
    except ValueError:
        return None
    return local.astimezone(timezone.utc).replace(second=0, microsecond=0)


def _number(row: Mapping[str, Any], *keys: str) -> float:
    for key in keys:
        value = row.get(key)
        if value not in (None, ""):
            try:
                return float(str(value).replace(",", ""))
            except (TypeError, ValueError):
                continue
    return 0.0


def _valid_bar(bar: RealtimeMinuteBar) -> bool:
    return (
        bar.minute_start.tzinfo is not None
        and min(bar.open, bar.high, bar.low, bar.close) > 0
        and bar.high >= max(bar.open, bar.close)
        and bar.low <= min(bar.open, bar.close)
        and bar.volume >= 0
    )
