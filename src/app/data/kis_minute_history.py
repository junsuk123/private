"""KIS minute-history adapter and persistent repository bridge."""

from __future__ import annotations

from datetime import datetime, time, timezone
from typing import Any, Iterable
from zoneinfo import ZoneInfo

from app.data.market_capabilities import FeedScope, MarketGroup, SessionId, Venue
from app.data.realtime_types import FeedMetadata, RealtimeMinuteBar


_SEOUL = ZoneInfo("Asia/Seoul")
_NEW_YORK = ZoneInfo("America/New_York")


def _number(row: dict[str, Any], *keys: str) -> float:
    for key in keys:
        value = row.get(key)
        if value not in (None, ""):
            try:
                return float(str(value).replace(",", ""))
            except ValueError:
                continue
    return 0.0


def _text(row: dict[str, Any], *keys: str) -> str:
    for key in keys:
        value = str(row.get(key) or "").strip()
        if value:
            return value
    return ""


def _parse_rows(
    rows: Iterable[dict[str, Any]],
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
    for raw in rows:
        row = dict(raw or {})
        day = _text(row, "stck_bsop_date", "xymd", "date", "bsop_date")
        clock = _text(row, "stck_cntg_hour", "xhms", "time", "bsop_hour")
        if not day:
            day = default_day.astimezone(zone).strftime("%Y%m%d")
        digits = "".join(character for character in clock if character.isdigit())
        if len(digits) == 4:
            digits += "00"
        if len(day) != 8 or len(digits) < 6:
            continue
        try:
            local = datetime.strptime(day + digits[:6], "%Y%m%d%H%M%S").replace(
                tzinfo=zone
            )
        except ValueError:
            continue
        minute_start = local.astimezone(timezone.utc).replace(second=0, microsecond=0)
        open_price = _number(row, "stck_oprc", "open", "ovrs_nmix_oprc")
        high = _number(row, "stck_hgpr", "high", "ovrs_nmix_hgpr")
        low = _number(row, "stck_lwpr", "low", "ovrs_nmix_lwpr")
        close = _number(row, "stck_prpr", "last", "close", "ovrs_nmix_prpr")
        if (
            min(open_price, high, low, close) <= 0
            or high < max(open_price, close, low)
            or low > min(open_price, close, high)
        ):
            continue
        volume = max(0, int(_number(row, "cntg_vol", "evol", "volume", "acml_vol")))
        meta = FeedMetadata(
            market_group=market,
            exchange=str(exchange),
            venue=venue,
            session=SessionId.UNKNOWN,
            currency="KRW" if market is MarketGroup.KR else "USD",
            feed_scope=FeedScope.HISTORICAL,
            tr_id=str(tr_id),
            is_consolidated=False,
            is_tradeable=False,
            metadata_inferred=False,
        )
        parsed[minute_start] = RealtimeMinuteBar(
            symbol=str(symbol).strip().upper(),
            minute_start=minute_start,
            open=open_price,
            high=high,
            low=low,
            close=close,
            volume=volume,
            vwap=close,
            trade_count=max(0, int(_number(row, "cntg_cnt", "trade_count"))),
            spread_bps=0.0,
            orderbook_imbalance=0.0,
            liquidity_score=0.0,
            volatility=0.0,
            last_update_age_ms=0.0,
            meta=meta,
        )
    return tuple(parsed[key] for key in sorted(parsed))


def expected_session_bar(stamp: datetime, market: str) -> bool:
    group = str(market or "").strip().upper()
    zone = _SEOUL if group in {"KR", "KRX", "DOMESTIC"} else _NEW_YORK
    local = stamp.astimezone(zone)
    if local.weekday() >= 5:
        return False
    current = local.time().replace(tzinfo=None)
    if group in {"KR", "KRX", "DOMESTIC"}:
        return time(9, 0) <= current < time(15, 30)
    return time(9, 30) <= current < time(16, 0)


class PersistentBarRepository:
    def __init__(self, store: Any) -> None:
        self.store = store

    def bars_for_requirement(self, requirement: Any, *, as_of: datetime) -> tuple[Any, ...]:
        from datetime import timedelta

        since = as_of - timedelta(
            minutes=max(240, int(requirement.preferred_observations) * 20)
        )
        reconciled = getattr(self.store, "reconciled_minute_bars", None)
        if callable(reconciled):
            return tuple(
                reconciled(
                    requirement.symbol,
                    since,
                    limit=requirement.preferred_observations,
                    market=requirement.market,
                )
            )
        return tuple(
            self.store.recent_minute_bars(
                requirement.symbol, since, limit=requirement.preferred_observations
            )
        )

    def merge_bars(self, bars: Iterable[RealtimeMinuteBar]) -> int:
        return int(self.store.save_minute_bars(tuple(bars)))


class KisMinuteHistoryProvider:
    def __init__(self, client: Any | None = None) -> None:
        self._client = client

    def _get_client(self) -> Any:
        if self._client is None:
            from app.execution.kis_auth import build_kis_client

            self._client = build_kis_client(enabled=False)
        return self._client

    def fetch(self, requirement: Any, missing: tuple[Any, ...]) -> tuple[RealtimeMinuteBar, ...]:
        if not missing:
            return ()
        client = self._get_client()
        market = str(requirement.market or "").strip().upper()
        if market in {"KR", "KRX", "DOMESTIC"}:
            tr_id = "FHKST03010200"
            response = client._get(  # noqa: SLF001 - same package's authenticated read adapter.
                "/uapi/domestic-stock/v1/quotations/inquire-time-itemchartprice",
                tr_id=tr_id,
                params={
                    "FID_ETC_CLS_CODE": "",
                    "FID_COND_MRKT_DIV_CODE": "J",
                    "FID_INPUT_ISCD": requirement.symbol,
                    "FID_INPUT_HOUR_1": missing[-1].end.astimezone(_SEOUL).strftime("%H%M%S"),
                    "FID_PW_DATA_INCU_YN": "Y",
                },
            )
            client._ensure_success(response, "KIS domestic minute history failed")  # noqa: SLF001
            rows = response.get("output2") or response.get("output") or ()
            return _parse_rows(
                rows if isinstance(rows, list) else (rows,),
                symbol=requirement.symbol,
                market=MarketGroup.KR,
                exchange="KRX",
                venue=Venue.KRX,
                tr_id=tr_id,
                default_day=missing[-1].end,
                zone=_SEOUL,
            )

        tr_id = "HHDFS76950200"
        response = client._get(  # noqa: SLF001
            "/uapi/overseas-price/v1/quotations/inquire-time-itemchartprice",
            tr_id=tr_id,
            params={
                "AUTH": "",
                "EXCD": "NAS",
                "SYMB": requirement.symbol,
                "NMIN": str(requirement.timeframe_minutes),
                "PINC": "1",
                "NEXT": "",
                "NREC": str(min(120, max(1, requirement.preferred_observations))),
                "FILL": "Y",
                "KEY": "",
            },
        )
        client._ensure_success(response, "KIS overseas minute history failed")  # noqa: SLF001
        rows = response.get("output2") or response.get("output") or ()
        return _parse_rows(
            rows if isinstance(rows, list) else (rows,),
            symbol=requirement.symbol,
            market=MarketGroup.US,
            exchange="NAS",
            venue=Venue.NASDAQ,
            tr_id=tr_id,
            default_day=missing[-1].end,
            zone=_NEW_YORK,
        )
