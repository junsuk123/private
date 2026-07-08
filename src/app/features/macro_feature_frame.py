"""Market-level (macro) feature frame for MacroMarketReasoner.

Derives market context — index trend, breadth, realized volatility, total
trading value, and per-sector strength — from the realtime data the system
already collects, WITHOUT a dedicated index feed: it aggregates across the
tracked symbol universe (candidates + holdings). Pure and NaN-safe; when there
is insufficient realtime data (e.g. off-hours, no ticks) the fields are ``None``
and the macro reasoner falls back to its conservative regime.

No graph imports here (avoids a cycle); the caller maps the frame into a
``MacroReasoningInput`` via :meth:`MacroFeatureFrame.as_macro_kwargs`.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from statistics import fmean, pstdev
from typing import Any, Mapping, Sequence


@dataclass(frozen=True)
class MacroFeatureFrame:
    timestamp: datetime
    index_trend: float | None
    market_breadth: float | None
    market_volatility: float | None
    total_trading_value: float | None
    per_symbol_return: Mapping[str, float]
    sector_snapshots: Mapping[str, Mapping[str, float]]
    sector_of: Mapping[str, str]
    symbol_count: int

    def as_macro_kwargs(self) -> dict[str, Any]:
        """Keyword args to splat into a MacroReasoningInput."""
        index_snapshots = {"COMPOSITE": {"trend": self.index_trend}} if self.index_trend is not None else {}
        return {
            "index_snapshots": index_snapshots,
            "sector_snapshots": dict(self.sector_snapshots),
            "market_breadth": self.market_breadth,
            "market_volatility": self.market_volatility,
            "total_trading_value": self.total_trading_value,
        }


def _finite(values: Sequence[float]) -> list[float]:
    out: list[float] = []
    for v in values:
        try:
            f = float(v)
        except (TypeError, ValueError):
            continue
        if f == f:  # NaN-safe
            out.append(f)
    return out


def _rolling_return(closes: list[float], window: int) -> float | None:
    if len(closes) < window + 1 or closes[-window - 1] == 0:
        return None
    return closes[-1] / closes[-window - 1] - 1.0


def _realized_vol(closes: list[float], window: int) -> float | None:
    if len(closes) < window + 1:
        return None
    tail = closes[-window - 1 :]
    rets = [tail[i] / tail[i - 1] - 1.0 for i in range(1, len(tail)) if tail[i - 1]]
    return pstdev(rets) if len(rets) >= 2 else None


def build_macro_feature_frame(
    closes_by_symbol: Mapping[str, Sequence[float]],
    *,
    timestamp: datetime | None = None,
    volumes_by_symbol: Mapping[str, Sequence[float]] | None = None,
    sector_of: Mapping[str, str] | None = None,
    trend_window: int = 5,
    vol_window: int = 20,
) -> MacroFeatureFrame:
    timestamp = timestamp or datetime.now(timezone.utc)
    sector_of = {str(k): str(v) for k, v in (sector_of or {}).items() if v and str(v).lower() != "unknown"}
    volumes_by_symbol = volumes_by_symbol or {}

    per_symbol_return: dict[str, float] = {}
    vols: list[float] = []
    total_value = 0.0
    have_value = False
    for symbol, raw in closes_by_symbol.items():
        closes = _finite(raw)
        if not closes:
            continue
        r = _rolling_return(closes, trend_window)
        if r is not None:
            per_symbol_return[str(symbol)] = r
        v = _realized_vol(closes, vol_window)
        if v is not None:
            vols.append(v)
        vseq = _finite(volumes_by_symbol.get(symbol, ()))
        if vseq:
            total_value += closes[-1] * sum(vseq)
            have_value = True

    index_trend = fmean(per_symbol_return.values()) if per_symbol_return else None
    breadth = (sum(1 for r in per_symbol_return.values() if r > 0) / len(per_symbol_return)) if per_symbol_return else None
    market_vol = fmean(vols) if vols else None

    # Per-sector aggregation (mean return of the sector's symbols).
    sector_returns: dict[str, list[float]] = {}
    for symbol, r in per_symbol_return.items():
        sector = sector_of.get(symbol)
        if sector:
            sector_returns.setdefault(sector, []).append(r)
    sector_snapshots = {
        sector: {"strength": fmean(rs), "volume_change": 0.0}
        for sector, rs in sector_returns.items()
    }

    return MacroFeatureFrame(
        timestamp=timestamp,
        index_trend=index_trend,
        market_breadth=breadth,
        market_volatility=market_vol,
        total_trading_value=total_value if have_value else None,
        per_symbol_return=per_symbol_return,
        sector_snapshots=sector_snapshots,
        sector_of=sector_of,
        symbol_count=len(per_symbol_return),
    )


def _series_from_store(store: Any, symbol: str, since: datetime) -> tuple[list[float], list[float]]:
    """Best-available realtime price/volume series for a symbol, market-agnostic.

    Prefers trade ticks (KR websocket); falls back to orderbook mid-prices
    (used by REST-polled US quotes and any market without a tick stream) — the
    same fallback the live feature frame uses. Either market's live data counts.
    """
    try:
        ticks = store.recent_ticks(symbol, since)
    except Exception:  # noqa: BLE001
        ticks = ()
    prices = [float(t.price) for t in ticks if getattr(t, "price", 0) and float(t.price) > 0]
    if prices:
        volumes = [float(max(0, getattr(t, "volume", 0) or 0)) for t in ticks if getattr(t, "price", 0) and float(t.price) > 0]
        return prices, volumes
    # Fallback: orderbook mid-price series (US REST-polled quotes, etc.).
    try:
        books = store.recent_orderbooks(symbol, since)
    except Exception:  # noqa: BLE001
        books = ()
    mids: list[float] = []
    vols: list[float] = []
    for b in books:
        bid = float(getattr(b, "best_bid", 0.0) or 0.0)
        ask = float(getattr(b, "best_ask", 0.0) or 0.0)
        if bid > 0 and ask > 0:
            mids.append((bid + ask) / 2.0)
            vols.append(float((getattr(b, "total_bid_volume", 0.0) or 0.0) + (getattr(b, "total_ask_volume", 0.0) or 0.0)))
    return mids, vols


def macro_feature_frame_from_store(
    store: Any,
    symbols: Sequence[str],
    *,
    now: datetime | None = None,
    lookback_minutes: int = 10,
    sector_of: Mapping[str, str] | None = None,
) -> MacroFeatureFrame:
    """Build a macro frame from realtime data for EITHER market.

    Uses the best-available series per symbol (ticks, else orderbook mids), so
    KR (websocket ticks) and US (REST-polled quotes/orderbooks) both contribute.
    Symbols with no realtime data are omitted; if none have data the frame is
    all-``None`` (conservative macro fallback). Best-effort and error-isolated.
    """
    now = now or datetime.now(timezone.utc)
    since = now - timedelta(minutes=max(1, lookback_minutes))
    closes: dict[str, list[float]] = {}
    volumes: dict[str, list[float]] = {}
    for symbol in dict.fromkeys(str(s) for s in symbols if str(s)):
        prices, vols = _series_from_store(store, symbol, since)
        if prices:
            closes[symbol] = prices
            volumes[symbol] = vols
    return build_macro_feature_frame(
        closes, timestamp=now, volumes_by_symbol=volumes, sector_of=sector_of,
    )
