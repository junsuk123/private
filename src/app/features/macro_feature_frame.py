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
    # --- Regime-shape channels --------------------------------------------------
    # Volatility magnitude alone cannot say WHY a market is volatile. These are the
    # cheapest additional statistics that separate a one-sided repricing from an
    # overshoot from an impaired market, all computed from the same realtime series
    # already loaded above. ``None`` wherever the data cannot support them.
    breadth_momentum: float | None = None
    cross_sectional_dispersion: float | None = None
    average_market_correlation: float | None = None
    volatility_of_volatility: float | None = None
    jump_ratio: float | None = None
    per_symbol_residual_return: Mapping[str, float] = field(default_factory=dict)
    # Same residual over a longer window. A relative-strength thesis needs BOTH to
    # be positive: one window alone cannot distinguish persistent idiosyncratic
    # strength from a single-window bounce.
    per_symbol_residual_return_long: Mapping[str, float] = field(default_factory=dict)
    per_symbol_market_beta: Mapping[str, float] = field(default_factory=dict)
    sector_returns: Mapping[str, float] = field(default_factory=dict)

    def as_macro_kwargs(self) -> dict[str, Any]:
        """Keyword args to splat into a MacroReasoningInput."""
        index_snapshots = {"COMPOSITE": {"trend": self.index_trend}} if self.index_trend is not None else {}
        return {
            "index_snapshots": index_snapshots,
            "sector_snapshots": dict(self.sector_snapshots),
            "market_breadth": self.market_breadth,
            "market_volatility": self.market_volatility,
            "total_trading_value": self.total_trading_value,
            "breadth_momentum": self.breadth_momentum,
            "cross_sectional_dispersion": self.cross_sectional_dispersion,
            "average_market_correlation": self.average_market_correlation,
            "volatility_of_volatility": self.volatility_of_volatility,
            "jump_ratio": self.jump_ratio,
        }

    def as_change_point_channels(self) -> dict[str, float]:
        """The market-state vector handed to the change-point detector.

        Only measured channels are included: an imputed zero would look to the
        detector like a real, stable observation.
        """
        channels = {
            "index_return": self.index_trend,
            "market_volatility": self.market_volatility,
            "market_breadth": self.market_breadth,
            "cross_sectional_dispersion": self.cross_sectional_dispersion,
            "average_correlation": self.average_market_correlation,
        }
        return {name: float(value) for name, value in channels.items() if value is not None}


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


def _return_series(closes: list[float]) -> list[float]:
    return [
        closes[index] / closes[index - 1] - 1.0
        for index in range(1, len(closes))
        if closes[index - 1]
    ]


def _correlation(left: Sequence[float], right: Sequence[float]) -> float | None:
    size = min(len(left), len(right))
    if size < 3:
        return None
    a = list(left)[-size:]
    b = list(right)[-size:]
    mean_a, mean_b = fmean(a), fmean(b)
    cov = sum((x - mean_a) * (y - mean_b) for x, y in zip(a, b))
    var_a = sum((x - mean_a) ** 2 for x in a)
    var_b = sum((y - mean_b) ** 2 for y in b)
    if var_a <= 0 or var_b <= 0:
        return None
    return cov / ((var_a * var_b) ** 0.5)


def _jump_ratio(returns: Sequence[float]) -> float | None:
    """Share of total variation contributed by the single largest move.

    A crude but honest jump proxy: continuous diffusion spreads variation across
    many observations, a jump concentrates it in one. Bipower variation would be
    better but needs far more observations than a ten-minute realtime window has.
    """
    values = [abs(float(value)) for value in returns if value == value]
    if len(values) < 4:
        return None
    total = sum(value**2 for value in values)
    if total <= 0:
        return None
    return max(value**2 for value in values) / total


def build_macro_feature_frame(
    closes_by_symbol: Mapping[str, Sequence[float]],
    *,
    timestamp: datetime | None = None,
    volumes_by_symbol: Mapping[str, Sequence[float]] | None = None,
    sector_of: Mapping[str, str] | None = None,
    trend_window: int = 5,
    vol_window: int = 20,
    long_trend_window: int = 30,
) -> MacroFeatureFrame:
    timestamp = timestamp or datetime.now(timezone.utc)
    sector_of = {str(k): str(v) for k, v in (sector_of or {}).items() if v and str(v).lower() != "unknown"}
    volumes_by_symbol = volumes_by_symbol or {}

    per_symbol_return: dict[str, float] = {}
    per_symbol_long_return: dict[str, float] = {}
    per_symbol_returns_series: dict[str, list[float]] = {}
    per_symbol_previous_return: dict[str, float] = {}
    vols: list[float] = []
    recent_vols: list[float] = []
    total_value = 0.0
    have_value = False
    for symbol, raw in closes_by_symbol.items():
        closes = _finite(raw)
        if not closes:
            continue
        r = _rolling_return(closes, trend_window)
        if r is not None:
            per_symbol_return[str(symbol)] = r
        long_r = _rolling_return(closes, long_trend_window)
        if long_r is not None:
            per_symbol_long_return[str(symbol)] = long_r
        # Breadth momentum needs the same statistic one window earlier, computed
        # from the same series so the comparison is apples to apples.
        if len(closes) > trend_window + 1:
            previous = _rolling_return(closes[:-1], trend_window)
            if previous is not None:
                per_symbol_previous_return[str(symbol)] = previous
        series = _return_series(closes)
        if len(series) >= 2:
            per_symbol_returns_series[str(symbol)] = series
        v = _realized_vol(closes, vol_window)
        if v is not None:
            vols.append(v)
        short_v = _realized_vol(closes, max(3, vol_window // 4))
        if short_v is not None:
            recent_vols.append(short_v)
        vseq = _finite(volumes_by_symbol.get(symbol, ()))
        if vseq:
            total_value += closes[-1] * sum(vseq)
            have_value = True

    index_trend = fmean(per_symbol_return.values()) if per_symbol_return else None
    breadth = (sum(1 for r in per_symbol_return.values() if r > 0) / len(per_symbol_return)) if per_symbol_return else None
    market_vol = fmean(vols) if vols else None

    previous_breadth = (
        sum(1 for r in per_symbol_previous_return.values() if r > 0)
        / len(per_symbol_previous_return)
        if per_symbol_previous_return
        else None
    )
    breadth_momentum = (
        breadth - previous_breadth
        if breadth is not None and previous_breadth is not None
        else None
    )
    dispersion = (
        pstdev(per_symbol_return.values()) if len(per_symbol_return) >= 2 else None
    )
    # Vol-of-vol: how unstable the volatility itself is. A stable-but-high
    # volatility is a trend; an unstable one is a market searching for a price.
    volatility_of_volatility = pstdev(recent_vols) if len(recent_vols) >= 2 else None

    # Market factor = equal-weighted return series across the tracked universe.
    market_series: list[float] = []
    if per_symbol_returns_series:
        length = min(len(series) for series in per_symbol_returns_series.values())
        if length >= 3:
            market_series = [
                fmean([series[-length:][index] for series in per_symbol_returns_series.values()])
                for index in range(length)
            ]
    correlations = [
        value
        for series in per_symbol_returns_series.values()
        if (value := _correlation(series, market_series)) is not None
    ]
    average_correlation = fmean(correlations) if correlations else None
    jump_ratio = _jump_ratio(market_series) if market_series else None

    # Per-sector aggregation (mean return of the sector's symbols).
    sector_return_lists: dict[str, list[float]] = {}
    for symbol, r in per_symbol_return.items():
        sector = sector_of.get(symbol)
        if sector:
            sector_return_lists.setdefault(sector, []).append(r)
    sector_returns = {sector: fmean(rs) for sector, rs in sector_return_lists.items()}
    sector_snapshots = {
        sector: {"strength": strength, "volume_change": 0.0}
        for sector, strength in sector_returns.items()
    }

    # Residual return: what is left of a name's move after the market and its
    # sector are removed. This is what a relative-strength thesis should rank on;
    # ranking raw return in an index-dominated tape ranks market beta instead.
    betas: dict[str, float] = {}
    residuals: dict[str, float] = {}
    for symbol, r in per_symbol_return.items():
        beta = 1.0
        series = per_symbol_returns_series.get(symbol)
        if series and market_series:
            estimated = _beta(series, market_series)
            if estimated is not None:
                beta = estimated
        betas[symbol] = beta
        residual = r - beta * (index_trend or 0.0)
        sector = sector_of.get(symbol)
        if sector and sector in sector_returns:
            # Sector excess over the market, so the market term is not counted twice.
            sector_excess = sector_returns[sector] - (index_trend or 0.0)
            residual -= sector_excess
        residuals[symbol] = residual

    long_index_trend = (
        fmean(per_symbol_long_return.values()) if per_symbol_long_return else None
    )
    long_sector_lists: dict[str, list[float]] = {}
    for symbol, r in per_symbol_long_return.items():
        sector = sector_of.get(symbol)
        if sector:
            long_sector_lists.setdefault(sector, []).append(r)
    long_sector_returns = {sector: fmean(rs) for sector, rs in long_sector_lists.items()}
    long_residuals: dict[str, float] = {}
    for symbol, r in per_symbol_long_return.items():
        beta = betas.get(symbol, 1.0)
        residual = r - beta * (long_index_trend or 0.0)
        sector = sector_of.get(symbol)
        if sector and sector in long_sector_returns:
            residual -= long_sector_returns[sector] - (long_index_trend or 0.0)
        long_residuals[symbol] = residual

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
        breadth_momentum=breadth_momentum,
        cross_sectional_dispersion=dispersion,
        average_market_correlation=average_correlation,
        volatility_of_volatility=volatility_of_volatility,
        jump_ratio=jump_ratio,
        per_symbol_residual_return=residuals,
        per_symbol_residual_return_long=long_residuals,
        per_symbol_market_beta=betas,
        sector_returns=sector_returns,
    )


def _beta(series: Sequence[float], market: Sequence[float]) -> float | None:
    size = min(len(series), len(market))
    if size < 3:
        return None
    a = list(series)[-size:]
    b = list(market)[-size:]
    mean_a, mean_b = fmean(a), fmean(b)
    cov = sum((x - mean_a) * (y - mean_b) for x, y in zip(a, b))
    var_b = sum((y - mean_b) ** 2 for y in b)
    if var_b <= 0:
        return None
    # Clamped: an unstable ten-minute regression can produce absurd betas, and an
    # absurd beta silently turns the residual into noise.
    return max(-3.0, min(3.0, cov / var_b))


def _series_from_store(store: Any, symbol: str, since: datetime) -> tuple[list[float], list[float]]:
    """Best-available realtime price/volume series for a symbol, market-agnostic.

    Prefers trade ticks (KR websocket); falls back to orderbook mid-prices
    (used by REST-polled US quotes and any market without a tick stream) — the
    same fallback the live feature frame uses. Either market's live data counts.
    """
    # Macro windows are minute-scale. Prefer completed bars so a five-period
    # trend means five minutes, not five high-frequency websocket ticks.
    try:
        bars = store.recent_minute_bars(symbol, since, limit=120)
    except Exception:  # noqa: BLE001
        bars = ()
    valid_bars = tuple(bar for bar in bars if float(bar.close) > 0)
    if valid_bars:
        # Five-period trend needs six closes.  Returning one to five freshly
        # rebuilt bars here hid the already-streaming tick tape and made every
        # restart look like ``MACRO_INSUFFICIENT_DATA`` for roughly five
        # minutes.  Prefer completed minutes once they can support the trend;
        # until then continue to the causal tick fallback below.
        if len(valid_bars) >= 6:
            return (
                [float(bar.close) for bar in valid_bars],
                [float(max(0, bar.volume)) for bar in valid_bars],
            )
    try:
        ticks = store.recent_ticks(symbol, since)
    except Exception:  # noqa: BLE001
        ticks = ()
    # During minute-bar warmup, collapse a high-frequency tape into ten-second
    # closes.  This prevents five adjacent prints from masquerading as a
    # five-period market trend while still giving the macro layer real, causal
    # visibility within about a minute of restart.
    bucketed: dict[int, Any] = {}
    for tick in ticks:
        price = getattr(tick, "price", 0)
        if not price or float(price) <= 0:
            continue
        observed_at = getattr(tick, "received_at", None) or getattr(
            tick, "exchange_timestamp", None
        )
        if isinstance(observed_at, datetime):
            bucketed[int(observed_at.timestamp()) // 10] = tick
    usable_ticks = list(bucketed.values()) if len(bucketed) >= 6 else list(ticks)
    prices = [
        float(t.price)
        for t in usable_ticks
        if getattr(t, "price", 0) and float(t.price) > 0
    ]
    if prices:
        volumes = [
            float(max(0, getattr(t, "volume", 0) or 0))
            for t in usable_ticks
            if getattr(t, "price", 0) and float(t.price) > 0
        ]
        return prices, volumes
    # A sparse market may have completed bars but no recent prints. Preserve
    # those real observations; the builder will correctly keep the trend empty
    # until at least two/required observations exist.
    if valid_bars:
        return (
            [float(bar.close) for bar in valid_bars],
            [float(max(0, bar.volume)) for bar in valid_bars],
        )
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
