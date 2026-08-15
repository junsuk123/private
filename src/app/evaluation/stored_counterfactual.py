from __future__ import annotations

import hashlib
import json
import math
import sqlite3
from collections import defaultdict
from dataclasses import asdict, dataclass, replace
from datetime import datetime, timedelta, timezone
from pathlib import Path
from statistics import fmean
from typing import Any, Mapping, Sequence

from app.backtesting.event_simulator import EventDrivenFillSimulator
from app.data.investor_flow_store import InvestorFlowStore, business_date_for
from app.evaluation.purged_walk_forward import purged_walk_forward_splits
from app.strategy.experts import ALL_EXPERT_TYPES, ExpertContext
from app.trading.contracts import Bar
from app.trading.directional import parse_direction, stop_price, target_price
from app.features.schemas import OHLCVBar
from app.features import session_structure
from app.features import strategy_graph_context
from app.features.strategy_graph_context import (
    STRATEGY_GRAPH_CONTEXT_SCHEMA,
    StrategyGraphContextError,
)
from app.technical import indicators as ti


@dataclass(frozen=True)
class EvaluationConfig:
    history_bars: int = 30
    # US BanKIS round-trip expenses are roughly 40bp before spread/slippage.
    # A 4-10 minute label mostly teaches "cost exceeds move"; use a one-hour
    # causal window so strategies can learn moves large enough to clear costs.
    horizon_bars: int = 60
    stride_bars: int = 5
    minimum_symbol_bars: int = 100
    maximum_bar_gap_seconds: float = 120.0
    minimum_active_history_fraction: float = 0.10
    minimum_future_bars: int = 5
    quantity: int = 1
    venue: str = "NASD"
    market: str = "US"
    instrument_type: str = "overseas_stock"
    # v2 replaces legacy bar-count opening features with the causal, clock-based
    # 30-minute authority shared by live serving. Old labels are not compatible.
    feature_schema_name: str = "counterfactual_quantiles_v2_session_structure"
    # When enabled, each strategy is observed for its own live max-holding
    # period instead of forcing every thesis into the generic 60-bar window.
    # Kept opt-in so compact unit-test fixtures and historical comparisons can
    # explicitly retain their requested horizon.
    align_strategy_horizons: bool = False
    infer_market_from_symbol: bool = True


@dataclass(frozen=True)
class CounterfactualLabel:
    as_of: datetime
    label_end: datetime
    symbol: str
    strategy_id: str
    triggered: bool
    filled: bool
    net_return_bps: float
    cost_bps: float
    exit_reason: str
    features: tuple[float, ...] = ()
    # --- Direction and borrow --------------------------------------------- #
    # ``net_return_bps`` is direction-SIGNED (see ``directional.gross_return_bps``),
    # so a short whose price fell carries a positive label. The sign is applied once,
    # at label construction; nothing downstream re-applies it.
    #
    # A short label is NEVER produced by negating a long label. The two differ by more
    # than a sign: a short pays an accruing borrow fee, sells the bid and covers the
    # ask (not the reverse), and can be recalled mid-horizon. Negation would train the
    # model on a cost structure that does not exist.
    direction: str = "LONG"
    execution_product: str = "CASH"
    borrow_available: bool | None = None
    borrow_quantity: int | None = None
    borrow_fee_bps_annualised: float | None = None
    borrow_cost_bps: float = 0.0
    # False when 대주 was unavailable at signal time. Such rows are kept for
    # signal-quality analysis and EXCLUDED from training and promotion: a strategy may
    # not be judged on trades it could not have taken.
    short_restriction_passed: bool = True
    # Both timestamps are stored because the gap between them IS the leak surface. A
    # borrow observed after the feature snapshot is future information relative to the
    # decision, and a label built on it is unachievable live.
    borrow_observed_at: datetime | None = None
    feature_snapshot_at: datetime | None = None
    # Partial fills. A label whose fill ratio is below 1 describes a smaller position
    # than the strategy asked for, and the unfilled remainder earns nothing.
    fill_ratio: float = 1.0
    gross_return_bps: float = 0.0
    mae_bps: float = 0.0
    mfe_bps: float = 0.0
    holding_seconds: float = 0.0
    cost_floor_dominated: bool = False

    @property
    def is_short(self) -> bool:
        return str(self.direction or "LONG").upper() == "SHORT"

    @property
    def usable_for_training(self) -> bool:
        """Executable AND leak-free.

        A label is dropped when the borrow observation post-dates the feature
        snapshot: that ordering means the label knows something the live decision
        could not have known.
        """
        if not self.short_restriction_passed:
            return False
        if self.exit_reason == "FUTURE_WINDOW_CENSORED":
            return False
        if self.borrow_observed_at is not None and self.feature_snapshot_at is not None:
            if self.borrow_observed_at > self.feature_snapshot_at:
                return False
        return True

    @property
    def all_in_cost_bps(self) -> float:
        return float(self.cost_bps) + float(self.borrow_cost_bps)


@dataclass(frozen=True)
class _Interval:
    as_of: datetime
    label_end: datetime


def causal_percentile(value: float, history: Sequence[float]) -> float:
    """Empirical percentile using prior observations only, with ties at midrank.

    The old definition was ``sum(item <= value) / n``, which counts a tie as
    "below" and therefore scores a value equal to its entire history at 1.0 —
    the top of the distribution.

    On this data that is not a corner case, it is the common case. Measured
    2026-08-07 on stored minute bars: 52.3% of US bars and 29.8% of KR bars print
    EXACTLY zero return, because a thin name's price simply does not change from
    one minute to the next. Under ``<=`` every one of those flat bars scored 1.0
    on the return percentile, so ``intraday_momentum``'s "return in the top 20% of
    its history" fired on a bar that did not move. **72.8% of all US momentum
    triggers were flat bars** (36.4% on KR). The strategy family that is supposed
    to buy strength was mostly buying non-events in illiquid symbols, and every
    other percentile feature — volume, breakout, spread, relative strength —
    carried the same distortion wherever its values repeat.

    Midrank (``below + 0.5 * equal``) is the standard tie correction: a value tied
    with its whole history lands at 0.5, which is what "no information" should
    look like, and a genuine move still ranks at the top.
    """
    if not history:
        return 0.5
    below = sum(item < value for item in history)
    equal = sum(item == value for item in history)
    return (below + 0.5 * equal) / len(history)


@dataclass(frozen=True)
class BarMicrostructure:
    """Per-bar microstructure already persisted alongside the OHLCV columns.

    ``realtime_minute_bars`` carries ``vwap``, ``spread_bps``,
    ``orderbook_imbalance``, ``liquidity_score``, ``volatility`` and
    ``trade_count``, but :func:`load_minute_bars` selected only OHLCV and dropped
    them. Six of eleven strategies need exactly these inputs, so dropping them
    silently reduced the training catalogue by more than half -- see
    :func:`build_labels` for the quantiles this now makes derivable.
    """

    vwap: float | None
    spread_bps: float | None
    orderbook_imbalance: float | None
    liquidity_score: float | None
    volatility: float | None
    trade_count: float | None


def _preferred_minute_bar_streams(
    connection: sqlite3.Connection,
) -> dict[tuple[str, str], str]:
    """One stream per (symbol, UTC date), so a series never mixes feeds within a day.

    ``realtime_minute_bars`` stores a row per (symbol, minute, stream), so a symbol
    carried by both the websocket feed and the REST snapshot holds two rows for the
    same minute. Measured over the current store that is 9,104 of 109,247 rows
    (8.3%), across 15 symbols with two to four streams each -- and they are the
    symbols with the MOST training data: F, SOFI, PFE, T, 005930, INTC.

    Reading them mixed is not a rounding error. ``build_labels`` walks each series
    consecutively, so every duplicated minute:

    - injects a return measured BETWEEN TWO FEEDS. ACDC at 2026-08-05T16:59 holds
      close 4.425 (REST snapshot) and 4.421 (websocket); consecutive division
      turns a stream switch into a fabricated -9bp minute that never happened.
    - inflates ``realized_volatility_30m`` with those switches.
    - double counts the symbol in :func:`_market_return_index`, which is the
      cross-sectional benchmark the relative-strength experts measure against.
    - makes the forward path the triple-barrier labels walk traverse the same
      minute twice, so MFE/MAE and the simulated fill are read off a series that
      is not a time series.

    ``RealtimeMarketDataStore.preferred_minute_bar_stream`` already refuses to mix
    them on the live side for exactly this reason, and this mirrors its ranking:
    widest minute coverage, then tradeable, then trade count, then stream id.

    The grouping is per DAY rather than per symbol because the streams are
    time-disjoint eras, not steady duplicates -- the ``stream_id`` migration lands
    around 2026-08-04, so a symbol's July bars carry an empty id and its August bars
    a labelled one. Choosing one stream for a symbol's whole history therefore
    discards an entire era: F keeps either 2,626 July minutes or 2,295 August ones.
    Measured over the current store, the three candidate rules give:

    ==========================  ==========  ====================
    rule                        retained    intra-day feed switch
    ==========================  ==========  ====================
    one stream per symbol       78.6%       0
    one row per symbol-minute   100.0%      1,346
    one stream per symbol-day   **96.5%**   **0**
    ==========================  ==========  ====================

    Per-day wins on both axes. A switch at a date boundary is free because no
    feature window survives an overnight gap -- :func:`_bars_are_contiguous` rejects
    it -- whereas a switch inside a day injects a return measured between two feeds,
    and those feeds disagree by a median 0.71bps but a p90 of 12.5bps, which is the
    size of the edge this system is trying to measure.

    Returns an empty mapping for a store with no ``stream_id`` column, which leaves
    an older database on its previous behaviour rather than failing the run.
    """
    try:
        rows = connection.execute(
            """
            SELECT symbol, substr(minute_start, 1, 10) AS day,
                   COALESCE(stream_id, '') AS stream,
                   COUNT(DISTINCT minute_start) AS minutes,
                   MAX(COALESCE(is_tradeable, 0)) AS tradeable,
                   SUM(COALESCE(trade_count, 0)) AS trades
            FROM realtime_minute_bars
            GROUP BY symbol, day, stream
            """
        ).fetchall()
    except sqlite3.Error:
        return {}
    best: dict[tuple[str, str], tuple[tuple[int, int, int, str], str]] = {}
    for symbol, day, stream_id, minutes, tradeable, trades in rows:
        stream = str(stream_id or "")
        rank = (int(minutes or 0), int(tradeable or 0), int(trades or 0), stream)
        key = (str(symbol), str(day))
        current = best.get(key)
        if current is None or rank > current[0]:
            best[key] = (rank, stream)
    return {key: chosen for key, (_rank, chosen) in best.items()}


def _rows_on_preferred_stream(
    connection: sqlite3.Connection,
    columns: str,
    preferred: dict[tuple[str, str], str],
):
    """Iterate ``columns`` for the bars on each (symbol, day)'s preferred stream.

    ``columns`` must start with ``symbol, minute_start``; the day is taken from the
    timestamp's date prefix, matching the grouping in
    :func:`_preferred_minute_bar_streams`. The stream column is appended to the
    projection and stripped here, so callers unpack exactly what they asked for
    whether or not filtering is active.
    """
    if not preferred:
        yield from connection.execute(
            f"SELECT {columns} FROM realtime_minute_bars ORDER BY symbol, minute_start"
        )
        return
    rows = connection.execute(
        f"SELECT {columns}, COALESCE(stream_id, '') FROM realtime_minute_bars "
        "ORDER BY symbol, minute_start"
    )
    for row in rows:
        key = (str(row[0]), str(row[1])[:10])
        if preferred.get(key, "") == str(row[-1] or ""):
            yield row[:-1]


def load_minute_microstructure(
    database: Path,
) -> dict[str, dict[datetime, BarMicrostructure]]:
    """Microstructure columns keyed by ``(symbol, minute_start)``.

    Keyed by timestamp rather than positional index on purpose: ``build_labels``
    filters bars for contiguity and activity, so a positional pairing would
    silently misalign a symbol's features with another bar's outcome.

    Restricted to the same preferred stream as :func:`load_minute_bars`. Keying by
    timestamp made the LAST row of a duplicated minute win, and the winner was
    arbitrary among streams -- so a bar's OHLCV could come from the REST snapshot
    while its spread and book imbalance came from the websocket. That also flipped
    ``microstructure_available``, which keys off ``spread_bps > 0``, between the two
    feeds for the same minute.
    """
    by_symbol: dict[str, dict[datetime, BarMicrostructure]] = defaultdict(dict)
    connection = sqlite3.connect(database)
    try:
        preferred = _preferred_minute_bar_streams(connection)
        rows = _rows_on_preferred_stream(
            connection,
            "symbol, minute_start, vwap, spread_bps, orderbook_imbalance, "
            "liquidity_score, volatility, trade_count",
            preferred,
        )
        for (
            symbol,
            start,
            vwap,
            spread_bps,
            imbalance,
            liquidity,
            volatility,
            trade_count,
        ) in rows:
            by_symbol[symbol][datetime.fromisoformat(start)] = BarMicrostructure(
                vwap=_optional_float(vwap),
                spread_bps=_optional_float(spread_bps),
                orderbook_imbalance=_optional_float(imbalance),
                liquidity_score=_optional_float(liquidity),
                volatility=_optional_float(volatility),
                trade_count=_optional_float(trade_count),
            )
    except sqlite3.Error:
        # An older store without these columns must degrade to the previous
        # behaviour (features absent -> those strategies simply do not fire),
        # never to a crash in the training path.
        return {}
    finally:
        connection.close()
    return {symbol: dict(values) for symbol, values in by_symbol.items()}


DEFAULT_NEWS_SENTIMENT_PATH = "data/store/news_sentiment.sqlite3"


def load_news_sentiment(
    path: str | Path = DEFAULT_NEWS_SENTIMENT_PATH,
) -> dict[str, tuple[tuple[datetime, float], ...]]:
    """Point-in-time news sentiment per ticker, ascending by observation time.

    Read from the separate ``news_sentiment`` store the event-LLM pipeline writes.
    Its score is effectively BINARY in practice — measured over 180,699 rows,
    96.4% are exactly +1.0 and 3.6% are exactly -1.0 — so the absolute level is
    close to uninformative and only intensity and the rare negative carry signal.
    :func:`_event_quantiles` is built around that measurement rather than pretending
    the score is a graded sentiment.
    """
    store = Path(path)
    if not store.exists():
        return {}
    by_ticker: dict[str, list[tuple[datetime, float]]] = defaultdict(list)
    connection = sqlite3.connect(store)
    try:
        rows = connection.execute(
            "SELECT ticker, observed_at, score FROM news_sentiment ORDER BY ticker, observed_at"
        )
        for ticker, observed_at, score in rows:
            try:
                moment = datetime.fromisoformat(str(observed_at))
            except (TypeError, ValueError):
                continue
            if moment.tzinfo is None:
                moment = moment.replace(tzinfo=timezone.utc)
            value = _optional_float(score)
            if value is None:
                continue
            # Suffixed vendor symbols and bare exchange codes describe the same issuer.
            key = str(ticker or "").upper().split(".", 1)[0].strip()
            if key:
                by_ticker[key].append((moment, value))
    except sqlite3.Error:
        return {}
    finally:
        connection.close()
    return {ticker: tuple(values) for ticker, values in by_ticker.items()}


def _event_quantiles(
    series: Sequence[tuple[datetime, float]],
    as_of: datetime,
    *,
    window_seconds: float = 900.0,
    baseline_seconds: float = 21_600.0,
) -> dict[str, float]:
    """Event relevance and direction from a saturated sentiment feed.

    ``relevance`` is news INTENSITY against the ticker's own recent baseline: a
    burst of coverage is evidence that something happened, and unlike the score
    itself it is not saturated.

    ``direction`` is the absence of negative coverage in the window. With 96.4% of
    scores pinned at +1.0, a "positive mean" is true almost always and would gate
    nothing; the 3.6% negatives are the only discriminating observations, so a
    window containing one must not read as bullish confirmation.

    Strictly causal: only observations at or before ``as_of`` are consulted.
    """
    if not series:
        return {"event_relevance": 0.0, "event_direction": 0.0}
    window_start = as_of - timedelta(seconds=window_seconds)
    baseline_start = as_of - timedelta(seconds=baseline_seconds)
    window: list[float] = []
    baseline_count = 0
    for moment, score in series:
        if moment > as_of:
            break  # ascending: nothing later can qualify
        if moment >= baseline_start:
            baseline_count += 1
        if moment >= window_start:
            window.append(score)
    if not window:
        return {"event_relevance": 0.0, "event_direction": 0.0}

    # Expected count in a window of this length if coverage were uniform across
    # the baseline. Above 1.0 means unusually heavy coverage right now.
    expected = baseline_count * (window_seconds / baseline_seconds)
    intensity = len(window) / expected if expected > 0 else 0.0
    negatives = sum(1 for score in window if score < 0)
    return {
        # Saturates at 3x normal coverage, so a genuine burst reaches the 0.8
        # entry quantile while routine background coverage does not.
        "event_relevance": max(0.0, min(1.0, intensity / 3.0)),
        "event_direction": max(0.0, 1.0 - negatives / len(window)),
    }


def _optional_float(value: object) -> float | None:
    try:
        if value is None:
            return None
        result = float(value)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return None
    return result if result == result else None  # drop NaN


def load_minute_bars(database: Path) -> dict[str, tuple[Bar, ...]]:
    """Every symbol's minute series, one feed per symbol.

    See :func:`_preferred_minute_bar_streams` for why mixing streams here is not
    survivable: a symbol carried by two feeds had two rows per minute, and this
    loader's consecutive series turned each stream switch into a return that never
    happened.
    """
    by_symbol: dict[str, list[Bar]] = defaultdict(list)
    connection = sqlite3.connect(database)
    try:
        preferred = _preferred_minute_bar_streams(connection)
        rows = _rows_on_preferred_stream(
            connection,
            "symbol, minute_start, open, high, low, close, volume",
            preferred,
        )
        for symbol, start, open_, high, low, close, volume in rows:
            start_time = datetime.fromisoformat(start)
            normalized_symbol = str(symbol).upper()
            venue = (
                "KRX"
                if normalized_symbol.isdigit() and len(normalized_symbol) == 6
                else "NASD"
            )
            by_symbol[symbol].append(
                Bar(
                    symbol=symbol,
                    venue=venue,
                    interval="1m",
                    start_time=start_time,
                    end_time=start_time + timedelta(minutes=1),
                    open=float(open_),
                    high=float(high),
                    low=float(low),
                    close=float(close),
                    volume=float(volume),
                )
            )
    finally:
        connection.close()
    return {symbol: tuple(bars) for symbol, bars in by_symbol.items()}


def _market_return_index(
    bars_by_symbol: dict[str, tuple[Bar, ...]],
) -> dict[datetime, float]:
    """Equal-weighted cross-sectional mean 1-bar return at each timestamp.

    This is the benchmark that turns a raw return into *relative* strength and a
    market-neutral residual. It is a mean across whatever symbols traded in that
    minute, which is the honest closed-world answer for this store -- there is no
    index series and no sector membership in it.
    """
    sums: dict[datetime, float] = defaultdict(float)
    counts: dict[datetime, int] = defaultdict(int)
    for bars in bars_by_symbol.values():
        for position in range(1, len(bars)):
            previous, current = bars[position - 1], bars[position]
            if previous.close <= 0:
                continue
            sums[current.start_time] += current.close / previous.close - 1
            counts[current.start_time] += 1
    return {
        moment: sums[moment] / counts[moment]
        for moment in sums
        if counts[moment] >= 2  # one symbol is not a cross-section
    }


def _relative_volume(current_volume: float, history: Sequence[float]) -> float:
    """RVOL: current volume over its own recent average.

    The day-trading literature treats this as the primary selection filter --
    "stocks in play". Zarattini/Barbon/Aziz restrict an opening-range breakout to
    the top names by opening relative volume and report a Sharpe of 2.81 where the
    unrestricted version is unprofitable, and practitioner studies put the useful
    threshold around 1.5-2.0x. It is returned as a raw ratio; the caller maps it
    to a quantile so it stays comparable with every other feature.
    """
    usable = [value for value in history if value > 0]
    if not usable:
        return 1.0
    average = fmean(usable)
    if average <= 0:
        return 1.0
    return max(0.0, current_volume / average)


def _session_date_changed(previous: Bar | None, current: Bar) -> bool:
    """Did a new trading session start at ``current``?

    Derived from the bar timestamps rather than an exchange calendar: the store
    holds no session metadata, and a UTC date change is a sound proxy for both
    venues here because neither KRX nor US regular hours straddle UTC midnight.
    """
    if previous is None:
        return True
    return previous.start_time.date() != current.start_time.date()


def _slope(values: Sequence[float]) -> float:
    """Least-squares slope over evenly spaced samples; 0.0 when undefined."""
    n = len(values)
    if n < 2:
        return 0.0
    mean_x = (n - 1) / 2.0
    mean_y = fmean(values)
    denominator = sum((index - mean_x) ** 2 for index in range(n))
    if denominator <= 0:
        return 0.0
    numerator = sum((index - mean_x) * (value - mean_y) for index, value in enumerate(values))
    return numerator / denominator


def _gap_quantile(bars: Sequence[Bar], index: int, session_start: int) -> float:
    """Overnight gap, as a percentile of this symbol's own recent gaps."""
    if session_start <= 0 or session_start >= len(bars):
        return 0.0
    previous_close = bars[session_start - 1].close
    if previous_close <= 0:
        return 0.0
    gap = bars[session_start].open / previous_close - 1
    history = []
    for position in range(1, min(session_start, len(bars))):
        prior = bars[position - 1].close
        if prior > 0:
            history.append(bars[position].open / prior - 1)
    if not history:
        # A single observed gap cannot be ranked; a neutral 0.5 would falsely
        # satisfy nothing and a 1.0 would fire on no evidence.
        return 0.0
    return causal_percentile(gap, history)


def _opening_confirmation_quantile(
    bars: Sequence[Bar], index: int, session_start: int
) -> float:
    """Has price held the gap's direction since the session opened?

    The gap_context thesis is explicitly "continuation only after price-discovery
    confirmation", so an unconfirmed gap must not fire.
    """
    if session_start <= 0 or index <= session_start:
        return 0.0
    open_price = bars[session_start].open
    if open_price <= 0:
        return 0.0
    previous_close = bars[session_start - 1].close
    if previous_close <= 0:
        return 0.0
    gap_up = bars[session_start].open >= previous_close
    drift = bars[index].close / open_price - 1
    # Confirmation is directional agreement with the gap, scaled so that a drift
    # of one half-percent in the gap's direction is full confirmation.
    aligned = drift if gap_up else -drift
    return max(0.0, min(1.0, 0.5 + aligned / 0.01))


def _opening_range_breakout_quantile(
    bars: Sequence[Bar], index: int, session_start: int, range_bars: int = 5
) -> float:
    """Position of price relative to the session's opening range.

    The published ORB rule (Zarattini/Barbon/Aziz) buys when price clears the
    first N minutes' range, and its profitability rests on pairing that with a
    relative-volume selection filter rather than on the breakout alone. Returned
    as 0..1 where >0.5 means "above the opening range high".
    """
    del range_bars  # retained for compatibility; the authoritative range is 30 minutes.
    if session_start < 0 or session_start >= len(bars) or index >= len(bars):
        return 0.0
    observed = session_structure.opening_range(
        bars[: index + 1],
        session_open=bars[session_start].start_time,
        minutes=30,
        now=bars[index].end_time,
    )
    if observed is None:
        return 0.0
    high = observed.high
    low = observed.low
    span = high - low
    if span <= 0 or high <= 0:
        return 0.0
    excess = (bars[index].close - high) / span
    # The published trigger is clearing the opening range at all, not clearing it
    # by some margin, so any positive excess must land in admissible territory
    # (>= the 0.8 entry quantile) and deeper excess merely ranks higher. Inside
    # the range stays below the threshold.
    if excess > 0:
        return min(1.0, 0.8 + excess)
    return max(0.0, 0.4 + excess)


# Session geometry comes from ``session_structure.regular_session`` so KRX and US
# rows are boxed by their own exchange clock. KRX continuous trading ends at 15:20
# and 15:20-15:30 is a closing single-price auction, so the tradable "last
# half-hour" is 14:50-15:20 — NOT 15:00-15:30. Entering into the auction would price
# against a mechanism this system does not model.
_KST = timezone(timedelta(hours=9))
_PENULTIMATE_HALF_HOUR = (14 * 60 + 20, 14 * 60 + 50)


def _kst_minute_of_day(moment: datetime) -> int:
    local = moment.astimezone(_KST)
    return local.hour * 60 + local.minute


def _intraday_momentum_quantiles(
    bars: Sequence[Bar],
    index: int,
    session_start: int,
) -> dict[str, float]:
    """Market-intraday-momentum features for the bar at ``index``.

    The published effect is that the FIRST half-hour return (measured from the
    previous close, so it includes the overnight gap) predicts the LAST half-hour
    return, with the relationship strongest on high-volatility days. Reported R² is
    1.6%, rising to 3.3% on high first-half-hour-volatility days.

    Three features are produced, all strictly causal:

    ``intraday_momentum_signal``
        0..1, above 0.5 when the first half-hour return was positive. Magnitude is
        scaled so a 1% first half-hour move saturates.
    ``intraday_momentum_window``
        1.0 only inside 14:50-15:20 Korea time — the last half-hour of CONTINUOUS
        trading. Zero elsewhere, so the strategy cannot fire at any other time.
    ``first_half_hour_volatility``
        0..1 percentile of the day's first-half-hour realised volatility against the
        same measure on previous days. This is the paper's strongest condition and is
        independently the cost condition: only a volatile day travels far enough to
        clear a ~33bps round trip.
    """
    absent = {
        "intraday_momentum_signal": 0.0,
        "intraday_momentum_window": 0.0,
        "first_half_hour_volatility": 0.0,
    }
    if session_start <= 0 or index <= session_start:
        return absent
    current = bars[index]
    session = session_structure.regular_session(current.symbol)
    in_window = session.in_last_continuous_half_hour(current.end_time)
    if not in_window:
        # Cheap exit: outside the entry window nothing else matters.
        return absent

    previous_close = bars[session_start - 1].close
    observed = session_structure.opening_range(
        bars[: index + 1],
        session_open=bars[session_start].start_time,
        minutes=30,
        now=current.end_time,
    )
    r1_bps = session_structure.first_half_hour_return_bps(
        bars[: index + 1],
        previous_close=previous_close,
        session_open=bars[session_start].start_time,
        minutes=30,
        now=current.end_time,
    )
    if observed is None or r1_bps is None:
        return absent
    r1 = r1_bps / 10_000.0
    if r1 <= 0:
        # Long-only: a negative first half-hour predicts a negative last half-hour,
        # which is not expressible here.
        return {**absent, "intraday_momentum_window": 1.0}

    # Same measure on earlier sessions, for a like-for-like percentile.
    prior_vols = _prior_session_opening_volatility(bars, session_start)
    volatility_q = session_structure.first_half_hour_volatility_percentile(
        prior_vols, observed.volatility, minimum_samples=3
    )

    return {
        "intraday_momentum_signal": max(0.0, min(1.0, 0.5 + r1 / 0.02)),
        "intraday_momentum_window": 1.0,
        # Counterfactual quantile tensors are finite by contract.  An unrankable
        # volatility therefore remains a non-triggering zero here; the live
        # election context preserves the stronger None/absent distinction.
        "first_half_hour_volatility": volatility_q if volatility_q is not None else 0.0,
    }


def _overnight_carry_window(bars: Sequence[Bar], index: int) -> float:
    """1.0 inside the last continuous half hour of this symbol's own session.

    Deliberately independent of the opening range. ``intraday_momentum_window``
    also encodes the same clock, but only after an opening range could be
    measured, so reusing it would have made an overnight carry inherit a
    first-half-hour data requirement it does not have — and the opening window is
    exactly the one this feed most often misses.
    """
    current = bars[index]
    session = session_structure.regular_session(current.symbol)
    return 1.0 if session.in_last_continuous_half_hour(current.end_time) else 0.0


def _prior_session_opening_volatility(
    bars: Sequence[Bar], session_start: int
) -> list[float]:
    """First-half-hour range/price for every session strictly before this one."""
    if session_start <= 0:
        return []
    session = session_structure.regular_session(bars[0].symbol)
    zone = session.zone
    by_day: dict[int, list[Bar]] = defaultdict(list)
    for position in range(session_start):
        bar = bars[position]
        by_day[bar.start_time.astimezone(zone).date().toordinal()].append(bar)
    values: list[float] = []
    for day in sorted(by_day):
        window = by_day[day]
        local_day = window[0].start_time.astimezone(zone)
        session_open = local_day.replace(
            hour=session.open_time.hour,
            minute=session.open_time.minute,
            second=0,
            microsecond=0,
        )
        observed = session_structure.opening_range(
            window,
            session_open=session_open,
            minutes=30,
            now=window[-1].end_time,
        )
        if observed is not None:
            values.append(observed.volatility)
    return values


def _microstructure_quantiles(
    *,
    bars: Sequence[Bar],
    index: int,
    history_start: int,
    micro_by_time: dict[datetime, BarMicrostructure],
    spreads: Sequence[float],
    spread_history: Sequence[float],
) -> dict[str, float]:
    """Order-flow features from the persisted per-bar microstructure columns.

    Order-flow imbalance is the best-documented short-horizon predictor in the
    microstructure literature -- near-linear in contemporaneous price change, with
    order-book imbalance explaining the majority of short-interval moves and the
    signal strongest under ~3 minutes. The store keeps ``orderbook_imbalance`` per
    bar, so its level and slope are recoverable here.

    When the columns are absent (older store) every value falls back to the
    never-fire constant the previous implementation used, so a missing column
    degrades to "this strategy is untrainable" rather than to a fabricated signal.
    """
    absent = {
        "vwap_zscore": 1.0,
        "liquidity_recovery": 0.0,
        "microprice_edge": 0.0,
        "ofi_slope": 0.0,
        "depth_recovery": 0.0,
        "flow_toxicity": 1.0,
        "liquidity_micro": 0.0,
    }
    current_micro = micro_by_time.get(bars[index].start_time)
    if current_micro is None:
        return absent

    window = [
        micro_by_time.get(bars[position].start_time)
        for position in range(history_start, index + 1)
    ]
    imbalances = [m.orderbook_imbalance for m in window if m and m.orderbook_imbalance is not None]
    spread_values = [m.spread_bps for m in window if m and m.spread_bps is not None]
    liquidity_values = [m.liquidity_score for m in window if m and m.liquidity_score is not None]

    result = dict(absent)

    # --- Anchored-VWAP displacement, volatility-normalised ------------------
    # Uses the bar's own recorded VWAP rather than the close-weighted proxy.
    vwaps = [m.vwap for m in window if m and m.vwap is not None and m.vwap > 0]
    if vwaps and current_micro.vwap and current_micro.vwap > 0:
        anchor = fmean(vwaps)
        volatility = current_micro.volatility or 0.0
        displacement = bars[index].close / anchor - 1
        scale = volatility if volatility > 0 else 0.005
        zscore = displacement / scale
        # 0..1 with LOW meaning deeply displaced below the anchor, matching the
        # expert's ``vwap_zscore <= 1 - entry_quantile`` test.
        result["vwap_zscore"] = max(0.0, min(1.0, 0.5 + zscore / 6.0))

    # --- Order-flow imbalance level and slope -------------------------------
    if len(imbalances) >= 3:
        result["ofi_slope"] = causal_percentile(
            _slope(imbalances[-5:]), [_slope(imbalances[max(0, j - 5) : j]) for j in range(3, len(imbalances))]
        )
        # Microprice tilt: a bid-heavy book prices the true mid above the midpoint.
        result["microprice_edge"] = max(0.0, min(1.0, 0.5 + imbalances[-1] / 2.0))

    # --- Depth / liquidity recovery -----------------------------------------
    if len(liquidity_values) >= 3:
        recovering = liquidity_values[-1] - fmean(liquidity_values[:-1])
        result["depth_recovery"] = max(0.0, min(1.0, 0.5 + recovering * 50.0))
        result["liquidity_micro"] = causal_percentile(
            liquidity_values[-1], liquidity_values[:-1]
        )

    # --- Spread normalisation and flow toxicity -----------------------------
    if len(spread_values) >= 3:
        baseline = fmean(spread_values[:-1])
        latest = spread_values[-1]
        # Spread back to or below its baseline == liquidity has fully returned.
        # The previous ``1 - latest / baseline`` formula returned zero at the
        # exact point described as recovered, so no normalised tape could ever
        # clear a 0.65 confirmation gate.  Score 1.0 at/below baseline and decay
        # linearly to zero as the spread widens from 1x to 2x baseline.
        result["liquidity_recovery"] = (
            max(0.0, min(1.0, 2.0 - (latest / baseline))) if baseline > 0 else 0.0
        )
        # Toxicity: a spread blowing out relative to baseline means the
        # counterparty is better informed than a reversion thesis assumes.
        result["flow_toxicity"] = (
            max(0.0, min(1.0, (latest / baseline) / 3.0)) if baseline > 0 else 1.0
        )
    return result


def _rolling_mean_percentile(
    series: Sequence[float],
    index: int,
    history_start: int,
    window: int,
) -> float:
    """Percentile of a ``window``-bar mean against prior ``window``-bar MEANS.

    Ranking a mean against a distribution of single observations is invalid and
    silently one-sided: averaging shrinks variance, so a window mean lands near the
    centre of the singles distribution almost always and its percentile clusters
    around 0.5. Measured consequence — ``residual_strength_long`` could then never
    reach its 0.65 confirmation threshold, so ``residual_relative_strength`` never
    fired even with every other input satisfied. Compare like with like.
    """
    if window < 1 or index < history_start + window:
        return 0.0
    current = fmean(series[index - window + 1 : index + 1])
    history = [
        fmean(series[position - window + 1 : position + 1])
        for position in range(history_start + window - 1, index)
        if position - window + 1 >= 0
    ]
    if len(history) < 3:
        return 0.0
    return causal_percentile(current, history)


def _investor_flow_quantile(
    history: Mapping[str, Any] | None,
    business_date: str,
) -> float:
    """Informed net buying on this business day, ranked against prior days.

    Ranked rather than thresholded because net-buy value is denominated in KRW and
    is not comparable across symbols: 3.6bn won into Samsung is routine, the same
    number into a mid-cap is not. A percentile of the symbol's own history is.

    Strictly causal: only business days strictly BEFORE the current one form the
    comparison set, so a day never ranks against its own future.
    """
    if not history:
        return 0.0
    prior = [
        day.informed_net_buy_value
        for date_key, day in history.items()
        if date_key < business_date
    ]
    today = history.get(business_date)
    if today is None or len(prior) < 3:
        return 0.0
    return causal_percentile(today.informed_net_buy_value, prior)


def build_labels(
    bars_by_symbol: dict[str, tuple[Bar, ...]],
    config: EvaluationConfig,
    microstructure_by_symbol: dict[str, dict[datetime, BarMicrostructure]] | None = None,
    investor_flow_by_symbol: Mapping[str, Mapping[str, Any]] | None = None,
    news_by_ticker: Mapping[str, Sequence[tuple[datetime, float]]] | None = None,
) -> tuple[CounterfactualLabel, ...]:
    simulator = EventDrivenFillSimulator()
    labels: list[CounterfactualLabel] = []
    experts = tuple(expert_type() for expert_type in ALL_EXPERT_TYPES)
    future_bars_by_strategy = {
        expert.strategy_id: (
            max(1, math.ceil(expert.config.max_holding_seconds / 60.0) + 1)
            if config.align_strategy_horizons
            else config.horizon_bars
        )
        for expert in experts
    }
    maximum_future_bars = max(future_bars_by_strategy.values())
    # The loop below is bounded by the SHORTEST horizon, not the longest.
    #
    # Bounding it by the longest let one strategy truncate every other strategy's
    # labels: adding an 18-hour overnight thesis (1,081 bars) cut the stored label
    # set from 60,108 rows to 25,537 and moved the last labelled bar three days
    # earlier, which in turn dropped every strategy below the upside-supervision
    # floor and de-authorized the whole checkpoint. A per-strategy horizon has to
    # cost that strategy its unobservable tail and nobody else's; each expert
    # already slices ``future`` to its own window, so the only change needed is to
    # skip the strategies whose window runs past the end.
    minimum_future_bars = min(future_bars_by_strategy.values())
    microstructure_by_symbol = microstructure_by_symbol or {}
    investor_flow_by_symbol = investor_flow_by_symbol or {}
    news_by_ticker = news_by_ticker or {}
    # Cross-sectional context: each symbol's return over a common window, so
    # relative strength is measured against the actual universe rather than
    # against a hardcoded 0.0. Built once, keyed by bar timestamp.
    market_returns_by_time = _market_return_index(bars_by_symbol)
    for symbol, bars in sorted(bars_by_symbol.items()):
        if len(bars) < config.minimum_symbol_bars:
            continue
        venue, market, instrument_type = _execution_market_for_symbol(
            symbol,
            config,
        )
        returns = [0.0]
        returns.extend(bars[i].close / bars[i - 1].close - 1 for i in range(1, len(bars)))
        volumes = [bar.volume for bar in bars]
        spreads = [max(0.0, (bar.high - bar.low) / bar.close) for bar in bars]
        micro_by_time = microstructure_by_symbol.get(symbol) or {}
        flow_history = investor_flow_by_symbol.get(symbol) or {}
        news_series = news_by_ticker.get(str(symbol).upper().split(".", 1)[0]) or ()
        # Residual return = own return minus the cross-sectional mean. This is the
        # market-neutral component the residual/relative-strength experts score.
        residuals = [
            returns[position]
            - market_returns_by_time.get(bars[position].start_time, 0.0)
            for position in range(len(bars))
        ]
        # Session-open reference for the gap / opening-range features.
        session_open_index: list[int] = []
        current_session_start = 0
        for position, bar in enumerate(bars):
            if _session_date_changed(bars[position - 1] if position else None, bar):
                current_session_start = position
            session_open_index.append(current_session_start)
        for index in range(
            config.history_bars,
            len(bars) - minimum_future_bars,
            config.stride_bars,
        ):
            current = bars[index]
            history_start = index - config.history_bars
            history_window = bars[history_start : index + 1]
            if not _bars_are_contiguous(
                history_window,
                maximum_gap_seconds=config.maximum_bar_gap_seconds,
            ):
                continue
            active_history = sum(
                bar.high > bar.low
                or (
                    position > 0
                    and bar.close != history_window[position - 1].close
                )
                for position, bar in enumerate(history_window)
            )
            if (
                active_history / max(1, len(history_window))
                < config.minimum_active_history_fraction
            ):
                continue
            return_history = returns[history_start:index]
            volume_history = volumes[history_start:index]
            spread_history = spreads[history_start:index]
            prior_high = max(bar.high for bar in bars[max(0, index - 10) : index])
            vwap_proxy = sum(
                bar.close * max(bar.volume, 1.0) for bar in bars[history_start:index]
            ) / sum(max(bar.volume, 1.0) for bar in bars[history_start:index])
            current_return = returns[index]
            completed = tuple(
                OHLCVBar(
                    ticker=bar.symbol,
                    as_of=bar.end_time,
                    open=bar.open,
                    high=bar.high,
                    low=bar.low,
                    close=bar.close,
                    volume=bar.volume,
                )
                for bar in bars[: index + 1]
            )
            rvgi_result = ti.rvgi(completed, 10)
            box = ti.causal_box_geometry(completed, 20)
            rvgi_diff = (
                rvgi_result.main - rvgi_result.signal
                if rvgi_result.ok
                and rvgi_result.main is not None
                and rvgi_result.signal is not None
                else None
            )
            box_breakout = bool(
                box.ok and box.high is not None and current.close > box.high
            )
            volume_confirmed = (
                causal_percentile(current.volume, volume_history) >= 0.65
            )
            breakout_move = current.close / prior_high - 1
            prior_range_high = max(
                bar.high for bar in bars[max(0, index - 11) : index - 1]
            )
            prior_breakout_accepted = bool(
                index > 1
                and bars[index - 1].close > prior_range_high
                and current.close >= bars[index - 1].close
            )
            minutes_from_open = max(
                0.0,
                (
                    current.start_time
                    - bars[session_open_index[index]].start_time
                ).total_seconds()
                / 60.0,
            )
            quantiles = {
                # Directional LONG signals must not turn a tied zero move into a
                # top-quantile observation (the empirical CDF returns 1.0 for a
                # value tied with an all-zero history).  Preserve the percentile
                # only on the thesis-consistent side of zero.
                "return": (
                    causal_percentile(current_return, return_history)
                    if current_return > 0
                    else 0.0
                ),
                "volume": causal_percentile(current.volume, volume_history),
                "breakout": causal_percentile(
                    breakout_move,
                    [
                        bars[j].close / max(bar.high for bar in bars[max(0, j - 10) : j]) - 1
                        for j in range(max(10, history_start), index)
                    ],
                ) if breakout_move > 0 else 0.0,
                "breakout_acceptance": 1.0 if prior_breakout_accepted else 0.0,
                "vwap_deviation": causal_percentile(
                    current.close / vwap_proxy - 1,
                    [bar.close / vwap_proxy - 1 for bar in bars[history_start:index]],
                ),
                "reversion": causal_percentile(current_return, return_history),
                "liquidity_shock": (
                    causal_percentile(
                        spreads[index - 1], spreads[history_start : index - 1]
                    )
                    if spreads[index - 1] > 0
                    else 0.0
                ),
                # A reversal is a two-step event: the PREVIOUS bar supplies the
                # sell-off and the CURRENT bar supplies the recovery.  Ranking the
                # same return once with each sign made the two conditions mutually
                # exclusive for normal observations, while tied zero-return bars
                # scored 1.0 on both sides.  That selected stationary/thin bars as
                # "reversals" and was the dominant source of the strategy's
                # measured losses.
                "price_drop": (
                    causal_percentile(
                        -returns[index - 1],
                        [-item for item in returns[history_start : index - 1]],
                    )
                    if returns[index - 1] < 0
                    else 0.0
                ),
                "recovery": (
                    causal_percentile(current_return, return_history)
                    if current_return > 0
                    else 0.0
                ),
                "liquidity": causal_percentile(current.volume, volume_history),
                # Event features come from the separate news-sentiment store the
                # event-LLM pipeline writes. Absent (no coverage for this ticker at
                # this moment) still yields 0.0 -> the strategy does not fire.
                **_event_quantiles(news_series, current.end_time),
                # Relative strength IS derivable: the store holds every symbol, so
                # the cross-sectional mean return is a real benchmark. Previously
                # hardcoded to 0.0, which made this strategy untrainable.
                "relative_strength": causal_percentile(
                    residuals[index], residuals[history_start:index]
                ),
                # Gap and opening confirmation come from the session-open bar,
                # located from bar timestamps rather than an exchange calendar.
                "overnight_carry_window": _overnight_carry_window(bars, index),
                "gap": _gap_quantile(bars, index, session_open_index[index]),
                "opening_confirmation": _opening_confirmation_quantile(
                    bars, index, session_open_index[index]
                ),
                # Gap continuation is an opening price-discovery thesis, not a
                # permanent all-day attribute. Wait five minutes for discovery
                # and stop admitting new entries after the first half-hour.
                "gap_entry_window": (
                    1.0 if 5.0 <= minutes_from_open <= 30.0 else 0.0
                ),
                # Relative volume -- the "stocks in play" filter. Not consumed by
                # the legacy experts, but the opening-range expert requires it and
                # the model gets it as a feature either way.
                "relative_volume": causal_percentile(
                    _relative_volume(current.volume, volume_history),
                    [
                        _relative_volume(volumes[j], volumes[max(0, j - 20) : j])
                        for j in range(max(20, history_start), index)
                    ],
                ),
                "momentum_persistence_short": _rolling_mean_percentile(
                    returns, index, history_start, 5
                ),
                "momentum_persistence_long": _rolling_mean_percentile(
                    returns, index, history_start, 15
                ),
                "opening_range_breakout": _opening_range_breakout_quantile(
                    bars, index, session_open_index[index]
                ),
                **_intraday_momentum_quantiles(bars, index, session_open_index[index]),
                "rvgi_diff": (
                    max(0.0, min(1.0, 0.5 + rvgi_diff * 5.0))
                    if rvgi_diff is not None
                    else 0.0
                ),
                "rvgi_cross": 1.0 if rvgi_result.bullish_cross else 0.0,
                "box_position": float(box.position or 0.0) if box.ok else 0.0,
                "false_breakout_risk": (
                    0.0 if box_breakout and volume_confirmed else 1.0
                ),
                # Residual strength over two horizons, both market-neutral. Each is
                # ranked against prior means of the SAME window length; see
                # _rolling_mean_percentile for why the naive comparison is invalid.
                "residual_strength_short": _rolling_mean_percentile(
                    residuals, index, history_start, 5
                ),
                "residual_strength_long": _rolling_mean_percentile(
                    residuals, index, history_start, 15
                ),
                # Real broker investor-type data (외국인/기관 net buying), fetched
                # from KIS and persisted per business day. An orderbook imbalance is
                # NOT a substitute — it describes resting quotes over seconds, not
                # who accumulated over a day — so this reads the actual series and
                # falls back to 0.0 (does not fire) when a symbol is uncollected.
                "investor_flow": _investor_flow_quantile(
                    flow_history, business_date_for(current.end_time)
                ),
                **_microstructure_quantiles(
                    bars=bars,
                    index=index,
                    history_start=history_start,
                    micro_by_time=micro_by_time,
                    spreads=spreads,
                    spread_history=spread_history,
                ),
            }
            as_of = current.end_time
            future = _contiguous_future_bars(
                bars[index + 1 : index + 1 + maximum_future_bars],
                expected_start=current.end_time,
                maximum_gap_seconds=config.maximum_bar_gap_seconds,
            )
            required_future_bars = max(
                1,
                min(config.minimum_future_bars, config.horizon_bars),
            )
            if len(future) < required_future_bars:
                continue
            # Built once per snapshot rather than once per strategy: the vector
            # does not depend on which expert is being labelled. A snapshot whose
            # contract cannot be built drops out of the label set entirely — the
            # alternative is a row carrying a fabricated field, which is how an
            # untrained weight reaches production.
            try:
                label_features = _label_features(
                    config.feature_schema_name,
                    current=current,
                    current_return=current_return,
                    return_history=return_history,
                    close_history=[bar.close for bar in bars[history_start:index]],
                    volume_history=volume_history,
                    vwap_proxy=vwap_proxy,
                    microstructure=micro_by_time.get(current.start_time),
                    rvgi_result=rvgi_result,
                    box=box,
                    quantiles=quantiles,
                )
            except StrategyGraphContextError:
                continue
            context = ExpertContext(
                symbol=symbol,
                as_of=as_of,
                price=current.close,
                proposed_quantity=config.quantity,
                feature_snapshot_id=f"stored:{symbol}:{as_of.isoformat()}",
                utility_evidence_id="counterfactual",
                quantiles=quantiles,
            )
            for expert in experts:
                plan = expert.propose(context)
                triggered = plan is not None
                strategy_future_bars = future_bars_by_strategy[expert.strategy_id]
                if index + strategy_future_bars >= len(bars):
                    # This strategy's own horizon runs past the end of the stored
                    # series, so its outcome is unobservable at this bar. Skipping
                    # it here is what keeps a long-horizon thesis from deleting
                    # every short-horizon label in the dataset.
                    continue
                strategy_future = future[:strategy_future_bars]
                # A strategy that did not fire is a useful classification
                # negative, but it is not a hypothetical filled trade.  The
                # previous implementation forced every expert to propose with
                # synthetic all-one quantiles, then simulated P&L even when
                # the real point-in-time context was inadmissible.  That
                # contaminated the return/cost heads with fabricated losses.
                if plan is not None:
                    baseline_cost = simulator.cost_engine.estimate(
                        symbol=symbol,
                        market=market,
                        venue=venue,
                        instrument_type=instrument_type,
                        entry_price=current.close,
                        expected_exit_price=current.close,
                        quantity=config.quantity,
                    )
                    fee_policy = simulator.cost_engine.policy_for(
                        venue=venue,
                        instrument_type=instrument_type,
                    )
                    cost_bps = baseline_cost.total_cost_rate * 10_000.0
                    # Size BOTH barriers to the horizon's own volatility, floored
                    # at the round trip.
                    #
                    # The previous target was cost + max(25, cost) and the stop was
                    # whatever the expert declared, so one fixed pair (stop 60 /
                    # target 160bps) was applied to both markets. Measured
                    # 2026-08-06 on stored bars: that pair resolves 7.3% target /
                    # 23.1% stop / 69.6% timeout on US names, and 22.4% / 68.4% /
                    # 9.2% on KR ones. Opposite pathologies, same constants — the
                    # label was recording the symbol's volatility, not the signal.
                    #
                    # That is also why the MFE head starved: only 9-25% of fills
                    # ended profitable, so the upside channel (trained on
                    # profitable fills alone) got 0-21 rows per strategy while the
                    # downside channel got 5-112.
                    sigma_bps = _causal_horizon_sigma_bps(
                        bars,
                        index,
                        history_start,
                        horizon_bars=strategy_future_bars,
                    )
                    target_bps, stop_bps, cost_floor_dominated = _barrier_bps(
                        sigma_bps=sigma_bps,
                        cost_bps=cost_bps,
                        safety_margin_bps=fee_policy.safety_margin_rate * 10_000.0,
                        configured_target_bps=float(
                            plan.profit_policy.get("bps", 0.0)
                        ),
                        configured_stop_bps=float(
                            plan.initial_stop.get("bps", 0.0)
                        ),
                    )
                    plan = replace(
                        plan,
                        profit_policy={
                            **plan.profit_policy,
                            "bps": target_bps,
                            "price": target_price(
                                current.close,
                                target_bps / 10_000.0,
                                parse_direction(plan.position_direction),
                            ),
                            "cost_floor_dominated": cost_floor_dominated,
                            "horizon_sigma_bps": sigma_bps,
                        },
                        initial_stop={
                            **plan.initial_stop,
                            "bps": stop_bps,
                            "price": stop_price(
                                current.close,
                                stop_bps / 10_000.0,
                                parse_direction(plan.position_direction),
                            ),
                        },
                    )
                outcome = (
                    simulator.simulate(
                        plan,
                        strategy_future,
                        as_of=as_of,
                        venue=venue,
                        market=market,
                        instrument_type=instrument_type,
                    )
                    if plan is not None
                    else None
                )
                labels.append(
                    CounterfactualLabel(
                        as_of=as_of,
                        label_end=strategy_future[-1].end_time,
                        symbol=symbol,
                        strategy_id=expert.strategy_id,
                        triggered=triggered,
                        filled=outcome.filled if outcome is not None else False,
                        net_return_bps=(
                            outcome.net_return_bps if outcome is not None else 0.0
                        ),
                        cost_bps=outcome.cost_bps if outcome is not None else 0.0,
                        exit_reason=(
                            outcome.exit_reason
                            if outcome is not None
                            else "STRATEGY_NOT_TRIGGERED"
                        ),
                        features=label_features,
                        direction=(
                            plan.position_direction if plan is not None else "LONG"
                        ),
                        execution_product=(
                            plan.execution_product if plan is not None else "CASH"
                        ),
                        gross_return_bps=(
                            outcome.gross_return_bps if outcome is not None else 0.0
                        ),
                        mae_bps=(outcome.mae_bps if outcome is not None else 0.0),
                        mfe_bps=(outcome.mfe_bps if outcome is not None else 0.0),
                        holding_seconds=(
                            outcome.holding_seconds if outcome is not None else 0.0
                        ),
                        cost_floor_dominated=bool(
                            plan is not None
                            and plan.profit_policy.get("cost_floor_dominated", False)
                        ),
                    )
                )
    return tuple(labels)


#: Barrier widths as a multiple of the horizon's typical excursion. A target at
#: ~1 sigma and a stop at ~0.7 sigma keeps both barriers inside what the horizon
#: actually delivers, so the label resolves on the SIGNAL rather than on which
#: market the symbol happens to trade in.
_VOLATILITY_TARGET_K = 1.0
_VOLATILITY_STOP_K = 0.7
#: Never let a volatility estimate collapse the stop into the tick grid.
_MINIMUM_STOP_BPS = 15.0


def _causal_horizon_sigma_bps(
    bars: Sequence[Bar],
    index: int,
    history_start: int,
    *,
    horizon_bars: int,
) -> float | None:
    """Typical excursion over ``horizon_bars``, estimated from PAST bars only.

    Uses the MEAN absolute per-bar return scaled by ``sqrt(horizon)``.

    Not the median: in a thin name more than half the bars print no price change at
    all, so the median absolute return is exactly 0 and the estimator returns None
    for precisely the symbols that need it most. That silently reinstated the fixed
    160/60 pair on every US name — measured while building this: the median target
    came out at exactly 160.0bps, unchanged. The mean is non-zero whenever any bar
    moved, and for a roughly normal return it is ``sigma * sqrt(2/pi)``, so it is a
    scale estimate rather than a quantile.

    ``sqrt`` scaling because there is not enough history to measure overlapping
    horizon-length windows directly — history is 30 bars, the horizon is 60.

    Reads ``bars[history_start:index + 1]`` and nothing after ``index``, so it is
    usable at decision time.
    """
    window = [bar for bar in bars[history_start : index + 1] if bar.close > 0]
    if len(window) < 5:
        return None
    moves = [
        abs(later.close / earlier.close - 1.0)
        for earlier, later in zip(window, window[1:])
        if earlier.close > 0
    ]
    if not moves:
        return None
    per_bar = fmean(moves)
    if per_bar <= 0:
        # Every bar in the window printed the same price. There is no scale to
        # estimate, and inventing one would be worse than saying so.
        return None
    return per_bar * math.sqrt(max(1, int(horizon_bars))) * 10_000.0


def _barrier_bps(
    *,
    sigma_bps: float | None,
    cost_bps: float,
    safety_margin_bps: float,
    configured_target_bps: float,
    configured_stop_bps: float,
    stop_volatility_k: float = _VOLATILITY_STOP_K,
) -> tuple[float, float, bool]:
    """``(target_bps, stop_bps, cost_floor_dominated)`` for one labelled trade.

    Two requirements pull in opposite directions and both are real:

    * a target below the round-trip cost teaches a trade that cannot pay, so the
      cost floor is a hard minimum;
    * a target above what the horizon delivers is never touched, so the label
      degenerates into "stopped out or timed out" and carries no signal.

    When the cost floor wins, that IS the finding — measured 2026-08-06: US names
    need ~88bps to clear a 63bps round trip while the median 60-minute favourable
    excursion is 15.7bps, so no barrier pair can be both payable and reachable.
    ``cost_floor_dominated`` is returned rather than hidden so the caller can
    record it instead of presenting a coin flip as a trading opportunity.

    KR, for contrast, has a 28bps round trip against a 133bps median excursion —
    there the volatility term leads and the barriers land inside the distribution.
    """
    cost_floor = cost_bps + safety_margin_bps + max(8.0, 0.25 * cost_bps)
    if sigma_bps is None or sigma_bps <= 0:
        # No usable history: fall back to the strategy's declared geometry, still
        # floored at cost. This is the old behaviour, kept for the cold-start bars.
        target = max(configured_target_bps, cost_floor)
        return (
            target,
            max(_MINIMUM_STOP_BPS, configured_stop_bps),
            configured_target_bps < cost_floor,
        )
    volatility_target = _VOLATILITY_TARGET_K * sigma_bps
    # The live geometry is a floor, not a comment. Ignoring the configured
    # barriers let training optimise a different stop/target pair than the
    # executor used. Volatility may widen either side, never narrow it.
    target = max(configured_target_bps, volatility_target, cost_floor)
    stop = max(
        _MINIMUM_STOP_BPS,
        configured_stop_bps,
        max(0.1, stop_volatility_k) * sigma_bps,
    )
    return target, stop, cost_floor > volatility_target


def _bars_are_contiguous(
    bars: Sequence[Bar],
    *,
    maximum_gap_seconds: float,
) -> bool:
    return all(
        0.0
        <= (current.start_time - previous.start_time).total_seconds()
        <= maximum_gap_seconds
        for previous, current in zip(bars, bars[1:])
    )


def _contiguous_future_bars(
    bars: Sequence[Bar],
    *,
    expected_start: datetime,
    maximum_gap_seconds: float,
) -> tuple[Bar, ...]:
    selected: list[Bar] = []
    previous_start = expected_start - timedelta(minutes=1)
    for bar in bars:
        gap = (bar.start_time - previous_start).total_seconds()
        if gap < 0.0 or gap > maximum_gap_seconds:
            break
        selected.append(bar)
        previous_start = bar.start_time
    return tuple(selected)


def _execution_market_for_symbol(
    symbol: str,
    config: EvaluationConfig,
) -> tuple[str, str, str]:
    normalized = str(symbol or "").upper().strip()
    if config.infer_market_from_symbol and normalized.isdigit() and len(normalized) == 6:
        return "KRX", "KR", "domestic_stock"
    if config.infer_market_from_symbol and normalized and normalized[0].isalpha():
        return "NASD", "US", "overseas_stock"
    return config.venue, config.market, config.instrument_type


def _label_features(
    schema_name: str,
    *,
    current,
    current_return: float,
    return_history: Sequence[float],
    close_history: Sequence[float],
    volume_history: Sequence[float],
    vwap_proxy: float,
    microstructure: BarMicrostructure | None,
    rvgi_result,
    box,
    quantiles: dict[str, float],
) -> tuple[float, ...]:
    if schema_name == STRATEGY_GRAPH_CONTEXT_SCHEMA:
        return _strategy_graph_context_features(
            current=current,
            current_return=current_return,
            close_history=close_history,
            volume_history=volume_history,
            microstructure=microstructure,
            rvgi_result=rvgi_result,
            box=box,
        )
    micro = _realtime_microstructure_proxy_features(
        current=current,
        current_return=current_return,
        return_history=return_history,
        vwap_proxy=vwap_proxy,
    )
    if schema_name == "realtime_microstructure_v1":
        return micro
    if schema_name in {
        "realtime_strategy_context_v2",
        "realtime_strategy_graph_v3",
    }:
        return (*micro, *_rvgi_box_proxy_features(current, rvgi_result, box))
    if schema_name == "realtime_strategy_graph_v4_market":
        is_krx = (
            str(getattr(current, "symbol", "") or "").isdigit()
            and len(str(getattr(current, "symbol", "") or "")) == 6
        )
        return (
            *micro,
            *_rvgi_box_proxy_features(current, rvgi_result, box),
            1.0 if is_krx else 0.0,
        )
    return (
        quantiles["return"],
        quantiles["volume"],
        quantiles["breakout"],
        quantiles["vwap_deviation"],
        quantiles["reversion"],
        quantiles["liquidity_shock"],
        quantiles["price_drop"],
        quantiles["recovery"],
        quantiles["liquidity"],
        quantiles["event_relevance"],
        quantiles["relative_strength"],
        quantiles["gap"],
    )


def _strategy_graph_context_features(
    *,
    current,
    current_return: float,
    close_history: Sequence[float],
    volume_history: Sequence[float],
    microstructure: BarMicrostructure | None,
    rvgi_result,
    box,
) -> tuple[float, ...]:
    """The aligned GNN context, built from the SAME source columns as serving.

    Every field here has a counterpart in
    :func:`app.features.live_feature_frame._strategy_graph_context_columns`
    computed by the same estimator over the same window, and the two are pinned
    together by ``tests/test_strategy_graph_context.py``.

    The v4 builder this replaces derived a "spread" from the bar's high-low range
    and an order-flow proxy from where the bar closed inside that range, while
    serving supplied a real spread and a real book imbalance in those slots --
    even though ``realtime_minute_bars`` had been persisting the real columns all
    along and ``load_minute_microstructure`` was already loading them. They were
    simply never routed into the feature vector.

    A bar with no microstructure ROW at all raises, because the snapshot cannot
    be placed. A row whose book sample is missing is a different thing and is
    reported through ``microstructure_available``: the store writes 0.0 rather
    than NULL for a minute it never sampled, and nine of ten KRX minutes are in
    that state, so raising there would discard almost the entire KRX catalogue.
    """
    if microstructure is None:
        raise StrategyGraphContextError(
            "strategy graph context requires persisted bar microstructure"
        )
    price = float(current.close)
    vwap = strategy_graph_context.volume_weighted_close(
        close_history,
        volume_history,
    )
    # Returns are derived from the window's own closes rather than taken from the
    # caller's ``return_history``. That list starts one bar EARLIER (its first
    # entry is the history window's opening bar measured against the bar before
    # it), so it holds 30 returns where the serving path holds 29 — a volatility
    # computed over a different sample on each side, which is the same class of
    # drift as the slot mismatch even though both are called "realized_volatility".
    window_returns = [
        float(close_history[position]) / float(close_history[position - 1]) - 1.0
        for position in range(1, len(close_history))
        if float(close_history[position - 1]) > 0
    ]
    box_high = float(box.high) if box.ok and box.high else 0.0
    return strategy_graph_context.build_strategy_graph_context(
        {
            **strategy_graph_context.microstructure_columns(
                microstructure.spread_bps,
                microstructure.orderbook_imbalance,
                microstructure.liquidity_score,
            ),
            "return_1m_scaled": strategy_graph_context.scaled_return(current_return),
            "realized_volatility_30m": strategy_graph_context.realized_volatility(
                window_returns
            ),
            "distance_from_vwap": (
                strategy_graph_context.safe_ratio(price, vwap) - 1.0 if vwap else 0.0
            ),
            "volume_spike_ratio": strategy_graph_context.volume_spike_ratio(
                float(current.volume),
                volume_history,
            ),
            "is_krx": strategy_graph_context.is_krx_symbol(
                getattr(current, "symbol", "")
            ),
            "rvgi_available": 1.0 if rvgi_result.ok else 0.0,
            "rvgi": float(rvgi_result.main or 0.0),
            "rvgi_signal": float(rvgi_result.signal or 0.0),
            "rvgi_diff": (
                float((rvgi_result.main or 0.0) - (rvgi_result.signal or 0.0))
                if rvgi_result.ok
                else 0.0
            ),
            "rvgi_slope": float(rvgi_result.slope or 0.0),
            "rvgi_bullish_cross": 1.0 if rvgi_result.bullish_cross else 0.0,
            "box_available": 1.0 if box.ok else 0.0,
            "box_high_ratio": strategy_graph_context.safe_ratio(box_high, price),
            "box_low_ratio": strategy_graph_context.safe_ratio(
                float(box.low) if box.ok and box.low else 0.0, price
            ),
            "box_mid_ratio": strategy_graph_context.safe_ratio(
                float(box.mid) if box.ok and box.mid else 0.0, price
            ),
            "box_width_pct": float(box.width_pct or 0.0),
            "box_position": float(box.position or 0.0),
            "breakout_distance_pct": (
                (price / box_high - 1.0) * 100.0 if box_high > 0 else 0.0
            ),
            # v4 hardcoded 0.0 in this slot while serving divided a real previous
            # close by the price, so the two sides disagreed by construction.
            "box_previous_close_ratio": strategy_graph_context.safe_ratio(
                close_history[-1] if close_history else 0.0, price
            ),
            "box_context_available": (
                1.0 if box.source_timestamp is not None else 0.0
            ),
        }
    )


def _rvgi_box_proxy_features(current, rvgi_result, box) -> tuple[float, ...]:
    scale = max(float(current.close), 1e-12)
    rvgi_available = 1.0 if rvgi_result.ok else 0.0
    box_available = 1.0 if box.ok else 0.0
    main = float(rvgi_result.main or 0.0)
    signal = float(rvgi_result.signal or 0.0)
    return (
        rvgi_available,
        main,
        signal,
        main - signal if rvgi_result.ok else 0.0,
        float(rvgi_result.slope or 0.0),
        1.0 if rvgi_result.bullish_cross else 0.0,
        box_available,
        float(box.high or 0.0) / scale,
        float(box.low or 0.0) / scale,
        float(box.mid or 0.0) / scale,
        float(box.width_pct or 0.0),
        float(box.position or 0.0),
        ((float(current.close) / float(box.high) - 1.0) * 100.0)
        if box.ok and box.high
        else 0.0,
        0.0,
        1.0 if box.source_timestamp is not None else 0.0,
    )


def _realtime_microstructure_proxy_features(
    *,
    current: Bar,
    current_return: float,
    return_history: Sequence[float],
    vwap_proxy: float,
) -> tuple[float, ...]:
    """Causal minute-bar proxy for the 12-field live slow-intelligence schema.

    Historical minute bars do not contain L2 depth.  Price, spread, signed-flow,
    VWAP, and volatility fields retain the live units while unavailable depth
    fields use neutral closed-world values.  Metadata records this proxy
    provenance so authorization remains limited to order-free shadow inference.
    """

    price_scale = 100_000.0
    bar_range = max(0.0, current.high - current.low)
    spread_bps = (bar_range / current.close * 10_000.0) if current.close else 0.0
    close_location = (
        ((current.close - current.low) - (current.high - current.close)) / bar_range
        if bar_range
        else 0.0
    )
    mean_return = fmean(return_history) if return_history else 0.0
    realized_volatility = (
        (
            sum((item - mean_return) ** 2 for item in return_history)
            / max(1, len(return_history))
        )
        ** 0.5
        if return_history
        else 0.0
    )
    signed_flow = current.volume * (1.0 if current_return > 0 else -1.0 if current_return < 0 else 0.0)
    return (
        current.close / price_scale,
        current.close / price_scale,
        spread_bps / 100.0,
        current.close / price_scale,
        max(-1.0, min(1.0, close_location)),
        signed_flow / 10_000.0,
        max(-1.0, min(1.0, current_return * 10_000.0)),
        vwap_proxy / price_scale,
        realized_volatility * 100.0,
        1.0,
        0.0,
        1.0,
    )


def build_report(
    database: Path,
    *,
    config: EvaluationConfig | None = None,
) -> dict[str, object]:
    selected_config = config or EvaluationConfig(align_strategy_horizons=True)
    bars_by_symbol = load_minute_bars(database)
    labels = build_labels(
        bars_by_symbol,
        selected_config,
        microstructure_by_symbol=load_minute_microstructure(database),
        investor_flow_by_symbol=InvestorFlowStore().load_all(),
        news_by_ticker=load_news_sentiment(),
    )
    config_json = json.dumps(asdict(selected_config), sort_keys=True, separators=(",", ":"))
    data_digest = hashlib.sha256()
    for row in labels:
        data_digest.update(repr(row).encode())
    code_digest = hashlib.sha256()
    for source in (
        Path(__file__),
        Path(__file__).parents[1] / "backtesting" / "event_simulator.py",
        Path(__file__).parents[1] / "strategy" / "experts.py",
    ):
        code_digest.update(source.read_bytes())

    by_strategy: dict[str, list[CounterfactualLabel]] = defaultdict(list)
    for row in labels:
        by_strategy[row.strategy_id].append(row)
    metrics = {}
    for strategy_id, rows in sorted(by_strategy.items()):
        fired = [row for row in rows if row.triggered]
        filled = [row for row in fired if row.filled]
        metrics[strategy_id] = {
            "labels": len(rows),
            "triggered": len(fired),
            "filled": len(filled),
            "fill_rate_when_triggered": len(filled) / len(fired) if fired else None,
            "mean_net_return_bps_when_filled": fmean(
                row.net_return_bps for row in filled
            )
            if filled
            else None,
            "positive_net_rate_when_filled": sum(row.net_return_bps > 0 for row in filled)
            / len(filled)
            if filled
            else None,
        }

    snapshots: dict[tuple[datetime, str], list[CounterfactualLabel]] = defaultdict(list)
    for row in labels:
        snapshots[(row.as_of, row.symbol)].append(row)
    ordered_keys = sorted(snapshots)
    intervals = [
        _Interval(key[0], max(row.label_end for row in snapshots[key])) for key in ordered_keys
    ]
    train_size = max(1, int(len(intervals) * 0.5))
    test_size = max(1, int(len(intervals) * 0.2))
    splits = (
        purged_walk_forward_splits(
            intervals,
            train_size=train_size,
            test_size=test_size,
            embargo_count=max(1, int(len(intervals) * 0.01)),
            step_size=test_size,
        )
        if len(intervals) >= train_size + test_size
        else ()
    )
    selected_returns: list[float] = []
    selected_trades = 0
    for split in splits:
        training_rows = [
            row
            for index in split.train_indices
            for row in snapshots[ordered_keys[index]]
            if row.triggered and row.filled
        ]
        strategy_means = {
            strategy_id: fmean(row.net_return_bps for row in training_rows if row.strategy_id == strategy_id)
            for strategy_id in {row.strategy_id for row in training_rows}
        }
        for index in split.test_indices:
            candidates = [
                row
                for row in snapshots[ordered_keys[index]]
                if row.triggered and row.filled
            ]
            candidate = max(
                candidates,
                key=lambda row: strategy_means.get(row.strategy_id, float("-inf")),
                default=None,
            )
            expected = (
                strategy_means.get(candidate.strategy_id, float("-inf"))
                if candidate is not None
                else float("-inf")
            )
            if candidate is not None and expected > 0:
                selected_returns.append(candidate.net_return_bps)
                selected_trades += 1
            else:
                selected_returns.append(0.0)

    unavailable = {
        "event_momentum": "point-in-time event feed absent",
        "cross_sectional_relative_strength": "point-in-time sector graph absent",
        "gap_context": "authoritative exchange session calendar absent",
        "legacy": "historical legacy decisions were not journaled against these snapshots",
        "temporal_rgcn": "no trained/calibrated checkpoint exists",
    }
    dense_symbols = sum(len(bars) >= selected_config.minimum_symbol_bars for bars in bars_by_symbol.values())
    observed_dates = {
        bar.start_time.date().isoformat() for bars in bars_by_symbol.values() for bar in bars
    }
    promotion_eligible = (
        len(observed_dates) >= 60
        and dense_symbols >= 100
        and not unavailable
        and selected_trades >= 100
    )
    return {
        "status": "NOT_PROMOTED" if not promotion_eligible else "ELIGIBLE_FOR_REVIEW",
        "promotion_eligible": promotion_eligible,
        "configuration": asdict(selected_config),
        "reproducibility": {
            "label_data_sha256": data_digest.hexdigest(),
            "config_sha256": hashlib.sha256(config_json.encode()).hexdigest(),
            "code_sha256": code_digest.hexdigest(),
        },
        "coverage": {
            "bars": sum(map(len, bars_by_symbol.values())),
            "symbols": len(bars_by_symbol),
            "symbols_meeting_minimum": dense_symbols,
            "distinct_utc_dates": len(observed_dates),
            "first_event": min(
                (bar.start_time.isoformat() for bars in bars_by_symbol.values() for bar in bars),
                default=None,
            ),
            "last_event": max(
                (bar.start_time.isoformat() for bars in bars_by_symbol.values() for bar in bars),
                default=None,
            ),
        },
        "labels": {"snapshots": len(snapshots), "strategy_labels": len(labels)},
        "leakage_audit": {
            "feature_cutoff_precedes_label_window": True,
            "rolling_quantiles_use_prior_slice_only": True,
            "walk_forward_splits": len(splits),
            "purging_enabled": True,
            "embargo_enabled": True,
        },
        "strategy_metrics": metrics,
        "walk_forward_tabular_baseline": {
            "policy": "train-window strategy mean; select only positive expected net utility",
            "observations": len(selected_returns),
            "selected_trades": selected_trades,
            "mean_net_return_bps": fmean(selected_returns) if selected_returns else None,
        },
        "mandatory_system_comparison": {
            "ontology_only": "NoTrade: required event/sector/session facts fail closed",
            "tabular_baseline": "evaluated above",
            "legacy": "UNAVAILABLE",
            "temporal_rgcn_cpu": "UNAVAILABLE_UNTRAINED",
            "temporal_rgcn_npu": "UNAVAILABLE_UNTRAINED_AND_NPU_BENCHMARK_REJECTED",
        },
        "unavailable_evidence": unavailable,
        "limitations": [
            "The local minute store is concentrated in a few US sessions, not the target KRX/NXT market.",
            "Minute OHLC cannot recover tick-level queue position or intrabar barrier ordering.",
            "Missing point-in-time events, sector graph, session calendar, and historical legacy decisions prevent the mandatory seven-strategy comparison.",
        ],
    }


def write_report(report: dict[str, object], output: Path) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
