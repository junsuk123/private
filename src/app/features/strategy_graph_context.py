"""The one definition of the strategy-graph GNN context vector.

Both producers of this vector -- the historical labelling path in
``app.evaluation.stored_counterfactual`` and the live path in
``app.routing.shadow_intelligence`` -- must build it through
:func:`build_strategy_graph_context`. Before this module they each assembled a
positional tuple independently, and the two drifted apart in two different ways
that nothing reported:

1. **Different quantities in the same slot.** Slot 4 held ``close_location``
   (where a minute bar closed inside its own range) during training and
   ``orderbook_imbalance`` at serving time. Slot 6 held a clipped bar return
   during training and ``aggressor_imbalance_5s`` at serving time. Slot 2 held a
   high-low range proxy during training and a real ``spread_bps`` at serving
   time. A weight fitted on one quantity was applied to another.

2. **Silently defaulted slots.** The live adapter read its inputs with
   ``values.get(name, default)`` against the live frame's feature dictionary.
   When ``LIVE_FEATURE_NAMES`` later dropped columns, five of the twenty-eight
   slots -- ``realized_volatility_3m``, ``box_high``, ``box_low``, ``box_mid``,
   ``box_previous_close`` -- became a permanent 0.0 at serving time while
   training kept supplying real values. A missing key looked exactly like a
   measured zero.

Two rules follow, and both are enforced here rather than by convention:

- **Every field is defined once, by name.** Positional order lives in
  :data:`STRATEGY_GRAPH_CONTEXT_FIELDS` and nowhere else. Consumers that need a
  slot index ask :func:`context_index`.
- **There are no defaults.** A missing or non-finite field raises
  :class:`StrategyGraphContextError`. A producer that cannot supply a field is a
  broken producer, not a producer of zeros.

The field list is deliberately the INTERSECTION of what both sides can compute
with the same estimator over the same window, all of it derived from completed
one-minute bars and the microstructure columns persisted alongside them in
``realtime_minute_bars``. Quantities only one side can produce are excluded
rather than padded: ``aggressor_imbalance_5s`` is real and useful at serving
time, but historical minute bars cannot produce it, so a weight for it would
never be trained -- and applying an untrained weight is the failure this module
exists to prevent.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence


class StrategyGraphContextError(ValueError):
    """A producer failed to supply the context contract."""


#: Schema identifier stamped into checkpoints and snapshots. Changing the field
#: list MUST change this string: ``ShadowIntelligenceService`` compares it against
#: the checkpoint's ``input_feature_schema`` and refuses to score on a mismatch
#: (``MODEL_INPUT_SCHEMA_MISMATCH``), which is what keeps a stale checkpoint from
#: being fed a vector whose slots have moved.
STRATEGY_GRAPH_CONTEXT_SCHEMA = "realtime_strategy_graph_v6_trend"

#: Ordered context fields. Every name documents an estimator and a window; both
#: producers must implement THAT estimator over THAT window, not something
#: nearby. Where a name carries a scale (``_scaled``, ``_ratio``, ``_pct``) the
#: scale is part of the contract.
STRATEGY_GRAPH_CONTEXT_FIELDS: tuple[str, ...] = (
    # --- Microstructure of the last completed one-minute bar ---------------- #
    # Persisted columns of ``realtime_minute_bars``, read by both sides. These
    # replace the v4 slots where a high-low range stood in for a spread and a
    # close-location stood in for book imbalance.
    #
    # The store writes 0.0, not NULL, when a minute had no book sample, and a
    # zero spread is not a market state — best_bid == best_ask cannot happen in a
    # live book. Measured over the current store, only 10.5% of KRX bars (2,088
    # of 19,920) and 78.6% of US bars carry a real sample. So availability is a
    # FIELD, following the same convention as ``rvgi_available`` /
    # ``box_available``: when it is 0.0 the three columns below are 0.0 and the
    # model can tell "no book" from "balanced book" instead of learning that
    # nine of ten KRX minutes had a zero spread.
    "microstructure_available",
    "spread_bps_scaled",          # bar spread_bps / 100
    "orderbook_imbalance",        # bar orderbook_imbalance, [-1, 1]
    "liquidity_score",            # bar liquidity_score, [0, 1]
    # --- Statistics over the shared 30-bar history ------------------------- #
    # v4 put the raw price level in three separate slots (and the VWAP level in a
    # fourth). A price level identifies the instrument, so those slots taught the
    # model which symbol it was looking at; every field here is scale-free.
    "return_1m_scaled",           # last bar return in bps / 50, clipped [-1, 1]
    "realized_volatility_30m",    # stdev of 30 bar returns * 100
    "distance_from_vwap",         # close / 30-bar volume-weighted close - 1
    "volume_spike_ratio",         # bar volume / 30-bar mean volume, clipped [0, 10]
    "is_krx",                     # 1.0 for a six-digit KRX code
    # --- RVGI over completed bars ------------------------------------------ #
    "rvgi_available",
    "rvgi",
    "rvgi_signal",
    "rvgi_diff",
    "rvgi_slope",
    "rvgi_bullish_cross",
    # --- Box geometry over completed bars ---------------------------------- #
    # Ratios rather than levels, for the same identity-leak reason as above.
    "box_available",
    "box_high_ratio",
    "box_low_ratio",
    "box_mid_ratio",
    "box_width_pct",
    "box_position",
    "breakout_distance_pct",      # (close / box_high - 1) * 100
    "box_previous_close_ratio",
    "box_context_available",
    # --- Trend structure over completed bars -------------------------------- #
    # v5 spent fifteen of its twenty-four slots on RVGI and box geometry, and
    # carried nothing a trend thesis could read. The ontology therefore had no
    # relation to express for supertrend_dmi_continuation, bar_trend_continuation,
    # keltner_volatility_breakout or choppiness_range_reversion, and the closed-
    # world gate turned "cannot express" into a permanent veto — every trend and
    # volatility-regime arm in the catalogue was unreachable on every pass.
    #
    # Each field below is produced by the SAME estimator on both sides, because
    # both call ``trend_structure_columns`` rather than reimplementing it.
    "trend_available",            # 1.0 when the window satisfies the longest warmup
    "adx_scaled",                 # ADX(14) / 100, [0, 1]
    "dmi_spread_scaled",          # (plus_di - minus_di) / 100, [-1, 1]
    "supertrend_direction",       # -1.0 / 0.0 / 1.0
    "supertrend_distance_pct",    # (close / supertrend_line - 1) * 100
    "ema_separation_pct",         # (EMA9 / EMA21 - 1) * 100
    "ema_fast_distance_pct",      # (close / EMA9 - 1) * 100
    "atr_pct",                    # ATR(14) / close * 100
    "keltner_available",
    "keltner_position",           # (close - mid) / (upper - mid), [-5, 5]
    "keltner_bandwidth_pct",      # (upper - lower) / mid * 100
    "oscillator_available",
    "rsi_scaled",                 # RSI(14) / 100, [0, 1]
    "choppiness_scaled",          # CHOP(14) / 100, [0, 1]
    "bb_percent_b",               # Bollinger %B, [-5, 5]
    "bb_bandwidth_pct",           # (upper - lower) / mid * 100
    # --- Session structure -------------------------------------------------- #
    # Anchored to the session open, not to the rolling window: at 14:00 an opening
    # range is five hours behind the anchor bar. Producing these is what lets the
    # ontology express opening_range_breakout / _breakdown, gap_context and the two
    # market_intraday_momentum arms, all of which were CONTEXT_UNAVAILABLE.
    "session_structure_available",
    "opening_range_high_ratio",   # opening-range high / close
    "opening_range_low_ratio",    # opening-range low / close
    "opening_range_position",     # 0 at the range low, 1 at the high, unclamped
    "opening_range_width_pct",    # (high - low) / close * 100
    "minutes_since_session_open",
    "first_half_hour_return_pct",
    "session_gap_available",
    "session_gap_pct",            # (session open / previous close - 1) * 100
)

STRATEGY_GRAPH_CONTEXT_DIM = len(STRATEGY_GRAPH_CONTEXT_FIELDS)

_FIELD_INDEX = {name: index for index, name in enumerate(STRATEGY_GRAPH_CONTEXT_FIELDS)}

#: Clamps applied after the producers hand over their values. They bound a
#: mis-scaled input rather than silently accepting it into the model, and they
#: are part of the contract so both sides land on identical numbers.
_BOUNDS: dict[str, tuple[float, float]] = {
    "spread_bps_scaled": (0.0, 100.0),
    "orderbook_imbalance": (-1.0, 1.0),
    "liquidity_score": (0.0, 1.0),
    "return_1m_scaled": (-1.0, 1.0),
    "realized_volatility_30m": (0.0, 100.0),
    "distance_from_vwap": (-1.0, 1.0),
    "volume_spike_ratio": (0.0, 10.0),
    "box_position": (-5.0, 5.0),
    "breakout_distance_pct": (-100.0, 100.0),
    "adx_scaled": (0.0, 1.0),
    "dmi_spread_scaled": (-1.0, 1.0),
    "supertrend_direction": (-1.0, 1.0),
    "supertrend_distance_pct": (-100.0, 100.0),
    "ema_separation_pct": (-100.0, 100.0),
    "ema_fast_distance_pct": (-100.0, 100.0),
    "atr_pct": (0.0, 100.0),
    "keltner_position": (-5.0, 5.0),
    "keltner_bandwidth_pct": (0.0, 100.0),
    "rsi_scaled": (0.0, 1.0),
    "choppiness_scaled": (0.0, 1.0),
    "bb_percent_b": (-5.0, 5.0),
    "bb_bandwidth_pct": (0.0, 100.0),
    "opening_range_position": (-20.0, 20.0),
    "opening_range_width_pct": (0.0, 100.0),
    "minutes_since_session_open": (0.0, 1440.0),
    "first_half_hour_return_pct": (-100.0, 100.0),
    "session_gap_pct": (-100.0, 100.0),
}


def context_index(field: str) -> int:
    """Positional index of ``field``, for consumers that must read a slot.

    Exists so no caller hardcodes an integer. The v4 compatibility priors read
    ``features[4]`` and ``features[6]`` directly, which is why they kept scoring
    after the meaning of those slots diverged between training and serving.
    """
    try:
        return _FIELD_INDEX[field]
    except KeyError:
        raise StrategyGraphContextError(f"unknown context field: {field}") from None


def build_strategy_graph_context(values: Mapping[str, float]) -> tuple[float, ...]:
    """Ordered context vector, or raise.

    Strict on purpose: an absent field raises rather than defaulting, because the
    silent-zero path is the specific defect this contract replaces. Extra keys
    are ignored so a producer may pass a wider dictionary.
    """
    missing = [name for name in STRATEGY_GRAPH_CONTEXT_FIELDS if name not in values]
    if missing:
        raise StrategyGraphContextError(
            "strategy graph context missing fields: " + ", ".join(missing)
        )
    context: list[float] = []
    for name in STRATEGY_GRAPH_CONTEXT_FIELDS:
        try:
            value = float(values[name])
        except (TypeError, ValueError):
            raise StrategyGraphContextError(
                f"strategy graph context field {name} is not numeric"
            ) from None
        if not math.isfinite(value):
            raise StrategyGraphContextError(
                f"strategy graph context field {name} is not finite"
            )
        low, high = _BOUNDS.get(name, (-1e9, 1e9))
        context.append(max(low, min(high, value)))
    return tuple(context)


def as_context_mapping(context: Sequence[float]) -> dict[str, float]:
    """Name the values of an already-built context vector."""
    if len(context) != STRATEGY_GRAPH_CONTEXT_DIM:
        raise StrategyGraphContextError(
            f"context has {len(context)} fields, expected {STRATEGY_GRAPH_CONTEXT_DIM}"
        )
    return {
        name: float(value)
        for name, value in zip(STRATEGY_GRAPH_CONTEXT_FIELDS, context, strict=True)
    }


# --------------------------------------------------------------------------- #
# Shared estimators.
#
# Both producers call these rather than reimplementing them, so "the same
# estimator over the same window" is a fact about the code and not a comment.
# --------------------------------------------------------------------------- #

#: Bars of history behind the window statistics. Matches
#: ``EvaluationConfig.history_bars`` so the training window and the serving
#: window are the same length; a 30-bar volatility compared against a 3-minute
#: one was another way v4 disagreed with itself.
CONTEXT_HISTORY_BARS = 30

#: Divisor that maps a one-minute return in bps onto [-1, 1]. v4 clipped
#: ``return * 10_000`` to [-1, 1], which saturates at ONE basis point: every move
#: larger than 0.01% collapsed to the sign. 50bp keeps an intraday minute bar
#: inside the linear region.
RETURN_SCALE_BPS = 50.0


def scaled_return(bar_return: float) -> float:
    """One-minute return mapped onto [-1, 1] at :data:`RETURN_SCALE_BPS`."""
    if not math.isfinite(bar_return):
        return 0.0
    return max(-1.0, min(1.0, bar_return * 10_000.0 / RETURN_SCALE_BPS))


def realized_volatility(returns: Sequence[float]) -> float:
    """Population stdev of bar returns, in percent.

    Population rather than sample so a producer holding exactly one return still
    yields 0.0 instead of a division by zero.
    """
    finite = [float(value) for value in returns if math.isfinite(float(value))]
    if not finite:
        return 0.0
    mean = sum(finite) / len(finite)
    variance = sum((value - mean) ** 2 for value in finite) / len(finite)
    return math.sqrt(max(0.0, variance)) * 100.0


def volume_weighted_close(
    closes: Sequence[float],
    volumes: Sequence[float],
) -> float:
    """Volume-weighted close over the history window.

    Volumes are floored at 1.0 so a window of zero-volume bars degrades to a
    simple mean rather than dividing by zero.
    """
    if not closes:
        return 0.0
    weights = [max(float(volume), 1.0) for volume in volumes] or [1.0] * len(closes)
    total = sum(weights)
    if total <= 0:
        return float(closes[-1])
    return sum(
        float(close) * weight for close, weight in zip(closes, weights, strict=True)
    ) / total


def volume_spike_ratio(current_volume: float, history_volumes: Sequence[float]) -> float:
    """Current bar volume over the mean of the history window.

    Scale-free by construction: a raw share count is close to an instrument id,
    which is why the live short-horizon schema dropped its raw depth columns.
    """
    finite = [float(value) for value in history_volumes if math.isfinite(float(value))]
    mean = sum(finite) / len(finite) if finite else 0.0
    if mean <= 0:
        return 0.0
    return max(0.0, min(10.0, float(current_volume) / mean))


def safe_ratio(value: float | None, reference: float) -> float:
    """``value / reference`` with a zero for an unusable reference."""
    if value is None or reference is None:
        return 0.0
    try:
        numerator = float(value)
        denominator = float(reference)
    except (TypeError, ValueError):
        return 0.0
    if not math.isfinite(numerator) or not math.isfinite(denominator):
        return 0.0
    if abs(denominator) < 1e-12:
        return 0.0
    return numerator / denominator


def microstructure_columns(
    spread_bps: float | None,
    orderbook_imbalance: float | None,
    liquidity_score: float | None,
) -> dict[str, float]:
    """The four microstructure fields, with availability decided in ONE place.

    Availability keys off ``spread_bps > 0`` alone. Measured over the store the
    three columns move together: of the 2,088 KRX bars with a positive spread,
    2,088 also carry a non-zero imbalance, and a zero spread with a real
    imbalance occurs on 46 KRX and 102 US bars out of 80,203. One rule applied by
    both producers is worth more than three near-identical ones that can drift.
    """
    try:
        spread = float(spread_bps) if spread_bps is not None else 0.0
        imbalance = float(orderbook_imbalance) if orderbook_imbalance is not None else 0.0
        liquidity = float(liquidity_score) if liquidity_score is not None else 0.0
    except (TypeError, ValueError):
        spread = imbalance = liquidity = 0.0
    available = (
        math.isfinite(spread)
        and math.isfinite(imbalance)
        and math.isfinite(liquidity)
        and spread > 0.0
    )
    if not available:
        return {
            "microstructure_available": 0.0,
            "spread_bps_scaled": 0.0,
            "orderbook_imbalance": 0.0,
            "liquidity_score": 0.0,
        }
    return {
        "microstructure_available": 1.0,
        "spread_bps_scaled": spread / 100.0,
        "orderbook_imbalance": imbalance,
        "liquidity_score": liquidity,
    }


def is_krx_symbol(symbol: str | None) -> float:
    """1.0 for a six-digit KRX code, 0.0 otherwise."""
    text = str(symbol or "").strip()
    return 1.0 if text.isdigit() and len(text) == 6 else 0.0


#: Longest warmup among the trend estimators below. Keltner needs 20 bars for its
#: EMA basis plus an ATR, Bollinger needs 20, ADX needs 2x14. The window is
#: CONTEXT_HISTORY_BARS + the anchor bar, so a full window clears all of them; a
#: short one reports ``trend_available = 0.0`` rather than a partially-warmed number.
TREND_WARMUP_BARS = 28


def trend_structure_columns(bars: Sequence[object]) -> dict[str, float]:
    """Trend, volatility-regime and oscillator structure of a completed-bar window.

    The one implementation. Both producers call THIS, for the same reason
    :func:`microstructure_columns` exists: a near-copy on each side is how v4 came to
    serve one quantity into a slot fitted on another.

    Everything is scale-free — ratios, percentages and bounded oscillators, never a
    price level — so the model cannot learn which instrument it is looking at from a
    trend field, the identity leak the v4 notes describe.

    A window too short for the longest estimator returns ``trend_available = 0.0`` with
    zeros beside it. That is the same availability convention as ``rvgi_available`` and
    ``box_available``: the model can separate "no reading" from "a reading of zero",
    which for a signed field like ``dmi_spread_scaled`` are opposite claims.
    """
    from app.technical.indicators import (
        atr_percent,
        bollinger,
        choppiness_index,
        dmi_adx,
        ema,
        keltner_channels,
        rsi,
        supertrend,
    )

    blank = {
        "trend_available": 0.0,
        "adx_scaled": 0.0,
        "dmi_spread_scaled": 0.0,
        "supertrend_direction": 0.0,
        "supertrend_distance_pct": 0.0,
        "ema_separation_pct": 0.0,
        "ema_fast_distance_pct": 0.0,
        "atr_pct": 0.0,
        "keltner_available": 0.0,
        "keltner_position": 0.0,
        "keltner_bandwidth_pct": 0.0,
        "oscillator_available": 0.0,
        "rsi_scaled": 0.0,
        "choppiness_scaled": 0.0,
        "bb_percent_b": 0.0,
        "bb_bandwidth_pct": 0.0,
    }
    window = tuple(bars or ())
    if len(window) < TREND_WARMUP_BARS:
        return blank
    closes = [float(bar.close) for bar in window]
    price = closes[-1]
    if not math.isfinite(price) or price <= 0:
        return blank

    def pct(numerator: float | None, denominator: float) -> float:
        if numerator is None or denominator == 0.0:
            return 0.0
        ratio = float(numerator) / denominator - 1.0
        return ratio * 100.0 if math.isfinite(ratio) else 0.0

    columns = dict(blank)

    # ``trend_available`` follows the estimators' own warmup, not the bar count. A
    # 28-bar window is long enough to call ``dmi_adx`` and long enough for it to
    # return not-ok, and reporting available beside an unwarmed ADX of 0.0 would
    # publish exactly the partially-warmed number this convention exists to avoid.
    dmi = dmi_adx(window)
    trend = supertrend(window)
    fast = ema(closes, 9)
    slow = ema(closes, 21)
    if not (dmi.ok and trend.ok and fast is not None and slow is not None and slow > 0):
        return blank

    columns["trend_available"] = 1.0
    columns["adx_scaled"] = float(dmi.adx or 0.0) / 100.0
    columns["dmi_spread_scaled"] = float(dmi.dmi_spread or 0.0) / 100.0
    columns["supertrend_direction"] = float(trend.direction or 0.0)
    columns["supertrend_distance_pct"] = pct(price, float(trend.line or 0.0))
    columns["ema_separation_pct"] = pct(fast, slow)
    columns["ema_fast_distance_pct"] = pct(price, float(fast))
    columns["atr_pct"] = float(atr_percent(window) or 0.0)

    channel = keltner_channels(window)
    if channel.ok and channel.mid:
        columns["keltner_available"] = 1.0
        columns["keltner_position"] = float(channel.position or 0.0)
        columns["keltner_bandwidth_pct"] = float(channel.bandwidth or 0.0)

    strength = rsi(closes)
    chop = choppiness_index(window)
    bands = bollinger(closes)
    if strength is not None and chop is not None and bands.ok:
        columns["oscillator_available"] = 1.0
        columns["rsi_scaled"] = float(strength) / 100.0
        columns["choppiness_scaled"] = float(chop) / 100.0
        columns["bb_percent_b"] = float(bands.percent_b or 0.0)
        columns["bb_bandwidth_pct"] = float(bands.bandwidth or 0.0)
    return columns


#: Bars from the session's first print that the opening range spans. KRX and the US
#: both run a 30-minute opening auction-to-continuous settling period, and every
#: opening-range strategy in the catalogue is parameterised on it.
OPENING_RANGE_BARS = 30

#: How far back a producer must load for the session block to be computable at any
#: point in the day. KRX runs 390 continuous minutes and the US 390; the margin
#: carries the previous session's last close for the gap. Both producers bound their
#: lookback by this, so neither can silently see a different amount of history.
SESSION_CONTEXT_BARS = 450


def session_structure_columns(
    session_bars: Sequence[object],
    previous_session_close: float | None = None,
) -> dict[str, float]:
    """Where the current bar sits relative to the session's own opening structure.

    ``session_bars`` is every completed bar of the CURRENT session, oldest first, the
    anchor bar last. Not the rolling 30-bar window the trend block uses: an opening
    range is anchored to the session open, so at 14:00 the facts it needs are five
    hours behind the anchor and a rolling window cannot see them. That is why this
    takes its own span.

    ``previous_session_close`` is the last close of the prior session, for the gap.
    ``None`` means the producer could not reach back that far, and the gap reports
    unavailable rather than defaulting to "no gap" — which is a claim, not an absence.

    Scale-free like the rest of the contract: ratios and percentages, never a level.
    """
    blank = {
        "session_structure_available": 0.0,
        "opening_range_high_ratio": 0.0,
        "opening_range_low_ratio": 0.0,
        "opening_range_position": 0.0,
        "opening_range_width_pct": 0.0,
        "minutes_since_session_open": 0.0,
        "first_half_hour_return_pct": 0.0,
        "session_gap_available": 0.0,
        "session_gap_pct": 0.0,
    }
    bars = tuple(session_bars or ())
    if len(bars) < OPENING_RANGE_BARS:
        return blank
    price = float(bars[-1].close)
    if not math.isfinite(price) or price <= 0:
        return blank

    opening = bars[:OPENING_RANGE_BARS]
    highs = [float(bar.high) for bar in opening]
    lows = [float(bar.low) for bar in opening]
    range_high = max(highs)
    range_low = min(lows)
    if not math.isfinite(range_high) or range_high <= 0 or range_low <= 0:
        return blank

    columns = dict(blank)
    columns["session_structure_available"] = 1.0
    columns["opening_range_high_ratio"] = safe_ratio(range_high, price)
    columns["opening_range_low_ratio"] = safe_ratio(range_low, price)
    width = range_high - range_low
    columns["opening_range_width_pct"] = (width / price) * 100.0 if price else 0.0
    # 0.0 at the range low, 1.0 at the high, and deliberately unclamped beyond that:
    # "how far outside the range" IS the breakout thesis, so clipping it to the band
    # would erase the only part an opening-range strategy cares about.
    columns["opening_range_position"] = (
        (price - range_low) / width if width > 0 else 0.0
    )
    session_open = float(opening[0].open)
    columns["first_half_hour_return_pct"] = (
        (float(opening[-1].close) / session_open - 1.0) * 100.0
        if session_open > 0
        else 0.0
    )
    # Minutes rather than bar count: a halted or untraded minute writes no bar, and
    # "how far into the session" must not shrink because the tape went quiet.
    span = getattr(bars[-1], "as_of", None) or getattr(bars[-1], "start_time", None)
    start = getattr(opening[0], "as_of", None) or getattr(opening[0], "start_time", None)
    if span is not None and start is not None:
        columns["minutes_since_session_open"] = max(
            0.0, (span - start).total_seconds() / 60.0
        )

    if previous_session_close is not None and float(previous_session_close) > 0:
        columns["session_gap_available"] = 1.0
        columns["session_gap_pct"] = (
            session_open / float(previous_session_close) - 1.0
        ) * 100.0
    return columns


def _bar_time(bar: object):
    return getattr(bar, "as_of", None) or getattr(bar, "start_time", None)


def session_slice(bars: Sequence[object]) -> tuple[tuple, float | None]:
    """The current session's bars, and the previous session's last close.

    One boundary rule for both producers. The labelling path already derived sessions
    from a UTC date change (``_session_date_changed``) because the store holds no
    session metadata and neither KRX nor US regular hours straddle UTC midnight; this
    is that rule, shared, so the live path cannot disagree about where a session began.

    ``bars`` is oldest-first with the anchor last. The previous close is ``None`` when
    the window does not reach back into the prior session — an absence the caller
    reports rather than fills.
    """
    window = tuple(bars or ())
    if not window:
        return (), None
    start = 0
    for position in range(1, len(window)):
        previous_time = _bar_time(window[position - 1])
        current_time = _bar_time(window[position])
        if previous_time is None or current_time is None:
            continue
        if previous_time.date() != current_time.date():
            start = position
    previous_close: float | None = None
    if start > 0:
        candidate = float(getattr(window[start - 1], "close", 0.0) or 0.0)
        previous_close = candidate if candidate > 0 else None
    return window[start:], previous_close
