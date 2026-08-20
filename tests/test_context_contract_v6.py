"""The v6 contract, and the coverage it buys.

v5 spent fifteen of its twenty-four slots on RVGI and box geometry and carried nothing
a trend, volatility-regime, range-regime or session thesis could read. The ontology
therefore had no relation to express for those strategies, and because ``compatible:{id}``
is a ``required_true`` gate fact, "cannot express" became a permanent veto: fifteen of
twenty-three arms were unreachable on every pass, and the fifteen were exactly the ones
covering the market regimes the surviving eight do not.
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pytest

from app.features import strategy_graph_context as ctx
from app.features.schemas import OHLCVBar
from app.routing.shadow_intelligence import compatibility_coverage

SYMBOL = "005930"


def _bars(count: int, *, drift: float, start: datetime, price: float = 10_000.0):
    out = []
    for index in range(count):
        price *= 1.0 + drift
        out.append(
            OHLCVBar(SYMBOL, start + timedelta(minutes=index),
                     price * 0.999, price * 1.003, price * 0.997, price, 1_000.0)
        )
    return out


DAY = datetime(2026, 8, 20, 0, 0, tzinfo=timezone.utc)


# --------------------------------------------------------------------------- #
# Availability is the estimator's warmup, not the bar count
# --------------------------------------------------------------------------- #
def test_a_short_window_publishes_no_trend_reading_at_all() -> None:
    columns = ctx.trend_structure_columns(_bars(12, drift=0.001, start=DAY))
    assert columns["trend_available"] == 0.0
    assert all(value == 0.0 for value in columns.values())


def test_an_unwarmed_estimator_does_not_report_available() -> None:
    """28 bars is enough to CALL dmi_adx and enough for it to decline.

    Reporting available beside an ADX of 0.0 is the partially-warmed number the
    availability convention exists to prevent.
    """
    assert ctx.trend_structure_columns(_bars(28, drift=0.0015, start=DAY))["trend_available"] == 0.0
    assert ctx.trend_structure_columns(_bars(31, drift=0.0015, start=DAY))["trend_available"] == 1.0


# --------------------------------------------------------------------------- #
# Direction
# --------------------------------------------------------------------------- #
def test_trend_fields_are_signed_and_symmetric() -> None:
    up = ctx.trend_structure_columns(_bars(60, drift=0.002, start=DAY))
    down = ctx.trend_structure_columns(_bars(60, drift=-0.002, start=DAY))

    assert up["supertrend_direction"] == 1.0
    assert down["supertrend_direction"] == -1.0
    assert up["dmi_spread_scaled"] > 0.0 > down["dmi_spread_scaled"]
    assert up["ema_separation_pct"] > 0.0 > down["ema_separation_pct"]
    assert up["supertrend_distance_pct"] > 0.0 > down["supertrend_distance_pct"]
    # Price above its own fast EMA in an uptrend, below it in a downtrend.
    assert up["ema_fast_distance_pct"] > 0.0 > down["ema_fast_distance_pct"]


def test_trend_fields_carry_no_price_level() -> None:
    """Doubling every price must not move a scale-free field."""
    cheap = ctx.trend_structure_columns(_bars(60, drift=0.001, start=DAY, price=1_000.0))
    dear = ctx.trend_structure_columns(_bars(60, drift=0.001, start=DAY, price=500_000.0))
    for field, value in cheap.items():
        assert value == pytest.approx(dear[field], abs=1e-6), field


# --------------------------------------------------------------------------- #
# Session structure
# --------------------------------------------------------------------------- #
def test_session_slice_finds_the_boundary_and_the_previous_close() -> None:
    first = _bars(60, drift=0.0, start=datetime(2026, 8, 19, 1, 0, tzinfo=timezone.utc))
    second = _bars(40, drift=0.0, start=datetime(2026, 8, 20, 1, 0, tzinfo=timezone.utc))
    current, previous_close = ctx.session_slice(first + second)

    assert len(current) == 40
    assert previous_close == pytest.approx(first[-1].close)


def test_a_window_that_never_reaches_the_prior_session_reports_no_gap() -> None:
    """Absent is not "no gap" — that would be a claim about the open."""
    columns = ctx.session_structure_columns(*ctx.session_slice(_bars(60, drift=0.0, start=DAY)))
    assert columns["session_structure_available"] == 1.0
    assert columns["session_gap_available"] == 0.0
    assert columns["session_gap_pct"] == 0.0


def test_the_gap_is_measured_open_against_the_previous_close() -> None:
    bars = _bars(60, drift=0.0, start=DAY, price=10_000.0)
    columns = ctx.session_structure_columns(bars, previous_session_close=9_800.0)
    assert columns["session_gap_available"] == 1.0
    assert columns["session_gap_pct"] == pytest.approx(
        (bars[0].open / 9_800.0 - 1.0) * 100.0
    )


def test_opening_range_position_is_not_clipped_at_the_band() -> None:
    """How far OUTSIDE the range is the entire breakout thesis."""
    columns = ctx.session_structure_columns(_bars(120, drift=0.0008, start=DAY))
    assert columns["opening_range_position"] > 1.0


def test_a_session_shorter_than_the_opening_range_is_unavailable() -> None:
    columns = ctx.session_structure_columns(_bars(10, drift=0.001, start=DAY))
    assert columns["session_structure_available"] == 0.0


# --------------------------------------------------------------------------- #
# What the contract buys
# --------------------------------------------------------------------------- #
def test_every_trend_and_session_arm_is_now_expressible() -> None:
    """These were CONTEXT_UNAVAILABLE, which the closed-world gate reads as a veto."""
    coverage = compatibility_coverage()
    for strategy_id in (
        "supertrend_dmi_continuation",
        "bar_trend_continuation",
        "keltner_volatility_breakout",
        "choppiness_range_reversion",
        "ofi_microprice_exhaustion_reversal",
        "adaptive_anchored_vwap_reversion",
        "opening_range_breakout",
        "opening_range_breakdown",
        "gap_context",
        "market_intraday_momentum",
        "market_intraday_momentum_short",
    ):
        assert coverage[strategy_id] == "COMPUTED", strategy_id


def test_the_remaining_gap_is_cross_sectional_and_event_facts_only() -> None:
    """Honest about what is still unreachable, and why.

    Peer ranking and residuals need OTHER symbols at the same instant, which neither
    producer's per-symbol window holds; event facts are a separate source entirely.
    Both are architectural, not a matter of adding a field.
    """
    blocked = {k: v for k, v in compatibility_coverage().items() if v != "COMPUTED"}
    assert set(blocked) == {
        "cross_sectional_relative_strength",
        "residual_relative_strength",
        "residual_relative_weakness",
        "event_momentum",
    }


# --------------------------------------------------------------------------- #
# The book-sample factor belongs to theses that read the book
# --------------------------------------------------------------------------- #
def _live_like_context(**overrides) -> tuple[float, ...]:
    """A context in a clean uptrend with a healthy book, before overrides."""
    values = {name: 0.0 for name in ctx.STRATEGY_GRAPH_CONTEXT_FIELDS}
    values.update(
        microstructure_available=1.0,
        spread_bps_scaled=0.05,
        orderbook_imbalance=0.30,
        liquidity_score=0.80,
        return_1m_scaled=0.40,
        realized_volatility_30m=0.30,
        distance_from_vwap=0.004,
        volume_spike_ratio=1.5,
        is_krx=1.0,
        rvgi_available=1.0,
        rvgi_bullish_cross=1.0,
        box_available=1.0,
        box_position=0.7,
        box_context_available=1.0,
        trend_available=1.0,
        adx_scaled=0.35,
        dmi_spread_scaled=0.20,
        supertrend_direction=1.0,
        supertrend_distance_pct=0.8,
        ema_separation_pct=0.5,
        ema_fast_distance_pct=0.3,
        atr_pct=0.4,
        keltner_available=1.0,
        keltner_position=0.9,
        keltner_bandwidth_pct=1.2,
        oscillator_available=1.0,
        rsi_scaled=0.62,
        choppiness_scaled=0.70,
        bb_percent_b=0.35,
        bb_bandwidth_pct=1.1,
        session_structure_available=1.0,
        opening_range_position=1.4,
        opening_range_width_pct=1.0,
        minutes_since_session_open=200.0,
        first_half_hour_return_pct=0.8,
        session_gap_available=1.0,
        session_gap_pct=0.9,
    )
    values.update(overrides)
    return ctx.build_strategy_graph_context(values)


def _passing(context) -> set[str]:
    from app.routing.shadow_intelligence import _compatibility_with_provenance

    scores, _ = _compatibility_with_provenance(context)
    return {name for name, score in scores.items() if score > 0.0}


def test_an_unsampled_book_does_not_veto_a_bar_thesis() -> None:
    """The store writes 0.0, not NULL, for a minute it never sampled — ~89.5% of KRX
    bars. ``spread_quality`` is zero there, so multiplying every relation by it says
    "we did not look" and reads as "the thesis failed". A three-hour trend does not
    stop being a trend because one minute went unsampled.
    """
    unsampled = _live_like_context(
        microstructure_available=0.0,
        spread_bps_scaled=0.0,
        orderbook_imbalance=0.0,
        liquidity_score=0.0,
    )
    survive = _passing(unsampled)
    for strategy_id in (
        "bar_trend_continuation",
        "keltner_volatility_breakout",
        "choppiness_range_reversion",
        "opening_range_breakout",
        "market_intraday_momentum",
        "supertrend_dmi_continuation",
    ):
        assert strategy_id in survive, strategy_id


def test_an_unsampled_book_still_withholds_a_book_thesis() -> None:
    """Scoping the factor must not become removing it.

    These read the book directly, so an unsampled minute genuinely means the relation
    cannot be evaluated.
    """
    unsampled = _live_like_context(
        microstructure_available=0.0,
        spread_bps_scaled=0.0,
        orderbook_imbalance=0.0,
        liquidity_score=0.0,
    )
    survive = _passing(unsampled)
    assert "ofi_microprice_exhaustion_reversal" not in survive
    assert "adaptive_anchored_vwap_reversion" not in survive


def test_a_healthy_book_admits_strictly_more_than_an_unsampled_one() -> None:
    sampled = _passing(_live_like_context())
    unsampled = _passing(
        _live_like_context(
            microstructure_available=0.0,
            spread_bps_scaled=0.0,
            orderbook_imbalance=0.0,
            liquidity_score=0.0,
        )
    )
    assert unsampled < sampled
