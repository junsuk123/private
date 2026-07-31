"""Market intraday momentum: one round trip per day, flat before the auction.

Chosen on cost grounds. 20bps of the 27.8bps KRX round-trip cost is statutory tax
charged per ROUND TRIP, so twelve scalps pay it twelve times against a measured
gross edge of ~0bps. This strategy takes one trip per day.

The published effect (first half-hour return predicts the last half-hour return) is
strongest on volatile days, which is independently the only condition under which
the last half-hour travels far enough to clear ~33bps. These tests pin that gate and
the KRX-specific hazard: 15:20-15:30 is a closing single-price auction, and a
position must never be carried into it.
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pytest

from app.evaluation.stored_counterfactual import _intraday_momentum_quantiles
from app.strategy.catalog import STRATEGY_IDS
from app.strategy.exit_geometry import exit_geometry
from app.strategy.experts import MarketIntradayMomentumExpert
from app.technical.strategy_algorithms import (
    AlgorithmConfig,
    ElectionContext,
    MarketIntradayMomentumAlgorithm,
)
from app.technical.signals import TechnicalFeatureSet
from app.trading.contracts import Bar

KST = timezone(timedelta(hours=9))


# --------------------------------------------------------------------------- #
# Catalogue wiring                                                             #
# --------------------------------------------------------------------------- #
def test_strategy_is_catalogued_with_a_time_boxed_geometry() -> None:
    assert "market_intraday_momentum" in STRATEGY_IDS
    geometry = exit_geometry("market_intraday_momentum")
    # Time-boxed: 14:50 -> 15:15 with margin before the 15:20 auction.
    assert geometry.max_holding_seconds == 1500
    # Still obeys the table's stop/spread invariant.
    assert geometry.stop_loss_bps >= 60.0


# --------------------------------------------------------------------------- #
# Labeller features                                                            #
# --------------------------------------------------------------------------- #
def _bar(day: int, hour: int, minute: int, close: float, *, high=None, low=None) -> Bar:
    start = datetime(2026, 7, day, hour, minute, tzinfo=KST)
    return Bar(
        symbol="005930",
        venue="KRX",
        interval="1m",
        start_time=start,
        end_time=start + timedelta(minutes=1),
        open=close,
        high=high if high is not None else close,
        low=low if low is not None else close,
        close=close,
        volume=1000.0,
    )


def _session(day: int, *, first_half_hour_close: float, prev_close: float, spread: float):
    """One session: a prior close, a first half-hour, and a last continuous window."""
    bars = [_bar(day - 1, 15, 10, prev_close)]
    for minute in range(0, 30):
        bars.append(
            _bar(
                day,
                9,
                minute,
                first_half_hour_close,
                high=first_half_hour_close + spread,
                low=first_half_hour_close - spread,
            )
        )
    for minute in range(50, 60):
        bars.append(_bar(day, 14, minute, first_half_hour_close))
    return bars


def test_window_gate_blocks_everything_outside_the_last_continuous_half_hour() -> None:
    bars = _session(15, first_half_hour_close=101.0, prev_close=100.0, spread=1.0)
    # Index of a 09:xx bar — inside the first half-hour, not the entry window.
    q = _intraday_momentum_quantiles(bars, index=10, session_start=1)
    assert q["intraday_momentum_window"] == 0.0
    assert q["intraday_momentum_signal"] == 0.0


def test_positive_first_half_hour_inside_the_window_produces_a_signal() -> None:
    bars = _session(15, first_half_hour_close=101.0, prev_close=100.0, spread=1.0)
    q = _intraday_momentum_quantiles(bars, index=len(bars) - 1, session_start=1)
    assert q["intraday_momentum_window"] == 1.0
    # +100bps first half-hour -> above the 0.5 midpoint.
    assert q["intraday_momentum_signal"] > 0.5


def test_negative_first_half_hour_cannot_fire_long_only() -> None:
    """A down day predicts a down last half-hour, which this account cannot express."""
    bars = _session(15, first_half_hour_close=99.0, prev_close=100.0, spread=1.0)
    q = _intraday_momentum_quantiles(bars, index=len(bars) - 1, session_start=1)
    assert q["intraday_momentum_window"] == 1.0
    assert q["intraday_momentum_signal"] == 0.0


def test_missing_opening_window_does_not_fabricate_a_signal() -> None:
    """Without the opening half-hour there is no signal to have."""
    bars = [_bar(14, 15, 10, 100.0)]
    bars += [_bar(15, 14, minute, 101.0) for minute in range(50, 60)]
    q = _intraday_momentum_quantiles(bars, index=len(bars) - 1, session_start=1)
    assert q == {
        "intraday_momentum_signal": 0.0,
        "intraday_momentum_window": 0.0,
        "first_half_hour_volatility": 0.0,
    }


# --------------------------------------------------------------------------- #
# Expert admissibility                                                         #
# --------------------------------------------------------------------------- #
def _expert_ctx(**overrides):
    from app.strategy.experts import ExpertContext

    quantiles = {
        "intraday_momentum_signal": 0.9,
        "intraday_momentum_window": 1.0,
        "first_half_hour_volatility": 0.9,
        "liquidity": 0.9,
    }
    quantiles.update(overrides)
    return ExpertContext(
        symbol="005930",
        as_of=datetime(2026, 7, 31, 5, 55, tzinfo=timezone.utc),
        price=80_000.0,
        proposed_quantity=1,
        feature_snapshot_id="f",
        utility_evidence_id="u",
        quantiles=quantiles,
    )


def test_expert_requires_window_signal_and_volatility() -> None:
    expert = MarketIntradayMomentumExpert()
    assert expert.propose(_expert_ctx()) is not None
    # Outside the window there is nothing to trade.
    assert expert.propose(_expert_ctx(intraday_momentum_window=0.0)) is None
    # A flat/down first half-hour is not a signal.
    assert expert.propose(_expert_ctx(intraday_momentum_signal=0.4)) is None
    # A quiet day does not travel far enough to clear the round trip.
    assert expert.propose(_expert_ctx(first_half_hour_volatility=0.2)) is None


# --------------------------------------------------------------------------- #
# Live algorithm: the auction hazard                                           #
# --------------------------------------------------------------------------- #
def _features(**overrides) -> TechnicalFeatureSet:
    values = {
        "symbol": "005930",
        "price": 80_000.0,
        "second_data_ready": 1.0,
        "tick_count_5s": 10.0,
        "return_1s": 0.0002,
        "return_5s": 0.0008,
        "return_10s": 0.0015,
        "aggressor_imbalance_5s": 0.3,
        "realized_volatility_10s": 0.002,
        "realized_volatility": 0.003,
        "spread_change_5s": -0.0001,
        "orderbook_imbalance": 0.4,
        "spread_bps": 8.0,
    }
    values.update(overrides)
    return TechnicalFeatureSet(**values)


def _ctx(**overrides) -> ElectionContext:
    values = {
        "strategy_id": "market_intraday_momentum",
        "first_half_hour_return_bps": 80.0,
        "first_half_hour_volatility_percentile": 0.9,
        "in_last_continuous_half_hour": True,
        "minutes_to_continuous_close": 25.0,
    }
    values.update(overrides)
    return ElectionContext(**values)


def _algo() -> MarketIntradayMomentumAlgorithm:
    return MarketIntradayMomentumAlgorithm(AlgorithmConfig())


def test_algorithm_fires_on_a_volatile_up_day_inside_the_window() -> None:
    decision = _algo().entry(_features(), _ctx())
    assert decision.triggered is True
    assert "MIM_FIRST_HALF_HOUR_CONTINUATION" in decision.reason_codes


def test_algorithm_refuses_outside_the_window() -> None:
    decision = _algo().entry(_features(), _ctx(in_last_continuous_half_hour=False))
    assert decision.triggered is False
    assert "MIM_OUTSIDE_ENTRY_WINDOW" in decision.reason_codes


def test_algorithm_refuses_too_close_to_the_auction() -> None:
    """KRX matches 15:20-15:30 as a single-price auction. Opening a position with
    four minutes of continuous trading left leaves no way out of it."""
    decision = _algo().entry(_features(), _ctx(minutes_to_continuous_close=4.0))
    assert decision.triggered is False
    assert "MIM_TOO_CLOSE_TO_AUCTION" in decision.reason_codes


def test_algorithm_refuses_a_quiet_day() -> None:
    decision = _algo().entry(
        _features(), _ctx(first_half_hour_volatility_percentile=0.1)
    )
    assert decision.triggered is False
    assert "MIM_DAY_NOT_VOLATILE_ENOUGH" in decision.reason_codes


def test_algorithm_refuses_a_flat_first_half_hour() -> None:
    decision = _algo().entry(_features(), _ctx(first_half_hour_return_bps=3.0))
    assert decision.triggered is False
    assert "MIM_FIRST_HALF_HOUR_NOT_UP" in decision.reason_codes


def test_algorithm_fails_closed_without_session_context() -> None:
    """Session structure is not recoverable from the tick window; absent means no."""
    decision = _algo().entry(_features(), _ctx(in_last_continuous_half_hour=None))
    assert decision.triggered is False
    assert "MIM_SESSION_CONTEXT_ABSENT" in decision.reason_codes


def test_holding_horizon_shrinks_to_the_continuous_close() -> None:
    """A late entry must not inherit a leash that runs into the auction."""
    algo = _algo()
    late = algo.exit_rule(80_000.0, _features(), _ctx(minutes_to_continuous_close=8.0))
    early = algo.exit_rule(80_000.0, _features(), _ctx(minutes_to_continuous_close=25.0))
    assert late.max_holding_seconds < early.max_holding_seconds
    # 8 minutes left, 2 reserved -> at most 6 minutes of holding.
    assert late.max_holding_seconds <= 6 * 60


def test_imminent_close_invalidates_the_thesis() -> None:
    codes = _algo().invalidation(_features(), _ctx(minutes_to_continuous_close=1.0))
    assert "MIM_CONTINUOUS_CLOSE_IMMINENT" in codes


def test_not_live_authorized_until_it_has_evidence() -> None:
    """Only 2 of 360 stored symbol-days can evaluate this, so it must start shadow."""
    config = AlgorithmConfig()
    assert config.get("market_intraday_momentum", "live_authorized") == 0.0
