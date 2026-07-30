from __future__ import annotations

import math
from datetime import datetime, timedelta, timezone

import pytest

from app.features.schemas import OHLCVBar
from app.strategy.catalog import STRATEGY_IDS
from app.technical.indicators import causal_box_geometry, rvgi
from app.technical.signals import TechnicalFeatureSet
from app.technical.strategy_algorithms import (
    ElectionContext,
    RvgiBoxBreakoutAlgorithm,
)


NOW = datetime(2026, 7, 29, tzinfo=timezone.utc)


def _bars(count: int = 30, *, phase: float = 5.1) -> tuple[OHLCVBar, ...]:
    rows = []
    for index in range(count):
        open_price = 100.0 + 0.05 * index
        close = open_price + 0.8 * math.sin(index * 0.7 + phase)
        rows.append(
            OHLCVBar(
                "TEST",
                NOW + timedelta(minutes=index),
                open_price,
                max(open_price, close) + 0.5,
                min(open_price, close) - 0.5,
                close,
                100.0 + index,
            )
        )
    return tuple(rows)


def test_strategy_catalog_is_append_only_and_rvgi_is_eighth() -> None:
    assert STRATEGY_IDS[:7] == (
        "intraday_momentum",
        "breakout_volume",
        "vwap_mean_reversion",
        "liquidity_shock_reversal",
        "event_momentum",
        "cross_sectional_relative_strength",
        "gap_context",
    )
    assert STRATEGY_IDS[7] == "rvgi_box_breakout"


def test_rvgi_hand_calculated_fixture_and_cross() -> None:
    result = rvgi(_bars(), 10)
    assert result.ok
    assert result.main == pytest.approx(-0.030514086171320523)
    assert result.signal == pytest.approx(-0.032639870638760414)
    assert result.previous_main == pytest.approx(-0.04244894094709963)
    assert result.previous_signal == pytest.approx(-0.01913962685968001)
    assert result.bullish_cross is True
    assert result.bearish_cross is False


def test_rvgi_insufficient_and_zero_range_fail_safe() -> None:
    assert not rvgi(_bars(15), 10).ok
    flat = tuple(
        OHLCVBar("TEST", NOW + timedelta(minutes=i), 100, 100, 100, 100, 1)
        for i in range(30)
    )
    result = rvgi(flat, 10)
    assert result.ok
    assert result.main == 0
    assert result.signal == 0
    assert result.bullish_cross is False
    assert math.isfinite(result.main)


def test_box_excludes_signal_bar_and_future_bars_do_not_change_past() -> None:
    bars = list(_bars())
    original = causal_box_geometry(tuple(bars), 20)
    assert original.ok
    bars[-1] = OHLCVBar("TEST", bars[-1].as_of, 100, 10_000, 1, 9_000, 1_000)
    changed_signal = causal_box_geometry(tuple(bars), 20)
    assert changed_signal.high == original.high
    assert changed_signal.low == original.low
    future = (*_bars(), OHLCVBar("TEST", NOW + timedelta(minutes=99), 1, 99_999, 1, 2, 1))
    assert causal_box_geometry(future[:-1], 20) == original


def _context() -> ElectionContext:
    return ElectionContext(
        strategy_id="rvgi_box_breakout",
        rvgi=0.10,
        rvgi_signal=0.05,
        rvgi_diff=0.05,
        rvgi_bullish_cross=True,
        box_high=100.0,
        box_low=98.0,
        box_mid=99.0,
        box_width_pct=2.0 / 99.0,
        box_position=1.0,
        box_context_timestamp=NOW.isoformat(),
        box_previous_close=99.8,
        volume_confirmed=True,
    )


def _features(**overrides) -> TechnicalFeatureSet:
    values = dict(
        symbol="TEST",
        price=100.10,
        volume_spike_ratio=2.0,
        return_1s=0.0001,
        return_5s=0.0003,
        tick_count_5s=5.0,
        aggressor_imbalance_5s=0.2,
        realized_volatility_10s=0.001,
        second_data_ready=1.0,
    )
    values.update(overrides)
    return TechnicalFeatureSet(**values)


def test_dedicated_rvgi_box_entry_and_rejections() -> None:
    algorithm = RvgiBoxBreakoutAlgorithm()
    assert algorithm.entry(_features(), _context()).triggered
    no_rvgi = ElectionContext(**{**_context().__dict__, "rvgi_bullish_cross": False})
    assert "RVGI_BOX_RVGI_NOT_CONFIRMED" in algorithm.entry(_features(), no_rvgi).reason_codes
    assert "RVGI_BOX_NOT_ABOVE_FROZEN_HIGH" in algorithm.entry(
        _features(price=100.01), _context()
    ).reason_codes
    assert "RVGI_BOX_VOLUME_NOT_CONFIRMED" in algorithm.entry(
        _features(volume_spike_ratio=1.0), _context()
    ).reason_codes
    assert "RVGI_BOX_OVEREXTENDED" in algorithm.entry(
        _features(price=101.0), _context()
    ).reason_codes


def test_exit_rule_and_invalidation_use_frozen_box() -> None:
    algorithm = RvgiBoxBreakoutAlgorithm()
    rule = algorithm.exit_rule(100.1, _features(), _context())
    assert rule.stop_price is not None and rule.stop_price <= 100.0
    assert rule.target_price == pytest.approx(101.1)
    assert "RVGI_BOX_FALSE_BREAKOUT" in algorithm.invalidation(
        _features(price=99.8), _context()
    )
