from __future__ import annotations

from datetime import datetime, timedelta, timezone
from types import SimpleNamespace
from zoneinfo import ZoneInfo

import pytest

from app.features.session_structure import (
    first_half_hour_return_bps,
    first_half_hour_volatility_percentile,
    opening_range,
)
from app.features.live_feature_frame import _session_structure_diagnostics
from app.evaluation.stored_counterfactual import _intraday_momentum_quantiles
from app.graph.micro_reasoner import MicroReasoningInput, MicroSymbolReasoner
from app.trading.contracts import Bar


KST = timezone(timedelta(hours=9))


def _bar(day: int, minute: int, close: float, *, high=None, low=None) -> Bar:
    start = datetime(2026, 7, day, 9, minute, tzinfo=KST)
    return Bar(
        symbol="005930",
        venue="KRX",
        interval="1m",
        start_time=start,
        end_time=start + timedelta(minutes=1),
        open=close,
        high=close if high is None else high,
        low=close if low is None else low,
        close=close,
        volume=100.0,
    )


def test_opening_range_is_absent_until_the_full_window_exists() -> None:
    session_open = datetime(2026, 7, 15, 9, 0, tzinfo=KST)
    bars = [_bar(15, minute, 100.0) for minute in range(29)]
    assert opening_range(
        bars, session_open=session_open, now=session_open + timedelta(minutes=29)
    ) is None


def test_opening_range_uses_only_visible_bars() -> None:
    session_open = datetime(2026, 7, 15, 9, 0, tzinfo=KST)
    bars = [
        _bar(15, minute, 100.0 + minute / 10, high=101.0 + minute, low=99.0)
        for minute in range(30)
    ]
    future = _bar(15, 30, 200.0, high=999.0, low=1.0)
    result = opening_range(
        [*bars, future],
        session_open=session_open,
        now=session_open + timedelta(minutes=30),
    )
    assert result is not None
    assert result.high == 130.0
    assert result.low == 99.0


def test_first_half_hour_return_is_previous_close_based_not_open_based() -> None:
    session_open = datetime(2026, 7, 15, 9, 0, tzinfo=KST)
    bars = [_bar(15, minute, 102.0) for minute in range(30)]
    result = first_half_hour_return_bps(
        bars,
        previous_close=100.0,
        session_open=session_open,
        now=session_open + timedelta(minutes=30),
    )
    assert result == pytest.approx(200.0)
    # Against a 102 open this would have been zero; pin the intended definition.
    assert result != 0.0


def test_first_half_hour_return_requires_a_valid_previous_close() -> None:
    session_open = datetime(2026, 7, 15, 9, 0, tzinfo=KST)
    bars = [_bar(15, minute, 102.0) for minute in range(30)]
    assert first_half_hour_return_bps(
        bars,
        previous_close=0.0,
        session_open=session_open,
        now=session_open + timedelta(minutes=30),
    ) is None


def test_volatility_percentile_requires_strictly_historical_samples() -> None:
    assert first_half_hour_volatility_percentile(
        [0.01, 0.02], 0.03, minimum_samples=3
    ) is None
    assert first_half_hour_volatility_percentile(
        [0.01, 0.02, 0.04], 0.03, minimum_samples=3
    ) == pytest.approx(2 / 3)


def test_live_and_stored_paths_share_the_same_session_values() -> None:
    stored: list[Bar] = []
    realtime = []
    for day, close, spread in (
        (10, 100.0, .5), (11, 101.0, 1.0), (12, 99.0, 1.5),
        (13, 100.0, 2.0), (14, 102.0, 1.2),
    ):
        # The prior close is visible before each session.  Only the last one's
        # value is used for the current first-half-hour return.
        close_start = datetime(2026, 7, day - 1, 15, 19, tzinfo=KST)
        stored.append(
            Bar("005930", "KRX", "1m", close_start, close_start + timedelta(minutes=1),
                close, close, close, close, 100.0)
        )
        realtime.append(SimpleNamespace(
            minute_start=close_start, open=close, high=close, low=close,
            close=close, volume=100,
        ))
        for minute in range(30):
            bar = _bar(day, minute, close + 1.0, high=close + 1.0 + spread, low=close + 1.0 - spread)
            stored.append(bar)
            realtime.append(SimpleNamespace(
                minute_start=bar.start_time, open=bar.open, high=bar.high,
                low=bar.low, close=bar.close, volume=100,
            ))
    for minute in range(50, 60):
        start = datetime(2026, 7, 14, 14, minute, tzinfo=KST)
        stored.append(Bar("005930", "KRX", "1m", start, start + timedelta(minutes=1),
                          103.0, 103.0, 103.0, 103.0, 100.0))
    stored.sort(key=lambda bar: bar.start_time)
    realtime.sort(key=lambda bar: bar.minute_start)
    current_start = next(
        index for index, bar in enumerate(stored)
        if bar.start_time == datetime(2026, 7, 14, 9, 0, tzinfo=KST)
    )
    now = datetime(2026, 7, 14, 15, 0, tzinfo=KST)
    live = _session_structure_diagnostics(
        SimpleNamespace(recent_minute_bars=lambda *_args, **_kwargs: tuple(realtime)),
        "005930", now,
    )
    stored_values = _intraday_momentum_quantiles(
        stored, len(stored) - 1, current_start
    )
    assert live["first_half_hour_return_bps"] == pytest.approx((103.0 / 102.0 - 1) * 10_000)
    assert live["first_half_hour_volatility_percentile"] == pytest.approx(
        stored_values["first_half_hour_volatility"]
    )
    assert stored_values["intraday_momentum_signal"] == pytest.approx(
        .5 + live["first_half_hour_return_bps"] / 200.0
    )


def test_us_gap_context_is_available_after_first_bar_not_after_thirty_minutes() -> None:
    ny = ZoneInfo("America/New_York")
    previous = datetime(2026, 8, 18, 15, 59, tzinfo=ny)
    current = datetime(2026, 8, 19, 9, 30, tzinfo=ny)
    rows = (
        SimpleNamespace(
            minute_start=previous, open=100.0, high=100.0, low=100.0,
            close=100.0, volume=100,
        ),
        SimpleNamespace(
            minute_start=current, open=102.0, high=102.5, low=101.8,
            close=102.2, volume=100,
        ),
    )
    result = _session_structure_diagnostics(
        SimpleNamespace(recent_minute_bars=lambda *_args, **_kwargs: rows),
        "INTC",
        datetime(2026, 8, 19, 13, 32, tzinfo=timezone.utc),
    )

    assert result["previous_close_price"] == 100.0
    assert result["session_open_price"] == 102.0
    assert result["gap_rate"] == pytest.approx(0.02)
    assert result["gap_submode"] == "continuation"
    assert "opening_range_high" not in result


def test_us_session_context_does_not_reuse_yesterday_before_regular_open() -> None:
    ny = ZoneInfo("America/New_York")
    old = datetime(2026, 8, 18, 15, 59, tzinfo=ny)
    rows = (
        SimpleNamespace(
            minute_start=old, open=100.0, high=100.0, low=100.0,
            close=100.0, volume=100,
        ),
    )
    result = _session_structure_diagnostics(
        SimpleNamespace(recent_minute_bars=lambda *_args, **_kwargs: rows),
        "INTC",
        datetime(2026, 8, 19, 13, 0, tzinfo=timezone.utc),
    )

    assert result == {}


def test_micro_diagnostics_preserve_measured_session_fields() -> None:
    frame = SimpleNamespace(
        diagnostics={
            "opening_range_high": 103.0,
            "opening_range_low": 99.0,
            "first_half_hour_return_bps": 98.0,
            "first_half_hour_volatility_percentile": .75,
        }
    )
    result = MicroSymbolReasoner().reason(
        MicroReasoningInput(
            timestamp=datetime(2026, 7, 14, 15, 0, tzinfo=KST),
            symbol="005930",
            live_feature_frame=frame,
            realtime_tick=SimpleNamespace(price=103.0),
        )
    )
    assert result.diagnostics["opening_range_high"] == 103.0
    assert result.diagnostics["first_half_hour_return_bps"] == 98.0
