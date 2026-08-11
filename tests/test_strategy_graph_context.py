"""The two producers of the GNN context must agree, field by field.

Nothing checked this before. The historical labelling path and the live serving
path each assembled a positional tuple of their own, and they disagreed in two
ways at once: slots 2/4/6 held different QUANTITIES on each side, and five slots
(``realized_volatility_3m``, ``box_high``, ``box_low``, ``box_mid``,
``box_previous_close``) were silently 0.0 at serving time because the live model
schema had dropped those column names and the adapter read them with
``.get(name, default)``.

The existing adapter test could not catch either failure: it fed a hand-written
dictionary through a stub frame, so every name it asked for was present by
construction. These tests drive BOTH producers from one set of minute-bar rows
and compare the vectors.
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pytest

from app.evaluation.stored_counterfactual import (
    BarMicrostructure,
    _strategy_graph_context_features,
)
from app.features import strategy_graph_context as ctx
from app.features.live_feature_frame import (
    _rvgi_box_columns,
    _strategy_graph_context_columns,
)
from app.technical import indicators as ti
from app.technical.causal_bars import completed_bars
from app.trading.contracts import Bar


SYMBOL = "005930"
START = datetime(2026, 8, 3, 0, 30, tzinfo=timezone.utc)


class _Row:
    """A ``realtime_minute_bars`` row, duck-typed as ``RealtimeMinuteBar``."""

    def __init__(self, index: int) -> None:
        # A deterministic but non-degenerate path: a drifting close with a
        # varying range, so no field collapses to a constant that would make the
        # two producers agree by accident.
        base = 70_000.0 + 90.0 * index + 220.0 * ((index % 7) - 3)
        self.symbol = SYMBOL
        self.minute_start = START + timedelta(minutes=index)
        self.open = base
        self.high = base + 130.0 + 12.0 * (index % 5)
        self.low = base - 115.0 - 9.0 * (index % 4)
        self.close = base + 45.0 * ((index % 3) - 1)
        self.volume = 1_200 + 130 * (index % 11)
        self.vwap = base + 8.0
        self.trade_count = 40 + index % 9
        self.spread_bps = 11.0 + 0.45 * (index % 6)
        self.orderbook_imbalance = 0.22 - 0.03 * (index % 5)
        self.liquidity_score = 0.61 + 0.01 * (index % 4)
        self.volatility = 0.0016
        self.last_update_age_ms = 120.0


def _rows(count: int = 90) -> tuple[_Row, ...]:
    return tuple(_Row(index) for index in range(count))


def _live_context(rows: tuple[_Row, ...], decision_time: datetime) -> dict[str, float]:
    bar_set = completed_bars(
        rows,
        symbol=SYMBOL,
        as_of=decision_time,
        timeframe_minutes=1,
        warmup_required=64,
    )
    price = float(bar_set.bars[-1].close)
    rvgi_box = _rvgi_box_columns(bar_set, SYMBOL, decision_time, price)
    return _strategy_graph_context_columns(
        bar_set,
        rows,
        symbol=SYMBOL,
        rvgi_box=rvgi_box,
    )


def _training_context(rows: tuple[_Row, ...], anchor: int) -> tuple[float, ...]:
    """The labelling path over the same rows, anchored on ``rows[anchor]``."""
    bars = tuple(
        Bar(
            symbol=SYMBOL,
            venue="KRX",
            interval="1m",
            start_time=row.minute_start,
            end_time=row.minute_start + timedelta(minutes=1),
            open=row.open,
            high=row.high,
            low=row.low,
            close=row.close,
            volume=float(row.volume),
        )
        for row in rows
    )
    current = bars[anchor]
    history_start = max(0, anchor - ctx.CONTEXT_HISTORY_BARS)
    history = bars[history_start:anchor]
    # The labelling path feeds indicators the bars completed BEFORE the anchor
    # plus the anchor itself, which is what ``completed_bars`` yields at a
    # decision time one minute past the anchor's start.
    completed = tuple(
        ti_bar
        for ti_bar in (
            _ohlcv(bar) for bar in bars[: anchor + 1]
        )
    )
    return _strategy_graph_context_features(
        current=current,
        current_return=current.close / history[-1].close - 1.0,
        close_history=[bar.close for bar in history],
        volume_history=[bar.volume for bar in history],
        microstructure=BarMicrostructure(
            vwap=rows[anchor].vwap,
            spread_bps=rows[anchor].spread_bps,
            orderbook_imbalance=rows[anchor].orderbook_imbalance,
            liquidity_score=rows[anchor].liquidity_score,
            volatility=rows[anchor].volatility,
            trade_count=rows[anchor].trade_count,
        ),
        rvgi_result=ti.rvgi(completed, 10),
        box=ti.causal_box_geometry(completed, 20),
    )


def _ohlcv(bar: Bar):
    from app.features.schemas import OHLCVBar

    return OHLCVBar(
        ticker=bar.symbol,
        as_of=bar.start_time,
        open=bar.open,
        high=bar.high,
        low=bar.low,
        close=bar.close,
        volume=bar.volume,
    )


def test_live_and_training_producers_emit_the_same_context() -> None:
    rows = _rows()
    anchor = len(rows) - 1
    # The decision must fall in the minute AFTER the anchor: ``completed_bars``
    # drops the minute containing ``as_of`` as still forming, so a decision
    # inside the anchor's own minute would leave the anchor out and silently
    # compare two different bars.
    decision_time = rows[anchor].minute_start + timedelta(seconds=90)

    live = ctx.build_strategy_graph_context(_live_context(rows, decision_time))
    training = _training_context(rows, anchor)

    assert len(live) == ctx.STRATEGY_GRAPH_CONTEXT_DIM
    named_live = ctx.as_context_mapping(live)
    named_training = ctx.as_context_mapping(training)
    disagreeing = {
        name: (named_training[name], named_live[name])
        for name in ctx.STRATEGY_GRAPH_CONTEXT_FIELDS
        if abs(named_training[name] - named_live[name]) > 1e-9
    }
    assert not disagreeing, f"training/serving disagree: {disagreeing}"


def test_microstructure_fields_are_the_persisted_columns_not_bar_range_proxies() -> None:
    """The v4 defect: a high-low range stood in for a spread during training.

    Pinned by value, so re-deriving these from the bar geometry fails here even
    if the vectors still happen to be the same length.
    """
    rows = _rows()
    anchor = len(rows) - 1
    named = ctx.as_context_mapping(_training_context(rows, anchor))

    assert named["microstructure_available"] == 1.0
    assert named["spread_bps_scaled"] == pytest.approx(rows[anchor].spread_bps / 100.0)
    assert named["orderbook_imbalance"] == pytest.approx(
        rows[anchor].orderbook_imbalance
    )
    assert named["liquidity_score"] == pytest.approx(rows[anchor].liquidity_score)


def test_unsampled_book_is_flagged_unavailable_not_reported_as_a_zero_spread() -> None:
    """The store writes 0.0, not NULL, for a minute it never sampled.

    ~90% of KRX bars are in that state. Passing the 0.0 through would teach the
    model that a zero spread — an impossible book — is the normal Korean market,
    so availability is a field and the columns behind it are zeroed together.
    """
    rows = _rows()
    anchor = len(rows) - 1
    rows[anchor].spread_bps = 0.0
    named = ctx.as_context_mapping(_training_context(rows, anchor))

    assert named["microstructure_available"] == 0.0
    assert named["spread_bps_scaled"] == 0.0
    assert named["orderbook_imbalance"] == 0.0
    assert named["liquidity_score"] == 0.0
    # The rest of the contract is unaffected: an unsampled book does not blank
    # the bar statistics, which come from OHLCV and are always present.
    assert named["realized_volatility_30m"] > 0.0


def test_both_producers_agree_when_the_book_was_never_sampled() -> None:
    rows = _rows()
    anchor = len(rows) - 1
    rows[anchor].spread_bps = 0.0
    decision_time = rows[anchor].minute_start + timedelta(seconds=90)

    live = ctx.as_context_mapping(
        ctx.build_strategy_graph_context(_live_context(rows, decision_time))
    )
    training = ctx.as_context_mapping(_training_context(rows, anchor))

    assert live["microstructure_available"] == 0.0
    assert training == pytest.approx(live)


def test_context_is_free_of_raw_price_levels() -> None:
    """Doubling every price must not move a single field.

    v4 carried the raw close in three separate slots and the VWAP level in a
    fourth, which encodes instrument identity — the same leak the live
    short-horizon schema removed when it dropped its raw depth columns.
    """
    rows = _rows()
    anchor = len(rows) - 1
    baseline = _training_context(rows, anchor)

    doubled = _rows()
    for row in doubled:
        row.open *= 2
        row.high *= 2
        row.low *= 2
        row.close *= 2
        row.vwap *= 2
    scaled = _training_context(doubled, anchor)

    named_base = ctx.as_context_mapping(baseline)
    named_scaled = ctx.as_context_mapping(scaled)
    moved = {
        name: (named_base[name], named_scaled[name])
        for name in ctx.STRATEGY_GRAPH_CONTEXT_FIELDS
        if abs(named_base[name] - named_scaled[name]) > 1e-9
    }
    assert not moved, f"price-level dependent fields: {moved}"


def test_missing_field_raises_instead_of_defaulting() -> None:
    values = {name: 0.0 for name in ctx.STRATEGY_GRAPH_CONTEXT_FIELDS}
    values.pop("orderbook_imbalance")
    with pytest.raises(ctx.StrategyGraphContextError) as excinfo:
        ctx.build_strategy_graph_context(values)
    assert "orderbook_imbalance" in str(excinfo.value)


def test_non_finite_field_raises() -> None:
    values = {name: 0.0 for name in ctx.STRATEGY_GRAPH_CONTEXT_FIELDS}
    values["realized_volatility_30m"] = float("nan")
    with pytest.raises(ctx.StrategyGraphContextError):
        ctx.build_strategy_graph_context(values)


def test_training_refuses_a_bar_without_persisted_microstructure() -> None:
    """No neutral value for "spread unknown" — the row drops out instead."""
    rows = _rows()
    anchor = len(rows) - 1
    bars = tuple(
        Bar(
            symbol=SYMBOL,
            venue="KRX",
            interval="1m",
            start_time=row.minute_start,
            end_time=row.minute_start + timedelta(minutes=1),
            open=row.open,
            high=row.high,
            low=row.low,
            close=row.close,
            volume=float(row.volume),
        )
        for row in rows
    )
    with pytest.raises(ctx.StrategyGraphContextError):
        _strategy_graph_context_features(
            current=bars[anchor],
            current_return=0.0,
            close_history=[bar.close for bar in bars[:anchor]],
            volume_history=[bar.volume for bar in bars[:anchor]],
            microstructure=None,
            rvgi_result=ti.rvgi((), 10),
            box=ti.causal_box_geometry((), 20),
        )


def test_volatility_uses_only_the_history_window_own_closes() -> None:
    """Both sides must measure the SAME sample, not just the same statistic.

    ``build_labels`` holds a ``return_history`` whose first entry measures the
    window's opening bar against the bar BEFORE the window — 30 returns where the
    serving path has 29. Feeding that in would have made every
    ``realized_volatility_30m`` disagree slightly while both sides still called
    it a 30-minute volatility, so the labelling builder derives its returns from
    ``close_history`` and takes no return series from its caller.
    """
    rows = _rows()
    anchor = len(rows) - 1
    named = ctx.as_context_mapping(_training_context(rows, anchor))

    history_closes = [
        row.close for row in rows[anchor - ctx.CONTEXT_HISTORY_BARS : anchor]
    ]
    expected = ctx.realized_volatility(
        [
            history_closes[position] / history_closes[position - 1] - 1.0
            for position in range(1, len(history_closes))
        ]
    )
    assert named["realized_volatility_30m"] == pytest.approx(expected)


def test_return_scale_does_not_saturate_at_one_basis_point() -> None:
    """v4 clipped ``return * 10_000`` to [-1, 1]: every move over 0.01% became a sign."""
    assert ctx.scaled_return(0.0005) == pytest.approx(0.1)   # 5bp
    assert ctx.scaled_return(0.002) == pytest.approx(0.4)    # 20bp
    assert ctx.scaled_return(0.01) == pytest.approx(1.0)     # saturates past 50bp
    assert ctx.scaled_return(0.0005) != ctx.scaled_return(0.002)
