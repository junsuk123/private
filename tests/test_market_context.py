"""MarketContext construction: deterministic, attributed, and never fabricated."""

from __future__ import annotations

from datetime import datetime, timezone

import pytest

from app.context import (
    CONTEXT_NO_ORDERBOOK_SAMPLE,
    CONTEXT_NO_REFERENCE_PRICE,
    CONTEXT_NO_TICK_WINDOW,
    MarketContextBuilder,
    MarketContextStore,
    SymbolContextInputs,
    declared_context_fields,
)
from app.technical.signals import TechnicalFeatureSet

AT = datetime(2026, 8, 11, 4, 0, 0, tzinfo=timezone.utc)


def _features(**overrides) -> TechnicalFeatureSet:
    values = {
        "symbol": "005930",
        "price": 70_000.0,
        "ema_fast": 70_100.0,
        "ema_slow": 69_900.0,
        "vwap_distance_bps": -30.0,
        "spread_bps": 12.0,
        "orderbook_imbalance": 0.2,
        "liquidity_score": 0.6,
        "aggressor_imbalance_5s": 0.15,
        "realized_volatility": 0.0009,
        "realized_volatility_10s": 0.0006,
        "return_5s": 0.0004,
        "return_30s": 0.0009,
        "tick_count_5s": 8.0,
        "second_data_ready": 1.0,
        "donchian_low": 69_500.0,
        "momentum_persistence": 0.6,
        "relative_volume": 1.4,
        "bid_depth": 1_000.0,
        "ask_depth": 800.0,
    }
    values.update(overrides)
    return TechnicalFeatureSet(**values)


def _inputs(**overrides) -> SymbolContextInputs:
    values = {
        "symbol": "005930",
        "features": _features(),
        "context_id": "ctx-fixed",
        "tick_freshness_sec": 0.5,
        "orderbook_freshness_sec": 0.8,
        "history_bar_count": 40,
        "election_inputs": {"sector_rank": 2, "sector_candidate_count": 5},
    }
    values.update(overrides)
    return SymbolContextInputs(**values)


def test_construction_is_deterministic() -> None:
    builder = MarketContextBuilder()
    first = builder.build(_inputs(), captured_at=AT)
    second = builder.build(_inputs(), captured_at=AT)
    assert first.as_dict() == second.as_dict()


def test_context_id_and_market_are_derived() -> None:
    builder = MarketContextBuilder()
    krx = builder.build(_inputs(), captured_at=AT)
    us = builder.build(
        _inputs(symbol="AAPL", features=_features(symbol="AAPL")), captured_at=AT
    )
    assert krx.market == "KR"
    assert us.market == "US"
    assert krx.context_id == "ctx-fixed"


def test_one_cycle_shares_captured_at_across_symbols() -> None:
    builder = MarketContextBuilder()
    contexts = builder.build_cycle(
        (
            _inputs(context_id=None),
            _inputs(symbol="000660", features=_features(symbol="000660"), context_id=None),
        ),
        captured_at=AT,
    )
    assert len({context.captured_at for context in contexts}) == 1
    # Distinct ids: two symbols in one cycle are two contexts, not one.
    assert len({context.context_id for context in contexts}) == 2


def test_zero_spread_is_absent_not_measured() -> None:
    builder = MarketContextBuilder()
    context = builder.build(
        _inputs(features=_features(spread_bps=0.0)), captured_at=AT
    )
    assert context.microstructure.spread_bps is None
    assert CONTEXT_NO_ORDERBOOK_SAMPLE in context.reason_codes
    assert not context.has("spread_bps")


def test_missing_fields_are_none_and_reported() -> None:
    builder = MarketContextBuilder()
    context = builder.build(
        _inputs(features=_features(price=None, second_data_ready=0.0)), captured_at=AT
    )
    assert context.symbol.reference_price is None
    assert CONTEXT_NO_REFERENCE_PRICE in context.reason_codes
    assert CONTEXT_NO_TICK_WINDOW in context.reason_codes
    # A regime label was never supplied, so it stays absent rather than defaulting.
    assert context.macro.market_regime is None


def test_feature_completeness_reflects_what_is_present() -> None:
    builder = MarketContextBuilder()
    rich = builder.build(_inputs(), captured_at=AT)
    sparse = builder.build(
        SymbolContextInputs(
            symbol="005930",
            features=TechnicalFeatureSet(symbol="005930"),
            context_id="ctx-sparse",
        ),
        captured_at=AT,
    )
    assert 0.0 < sparse.data_quality.feature_completeness < rich.data_quality.feature_completeness
    assert rich.data_quality.feature_completeness <= 1.0


def test_every_present_field_has_a_source() -> None:
    builder = MarketContextBuilder()
    context = builder.build(_inputs(), captured_at=AT)
    attributed = {"feature_completeness", "history_bar_count", "second_level_data_ready",
                  "tick_freshness_sec", "orderbook_freshness_sec"}
    for name in declared_context_fields():
        if name in attributed or not context.has(name):
            continue
        assert context.source_for(name) is not None, f"{name} has no provenance"


def test_flat_field_names_do_not_collide_across_groups() -> None:
    context = MarketContextBuilder().build(_inputs(), captured_at=AT)
    declared = declared_context_fields()
    assert len(declared) == len(set(declared))
    assert set(context.flat()) == set(declared)


def test_numeric_view_excludes_labels_and_includes_booleans() -> None:
    context = MarketContextBuilder().build(_inputs(), captured_at=AT)
    numeric = context.numeric()
    assert "session_phase" not in numeric
    assert numeric["is_opening_window"] in {0.0, 1.0}


def test_short_direction_geometry_of_context_is_not_assumed() -> None:
    """A context carries no direction; only strategies do."""
    context = MarketContextBuilder().build(_inputs(), captured_at=AT)
    assert not hasattr(context, "direction")


def test_store_is_bounded_and_keyed_by_id() -> None:
    store = MarketContextStore(max_contexts=2, retention_seconds=3600.0)
    builder = MarketContextBuilder()
    for index in range(3):
        store.put(
            builder.build(_inputs(context_id=f"ctx-{index}"), captured_at=AT)
        )
    assert len(store) == 2
    assert store.get("ctx-0") is None
    assert store.get("ctx-2") is not None
    assert store.latest_for_symbol("005930").context_id == "ctx-2"
