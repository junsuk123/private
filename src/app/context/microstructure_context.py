"""Microstructure slice: the book and the tape, not the chart.

``microprice_bias`` reuses ``TechnicalFeatureSet.microprice_edge_bps`` rather than
recomputing it. That property is an identity for a two-sided book
(``microprice - mid = (spread/2) * depth_imbalance``), and having two implementations of
an identity is how they drift.

The availability rule matches ``app.features.strategy_graph_context.microstructure_columns``:
a zero spread is an ABSENT book sample, not a market state — ``best_bid == best_ask``
cannot happen in a live book. Measured over the current store, only 10.5% of KRX minute
bars carry a real sample, so treating the zero as measured would teach every consumer
that nine of ten KRX minutes had a free round trip. Here that means ``spread_bps`` stays
``None`` and ``CONTEXT_NO_ORDERBOOK_SAMPLE`` is emitted, so the ontology's
``requiresDataQuality`` relation can block on it instead of a strategy silently passing
its ``max_spread_bps`` test against zero.
"""

from __future__ import annotations

import math
from typing import Any

from app.context.market_context import FeatureSource, MicrostructureContext

__all__ = ["CONTEXT_NO_ORDERBOOK_SAMPLE", "build_microstructure_context"]

CONTEXT_NO_ORDERBOOK_SAMPLE = "CONTEXT_NO_ORDERBOOK_SAMPLE"

_SOURCE = "orderbook_and_tape"


def _number(raw: Any) -> float | None:
    if raw is None or isinstance(raw, bool):
        return None
    try:
        value = float(raw)
    except (TypeError, ValueError):
        return None
    return value if math.isfinite(value) else None


def build_microstructure_context(
    features: Any,
    *,
    age_seconds: float | None = None,
) -> tuple[MicrostructureContext, dict[str, FeatureSource], tuple[str, ...]]:
    raw_spread = _number(getattr(features, "spread_bps", None))
    book_available = raw_spread is not None and raw_spread > 0.0

    tick_count_5s = _number(getattr(features, "tick_count_5s", None))
    depth_ratio = _number(getattr(features, "depth_ratio", None))
    bid_depth = _number(getattr(features, "bid_depth", None))
    ask_depth = _number(getattr(features, "ask_depth", None))

    depth_imbalance: float | None = None
    if bid_depth is not None and ask_depth is not None and (bid_depth + ask_depth) > 0:
        depth_imbalance = (bid_depth - ask_depth) / (bid_depth + ask_depth)
    elif depth_ratio is not None and depth_ratio > 0:
        # depth_ratio is bid/ask; map onto [-1, 1] so both producers land on one axis.
        depth_imbalance = (depth_ratio - 1.0) / (depth_ratio + 1.0)

    # Cost of crossing, per unit of the volatility that has to be captured to pay for
    # it. This is the quantity that actually decides whether a horizon is reachable,
    # and it is why a strategy can be right about direction and still unprofitable.
    volatility_10s = _number(getattr(features, "realized_volatility_10s", None))
    price_impact: float | None = None
    if book_available and volatility_10s is not None and volatility_10s > 0:
        price_impact = raw_spread / (volatility_10s * 10_000.0)
    elif book_available:
        price_impact = _number(getattr(features, "expected_slippage_bps", None))

    context = MicrostructureContext(
        spread_bps=raw_spread if book_available else None,
        orderflow_imbalance=_number(getattr(features, "aggressor_imbalance_5s", None)),
        microprice_bias=(
            _number(getattr(features, "microprice_edge_bps", None))
            if book_available
            else None
        ),
        trade_intensity=tick_count_5s,
        liquidity_score=(
            _number(getattr(features, "liquidity_score", None))
            if book_available
            else None
        ),
        bid_ask_depth_imbalance=depth_imbalance if book_available else None,
        short_term_price_impact=price_impact,
    )
    sources = {
        name: FeatureSource(source=_SOURCE, age_seconds=age_seconds)
        for name, value in (
            ("spread_bps", context.spread_bps),
            ("orderflow_imbalance", context.orderflow_imbalance),
            ("microprice_bias", context.microprice_bias),
            ("trade_intensity", context.trade_intensity),
            ("liquidity_score", context.liquidity_score),
            ("bid_ask_depth_imbalance", context.bid_ask_depth_imbalance),
            ("short_term_price_impact", context.short_term_price_impact),
        )
        if value is not None
    }
    reasons = () if book_available else (CONTEXT_NO_ORDERBOOK_SAMPLE,)
    return context, sources, reasons
