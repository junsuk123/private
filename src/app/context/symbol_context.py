"""Symbol and price-geometry slices, derived from one ``TechnicalFeatureSet``.

Everything here comes from the SAME feature object the strategy algorithms fire on
(``app.technical.signals.TechnicalFeatureSet``, built by
``app.technical.feature_builder.technical_feature_set_from_live_frame``). That is the
point: before this module the selector scored a candidate from the GNN context vector
while the algorithm triggered on the feature set, so the two could disagree about the
same instant.

Derivations, and why each is the form it is
------------------------------------------
``trend_strength``   EMA separation normalised by price, in bps. Scale-free, so a
                     150,000 KRW name and a 40 USD name land on the same axis. A raw
                     EMA difference encodes the instrument's price level, which is the
                     identity leak recorded in ``strategy_graph_context``.
``price_acceleration`` return_5s minus return_30s scaled to the same window. Not a
                     second derivative of price — the tick window is too short for that
                     to be anything but noise — but the change in short-horizon drift,
                     which is what the exhaustion theses actually condition on.
``return_percentile`` / ``volume_percentile`` are NOT computed here. A percentile needs
                     a history the feature set does not carry, and inventing one from a
                     single observation would make every context read 0.5. They stay
                     ``None`` unless a caller supplies measured values.
"""

from __future__ import annotations

import math
from typing import Any, Mapping

from app.context.market_context import (
    FeatureSource,
    PriceGeometryContext,
    SymbolContext,
)

__all__ = ["build_price_geometry_context", "build_symbol_context"]

_SOURCE = "technical_feature_set"


def _number(raw: Any) -> float | None:
    if raw is None or isinstance(raw, bool):
        return None
    try:
        value = float(raw)
    except (TypeError, ValueError):
        return None
    return value if math.isfinite(value) else None


def _bps_ratio(numerator: float | None, denominator: float | None) -> float | None:
    if numerator is None or denominator is None or abs(denominator) < 1e-12:
        return None
    return (numerator / denominator) * 10_000.0


def _sources(
    values: Mapping[str, Any], age_seconds: float | None
) -> dict[str, FeatureSource]:
    return {
        name: FeatureSource(source=_SOURCE, age_seconds=age_seconds)
        for name, value in values.items()
        if value is not None
    }


def build_symbol_context(
    features: Any,
    *,
    return_percentile: Any = None,
    volume_percentile: Any = None,
    age_seconds: float | None = None,
) -> tuple[SymbolContext, dict[str, FeatureSource]]:
    price = _number(getattr(features, "price", None))
    ema_fast = _number(getattr(features, "ema_fast", None))
    ema_slow = _number(getattr(features, "ema_slow", None))
    return_5s = _number(getattr(features, "return_5s", None))
    return_30s = _number(getattr(features, "return_30s", None))

    acceleration: float | None = None
    if return_5s is not None and return_30s is not None:
        # Same-window comparison: the 30s drift scaled down to 5s. Comparing the raw
        # numbers would call every trending tape "accelerating".
        acceleration = (return_5s - return_30s / 6.0) * 10_000.0

    context = SymbolContext(
        trend_strength=(
            _bps_ratio(ema_fast - ema_slow, price)
            if ema_fast is not None and ema_slow is not None
            else None
        ),
        trend_persistence=_number(getattr(features, "momentum_persistence", None)),
        realized_volatility=_number(getattr(features, "realized_volatility", None))
        or _number(getattr(features, "realized_volatility_10s", None)),
        return_percentile=_number(return_percentile),
        volume_percentile=_number(volume_percentile),
        relative_volume=_number(getattr(features, "relative_volume", None)),
        price_acceleration=acceleration,
        reference_price=price if price is not None and price > 0 else None,
    )
    return context, _sources(
        {
            "trend_strength": context.trend_strength,
            "trend_persistence": context.trend_persistence,
            "realized_volatility": context.realized_volatility,
            "return_percentile": context.return_percentile,
            "volume_percentile": context.volume_percentile,
            "relative_volume": context.relative_volume,
            "price_acceleration": context.price_acceleration,
            "reference_price": context.reference_price,
        },
        age_seconds,
    )


def build_price_geometry_context(
    features: Any,
    *,
    election_inputs: Mapping[str, Any] | None = None,
    session_high: Any = None,
    session_low: Any = None,
    age_seconds: float | None = None,
) -> tuple[PriceGeometryContext, dict[str, FeatureSource]]:
    inputs: Mapping[str, Any] = election_inputs or {}
    price = _number(getattr(features, "price", None))
    anchored_vwap = _number(inputs.get("anchored_vwap"))
    opening_high = _number(inputs.get("opening_range_high"))
    opening_low = _number(inputs.get("opening_range_low"))
    donchian_low = _number(getattr(features, "donchian_low", None))
    box_low = _number(getattr(features, "box_low", None))

    opening_position: float | None = None
    if (
        price is not None
        and opening_high is not None
        and opening_low is not None
        and opening_high > opening_low
    ):
        opening_position = (price - opening_low) / (opening_high - opening_low)

    # Nearest of the two structural floors this catalogue actually uses: the Donchian
    # low (range_support_reversion) and the box low (rvgi_box_breakout). Reporting the
    # nearer one keeps "how far above support" a single answerable question.
    support = None
    if donchian_low is not None and box_low is not None:
        support = max(donchian_low, box_low)
    else:
        support = donchian_low if donchian_low is not None else box_low

    context = PriceGeometryContext(
        vwap_distance_bps=_number(getattr(features, "vwap_distance_bps", None)),
        anchored_vwap_distance_bps=_bps_ratio(
            (price - anchored_vwap)
            if price is not None and anchored_vwap is not None
            else None,
            price,
        ),
        range_position=_number(getattr(features, "box_position", None)),
        breakout_distance_bps=_number(getattr(features, "breakout_distance_bps", None)),
        support_distance_bps=_bps_ratio(
            (price - support) if price is not None and support is not None else None,
            price,
        ),
        session_high_distance_bps=_bps_ratio(
            (price - _number(session_high))
            if price is not None and _number(session_high) is not None
            else None,
            price,
        ),
        session_low_distance_bps=_bps_ratio(
            (price - _number(session_low))
            if price is not None and _number(session_low) is not None
            else None,
            price,
        ),
        opening_range_position=opening_position,
    )
    return context, _sources(
        {
            "vwap_distance_bps": context.vwap_distance_bps,
            "anchored_vwap_distance_bps": context.anchored_vwap_distance_bps,
            "range_position": context.range_position,
            "breakout_distance_bps": context.breakout_distance_bps,
            "support_distance_bps": context.support_distance_bps,
            "session_high_distance_bps": context.session_high_distance_bps,
            "session_low_distance_bps": context.session_low_distance_bps,
            "opening_range_position": context.opening_range_position,
        },
        age_seconds,
    )
