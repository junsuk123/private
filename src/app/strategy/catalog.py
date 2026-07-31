"""Stable strategy identity contract shared by routing, models and execution."""

from __future__ import annotations


# Append-only: model output indices and persisted masks depend on this order.
STRATEGY_IDS: tuple[str, ...] = (
    "intraday_momentum",
    "breakout_volume",
    "vwap_mean_reversion",
    "liquidity_shock_reversal",
    "event_momentum",
    "cross_sectional_relative_strength",
    "gap_context",
    "rvgi_box_breakout",
    # Added for the current high-volatility, flow-driven tape. Appended, never
    # inserted: model output indices and persisted strategy masks depend on the
    # order above staying byte-stable.
    "residual_relative_strength",
    "adaptive_anchored_vwap_reversion",
    "ofi_microprice_exhaustion_reversal",
    # Opening-range breakout gated on relative volume ("stocks in play"). Added
    # from the published day-trading literature, where the unrestricted breakout
    # is unprofitable and the relative-volume restriction is what carries the
    # result — so the RVOL gate is part of the thesis, not a tuning knob.
    "opening_range_breakout",
    # Market intraday momentum: the first half-hour return predicts the last
    # half-hour return (Gao/Han/Li/Zhou, JFE 2018; confirmed in 12 of 16 developed
    # markets and in APAC). Added because it is ONE round trip per day, and this
    # account pays 20bps of statutory tax on every round trip — 72% of its KRX cost
    # is turnover-driven, so a strategy's trip count matters as much as its edge.
    "market_intraday_momentum",
)

STRATEGY_INDEX: dict[str, int] = {
    strategy_id: index for index, strategy_id in enumerate(STRATEGY_IDS)
}


def is_known_strategy(strategy_id: object) -> bool:
    return str(strategy_id or "") in STRATEGY_INDEX
