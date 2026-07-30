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
)

STRATEGY_INDEX: dict[str, int] = {
    strategy_id: index for index, strategy_id in enumerate(STRATEGY_IDS)
}


def is_known_strategy(strategy_id: object) -> bool:
    return str(strategy_id or "") in STRATEGY_INDEX
