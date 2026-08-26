from __future__ import annotations

import numpy as np

from app.strategy.catalog import STRATEGY_IDS


RELATION_NAMES = (
    "same_methodology_family",
    "confirming_methodology",
    "contrasting_methodology",
)
STRATEGY_NODE_COUNT = len(STRATEGY_IDS)

_MOMENTUM = {
    "intraday_momentum",
    "event_momentum",
    "cross_sectional_relative_strength",
    "bar_trend_continuation",
    "supertrend_dmi_continuation",
}
_BREAKOUT = {
    "breakout_volume",
    "gap_context",
    "rvgi_box_breakout",
    "opening_range_breakout",
    "keltner_volatility_breakout",
}
_REVERSION = {
    "vwap_mean_reversion",
    "liquidity_shock_reversal",
    "bar_confirmed_vwap_recovery",
    "adaptive_anchored_vwap_reversion",
    "range_support_reversion",
    "choppiness_range_reversion",
}


def strategy_relation_adjacency(
    allowed_strategy_ids: tuple[str, ...] = STRATEGY_IDS,
    *,
    market: str | None = None,
) -> np.ndarray:
    """Ontology-defined topology for the active target-market subgraph.

    Relation transforms remain shared. What changes is the active graph: a thesis
    whose ontology scope excludes the target market cannot send or receive messages.
    """
    allowed = set(allowed_strategy_ids) & set(strategy_ids_for_market(market))
    adjacency = np.zeros(
        (len(RELATION_NAMES), STRATEGY_NODE_COUNT, STRATEGY_NODE_COUNT),
        dtype=np.float32,
    )
    for source, source_id in enumerate(STRATEGY_IDS):
        if source_id not in allowed:
            continue
        for target, target_id in enumerate(STRATEGY_IDS):
            if target_id not in allowed or source == target:
                continue
            same_family = any(
                source_id in family and target_id in family
                for family in (_MOMENTUM, _BREAKOUT, _REVERSION)
            )
            confirming = (
                (source_id in _MOMENTUM and target_id in _BREAKOUT)
                or (source_id in _BREAKOUT and target_id in _MOMENTUM)
            )
            contrasting = (
                (source_id in (_MOMENTUM | _BREAKOUT) and target_id in _REVERSION)
                or (source_id in _REVERSION and target_id in (_MOMENTUM | _BREAKOUT))
            )
            adjacency[0, target, source] = 1.0 if same_family else 0.0
            adjacency[1, target, source] = 1.0 if confirming else 0.0
            adjacency[2, target, source] = 1.0 if contrasting else 0.0
    degrees = adjacency.sum(axis=-1, keepdims=True)
    return np.divide(
        adjacency,
        degrees,
        out=np.zeros_like(adjacency),
        where=degrees > 0,
    )


def strategy_ids_for_market(market: str | None) -> tuple[str, ...]:
    """Return strategies whose ontology permits a market name or symbol."""
    if market is None:
        return STRATEGY_IDS
    text = str(market or "").strip().upper()
    market_key = (
        "KR"
        if text in {"KR", "KRX", "KOSPI", "KOSDAQ", "KONEX"}
        or (text.isdigit() and len(text) == 6)
        else "US"
    )
    # Avoid making technical algorithm configuration an import-time dependency.
    from app.strategy.registry import default_strategy_registry

    registry = default_strategy_registry()
    return tuple(
        strategy_id
        for strategy_id in STRATEGY_IDS
        if (spec := registry.get(strategy_id)) is not None
        and spec.permits_market(market_key)
    )


def strategy_market_mask(market: str | None) -> np.ndarray:
    """One ontology market-applicability value per strategy node."""
    allowed = set(strategy_ids_for_market(market))
    return np.asarray(
        [1.0 if strategy_id in allowed else 0.0 for strategy_id in STRATEGY_IDS],
        dtype=np.float32,
    )


def strategy_node_features(context_features: tuple[float, ...]) -> np.ndarray:
    context = np.asarray(context_features, dtype=np.float32)
    identity = np.eye(STRATEGY_NODE_COUNT, dtype=np.float32)
    repeated = np.repeat(context[None, :], STRATEGY_NODE_COUNT, axis=0)
    return np.concatenate((repeated, identity), axis=1)


def diagonal_strategy_mask(
    allowed_strategy_ids: tuple[str, ...] = STRATEGY_IDS,
) -> np.ndarray:
    allowed = set(allowed_strategy_ids)
    mask = np.zeros((STRATEGY_NODE_COUNT, STRATEGY_NODE_COUNT), dtype=np.float32)
    for index, strategy_id in enumerate(STRATEGY_IDS):
        if strategy_id in allowed:
            mask[index, index] = 1.0
    return mask
