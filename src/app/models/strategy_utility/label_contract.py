"""Payoff provenance is separate from feature and tensor compatibility."""
from typing import Any, Mapping

LEGACY_BAR_POLICY = "legacy_strategy_geometry_v1"
ENTRY_FROZEN_SHADOW_POLICY = "ontology-risk-v1-entry-frozen-shadow"
BOUNDED_ADVISORY_SCOPE = "entry_frozen_shadow_bounded_risk_advisory_only"


def checkpoint_live_execution_authorized(metadata: Mapping[str, Any]) -> bool:
    """No currently implemented graph-label source validates dynamic live exits."""
    return False


def bounded_advisory_markets(metadata: Mapping[str, Any]) -> tuple[str, ...]:
    if (metadata.get("label_execution_policy") != ENTRY_FROZEN_SHADOW_POLICY
            or metadata.get("label_policy_provenance_matched") is not True
            or metadata.get("authorization_scope") != BOUNDED_ADVISORY_SCOPE):
        return ()
    declared = metadata.get("bounded_advisory_authorized_markets", ())
    if not isinstance(declared, (list, tuple)):
        return ()
    return tuple(market for market in ("KRX", "US") if market in declared)


checkpoint_bounded_advisory_markets = bounded_advisory_markets


def risk_advisory_strategy_markets(metadata: Mapping[str, Any]) -> dict[str, tuple[str, ...]]:
    """Require observed training support for both payoff sides of each arm."""
    result: dict[str, list[str]] = {}
    outcomes = metadata.get("label_outcomes_by_market", {})
    if not isinstance(outcomes, Mapping):
        return {}
    for market in bounded_advisory_markets(metadata):
        rows = outcomes.get(market, {})
        if not isinstance(rows, Mapping):
            continue
        for strategy, row in rows.items():
            if not isinstance(row, Mapping):
                continue
            try:
                positive, negative, filled = (int(row.get(key, 0)) for key in ("positive_net", "negative_net", "filled"))
            except (TypeError, ValueError, OverflowError):
                continue
            if positive >= 20 and negative >= 20 and filled >= positive + negative:
                result.setdefault(str(strategy), []).append(market)
    return {strategy: tuple(markets) for strategy, markets in result.items()}
