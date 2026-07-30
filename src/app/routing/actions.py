"""Canonical strategy-routing action contract."""

from __future__ import annotations

from enum import Enum


class StrategyRoutingAction(str, Enum):
    ACTIVATE_STRATEGY = "ACTIVATE_STRATEGY"
    NO_TRADE = "NO_TRADE"


_LEGACY_ACTIVATE = {"ACTIVATE", "ADMISSIBLE", "ALLOW", "ALLOWED", "BUY"}


def normalize_strategy_routing_action(value: object) -> StrategyRoutingAction:
    normalized = str(getattr(value, "value", value) or "").strip().upper()
    if normalized == StrategyRoutingAction.ACTIVATE_STRATEGY.value or normalized in _LEGACY_ACTIVATE:
        return StrategyRoutingAction.ACTIVATE_STRATEGY
    return StrategyRoutingAction.NO_TRADE


def is_actionable_strategy_route(value: object) -> bool:
    return normalize_strategy_routing_action(value) is StrategyRoutingAction.ACTIVATE_STRATEGY
