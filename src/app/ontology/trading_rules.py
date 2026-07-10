"""Deterministic, theory-driven inference rules for the trading ontology.

Each rule group inspects :class:`TradingFacts` and contributes support, conflict,
required-condition, and blocking reason codes plus a confidence delta. Rules NEVER
authorize — a BUY only survives if no rule blocks it AND it has genuine support
(execution feasibility + positive cost-adjusted edge + acceptable validation). This
encodes the spec's acceptance criteria as composable, testable predicates.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from app.ontology.trading_domain_ontology import ReasonCodes as RC
from app.ontology.trading_domain_ontology import DataTier, ValidationState
from app.ontology.trading_fact_builder import TradingFacts


@dataclass
class RuleContribution:
    support: list[str] = field(default_factory=list)
    conflict: list[str] = field(default_factory=list)
    required: list[str] = field(default_factory=list)
    blocked_by: list[str] = field(default_factory=list)
    confidence_delta: float = 0.0
    confidence_cap: float = 1.0

    def merge(self, other: "RuleContribution") -> None:
        self.support.extend(other.support)
        self.conflict.extend(other.conflict)
        self.required.extend(other.required)
        self.blocked_by.extend(other.blocked_by)
        self.confidence_delta += other.confidence_delta
        self.confidence_cap = min(self.confidence_cap, other.confidence_cap)


# --- BUY rule groups ---------------------------------------------------------
def execution_rules(facts: TradingFacts, cfg: dict[str, Any]) -> RuleContribution:
    c = RuleContribution()
    if not cfg.get("enabled", True):
        return c
    if not facts.is_buy:
        # Exits are not blocked on execution feasibility (handled by pricing policy).
        return c
    # last_price is a reference, not an executable price — always require a book-derived price.
    c.required.append("EXECUTABLE_PRICE_FROM_ORDER_BOOK")
    if not facts.has_orderbook:
        if cfg.get("require_orderbook_for_buy", True):
            c.blocked_by.append(RC.NO_ORDERBOOK)
        else:
            c.conflict.append(RC.NO_ORDERBOOK)
        return c
    if not facts.orderbook_fresh and cfg.get("require_fresh_orderbook_for_buy", True):
        c.blocked_by.append(RC.STALE_ORDERBOOK)
        return c
    if not facts.exchange_known and cfg.get("block_unknown_exchange_buy", True):
        c.blocked_by.append(RC.UNKNOWN_EXCHANGE)
        return c
    max_spread = float(cfg.get("max_spread_rate", 0.01))
    if facts.spread_rate > max_spread:
        c.blocked_by.append(RC.SPREAD_TOO_WIDE)
        return c
    c.support.append(RC.EXECUTION_FEASIBLE)
    c.confidence_delta += 0.1
    return c


def cost_rules(facts: TradingFacts, cfg: dict[str, Any]) -> RuleContribution:
    c = RuleContribution()
    if not cfg.get("enabled", True) or not facts.is_buy:
        return c
    c.required.append("POSITIVE_COST_ADJUSTED_NET_EDGE")
    if facts.net_edge_positive:
        c.support.append(RC.NET_EDGE_POSITIVE)
        c.confidence_delta += 0.15
        return c
    if facts.gross_positive and not facts.net_edge_positive:
        # Gross-positive but net-insufficient => cost-dominated.
        if cfg.get("block_cost_dominated_buy", True):
            c.blocked_by.append(RC.COST_DOMINATED)
        else:
            c.conflict.append(RC.COST_DOMINATED)
        return c
    if cfg.get("block_cost_dominated_buy", True):
        c.blocked_by.append(RC.NET_EDGE_INSUFFICIENT)
    else:
        c.conflict.append(RC.NET_EDGE_INSUFFICIENT)
    return c


def risk_rules(facts: TradingFacts, cfg: dict[str, Any]) -> RuleContribution:
    c = RuleContribution()
    if not cfg.get("enabled", True):
        return c
    if facts.is_buy:
        if facts.inventory_weight > float(cfg.get("max_inventory_weight", 0.30)):
            c.blocked_by.append(RC.INVENTORY_RISK_HIGH)
        if facts.downside_risk > float(cfg.get("high_downside_risk", 0.012)):
            c.blocked_by.append(RC.DOWNSIDE_RISK_HIGH)
        if facts.principal_floor_distance < float(cfg.get("principal_floor_min_distance", 0.1)):
            c.blocked_by.append(RC.PRINCIPAL_FLOOR_NEAR)
    else:
        # SELL: a triggered stop is genuine support for the exit.
        reason = facts.exit_reason.lower()
        if reason.startswith(("stop_loss", "hard_stop", "loss_exit", "domestic_emergency")):
            c.support.append(RC.STOP_LOSS_TRIGGERED)
    return c


def micro_rules(facts: TradingFacts, cfg: dict[str, Any]) -> RuleContribution:
    c = RuleContribution()
    if not cfg.get("enabled", True) or not facts.is_buy:
        return c
    family = facts.signal_family
    tier = facts.primary_data_tier
    # News/sentiment (T4) can never be standalone live-BUY evidence.
    if (family in {"sentiment", "news"} or tier == DataTier.T4_NEWS_SENTIMENT) and not cfg.get(
        "sentiment_standalone_buy_allowed", False
    ):
        c.blocked_by.append(RC.SIGNAL_ONLY_SENTIMENT)
        return c
    if family in {"momentum", "breakout"}:
        c.support.append(RC.MOMENTUM_SUPPORT)
        c.confidence_delta += 0.05
    elif family in {"mean_reversion", "vwap_reversion"}:
        c.support.append(RC.MEAN_REVERSION_SUPPORT)
        c.confidence_delta += 0.05
    min_conf = float(cfg.get("min_model_confidence_for_buy", 0.0))
    if facts.model_confidence >= min_conf and facts.model_confidence > 0:
        c.confidence_delta += 0.05 * facts.model_confidence
    return c


def macro_rules(facts: TradingFacts, cfg: dict[str, Any]) -> RuleContribution:
    c = RuleContribution()
    if not cfg.get("enabled", True) or not facts.is_buy:
        return c
    diag = facts.diagnostics or {}
    if diag.get("macro_block_buy") and cfg.get("block_buy_on_macro_risk", True):
        c.blocked_by.append(RC.MACRO_BLOCK_BUY)
    elif diag.get("macro_risk_high"):
        c.conflict.append(RC.MACRO_RISK_HIGH)
        c.confidence_cap = min(c.confidence_cap, 0.6)
    return c


def validation_rules(facts: TradingFacts, cfg: dict[str, Any], state: ValidationState) -> RuleContribution:
    c = RuleContribution()
    if not cfg.get("enabled", True) or not facts.is_buy:
        return c
    if state == ValidationState.NEGATIVE_EXPECTANCY and cfg.get("block_negative_expectancy", True):
        c.blocked_by.append(RC.NEGATIVE_EXPECTANCY_DISABLED)
        return c
    if state == ValidationState.BACKTEST_ONLY and not cfg.get("allow_backtest_only_live_buy", False):
        c.blocked_by.append(RC.BACKTEST_ONLY_NO_LIVE_BUY)
        return c
    cap = float(cfg.get("low_reliability_confidence_cap", 0.4))
    if state == ValidationState.OVERFIT_SUSPECTED:
        c.conflict.append(RC.OVERFIT_RISK)
        c.confidence_cap = min(c.confidence_cap, cap)
        c.required.append("OUT_OF_SAMPLE_EVIDENCE")
    elif state == ValidationState.UNVALIDATED:
        c.conflict.append(RC.LOW_VALIDATION_RELIABILITY)
        c.confidence_cap = min(c.confidence_cap, cap)
        c.required.append("VALIDATION_EVIDENCE")
    else:  # LIVE / PAPER validated
        c.support.append(RC.VALIDATION_OK)
        c.confidence_delta += 0.1
    return c


def classify_sell_reason(facts: TradingFacts) -> str:
    """Map a SELL exit_reason to a distinguishing reason code (spec requirement)."""
    r = facts.exit_reason.lower()
    if r.startswith("hard_stop") or "hard_stop" in r:
        return RC.SELL_HARD_STOP
    if r.startswith("stop_loss") or "stop_loss" in r or "loss_exit" in r or "emergency" in r:
        return RC.SELL_STOP_LOSS
    if "time_exit" in r or "time_stop" in r:
        return RC.SELL_TIME_STOP
    if "profit" in r or "take_profit" in r:
        return RC.SELL_TAKE_PROFIT
    return RC.SELL_MODEL_EXIT
