"""Theory-driven trading reasoner.

Runs the deterministic ontology rules over :class:`TradingFacts` and returns an
explainable :class:`OntologyReasoningResult`. It is ADVISORY: it can weaken or block
a signal, but never authorize one — RiskManager / ProfitabilityGate / FinalTradeGate
remain the sole execution gates.

Config is loaded from the five ``config/ontology/*.yaml`` files (one per ontology
module); missing files fall back to safe defaults (fail-safe, never fail-open).
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

from app.ontology.trading_domain_ontology import (
    IntentType,
    OntologyReasoningResult,
    OrderPolicyRecommendation,
    ReasonCodes as RC,
    ValidationState,
)
from app.ontology.trading_fact_builder import TradingFacts, classify_validation_state
from app.ontology import trading_rules as rules


_DEFAULTS: dict[str, dict[str, Any]] = {
    "execution": {
        "enabled": True,
        "require_orderbook_for_buy": True,
        "require_fresh_orderbook_for_buy": True,
        "block_unknown_exchange_buy": True,
        "max_spread_rate": 0.01,
    },
    "cost": {
        "enabled": True,
        "block_cost_dominated_buy": True,
    },
    "risk": {
        "enabled": True,
        "max_inventory_weight": 0.30,
        "high_downside_risk": 0.012,
        "principal_floor_min_distance": 0.1,
    },
    "micro": {
        "enabled": True,
        "min_model_confidence_for_buy": 0.0,
        "sentiment_standalone_buy_allowed": False,
    },
    "macro": {
        "enabled": True,
        "block_buy_on_macro_risk": True,
    },
    "validation": {
        "enabled": True,
        "min_sample_size": 30,
        "overfit_parameter_count": 20,
        "allow_backtest_only_live_buy": False,
        "block_negative_expectancy": True,
        "low_reliability_confidence_cap": 0.4,
    },
}

# Map module name -> yaml filename.
_CONFIG_FILES = {
    "execution": "execution_ontology_rules.yaml",
    "cost": "cost_ontology_rules.yaml",
    "risk": "risk_ontology_rules.yaml",
    "micro": "micro_ontology_rules.yaml",
    "macro": "macro_ontology_rules.yaml",
    "validation": "validation_ontology_rules.yaml",
}


def _deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    out = dict(base)
    for key, value in (override or {}).items():
        if isinstance(value, dict) and isinstance(out.get(key), dict):
            out[key] = _deep_merge(out[key], value)
        else:
            out[key] = value
    return out


def load_ontology_config(config_dir: str | None = None) -> dict[str, dict[str, Any]]:
    """Load and merge the five per-module YAML rule files over the safe defaults."""
    directory = Path(config_dir or os.getenv("TRADING_ONTOLOGY_CONFIG_DIR", "config/ontology"))
    merged: dict[str, dict[str, Any]] = {module: dict(cfg) for module, cfg in _DEFAULTS.items()}
    try:
        import yaml  # type: ignore
    except Exception:  # noqa: BLE001 - PyYAML missing: use defaults.
        return merged
    for module, filename in _CONFIG_FILES.items():
        path = directory / filename
        try:
            text = path.read_text(encoding="utf-8")
        except (FileNotFoundError, NotADirectoryError, OSError):
            continue
        try:
            data = yaml.safe_load(text) or {}
        except Exception:  # noqa: BLE001 - malformed yaml: keep defaults for this module.
            continue
        if isinstance(data, dict):
            # Accept either a flat mapping or one nested under the module name.
            section = data.get(module) if isinstance(data.get(module), dict) else data
            merged[module] = _deep_merge(merged[module], section)
    return merged


class TradingDomainReasoner:
    def __init__(self, config: dict[str, dict[str, Any]] | None = None, *, config_dir: str | None = None) -> None:
        self.config = config or load_ontology_config(config_dir)

    def reason(self, facts: TradingFacts) -> OntologyReasoningResult:
        cfg = self.config
        candidate = self._candidate_intent(facts)
        validation_state = classify_validation_state(
            facts,
            min_sample_size=int(cfg["validation"].get("min_sample_size", 30)),
            overfit_param_count=int(cfg["validation"].get("overfit_parameter_count", 20)),
        )

        agg = rules.RuleContribution()
        agg.merge(rules.execution_rules(facts, cfg["execution"]))
        agg.merge(rules.cost_rules(facts, cfg["cost"]))
        agg.merge(rules.risk_rules(facts, cfg["risk"]))
        agg.merge(rules.micro_rules(facts, cfg["micro"]))
        agg.merge(rules.macro_rules(facts, cfg["macro"]))
        agg.merge(rules.validation_rules(facts, cfg["validation"], validation_state))

        if not facts.is_buy:
            sell_code = rules.classify_sell_reason(facts)
            agg.support.append(sell_code)

        intent = self._resolve_intent(facts, candidate, agg)
        confidence = self._resolve_confidence(facts, agg, blocked=bool(agg.blocked_by))
        policy = self._order_policy(facts, intent)

        reason_codes = tuple(dict.fromkeys([*agg.support, *agg.conflict, *agg.blocked_by]))
        return OntologyReasoningResult(
            symbol=facts.symbol,
            intent=intent,
            candidate_intent=candidate,
            confidence=confidence,
            theory_support=tuple(dict.fromkeys(agg.support)),
            theory_conflict=tuple(dict.fromkeys(agg.conflict)),
            required_conditions=tuple(dict.fromkeys(agg.required)),
            blocked_by=tuple(dict.fromkeys(agg.blocked_by)),
            reason_codes=reason_codes,
            recommended_order_policy=policy,
            validation_state=validation_state,
            diagnostics={
                "confidence_cap": agg.confidence_cap,
                "signal_family": facts.signal_family,
                "primary_data_tier": facts.primary_data_tier,
            },
        )

    # -- helpers --------------------------------------------------------
    def _candidate_intent(self, facts: TradingFacts) -> IntentType:
        if facts.is_buy:
            return IntentType.BUY
        reason = facts.exit_reason.lower()
        if "reduce" in reason:
            return IntentType.REDUCE
        return IntentType.SELL

    def _resolve_intent(self, facts: TradingFacts, candidate: IntentType, agg: rules.RuleContribution) -> IntentType:
        # SELL/REDUCE exits are never blocked by the ontology (RiskManager still gates).
        if not facts.is_buy:
            return candidate
        if agg.blocked_by:
            return IntentType.BLOCK
        # A BUY must have BOTH execution feasibility and a positive cost-adjusted edge —
        # momentum / model confidence alone can never carry it.
        has_exec = RC.EXECUTION_FEASIBLE in agg.support
        has_edge = RC.NET_EDGE_POSITIVE in agg.support
        if has_exec and has_edge:
            return IntentType.BUY
        return IntentType.WATCH

    def _resolve_confidence(self, facts: TradingFacts, agg: rules.RuleContribution, *, blocked: bool) -> float:
        if blocked:
            return 0.0
        base = 0.3 + max(0.0, min(0.4, facts.model_confidence * 0.4))
        value = base + agg.confidence_delta
        value = min(value, agg.confidence_cap)
        return max(0.0, min(1.0, value))

    def _order_policy(self, facts: TradingFacts, intent: IntentType) -> OrderPolicyRecommendation:
        if intent == IntentType.BUY:
            return OrderPolicyRecommendation(side="BUY", price_policy="BUY_BEST_ASK", urgency="NORMAL")
        if intent in (IntentType.SELL, IntentType.REDUCE):
            sell_code = rules.classify_sell_reason(facts)
            if sell_code in (RC.SELL_HARD_STOP, RC.SELL_STOP_LOSS):
                return OrderPolicyRecommendation(
                    side="SELL", price_policy="SELL_STOP_MARKETABLE_BID", urgency="URGENT",
                    notes=(sell_code,),
                )
            return OrderPolicyRecommendation(
                side="SELL", price_policy="SELL_TP_BEST_BID", urgency="NORMAL", notes=(sell_code,)
            )
        return OrderPolicyRecommendation(side="NONE", price_policy="NONE")
