"""Trading-domain ontology vocabulary, semantic relations, and result type.

Defines the classes and the deterministic output contract for the theory-driven
trading reasoner. No numbers/thresholds live here (those are YAML config, read by
:mod:`app.ontology.trading_reasoner`) and no permission is granted here — the
ontology can only weaken or block, never authorize.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any


class IntentType(StrEnum):
    BUY = "BUY"
    SELL = "SELL"
    REDUCE = "REDUCE"
    HOLD = "HOLD"
    WATCH = "WATCH"
    BLOCK = "BLOCK"


class ValidationState(StrEnum):
    """How much out-of-sample evidence backs the strategy driving this signal."""

    LIVE_VALIDATED = "LIVE_VALIDATED"
    PAPER_VALIDATED = "PAPER_VALIDATED"
    BACKTEST_ONLY = "BACKTEST_ONLY"
    OVERFIT_SUSPECTED = "OVERFIT_SUSPECTED"
    NEGATIVE_EXPECTANCY = "NEGATIVE_EXPECTANCY"
    UNVALIDATED = "UNVALIDATED"


class DataTier(StrEnum):
    """Data-trust hierarchy (T0 highest). Mirrors the project data policy."""

    T0_BROKER_REALTIME = "T0"   # broker account/order/fill/balance
    T1_EXCHANGE = "T1"          # KRX official market data
    T2_DISCLOSURE = "T2"        # DART / OpenDART / XBRL
    T3_MACRO = "T3"             # rates / FX / index / macro
    T4_NEWS_SENTIMENT = "T4"    # news / sentiment (auxiliary only)
    T5_DERIVED = "T5"           # RSI / MACD / MA / VWAP-derived


# Trust level per tier (higher = more authoritative). Used for standalone-BUY gating.
DATA_TIER_TRUST: dict[str, int] = {
    DataTier.T0_BROKER_REALTIME: 5,
    DataTier.T1_EXCHANGE: 4,
    DataTier.T2_DISCLOSURE: 3,
    DataTier.T3_MACRO: 3,
    DataTier.T4_NEWS_SENTIMENT: 1,
    DataTier.T5_DERIVED: 2,
}


# --- Reason codes, grouped by ontology module --------------------------------
class ReasonCodes:
    # execution ontology
    NO_ORDERBOOK = "ONTO_EXEC_NO_ORDERBOOK"
    STALE_ORDERBOOK = "ONTO_EXEC_STALE_ORDERBOOK"
    UNKNOWN_EXCHANGE = "ONTO_EXEC_UNKNOWN_EXCHANGE"
    LAST_PRICE_NOT_EXECUTABLE = "ONTO_EXEC_LAST_PRICE_NOT_EXECUTABLE"
    SPREAD_TOO_WIDE = "ONTO_EXEC_SPREAD_TOO_WIDE"
    EXECUTION_FEASIBLE = "ONTO_EXEC_FEASIBLE"
    # cost ontology
    COST_DOMINATED = "ONTO_COST_DOMINATED"          # gross+ but net<=min
    NET_EDGE_INSUFFICIENT = "ONTO_COST_NET_EDGE_INSUFFICIENT"
    NET_EDGE_POSITIVE = "ONTO_COST_NET_EDGE_POSITIVE"
    # risk ontology
    INVENTORY_RISK_HIGH = "ONTO_RISK_INVENTORY_HIGH"
    DOWNSIDE_RISK_HIGH = "ONTO_RISK_DOWNSIDE_HIGH"
    PRINCIPAL_FLOOR_NEAR = "ONTO_RISK_PRINCIPAL_FLOOR_NEAR"
    STOP_LOSS_TRIGGERED = "ONTO_RISK_STOP_LOSS"
    # validation ontology
    BACKTEST_ONLY_NO_LIVE_BUY = "ONTO_VALID_BACKTEST_ONLY_NO_LIVE_BUY"
    NEGATIVE_EXPECTANCY_DISABLED = "ONTO_VALID_NEGATIVE_EXPECTANCY_DISABLED"
    OVERFIT_RISK = "ONTO_VALID_OVERFIT_RISK"
    LOW_VALIDATION_RELIABILITY = "ONTO_VALID_LOW_RELIABILITY"
    VALIDATION_OK = "ONTO_VALID_OK"
    # signal / micro ontology
    MOMENTUM_SUPPORT = "ONTO_MICRO_MOMENTUM_SUPPORT"
    MEAN_REVERSION_SUPPORT = "ONTO_MICRO_MEAN_REVERSION_SUPPORT"
    SIGNAL_ONLY_SENTIMENT = "ONTO_MICRO_SENTIMENT_NOT_STANDALONE"
    EXIT_DETERIORATION = "ONTO_MICRO_EXIT_DETERIORATION"
    # macro ontology
    MACRO_BLOCK_BUY = "ONTO_MACRO_BLOCK_BUY"
    MACRO_RISK_HIGH = "ONTO_MACRO_RISK_HIGH"
    # exit-reason classification (SELL)
    SELL_TAKE_PROFIT = "ONTO_SELL_TAKE_PROFIT"
    SELL_STOP_LOSS = "ONTO_SELL_STOP_LOSS"
    SELL_HARD_STOP = "ONTO_SELL_HARD_STOP"
    SELL_TIME_STOP = "ONTO_SELL_TIME_STOP"
    SELL_MODEL_EXIT = "ONTO_SELL_MODEL_EXIT"


@dataclass(frozen=True)
class OntologyClass:
    """A node in the trading ontology (for introspection / explainability)."""

    name: str
    module: str            # macro | micro | execution | cost | risk | validation
    parents: tuple[str, ...] = ()
    description: str = ""


# Compact class registry — the semantic backbone the rules reason over. Kept in
# Python (not OWL) so it is testable and dependency-free; the RDF/OWL schema in
# ``app/ontology/*.ttl`` remains the formal, richer vocabulary.
ONTOLOGY_CLASSES: tuple[OntologyClass, ...] = (
    # macro
    OntologyClass("MarketRegime", "macro", ("TradingEntity",), "Trend/range/volatility/news-shock market state"),
    OntologyClass("SectorRegime", "macro", ("TradingEntity",)),
    OntologyClass("MacroRisk", "macro", ("RiskState",)),
    OntologyClass("UniverseEligibility", "macro", ("TradingEntity",)),
    # micro
    OntologyClass("IntradayMomentum", "micro", ("StrategySignal",)),
    OntologyClass("MeanReversion", "micro", ("StrategySignal",)),
    OntologyClass("ExitDeterioration", "micro", ("StrategySignal",)),
    # execution
    OntologyClass("OrderBookState", "execution", ("ExecutionRisk",)),
    OntologyClass("NoOrderBook", "execution", ("OrderBookState",)),
    OntologyClass("StaleQuote", "execution", ("OrderBookState",)),
    OntologyClass("UnknownExchange", "execution", ("ExecutionRisk",)),
    OntologyClass("ExecutablePrice", "execution"),
    # cost
    OntologyClass("ExpectedGrossReturn", "cost"),
    OntologyClass("ExpectedNetReturn", "cost"),
    OntologyClass("CostDominated", "cost", ("ExecutionRisk",)),
    # risk
    OntologyClass("InventoryRisk", "risk", ("RiskState",)),
    OntologyClass("DrawdownRisk", "risk", ("RiskState",)),
    OntologyClass("PrincipalFloor", "risk", ("RiskState",)),
    # validation
    OntologyClass("BacktestResult", "validation"),
    OntologyClass("PaperTradingResult", "validation"),
    OntologyClass("LiveTradingResult", "validation"),
    OntologyClass("OverfittingRisk", "validation"),
    OntologyClass("NegativeNetExpectancy", "validation"),
)


@dataclass(frozen=True)
class OrderPolicyRecommendation:
    """Recommended execution policy for the resolved intent (advisory)."""

    side: str                       # BUY / SELL / NONE
    price_policy: str               # e.g. BUY_BEST_ASK, SELL_TP_BEST_BID, SELL_STOP_MARKETABLE_BID, NONE
    urgency: str = "NORMAL"         # NORMAL / URGENT
    notes: tuple[str, ...] = ()

    def as_dict(self) -> dict[str, Any]:
        return {"side": self.side, "price_policy": self.price_policy, "urgency": self.urgency, "notes": list(self.notes)}


@dataclass(frozen=True)
class OntologyReasoningResult:
    """Explainable, theory-driven trade reasoning result (the spec contract)."""

    symbol: str
    intent: IntentType                       # resolved intent after all rules
    candidate_intent: IntentType             # the pre-reasoning candidate (from inputs)
    confidence: float                        # [0,1]
    theory_support: tuple[str, ...]          # supporting theory / reason codes
    theory_conflict: tuple[str, ...]         # conflicting / weakening reason codes
    required_conditions: tuple[str, ...]     # conditions that must hold for the intent
    blocked_by: tuple[str, ...]              # reason codes that blocked the intent (empty = not blocked)
    reason_codes: tuple[str, ...]            # full flat list of emitted reason codes
    recommended_order_policy: OrderPolicyRecommendation
    validation_state: ValidationState
    diagnostics: dict[str, Any] = field(default_factory=dict)

    @property
    def approved(self) -> bool:
        """Advisory approval: not BLOCK and not empty of support. Never authorizes on
        its own — downstream RiskManager / ProfitabilityGate remain the real gates."""
        return self.intent not in (IntentType.BLOCK,) and not self.blocked_by

    def as_dict(self) -> dict[str, Any]:
        return {
            "symbol": self.symbol,
            "intent": str(self.intent),
            "candidate_intent": str(self.candidate_intent),
            "confidence": round(float(self.confidence), 4),
            "theory_support": list(self.theory_support),
            "theory_conflict": list(self.theory_conflict),
            "required_conditions": list(self.required_conditions),
            "blocked_by": list(self.blocked_by),
            "reason_codes": list(self.reason_codes),
            "recommended_order_policy": self.recommended_order_policy.as_dict(),
            "validation_state": str(self.validation_state),
            "diagnostics": dict(self.diagnostics),
        }
