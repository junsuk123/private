from __future__ import annotations

import hashlib
from dataclasses import dataclass, field

from app.graph.knowledge_graph import KnowledgeGraph, Triple
from app.graph.runtime import OntologyRuntime, get_ontology_runtime
from app.graph.action_aggregator import ActionAggregator
from app.graph.theory_registry import TheoryRegistry, get_theory_registry
from app.graph.theory_vote import FinalActionDecision, PositionContext, TheoryVote
from app.evaluation.theory_validation import TheoryValidationStore, get_theory_validation_store
from app.schemas.domain import ReasoningPath

SUPPORT_WEIGHTS = {
    "EarningsGrowth": 0.18,
    "ProfitabilityQuality": 0.14,
    "PositiveEventImpact": 0.10,
    "PositiveInvestorFlow": 0.10,
    "InformedOrderFlowImbalance": 0.13,
    "ForeignInstitutionJointBuying": 0.12,
    "RetailSupplyAbsorbedByInformedFlow": 0.10,
    "OrderFlowPriceConfirmation": 0.09,
    "SuspectedSmartMoneyAccumulation": 0.08,
    "OrderFlowConfirmedBuyCandidate": 0.10,
    "CashFitOneShare": 0.07,
    "AffordableByAccountCash": 0.08,
    "FreshBrokerQuote": 0.08,
    "LiveBrokerRealtimeQuote": 0.10,
    "SectorMomentum": 0.08,
    "BuyCandidate": 0.08,
    "AccountCashFeasibleBuyCandidate": 0.06,
    "ExecutableBuyCandidate": 0.10,
}
CONTRADICTION_WEIGHTS = {
    "ValuationDiscipline": 0.12,
    "ValuationSlightlyHigh": 0.10,
    "InformedOrderFlowDistribution": 0.15,
    "ForeignInstitutionJointSelling": 0.13,
    "RetailDemandMeetsInformedSelling": 0.11,
    "OrderFlowPriceDivergence": 0.10,
    "SuspectedSmartMoneyDistribution": 0.12,
    "CashBelowOneSharePrice": 0.18,
    "MissingMarketData": 0.22,
    "BuyCandidate": 0.20,
    "AggressiveBuy": 0.10,
}
RISK_WEIGHTS = {
    "MacroRateRisk": 0.14,
    "VolatilityRisk": 0.18,
    "NegativeEventRisk": 0.18,
    "LiquidityRisk": 0.20,
    "OrderFlowDistributionRisk": 0.16,
    "ThinLiquidityPriceImpactRisk": 0.14,
    "InsufficientAccountCashRisk": 0.20,
    "MissingMarketDataRisk": 0.24,
    "WeakMarketDataQualityRisk": 0.12,
    "SellCandidate": 0.22,
    "ReduceRiskCandidate": 0.18,
    "TradeForbidden": 0.30,
}


@dataclass(frozen=True)
class OntologyReasoningPolicy:
    base_confidence: float = 0.40
    buy_threshold: float = 0.58
    support_weights: dict[str, float] = field(default_factory=lambda: dict(SUPPORT_WEIGHTS))
    contradiction_weights: dict[str, float] = field(default_factory=lambda: dict(CONTRADICTION_WEIGHTS))
    risk_weights: dict[str, float] = field(default_factory=lambda: dict(RISK_WEIGHTS))
    default_support_weight: float = 0.05
    default_contradiction_weight: float = 0.05
    default_risk_weight: float = 0.06
    min_confidence: float = 0.05
    max_confidence: float = 0.95


class OntologyReasoner:
    def __init__(
        self,
        graph: KnowledgeGraph,
        runtime: OntologyRuntime | None = None,
        policy: OntologyReasoningPolicy | None = None,
        registry: TheoryRegistry | None = None,
        validation_store: TheoryValidationStore | None = None,
    ) -> None:
        self.graph = graph
        self.runtime = runtime or get_ontology_runtime()
        self.policy = policy or OntologyReasoningPolicy()
        self.registry = registry or get_theory_registry()
        self.validation_store = validation_store or get_theory_validation_store()
        self.action_aggregator = ActionAggregator(self.registry)

    def infer(self) -> KnowledgeGraph:
        self._infer_buy_candidates()
        self._infer_risk_adjustments()
        return self.graph

    def build_reasoning_paths(self, tickers: tuple[str, ...]) -> tuple[ReasoningPath, ...]:
        paths = []
        decisions = {decision.ticker: decision for decision in self.build_action_decisions(tickers)}
        for ticker in tickers:
            support = self.graph.matching(subject=ticker, predicate="supportsSignal")
            contradiction = self.graph.matching(subject=ticker, predicate="contradictsSignal")
            risk = self.graph.matching(subject=ticker, predicate="increasesRiskOf")
            sizing = self.graph.matching(subject=ticker, predicate="requiresSizingAdjustment")
            support_score = sum(
                self.policy.support_weights.get(triple.object, self.policy.default_support_weight)
                for triple in support
                if triple.object != "RiskAdjustedSizing"
            )
            contradiction_score = sum(
                self.policy.contradiction_weights.get(triple.object, self.policy.default_contradiction_weight)
                for triple in contradiction
            )
            risk_score = sum(
                self.policy.risk_weights.get(triple.object, self.policy.default_risk_weight)
                for triple in risk
            )
            confidence = max(
                self.policy.min_confidence,
                min(self.policy.max_confidence, self.policy.base_confidence + support_score - contradiction_score - risk_score),
            )
            decision = decisions.get(ticker)
            conclusion = _legacy_conclusion(decision, confidence, self.policy.buy_threshold)
            path_id = hashlib.sha256(
                f"{ticker}:{support}:{contradiction}:{risk}:{conclusion}".encode("utf-8")
            ).hexdigest()[:16]
            paths.append(
                ReasoningPath(
                    path_id=path_id,
                    ticker=ticker,
                    conclusion=conclusion,
                    confidence=confidence,
                    supporting_triples=_format_triples(support),
                    contradicting_triples=_format_triples(contradiction),
                    risk_triples=_format_triples(risk),
                    explanation=_explain(ticker, conclusion, confidence, support, contradiction, risk, sizing, decision),
                )
            )
        return tuple(paths)

    def build_action_decisions(
        self,
        tickers: tuple[str, ...],
        *,
        positions: dict[str, PositionContext] | None = None,
    ) -> tuple[FinalActionDecision, ...]:
        positions = positions or {}
        decisions: list[FinalActionDecision] = []
        for ticker in tickers:
            votes = self._theory_votes_for_ticker(ticker)
            decisions.append(
                self.action_aggregator.decide(
                    ticker,
                    votes,
                    position_context=positions.get(ticker, PositionContext()),
                    npu_profile=self.runtime.as_dict(),
                )
            )
        return tuple(decisions)

    def _theory_votes_for_ticker(self, ticker: str) -> tuple[TheoryVote, ...]:
        votes: list[TheoryVote] = []
        for triple in self.graph.matching(subject=ticker, predicate="supportsSignal"):
            vote = self._vote_from_triple(ticker, triple, "support")
            if vote is not None:
                votes.append(vote)
        for triple in self.graph.matching(subject=ticker, predicate="contradictsSignal"):
            vote = self._vote_from_triple(ticker, triple, "contradiction")
            if vote is not None:
                votes.append(vote)
        for triple in self.graph.matching(subject=ticker, predicate="increasesRiskOf"):
            vote = self._vote_from_triple(ticker, triple, "risk")
            if vote is not None:
                votes.append(vote)
        return tuple(votes)

    def _vote_from_triple(self, ticker: str, triple: Triple, kind: str) -> TheoryVote | None:
        theory_id = _theory_id_for_object(triple.object, kind)
        metadata = self.registry.get(theory_id)
        if metadata is None:
            return None
        raw_signal = _weight_for_triple(triple, kind, self.policy)
        action = _action_for_triple(triple.object, kind, metadata.default_action_bias)
        return TheoryVote(
            ticker=ticker,
            theory_id=theory_id,
            theory_family=metadata.family,
            style=metadata.style,
            action=action,
            horizon_bucket=metadata.horizon_bucket,
            expected_holding_minutes=metadata.expected_holding_minutes,
            raw_signal=raw_signal,
            confidence=min(0.95, max(0.05, 0.50 + raw_signal)),
            expected_net_return=None,
            evidence_cluster_id=metadata.evidence_cluster,
            regime_gate=1.0,
            data_quality_weight=1.0,
            validation_weight=self.validation_store.weight_for(theory_id),
            horizon_compatibility=1.0,
            evidence_ids=(triple.evidence_id or "",),
            explanation=f"{triple.predicate} {triple.object} mapped to {theory_id}/{action}.",
        )

    def _infer_buy_candidates(self) -> None:
        subjects = {triple.subject for triple in self.graph.triples()}
        for subject in subjects:
            support_objects = set(self.graph.objects(subject, "supportsSignal"))
            contra_objects = set(self.graph.objects(subject, "contradictsSignal"))
            risk_objects = set(self.graph.objects(subject, "increasesRiskOf"))
            if {"EarningsGrowth", "ProfitabilityQuality"}.issubset(support_objects):
                self.graph.add(subject, "supportsSignal", "BuyCandidate", "reasoner:growth-quality")
            if "BuyCandidate" in support_objects and "MacroRateRisk" in risk_objects:
                self.graph.add(subject, "contradictsSignal", "AggressiveBuy", "reasoner:macro-risk")
            if {"InformedOrderFlowImbalance", "OrderFlowPriceConfirmation"}.issubset(support_objects):
                self.graph.add(subject, "supportsSignal", "OrderFlowConfirmedBuyCandidate", "reasoner:ofi-price-impact")
            if "InformedOrderFlowDistribution" in contra_objects:
                self.graph.add(subject, "contradictsSignal", "BuyCandidate", "reasoner:ofi-distribution")
            if {"CashFitOneShare", "AffordableByAccountCash"}.issubset(support_objects):
                self.graph.add(subject, "supportsSignal", "AccountCashFeasibleBuyCandidate", "reasoner:account-cash")
            if (
                "AccountCashFeasibleBuyCandidate" in set(self.graph.objects(subject, "supportsSignal"))
                and {"LiveBrokerRealtimeQuote", "FreshBrokerQuote"} & set(self.graph.objects(subject, "supportsSignal"))
                and "MissingMarketDataRisk" not in risk_objects
                and "InsufficientAccountCashRisk" not in risk_objects
            ):
                self.graph.add(subject, "supportsSignal", "ExecutableBuyCandidate", "reasoner:execution-readiness")
            if "CashBelowOneSharePrice" in contra_objects:
                self.graph.add(subject, "contradictsSignal", "BuyCandidate", "reasoner:account-cash")
            if "MissingMarketData" in contra_objects or "MissingMarketDataRisk" in risk_objects:
                self.graph.add(subject, "contradictsSignal", "BuyCandidate", "reasoner:market-data")
                self.graph.add(subject, "increasesRiskOf", "TradeForbidden", "reasoner:market-data")

    def _infer_risk_adjustments(self) -> None:
        for triple in self.graph.matching(predicate="increasesRiskOf"):
            if triple.object in {"MacroRateRisk", "VolatilityRisk", "NegativeEventRisk"}:
                self.graph.add(triple.subject, "requiresSizingAdjustment", "RiskAdjustedSizing", "reasoner:risk-sizing")
            if triple.object in {"OrderFlowDistributionRisk", "ThinLiquidityPriceImpactRisk", "InsufficientAccountCashRisk"}:
                self.graph.add(triple.subject, "requiresSizingAdjustment", "RiskAdjustedSizing", "reasoner:flow-risk-sizing")


def _format_triples(triples: tuple[Triple, ...]) -> tuple[str, ...]:
    return tuple(f"{triple.subject} --{triple.predicate}--> {triple.object}" for triple in triples)


def _explain(
    ticker: str,
    conclusion: str,
    confidence: float,
    support: tuple[Triple, ...],
    contradiction: tuple[Triple, ...],
    risk: tuple[Triple, ...],
    sizing: tuple[Triple, ...] = (),
    decision: FinalActionDecision | None = None,
) -> str:
    base = (
        f"{ticker} conclusion={conclusion}, confidence={confidence:.2f}. "
        f"Support={len(support)}, contradiction={len(contradiction)}, risk={len(risk)}, sizing_adjustments={len(sizing)}."
    )
    if decision is None:
        return base
    return f"{base} {decision.final_explanation}"


def _legacy_conclusion(decision: FinalActionDecision | None, confidence: float, buy_threshold: float) -> str:
    if decision is None:
        return "BuyCandidate" if confidence >= buy_threshold else "HoldOrWatch"
    if decision.selected_action == "BUY" and confidence >= buy_threshold:
        return "BuyCandidate"
    if decision.selected_action in {"HOLD", "WATCH"} and confidence >= buy_threshold:
        return "BuyCandidate"
    if decision.selected_action == "SELL":
        return "SellCandidate"
    if decision.selected_action == "REDUCE":
        return "ReduceRiskCandidate"
    if decision.selected_action == "WATCH":
        return "HoldOrWatch"
    return "HoldOrWatch"


def _theory_id_for_object(object_name: str, kind: str) -> str:
    direct = {
        "ShortTermReversalBuy": "jegadeesh_1990_short_term_reversal",
        "ShortTermReversalCandidate": "jegadeesh_1990_short_term_reversal",
        "LiquiditySupportedReversal": "jegadeesh_1990_short_term_reversal",
        "RSIOversold": "jegadeesh_1990_short_term_reversal",
        "RangeOversold": "jegadeesh_1990_short_term_reversal",
        "IntradayMomentumBuy": "gao_2018_intraday_momentum",
        "IntradayMomentum": "gao_2018_intraday_momentum",
        "OpeningReturnStrength": "gao_2018_intraday_momentum",
        "VolumeConfirmedMomentum": "gao_2018_intraday_momentum",
        "MarketDirectionAligned": "gao_2018_intraday_momentum",
        "TechnicalBreakoutBuy": "brock_1992_technical_breakout",
        "BreakoutWatch": "brock_1992_technical_breakout",
        "VolumeConfirmedBreakout": "brock_1992_technical_breakout",
        "MovingAverageBreakout": "brock_1992_technical_breakout",
        "TradingRangeBreakout": "brock_1992_technical_breakout",
        "InformedOrderFlowImbalance": "microstructure_liquidity_imbalance",
        "OrderFlowPriceConfirmation": "microstructure_liquidity_imbalance",
        "OrderFlowConfirmedBuyCandidate": "microstructure_liquidity_imbalance",
        "InformedOrderFlowDistribution": "microstructure_liquidity_imbalance",
        "OrderFlowDistributionRisk": "microstructure_liquidity_imbalance",
        "SellCandidate": "profit_taking_exit",
        "WaitOrTakeProfit": "profit_taking_exit",
        "ReduceRiskCandidate": "risk_reduction_exit",
        "RiskAdjustedSizing": "risk_reduction_exit",
        "TradeForbidden": "risk_reduction_exit",
    }
    if object_name in direct:
        return direct[object_name]
    if kind == "risk":
        return "risk_reduction_exit"
    if kind == "contradiction":
        return "profit_taking_exit"
    return "gao_2018_intraday_momentum"


def _action_for_triple(object_name: str, kind: str, default_action_bias: str) -> str:
    if object_name in {"SellCandidate", "WaitOrTakeProfit", "InformedOrderFlowDistribution"}:
        return "SELL"
    if kind == "risk" or object_name in {"RiskAdjustedSizing", "ReduceRiskCandidate", "TradeForbidden"}:
        return "REDUCE"
    if kind == "contradiction":
        return "SELL"
    if default_action_bias == "BUY_OR_SELL":
        return "BUY"
    return default_action_bias if default_action_bias in {"BUY", "SELL", "HOLD", "REDUCE", "WATCH"} else "WATCH"


def _weight_for_triple(triple: Triple, kind: str, policy: OntologyReasoningPolicy) -> float:
    if kind == "support":
        return policy.support_weights.get(triple.object, policy.default_support_weight)
    if kind == "contradiction":
        return policy.contradiction_weights.get(triple.object, policy.default_contradiction_weight)
    return policy.risk_weights.get(triple.object, policy.default_risk_weight)
