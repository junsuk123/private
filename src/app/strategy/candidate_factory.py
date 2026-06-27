from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Mapping, Sequence

from app.cost import CostBreakdown, TradingCostEngine
from app.features.schemas import OHLCVBar
from app.features.short_horizon_features import ShortHorizonFeatures
from app.schemas.domain import OrderAction, OrderIntent
from app.strategy.candidates import StrategyCandidate
from app.strategy.pairs_relative_value import (
    PairAssetProfile,
    PairRelativeValueConfig,
    PairRelativeValueEngine,
    PairUniverseBuilder,
    PairUniverseMember,
)
from app.strategy.short_horizon import (
    IntradayMomentumConfig,
    IntradayMomentumEngine,
    ShortTermReversalConfig,
    ShortTermReversalEngine,
    TechnicalRuleConfig,
    TechnicalRuleEngine,
)


@dataclass(frozen=True)
class StrategyFactoryConfig:
    enabled: bool = True
    paper_only: bool = True
    enable_short_term_reversal: bool = True
    enable_intraday_momentum: bool = True
    enable_technical_rule: bool = True
    enable_pair_relative_value: bool = True
    target_net_return: float = 0.003
    max_cost_to_alpha_ratio: float = 0.5
    max_spread_rate: float = 0.0015
    min_liquidity_score: float = 0.5
    venue: str = "KRX"
    market: str = "KR"
    instrument_type: str = "domestic_stock"


@dataclass(frozen=True)
class StrategyCandidateFactoryInput:
    features_by_ticker: Mapping[str, ShortHorizonFeatures]
    price_history_by_ticker: Mapping[str, Sequence[OHLCVBar]] = field(default_factory=dict)
    entry_prices: Mapping[str, float] = field(default_factory=dict)
    pair_profiles: Mapping[str, PairAssetProfile] = field(default_factory=dict)
    pair_universe: Sequence[PairUniverseMember] | None = None


@dataclass(frozen=True)
class RankedStrategyCandidate:
    candidate: StrategyCandidate
    cost_breakdown: CostBreakdown
    ranking_score: float
    target_net_return: float
    ontology_score: float
    liquidity_score: float
    risk_adjustment: float

    def to_order_intent(
        self,
        *,
        market: str,
        suggested_weight: float,
        valid_until: datetime,
        source_data_ids: tuple[str, ...],
        action: OrderAction = OrderAction.BUY,
        model_uncertainty: float | None = None,
    ) -> OrderIntent:
        return OrderIntent(
            ticker=self.candidate.ticker,
            market=market,
            action=action,
            suggested_weight=suggested_weight,
            confidence=self.candidate.confidence,
            valid_until=valid_until,
            reasoning_summary=(self.candidate.reason or f"{self.candidate.strategy_family}:{self.candidate.signal_name}",),
            supporting_factors=tuple(self.candidate.ontology_tags),
            contradicting_factors=(),
            source_data_ids=source_data_ids,
            model_uncertainty=model_uncertainty,
            strategy_family=self.candidate.strategy_family,
            signal_name=self.candidate.signal_name,
            expected_exit_price=self.candidate.expected_exit_price,
            expected_holding_minutes=self.candidate.expected_holding_minutes,
            gross_expected_return=self.candidate.gross_expected_return,
            target_net_return=self.target_net_return,
            validation_id=self.candidate.validation_id,
            cost_breakdown=self.cost_breakdown.as_dict(),
            ontology_tags=tuple(self.candidate.ontology_tags),
            strategy_metadata={
                "features": dict(self.candidate.features),
                "entry_price": self.candidate.entry_price,
                "ranking_score": self.ranking_score,
                "ontology_score": self.ontology_score,
                "liquidity_score": self.liquidity_score,
                "risk_adjustment": self.risk_adjustment,
            },
        )


@dataclass(frozen=True)
class FilteredStrategyCandidate:
    candidate: StrategyCandidate
    reason: str
    cost_breakdown: CostBreakdown | None = None


@dataclass(frozen=True)
class StrategyCandidateFactoryResult:
    candidates: tuple[RankedStrategyCandidate, ...]
    filtered_candidates: tuple[FilteredStrategyCandidate, ...]


class StrategyCandidateFactory:
    def __init__(
        self,
        config: StrategyFactoryConfig | None = None,
        *,
        cost_engine: TradingCostEngine | None = None,
        short_term_reversal: ShortTermReversalEngine | None = None,
        intraday_momentum: IntradayMomentumEngine | None = None,
        technical_rule: TechnicalRuleEngine | None = None,
        pair_relative_value: PairRelativeValueEngine | None = None,
        pair_universe_builder: PairUniverseBuilder | None = None,
    ) -> None:
        self.config = config or StrategyFactoryConfig()
        self.cost_engine = cost_engine or TradingCostEngine()
        self.short_term_reversal = short_term_reversal or ShortTermReversalEngine(ShortTermReversalConfig())
        self.intraday_momentum = intraday_momentum or IntradayMomentumEngine(IntradayMomentumConfig())
        self.technical_rule = technical_rule or TechnicalRuleEngine(TechnicalRuleConfig())
        pair_config = PairRelativeValueConfig()
        self.pair_relative_value = pair_relative_value or PairRelativeValueEngine(pair_config, cost_engine=self.cost_engine)
        self.pair_universe_builder = pair_universe_builder or PairUniverseBuilder(pair_config)

    def build(self, inputs: StrategyCandidateFactoryInput, *, trading_mode: str = "paper") -> StrategyCandidateFactoryResult:
        if not self.config.enabled:
            return StrategyCandidateFactoryResult(candidates=(), filtered_candidates=())
        if self.config.paper_only and trading_mode != "paper":
            return StrategyCandidateFactoryResult(candidates=(), filtered_candidates=())

        raw_candidates = self._generate_raw_candidates(inputs, trading_mode=trading_mode)
        accepted: list[RankedStrategyCandidate] = []
        filtered: list[FilteredStrategyCandidate] = []
        for candidate in raw_candidates:
            ranked, rejection = self._rank_or_filter(candidate, inputs)
            if ranked is not None:
                accepted.append(ranked)
            else:
                filtered.append(rejection)
        return StrategyCandidateFactoryResult(
            candidates=tuple(sorted(accepted, key=lambda item: item.ranking_score, reverse=True)),
            filtered_candidates=tuple(filtered),
        )

    def _generate_raw_candidates(
        self,
        inputs: StrategyCandidateFactoryInput,
        *,
        trading_mode: str,
    ) -> list[StrategyCandidate]:
        candidates: list[StrategyCandidate] = []
        for ticker, features in inputs.features_by_ticker.items():
            entry_price = _entry_price(ticker, inputs)
            if entry_price is None:
                continue
            if self.config.enable_short_term_reversal:
                _append_candidate(
                    candidates,
                    self.short_term_reversal.generate_candidate(features, entry_price=entry_price, trading_mode=trading_mode),
                )
            if self.config.enable_intraday_momentum:
                _append_candidate(
                    candidates,
                    self.intraday_momentum.generate_candidate(features, entry_price=entry_price, trading_mode=trading_mode),
                )
            bars = inputs.price_history_by_ticker.get(ticker)
            if self.config.enable_technical_rule and bars:
                _append_candidate(
                    candidates,
                    self.technical_rule.generate_candidate(features, bars, entry_price=entry_price, trading_mode=trading_mode),
                )

        if self.config.enable_pair_relative_value:
            pairs = inputs.pair_universe
            if pairs is None:
                pairs = self.pair_universe_builder.build(
                    inputs.price_history_by_ticker,
                    profiles=inputs.pair_profiles,
                )
            for pair in pairs:
                _append_candidate(
                    candidates,
                    self.pair_relative_value.generate_candidate(
                        pair,
                        inputs.price_history_by_ticker,
                        inputs.features_by_ticker,
                        trading_mode=trading_mode,
                    ),
                )
        return candidates

    def _rank_or_filter(
        self,
        candidate: StrategyCandidate,
        inputs: StrategyCandidateFactoryInput,
    ) -> tuple[RankedStrategyCandidate | None, FilteredStrategyCandidate]:
        if candidate.expected_exit_price <= 0:
            return None, FilteredStrategyCandidate(candidate, "MISSING_EXPECTED_EXIT_PRICE")

        candidate_target = float(candidate.features.get("target_net_return", self.config.target_net_return))
        target_net_return = max(self.config.target_net_return, candidate_target)
        cost = self.cost_engine.estimate(
            symbol=candidate.ticker,
            market=self.config.market,
            venue=self.config.venue,
            instrument_type=self.config.instrument_type,
            entry_price=candidate.entry_price,
            expected_exit_price=candidate.expected_exit_price,
            quantity=1,
            target_net_return=target_net_return,
        )
        feature_snapshot = inputs.features_by_ticker.get(candidate.ticker)
        liquidity_score = _liquidity_score(candidate, feature_snapshot)
        spread_rate = _spread_rate(candidate, feature_snapshot)
        if cost.net_expected_return <= target_net_return:
            return None, FilteredStrategyCandidate(candidate, "BELOW_TARGET_NET_RETURN_AFTER_COST", cost)
        if cost.gross_expected_return <= cost.break_even_return + _safety_margin(self.cost_engine):
            return None, FilteredStrategyCandidate(candidate, "BELOW_BREAK_EVEN_WITH_MARGIN", cost)
        if cost.cost_to_alpha_ratio >= self.config.max_cost_to_alpha_ratio:
            return None, FilteredStrategyCandidate(candidate, "COST_BURDEN_HIGH", cost)
        if spread_rate is None or spread_rate >= self.config.max_spread_rate:
            return None, FilteredStrategyCandidate(candidate, "SPREAD_TOO_WIDE", cost)
        if liquidity_score <= self.config.min_liquidity_score:
            return None, FilteredStrategyCandidate(candidate, "LIQUIDITY_TOO_LOW", cost)

        ontology_score = _ontology_score(candidate.ontology_tags)
        risk_adjustment = _risk_adjustment(candidate, cost, spread_rate)
        excess_return_after_cost = cost.net_expected_return - target_net_return
        ranking_score = (
            excess_return_after_cost
            * candidate.confidence
            * ontology_score
            * liquidity_score
            * risk_adjustment
        )
        ranked = RankedStrategyCandidate(
            candidate=candidate,
            cost_breakdown=cost,
            ranking_score=ranking_score,
            target_net_return=target_net_return,
            ontology_score=ontology_score,
            liquidity_score=liquidity_score,
            risk_adjustment=risk_adjustment,
        )
        return ranked, FilteredStrategyCandidate(candidate, "ACCEPTED", cost)


def _append_candidate(candidates: list[StrategyCandidate], candidate: StrategyCandidate | None) -> None:
    if candidate is not None:
        candidates.append(candidate)


def _entry_price(ticker: str, inputs: StrategyCandidateFactoryInput) -> float | None:
    if ticker in inputs.entry_prices:
        price = inputs.entry_prices[ticker]
        return price if price > 0 else None
    bars = inputs.price_history_by_ticker.get(ticker)
    if bars:
        ordered = sorted(bars, key=lambda bar: bar.as_of)
        price = ordered[-1].close
        return price if price > 0 else None
    feature_price = inputs.features_by_ticker[ticker].as_feature_dict().get("entry_price")
    return feature_price if feature_price and feature_price > 0 else None


def _liquidity_score(candidate: StrategyCandidate, features: ShortHorizonFeatures | None) -> float:
    if features is not None and features.liquidity_score is not None:
        return max(0.0, min(1.0, features.liquidity_score))
    return max(0.0, min(1.0, float(candidate.features.get("liquidity_score", 0.0))))


def _spread_rate(candidate: StrategyCandidate, features: ShortHorizonFeatures | None) -> float | None:
    if features is not None:
        return features.spread_rate
    value = candidate.features.get("spread_rate")
    return float(value) if value is not None else None


def _ontology_score(tags: Sequence[str]) -> float:
    positive = {
        "CostEfficientReversal",
        "LiquiditySupportedReversal",
        "VolumeConfirmedMomentum",
        "MarketDirectionAligned",
        "VolumeConfirmedBreakout",
        "CloseSubstitutePair",
        "MeanReversionCandidate",
        "RelativeUndervaluation",
    }
    risk = {"BidAskBounceRisk", "FalseBreakoutRisk"}
    score = 1.0 + sum(0.04 for tag in tags if tag in positive) - sum(0.08 for tag in tags if tag in risk)
    return max(0.5, min(1.25, score))


def _risk_adjustment(candidate: StrategyCandidate, cost: CostBreakdown, spread_rate: float | None) -> float:
    cost_component = max(0.0, 1 - min(1.0, cost.cost_to_alpha_ratio))
    spread_component = 1.0 if spread_rate is None else max(0.0, 1 - min(1.0, spread_rate / 0.01))
    uncertainty_component = 0.9 if "BidAskBounceRisk" in candidate.ontology_tags else 1.0
    return max(0.0, min(1.0, cost_component * 0.55 + spread_component * 0.35 + uncertainty_component * 0.10))


def _safety_margin(cost_engine: TradingCostEngine) -> float:
    return float(cost_engine.config.get("safety_margin", {}).get("default_safety_margin_rate", 0.001))
