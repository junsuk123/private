from __future__ import annotations

from datetime import datetime
from typing import Any, Iterable

from app.features.schemas import SemanticFeatureRecord
from app.strategy.candidate_factory import RankedStrategyCandidate
from app.strategy.candidates import StrategyCandidate


POSITIVE_STRATEGY_SIGNALS = {
    "ShortTermReversalBuy",
    "IntradayMomentumBuy",
    "PairMeanReversionBuy",
    "TechnicalBreakoutBuy",
    "CostEfficientTrade",
    "RealityCheckPassed",
}
RISK_STRATEGY_SIGNALS = {
    "BidAskBounceRisk",
    "FalseBreakoutRisk",
    "SpreadTooWide",
    "SlippageRiskHigh",
    "CostBurdenHigh",
    "DataSnoopingRisk",
    "NoOutOfSampleValidation",
}
RISK_MANAGER_BLOCKING_TAGS = {"TradeForbidden", "SpreadTooWide", "SlippageRiskHigh", "CostBurdenHigh"}

_TAG_FEATURE_RULES: dict[str, tuple[str, str, str, str]] = {
    "BidAskBounceRisk": ("BidAskBounceRisk", "microstructure_risk", "increasesRiskOf", "RiskAdjustedSizing"),
    "FalseBreakoutRisk": ("FalseBreakoutRisk", "strategy_validation_risk", "increasesRiskOf", "RiskAdjustedSizing"),
    "SpreadTooWide": ("SpreadTooWide", "execution_cost_risk", "increasesRiskOf", "TradeForbidden"),
    "SlippageRiskHigh": ("SlippageRiskHigh", "execution_cost_risk", "increasesRiskOf", "TradeForbidden"),
    "CostBurdenHigh": ("CostBurdenHigh", "execution_cost_risk", "increasesRiskOf", "TradeForbidden"),
    "DataSnoopingRisk": ("DataSnoopingRisk", "strategy_validation_risk", "increasesRiskOf", "RiskAdjustedSizing"),
    "NoOutOfSampleValidation": ("NoOutOfSampleValidation", "strategy_validation_risk", "increasesRiskOf", "TradeForbidden"),
    "RealityCheckPassed": ("RealityCheckPassed", "strategy_validation", "supportsSignal", "TradeAllowed"),
    "CostEfficientTrade": ("CostEfficientTrade", "net_profitability", "supportsSignal", "TradeAllowed"),
}


def semantic_features_from_strategy_candidates(
    candidates: Iterable[StrategyCandidate | RankedStrategyCandidate],
    *,
    live_trading_requested: bool = False,
) -> tuple[SemanticFeatureRecord, ...]:
    features: list[SemanticFeatureRecord] = []
    for item in candidates:
        candidate, ranked = _unwrap_candidate(item)
        as_of = candidate.created_at or datetime.now()
        base_confidence = candidate.confidence
        target_net_return = _target_net_return(candidate, ranked)
        net_expected_return = _net_expected_return(candidate, ranked)
        tags = set(candidate.ontology_tags)

        signal = _strategy_buy_signal(candidate, tags, net_expected_return, target_net_return)
        if signal is not None and not _has_hard_cost_risk(tags):
            features.append(
                _feature(
                    candidate,
                    as_of,
                    signal,
                    "strategy_signal",
                    base_confidence,
                    "supportsSignal",
                    "BuyCandidate",
                )
            )

        if net_expected_return is not None:
            if net_expected_return > target_net_return:
                features.append(
                    _feature(
                        candidate,
                        as_of,
                        "CostEfficientTrade",
                        "net_profitability",
                        min(1.0, base_confidence + 0.05),
                        "supportsSignal",
                        "TradeAllowed",
                    )
                )
            else:
                features.append(
                    _feature(
                        candidate,
                        as_of,
                        "CostBurdenHigh",
                        "execution_cost_risk",
                        0.9,
                        "increasesRiskOf",
                        "TradeForbidden",
                    )
                )

        for tag in sorted(tags):
            rule = _TAG_FEATURE_RULES.get(tag)
            if rule is None:
                continue
            name, category, relation, target = rule
            features.append(_feature(candidate, as_of, name, category, base_confidence, relation, target))

        if live_trading_requested and not candidate.validation_id and "RealityCheckPassed" not in tags:
            features.append(
                _feature(
                    candidate,
                    as_of,
                    "NoOutOfSampleValidation",
                    "strategy_validation_risk",
                    1.0,
                    "increasesRiskOf",
                    "TradeForbidden",
                )
            )
    return tuple(_deduplicate(features))


def risk_manager_ontology_tags(
    semantic_features: Iterable[SemanticFeatureRecord],
) -> tuple[str, ...]:
    tags: set[str] = set()
    for feature in semantic_features:
        if feature.target_signal == "TradeForbidden":
            tags.add("TradeForbidden")
        if feature.feature_name in RISK_MANAGER_BLOCKING_TAGS:
            tags.add(feature.feature_name)
    return tuple(sorted(tags))


def _strategy_buy_signal(
    candidate: StrategyCandidate,
    tags: set[str],
    net_expected_return: float | None,
    target_net_return: float,
) -> str | None:
    profitable = net_expected_return is None or net_expected_return > target_net_return
    if not profitable:
        return None
    if {"ShortTermOverreaction", "LiquiditySupportedReversal"}.issubset(tags):
        return "ShortTermReversalBuy"
    if {"OpeningReturnStrength", "VolumeConfirmedMomentum", "MarketDirectionAligned"}.issubset(tags):
        return "IntradayMomentumBuy"
    if {"CloseSubstitutePair", "PairSpreadDivergence", "RelativeUndervaluation"}.issubset(tags):
        return "PairMeanReversionBuy"
    if "VolumeConfirmedBreakout" in tags and (
        "MovingAverageBreakout" in tags or "TradingRangeBreakout" in tags or candidate.signal_name == "brock_1992_technical_breakout"
    ):
        return "TechnicalBreakoutBuy"
    return None


def _unwrap_candidate(
    item: StrategyCandidate | RankedStrategyCandidate,
) -> tuple[StrategyCandidate, RankedStrategyCandidate | None]:
    if isinstance(item, RankedStrategyCandidate):
        return item.candidate, item
    return item, None


def _target_net_return(candidate: StrategyCandidate, ranked: RankedStrategyCandidate | None) -> float:
    if ranked is not None:
        return ranked.target_net_return
    return float(candidate.features.get("target_net_return", 0.0))


def _net_expected_return(candidate: StrategyCandidate, ranked: RankedStrategyCandidate | None) -> float | None:
    if ranked is not None:
        return ranked.cost_breakdown.net_expected_return
    value = candidate.features.get("net_expected_return_after_cost")
    return float(value) if value is not None else None


def _has_hard_cost_risk(tags: set[str]) -> bool:
    return bool(tags.intersection({"CostBurdenHigh", "SpreadTooWide", "SlippageRiskHigh"}))


def _feature(
    candidate: StrategyCandidate,
    as_of: datetime,
    name: str,
    category: str,
    confidence: float,
    relation: str,
    target_signal: str,
) -> SemanticFeatureRecord:
    return SemanticFeatureRecord(
        ticker=candidate.ticker,
        as_of=as_of,
        feature_name=name,
        feature_category=category,
        state="active",
        confidence=max(0.0, min(1.0, confidence)),
        supporting_indicators=tuple(candidate.ontology_tags),
        semantic_relation=relation,  # type: ignore[arg-type]
        target_signal=target_signal,
        ontology_node_id=f"strategy:{candidate.ticker}:{candidate.strategy_family}:{name}",
        generation_method="strategy_candidate_ontology",
        model_version=None,
        metadata={
            "strategy_family": candidate.strategy_family,
            "signal_name": candidate.signal_name,
            "expected_exit_price": candidate.expected_exit_price,
            "gross_expected_return": candidate.gross_expected_return,
        },
    )


def _deduplicate(features: list[SemanticFeatureRecord]) -> list[SemanticFeatureRecord]:
    by_key: dict[tuple[str, str, str], SemanticFeatureRecord] = {}
    for feature in features:
        key = (feature.ticker, feature.feature_name, feature.target_signal or "")
        current = by_key.get(key)
        if current is None or feature.confidence > current.confidence:
            by_key[key] = feature
    return list(by_key.values())
