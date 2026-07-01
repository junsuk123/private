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
            "theory_id": _theory_id_for_candidate(candidate, name, target_signal),
            "theory_family": _theory_family_for_candidate(candidate),
            "style": _style_for_candidate(candidate),
            "horizon_bucket": _horizon_for_candidate(candidate),
            "expected_holding_minutes": candidate.expected_holding_minutes,
            "evidence_cluster_id": _evidence_cluster_for_feature(name, category, target_signal),
            "action_bias": _action_bias_for_relation(relation, target_signal),
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


def _theory_id_for_candidate(candidate: StrategyCandidate, feature_name: str, target_signal: str | None) -> str:
    if candidate.signal_name in {
        "jegadeesh_1990_short_term_reversal",
        "brock_1992_technical_breakout",
    }:
        return candidate.signal_name
    if candidate.signal_name == "gao_2018_opening_return_momentum":
        return "gao_2018_intraday_momentum"
    if target_signal in {"SellCandidate", "WaitOrTakeProfit"}:
        return "profit_taking_exit"
    if target_signal in {"RiskAdjustedSizing", "TradeForbidden", "ReduceRiskCandidate"} or "Risk" in feature_name:
        return "risk_reduction_exit"
    if candidate.strategy_family == "intraday_momentum":
        return "gao_2018_intraday_momentum"
    if candidate.strategy_family == "technical_rule":
        return "brock_1992_technical_breakout"
    if candidate.strategy_family == "short_term_reversal":
        return "jegadeesh_1990_short_term_reversal"
    return candidate.signal_name


def _theory_family_for_candidate(candidate: StrategyCandidate) -> str:
    if candidate.strategy_family == "technical_rule":
        return "technical_breakout"
    return candidate.strategy_family


def _style_for_candidate(candidate: StrategyCandidate) -> str:
    if candidate.strategy_family == "short_term_reversal":
        return "contrarian"
    if candidate.strategy_family == "intraday_momentum":
        return "continuation"
    if candidate.strategy_family == "technical_rule":
        return "breakout"
    if candidate.strategy_family == "pair_relative_value":
        return "mean_reversion"
    return "unknown"


def _horizon_for_candidate(candidate: StrategyCandidate) -> str:
    minutes = candidate.expected_holding_minutes
    if minutes <= 10:
        return "scalp"
    if minutes <= 90:
        return "short_intraday"
    if minutes <= 390:
        return "late_intraday"
    return "swing"


def _evidence_cluster_for_feature(feature_name: str, category: str, target_signal: str | None) -> str:
    if target_signal in {"TradeForbidden", "RiskAdjustedSizing", "ReduceRiskCandidate"} or "risk" in category:
        return "risk_cluster"
    if "Reversal" in feature_name or "Overreaction" in feature_name:
        return "reversal_cluster"
    if "Momentum" in feature_name or "OpeningReturn" in feature_name:
        return "momentum_cluster"
    if "Breakout" in feature_name:
        return "breakout_cluster"
    if "Volume" in feature_name:
        return "volume_cluster"
    return "trend_cluster"


def _action_bias_for_relation(relation: str, target_signal: str | None) -> str:
    if target_signal in {"TradeForbidden", "RiskAdjustedSizing", "ReduceRiskCandidate"}:
        return "REDUCE"
    if target_signal in {"SellCandidate", "WaitOrTakeProfit"}:
        return "SELL"
    if relation == "supportsSignal" and target_signal in {"BuyCandidate", "TradeAllowed"}:
        return "BUY"
    return "WATCH"
