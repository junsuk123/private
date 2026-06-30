from __future__ import annotations

from collections import defaultdict
from datetime import datetime

from app.features.schemas import ReasoningPathRecord, SemanticFeatureRecord

POSITIVE_SIGNALS = {
    "BuyCandidate",
    "HoldWithTrailingStop",
    "BreakoutWatch",
    "Watchlist",
    "TradeAllowed",
}
NEGATIVE_SIGNALS = {"AggressiveBuy"}
RISK_SIGNALS = {
    "SellCandidate",
    "ReduceRiskCandidate",
    "RiskAdjustedSizing",
    "WaitOrTakeProfit",
    "TradeForbidden",
    "ConcentratedPositionRisk",
}
HARD_RISK_FEATURES = {
    "CostBurdenHigh",
    "SpreadTooWide",
    "SlippageRiskHigh",
    "NoOutOfSampleValidation",
}


def build_semantic_reasoning_paths(
    features: tuple[SemanticFeatureRecord, ...],
) -> tuple[ReasoningPathRecord, ...]:
    by_ticker: dict[str, list[SemanticFeatureRecord]] = defaultdict(list)
    for feature in features:
        by_ticker[feature.ticker].append(feature)

    paths: list[ReasoningPathRecord] = []
    for ticker, ticker_features in by_ticker.items():
        as_of = max(feature.as_of for feature in ticker_features)
        positive = tuple(
            feature.feature_name
            for feature in ticker_features
            if feature.semantic_relation == "supportsSignal" and feature.target_signal in POSITIVE_SIGNALS
        )
        negative = tuple(
            feature.feature_name
            for feature in ticker_features
            if feature.semantic_relation == "contradictsSignal" or feature.target_signal in NEGATIVE_SIGNALS
        )
        risk = tuple(
            feature.feature_name
            for feature in ticker_features
            if feature.semantic_relation == "increasesRiskOf" or feature.target_signal in RISK_SIGNALS
        )
        positive_score = sum(feature.confidence for feature in ticker_features if feature.feature_name in positive)
        negative_score = sum(feature.confidence for feature in ticker_features if feature.feature_name in negative)
        risk_score = sum(feature.confidence for feature in ticker_features if feature.feature_name in risk)
        contradiction_score = _bounded((negative_score + risk_score) / max(1.0, positive_score + negative_score + risk_score))
        confidence = _bounded(0.45 + positive_score * 0.08 - negative_score * 0.08 - risk_score * 0.06)
        signal = _select_signal(confidence, positive, risk)
        paths.append(
            ReasoningPathRecord(
                ticker=ticker,
                as_of=as_of,
                strategy_signal=signal,
                positive_features=positive,
                negative_features=negative,
                risk_features=risk,
                contradiction_score=round(contradiction_score, 6),
                final_confidence=round(confidence, 6),
                explanation=_explain(ticker, signal, as_of, positive, negative, risk, confidence),
            )
        )
    return tuple(paths)


def _select_signal(confidence: float, positive: tuple[str, ...], risk: tuple[str, ...]) -> str:
    if any(feature in HARD_RISK_FEATURES for feature in risk):
        return "TradeForbidden"
    if "ConcentratedPositionRisk" in risk:
        return "SellCandidate"
    if any(feature in {"SellCandidate"} for feature in risk):
        return "SellCandidate"
    if risk and confidence < 0.45:
        return "ReduceRiskCandidate"
    if "CostEfficientTrade" in positive and len(positive) >= 2 and not risk:
        return "TradeAllowed"
    if len(positive) >= 3 and confidence >= 0.55:
        return "BuyCandidate"
    if "BollingerSqueeze" in positive:
        return "BreakoutWatch"
    return "HoldOrWatch"


def _bounded(value: float) -> float:
    return max(0.0, min(1.0, value))


def _explain(
    ticker: str,
    signal: str,
    as_of: datetime,
    positive: tuple[str, ...],
    negative: tuple[str, ...],
    risk: tuple[str, ...],
    confidence: float,
) -> str:
    return (
        f"{ticker} as of {as_of.isoformat()} -> {signal} with confidence {confidence:.2f}. "
        f"Positive={len(positive)}, negative={len(negative)}, risk={len(risk)}."
    )
