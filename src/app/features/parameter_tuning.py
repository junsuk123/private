from __future__ import annotations

import math
from dataclasses import dataclass, replace
from datetime import datetime
from statistics import mean, pstdev
from typing import Protocol

from app.features.indicator_engine import IndicatorEngineConfig
from app.features.schemas import FormulaParameterRecommendation, OHLCVBar

PARAMETER_TUNER_VERSION = "formula-parameter-tuner-v1"


@dataclass(frozen=True)
class ParameterTuningExample:
    ticker: str
    as_of: datetime
    context_features: dict[str, float]
    parameters: dict[str, float | int]
    score: float


@dataclass(frozen=True)
class ParameterTunerState:
    model_version: str
    feature_means: dict[str, float]
    feature_stdevs: dict[str, float]
    parameter_centroid: dict[str, float]
    training_examples: int


class FormulaParameterTuner(Protocol):
    model_version: str

    def fit(self, examples: tuple[ParameterTuningExample, ...]) -> ParameterTunerState:
        ...

    def recommend(
        self,
        bars: tuple[OHLCVBar, ...],
        base_config: IndicatorEngineConfig,
        as_of: datetime | None = None,
    ) -> tuple[IndicatorEngineConfig, tuple[FormulaParameterRecommendation, ...]]:
        ...


class RegimeFormulaParameterTuner:
    """Dependency-free parameter tuner for formula engines.

    The tuner learns a weighted centroid of successful parameter sets and nudges
    it by the current as-of volatility/trend regime. It keeps formulas intact:
    only formula parameters are selected by the model.
    """

    def __init__(self, state: ParameterTunerState | None = None, model_version: str = PARAMETER_TUNER_VERSION) -> None:
        self.model_version = model_version
        self.state = state or ParameterTunerState(
            model_version=model_version,
            feature_means={},
            feature_stdevs={},
            parameter_centroid={},
            training_examples=0,
        )

    def fit(self, examples: tuple[ParameterTuningExample, ...]) -> ParameterTunerState:
        if not examples:
            return self.state
        feature_names = sorted({name for example in examples for name in example.context_features})
        parameter_names = sorted({name for example in examples for name in example.parameters})
        weights = [max(0.0, example.score) for example in examples]
        if sum(weights) == 0:
            weights = [1.0 for _ in examples]
        feature_means = {
            name: _mean([example.context_features[name] for example in examples if name in example.context_features])
            for name in feature_names
        }
        feature_stdevs = {
            name: _stdev([example.context_features[name] for example in examples if name in example.context_features])
            for name in feature_names
        }
        parameter_centroid = {
            name: _weighted_mean(
                [float(example.parameters[name]) for example in examples if name in example.parameters],
                [weights[index] for index, example in enumerate(examples) if name in example.parameters],
            )
            for name in parameter_names
        }
        self.state = ParameterTunerState(
            model_version=self.model_version,
            feature_means=feature_means,
            feature_stdevs=feature_stdevs,
            parameter_centroid=parameter_centroid,
            training_examples=len(examples),
        )
        return self.state

    def recommend(
        self,
        bars: tuple[OHLCVBar, ...],
        base_config: IndicatorEngineConfig,
        as_of: datetime | None = None,
    ) -> tuple[IndicatorEngineConfig, tuple[FormulaParameterRecommendation, ...]]:
        ordered = tuple(sorted((bar for bar in bars if as_of is None or bar.as_of <= as_of), key=lambda bar: bar.as_of))
        if not ordered:
            return base_config, ()
        context = build_parameter_context_features(ordered)
        volatility = context.get("realized_volatility_20", 0.0)
        trend = context.get("return_20", 0.0)
        centroid = self.state.parameter_centroid

        rsi_period = _bounded_int(_parameter_value(centroid, "rsi_period", base_config.rsi_period), 7, 28)
        bollinger_period = _bounded_int(_parameter_value(centroid, "bollinger_period", base_config.bollinger_period), 10, 40)
        bollinger_stddev = _bounded_float(_parameter_value(centroid, "bollinger_stddev", base_config.bollinger_stddev), 1.5, 3.0)
        atr_period = _bounded_int(_parameter_value(centroid, "atr_period", base_config.atr_period), 7, 28)
        volume_window = _bounded_int(_parameter_value(centroid, "volume_window", base_config.volume_window), 5, 40)

        if volatility > 0.45:
            rsi_period = max(7, rsi_period - 4)
            bollinger_stddev = min(3.0, bollinger_stddev + 0.35)
            atr_period = max(7, atr_period - 3)
            volume_window = max(5, volume_window - 5)
            reason = "high volatility regime"
        elif volatility < 0.15 and abs(trend) < 0.04:
            rsi_period = min(28, rsi_period + 5)
            bollinger_period = min(40, bollinger_period + 5)
            bollinger_stddev = max(1.5, bollinger_stddev - 0.2)
            reason = "low volatility range regime"
        else:
            reason = "normal trend regime"

        config = replace(
            base_config,
            rsi_period=rsi_period,
            bollinger_period=bollinger_period,
            bollinger_stddev=round(bollinger_stddev, 3),
            atr_period=atr_period,
            volume_window=volume_window,
            source=f"{base_config.source}+ai-parameters",
        )
        recommendations = tuple(
            FormulaParameterRecommendation(
                ticker=ordered[-1].ticker,
                as_of=ordered[-1].as_of,
                parameter_name=name,
                value=value,
                recommended_by="RegimeFormulaParameterTuner",
                model_version=self.model_version,
                confidence=_confidence(self.state.training_examples, volatility),
                reason=reason,
                source_features=tuple(sorted(context)),
            )
            for name, value in {
                "rsi_period": config.rsi_period,
                "bollinger_period": config.bollinger_period,
                "bollinger_stddev": config.bollinger_stddev,
                "atr_period": config.atr_period,
                "volume_window": config.volume_window,
            }.items()
        )
        return config, recommendations


def build_parameter_context_features(bars: tuple[OHLCVBar, ...]) -> dict[str, float]:
    closes = [bar.close for bar in bars]
    volumes = [bar.volume for bar in bars]
    returns = [closes[i] / closes[i - 1] - 1 for i in range(1, len(closes)) if closes[i - 1]]
    return_20 = closes[-1] / closes[-21] - 1 if len(closes) > 20 and closes[-21] else 0.0
    vol_window = returns[-20:]
    realized_volatility = pstdev(vol_window) * math.sqrt(252) if len(vol_window) > 1 else 0.0
    volume_ratio = volumes[-1] / mean(volumes[-21:-1]) if len(volumes) > 21 and mean(volumes[-21:-1]) else 1.0
    return {
        "return_20": return_20,
        "realized_volatility_20": realized_volatility,
        "volume_ratio_20": volume_ratio,
        "sample_size": float(len(bars)),
    }


def _parameter_value(values: dict[str, float], name: str, default: float | int) -> float:
    return values.get(name, float(default))


def _bounded_int(value: float, low: int, high: int) -> int:
    return max(low, min(high, int(round(value))))


def _bounded_float(value: float, low: float, high: float) -> float:
    return max(low, min(high, float(value)))


def _confidence(training_examples: int, volatility: float) -> float:
    sample_confidence = min(0.35, training_examples / 100)
    regime_confidence = 0.25 if volatility else 0.15
    return round(min(0.95, 0.35 + sample_confidence + regime_confidence), 6)


def _mean(values: list[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def _stdev(values: list[float]) -> float:
    if len(values) < 2:
        return 1.0
    avg = _mean(values)
    return math.sqrt(sum((value - avg) ** 2 for value in values) / (len(values) - 1)) or 1.0


def _weighted_mean(values: list[float], weights: list[float]) -> float:
    total = sum(weights)
    return sum(value * weight for value, weight in zip(values, weights)) / total if total else _mean(values)
