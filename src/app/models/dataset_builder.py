from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import Any

from app.features import IndicatorEngine, SemanticFeatureEngine
from app.features.ai_semantic_layer import AISemanticTrainingExample
from app.features.parameter_tuning import ParameterTuningExample, build_parameter_context_features
from app.features.schemas import OHLCVBar
from app.models.labeling import LabelConfig, future_return, triple_barrier_label


@dataclass(frozen=True)
class DatasetRow:
    ticker: str
    as_of: datetime
    features: dict[str, float | int]
    labels: dict[str, float | int | None]
    metadata: dict[str, Any]


class DatasetBuilder:
    def __init__(
        self,
        indicator_engine: IndicatorEngine | None = None,
        semantic_engine: SemanticFeatureEngine | None = None,
        label_config: LabelConfig | None = None,
    ) -> None:
        self.indicator_engine = indicator_engine or IndicatorEngine()
        self.semantic_engine = semantic_engine or SemanticFeatureEngine()
        self.label_config = label_config or LabelConfig()

    def build_rows(
        self,
        bars: tuple[OHLCVBar, ...],
        decision_times: tuple[datetime, ...],
    ) -> tuple[DatasetRow, ...]:
        rows: list[DatasetRow] = []
        ordered = tuple(sorted(bars, key=lambda bar: bar.as_of))
        if not ordered:
            return ()
        ticker = ordered[-1].ticker
        for as_of in sorted(decision_times):
            indicators = self.indicator_engine.calculate(ordered, as_of=as_of)
            semantic = self.semantic_engine.generate(indicators)
            feature_values: dict[str, float | int] = {}
            for indicator in indicators:
                if isinstance(indicator.value, (int, float)):
                    feature_values[indicator.indicator_name] = float(indicator.value)
            for feature in semantic:
                feature_values[f"semantic__{feature.feature_name}"] = 1
                feature_values[f"semantic_confidence__{feature.feature_name}"] = feature.confidence

            tb = triple_barrier_label(ordered, as_of, self.label_config)
            labels = {
                "future_return_5d": future_return(ordered, as_of, 5),
                "future_return_20d": future_return(ordered, as_of, 20),
                "triple_barrier_label": tb.label if tb is not None else None,
            }
            rows.append(
                DatasetRow(
                    ticker=ticker,
                    as_of=as_of,
                    features=feature_values,
                    labels=labels,
                    metadata={
                        "feature_cutoff": as_of.isoformat(),
                        "label_horizon_bars": self.label_config.horizon_bars,
                        "no_lookahead": True,
                    },
                )
            )
        return tuple(rows)

    def build_ai_training_examples(
        self,
        bars: tuple[OHLCVBar, ...],
        decision_times: tuple[datetime, ...],
        label_rules: dict[str, str],
    ) -> tuple[AISemanticTrainingExample, ...]:
        rows = self.build_rows(bars, decision_times)
        examples: list[AISemanticTrainingExample] = []
        for row in rows:
            labels = {
                target_name: _evaluate_label_rule(row.labels, rule)
                for target_name, rule in label_rules.items()
            }
            examples.append(
                AISemanticTrainingExample(
                    ticker=row.ticker,
                    as_of=row.as_of,
                    inputs={key: float(value) for key, value in row.features.items() if isinstance(value, (int, float))},
                    labels=labels,
                )
            )
        return tuple(examples)

    def build_parameter_tuning_examples(
        self,
        bars: tuple[OHLCVBar, ...],
        decision_times: tuple[datetime, ...],
    ) -> tuple[ParameterTuningExample, ...]:
        ordered = tuple(sorted(bars, key=lambda bar: bar.as_of))
        examples: list[ParameterTuningExample] = []
        for as_of in sorted(decision_times):
            visible = tuple(bar for bar in ordered if bar.as_of <= as_of)
            if len(visible) < 25:
                continue
            context = build_parameter_context_features(visible)
            future_5d = future_return(ordered, as_of, 5)
            score = abs(float(future_5d or 0.0))
            volatility = context.get("realized_volatility_20", 0.0)
            if volatility > 0.45:
                parameters = {
                    "rsi_period": 10,
                    "bollinger_period": 20,
                    "bollinger_stddev": 2.5,
                    "atr_period": 10,
                    "volume_window": 10,
                }
            elif volatility < 0.15:
                parameters = {
                    "rsi_period": 21,
                    "bollinger_period": 25,
                    "bollinger_stddev": 1.8,
                    "atr_period": 18,
                    "volume_window": 25,
                }
            else:
                parameters = {
                    "rsi_period": 14,
                    "bollinger_period": 20,
                    "bollinger_stddev": 2.0,
                    "atr_period": 14,
                    "volume_window": 20,
                }
            examples.append(
                ParameterTuningExample(
                    ticker=ordered[-1].ticker,
                    as_of=as_of,
                    context_features=context,
                    parameters=parameters,
                    score=score,
                )
            )
        return tuple(examples)


def _evaluate_label_rule(labels: dict[str, float | int | None], rule: str) -> int:
    if rule == "future_return_5d_positive":
        value = labels.get("future_return_5d")
        return int(value is not None and float(value) > 0)
    if rule == "future_return_5d_above_2pct":
        value = labels.get("future_return_5d")
        return int(value is not None and float(value) > 0.02)
    if rule == "triple_barrier_positive":
        return int(labels.get("triple_barrier_label") == 1)
    if rule == "triple_barrier_negative":
        return int(labels.get("triple_barrier_label") == -1)
    raise ValueError(f"Unknown AI label rule: {rule}")
