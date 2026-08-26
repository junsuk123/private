from __future__ import annotations

import math
from dataclasses import dataclass, replace
from typing import Any, Iterable, Sequence

import numpy as np

from app.graph.knowledge_graph import KnowledgeGraph
from app.quant.config import QuantConfig, default_quant_config
from app.quant.contracts import DataQuality, QuantEvidence, ValidationStatus

QUANT_GNN_SCHEMA_VERSION = "quant-evidence-gnn-v1"
QUANT_GNN_METRICS: tuple[str, ...] = (
    "simple_return", "zscore", "bollinger_width", "bollinger_position",
    "macd", "rsi", "trend", "realized_volatility", "max_drawdown",
)


@dataclass(frozen=True)
class QuantFeatureFrame:
    schema_version: str
    feature_names: tuple[str, ...]
    raw_values: np.ndarray
    normalized_values: np.ndarray
    freshness: np.ndarray
    quality: np.ndarray
    validation: np.ndarray
    mask: np.ndarray

    def compatible_with(self, version: str, feature_names: Sequence[str]) -> bool:
        return self.schema_version == version and self.feature_names == tuple(feature_names)


def to_gnn_feature_frame(
    evidence: Iterable[QuantEvidence],
    *,
    expected_schema_version: str = QUANT_GNN_SCHEMA_VERSION,
    expected_feature_names: Sequence[str] = QUANT_GNN_METRICS,
    config: QuantConfig | None = None,
) -> QuantFeatureFrame:
    """Build explicit raw/normalized/quality/mask channels.

    A caller requesting a different ordering/version is refused before inference; it
    never receives a same-shaped but semantically shifted vector.
    """
    if expected_schema_version != QUANT_GNN_SCHEMA_VERSION:
        raise ValueError("GNN quant schema version mismatch")
    if tuple(expected_feature_names) != QUANT_GNN_METRICS:
        raise ValueError("GNN quant feature ordering mismatch")
    cfg = config or default_quant_config()
    latest = {item.metric: item for item in evidence}
    size = len(QUANT_GNN_METRICS)
    raw = np.zeros(size, dtype=np.float64)
    normalized = np.zeros(size, dtype=np.float64)
    freshness = np.ones(size, dtype=np.float64)
    quality = np.zeros(size, dtype=np.float64)
    validation = np.zeros(size, dtype=np.float64)
    mask = np.zeros(size, dtype=np.float64)
    for index, metric in enumerate(QUANT_GNN_METRICS):
        item = latest.get(metric)
        if item is None or not item.usable:
            continue
        value = float(item.value)
        raw[index] = value
        normalized[index] = _normalize(metric, value)
        freshness[index] = min(1.0, item.freshness_ms / cfg.stale_after_ms)
        quality[index] = {DataQuality.GOOD: 1.0, DataQuality.DEGRADED: 0.5}.get(item.data_quality, 0.0)
        validation[index] = {
            ValidationStatus.PASSED: 1.0,
            ValidationStatus.UNVALIDATED: 0.5,
            ValidationStatus.UNAVAILABLE: 0.25,
        }.get(item.validation_status, 0.0)
        mask[index] = 1.0
    return QuantFeatureFrame(
        QUANT_GNN_SCHEMA_VERSION, QUANT_GNN_METRICS, raw, normalized,
        freshness, quality, validation, mask,
    )


def add_quant_evidence_to_graph(graph: KnowledgeGraph, evidence: Iterable[QuantEvidence]) -> int:
    """Reuse the graph's existing evidence relations; never creates order authority."""
    count = 0
    for item in evidence:
        evidence_id = f"quant:{item.implementation}:{item.metric}:{item.timestamp.isoformat()}"
        graph.add(item.symbol, "computedFrom", item.metric, evidence_id)
        graph.add(item.metric, "hasCalculationMethod", item.method_reference, evidence_id)
        graph.add(item.metric, "hasFreshness", str(round(item.freshness_ms, 3)), evidence_id)
        graph.add(item.metric, "hasDataQuality", item.data_quality.value, evidence_id)
        count += 4
        if not item.usable:
            graph.add(item.symbol, "increasesRiskOf", "InvalidQuantEvidence", evidence_id)
            count += 1
        elif item.metric in {"realized_volatility", "max_drawdown"} and float(item.value) < 0:
            graph.add(item.symbol, "increasesRiskOf", "VolatilityRisk", evidence_id)
            count += 1
        elif item.metric == "trend":
            predicate = "supportsSignal" if float(item.value) > 0 else "contradictsSignal"
            graph.add(item.symbol, predicate, "BuyCandidate", evidence_id)
            count += 1
    return count


def quant_risk_factor(evidence: Iterable[QuantEvidence], *, stale_after_ms: float) -> tuple[float, tuple[str, ...]]:
    """Conservative overlay: evidence may only keep or reduce requested exposure."""
    rows = tuple(evidence)
    if not rows:
        return 1.0, ()
    reasons: list[str] = []
    factor = 1.0
    for item in rows:
        if item.validation_status is ValidationStatus.FAILED:
            factor = min(factor, 0.0)
            reasons.append("QUANT_PARITY_FAILED")
        elif item.data_quality is DataQuality.INVALID or not item.usable:
            factor = min(factor, 0.5)
            reasons.append("QUANT_EVIDENCE_INVALID")
        if item.freshness_ms > stale_after_ms:
            factor = min(factor, 0.5)
            reasons.append("QUANT_EVIDENCE_STALE")
    return max(0.0, min(1.0, factor)), tuple(dict.fromkeys(reasons))


def apply_quant_to_gate_inputs(inputs: Any, evidence: Iterable[QuantEvidence], *, config: QuantConfig | None = None) -> Any:
    """Attach a <=1 sizing factor to FinalTradeGate inputs without authorizing anything."""
    cfg = config or default_quant_config()
    factor, reasons = quant_risk_factor(evidence, stale_after_ms=cfg.stale_after_ms)
    if not hasattr(inputs, "quant_evidence_factor"):
        raise TypeError("gate inputs do not support quant evidence")
    result = replace(inputs, quant_evidence_factor=factor)
    if factor <= 0 and hasattr(inputs, "stale_data_reasons"):
        result = replace(result, stale_data_reasons=tuple((*inputs.stale_data_reasons, *reasons)))
    return result


def _normalize(metric: str, value: float) -> float:
    scales = {
        "simple_return": 0.02, "zscore": 3.0, "bollinger_width": 0.10,
        "bollinger_position": 1.0, "macd": 1.0, "rsi": 100.0,
        "trend": 0.05, "realized_volatility": 0.50, "max_drawdown": 0.20,
    }
    scale = scales[metric]
    if metric == "rsi":
        value = (value - 50.0) / 50.0
        scale = 1.0
    elif metric == "bollinger_position":
        value -= 0.5
    return float(math.tanh(value / scale))
