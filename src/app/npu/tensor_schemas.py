from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, Sequence

import numpy as np


@dataclass(frozen=True)
class TensorSchema:
    name: str
    version: str
    feature_names: tuple[str, ...]

    @property
    def feature_dim(self) -> int:
        return len(self.feature_names)

    def validate(self, matrix: np.ndarray) -> None:
        if matrix.ndim != 2 or matrix.shape[1] != self.feature_dim:
            raise ValueError(f"{self.name} requires shape [N, {self.feature_dim}], got {tuple(matrix.shape)}")


SCHEMAS: dict[str, TensorSchema] = {
    "candidate_features": TensorSchema(
        "candidate_features",
        "v1",
        (
            "support_score",
            "risk_score",
            "momentum_score",
            "value_score",
            "liquidity_score",
            "confidence_score",
            "spread_rate",
            "depth_score",
            "volume_zscore",
            "volatility_5m",
            "market_alignment_score",
        ),
    ),
    "theory_vote_features": TensorSchema(
        "theory_vote_features",
        "v1",
        (
            "raw_signal",
            "theory_prior_weight",
            "regime_gate",
            "data_quality_weight",
            "validation_weight",
            "horizon_compatibility",
            "cluster_score",
            "expected_net_return",
            "uncertainty",
            "position_context_weight",
        ),
    ),
    "conflict_features": TensorSchema(
        "conflict_features",
        "v1",
        (
            "style_conflict",
            "horizon_conflict",
            "action_conflict",
            "regime_conflict",
            "duplicate_evidence_score",
            "validation_gap",
            "cost_conflict",
            "execution_risk_conflict",
        ),
    ),
    "execution_edge_features": TensorSchema(
        "execution_edge_features",
        "v1",
        (
            "strategy_expected_return",
            "broker_fee_rate",
            "tax_rate",
            "spread_rate",
            "expected_slippage_rate",
            "depth_score",
            "queue_imbalance",
            "order_flow_imbalance",
            "toxicity_proxy",
            "short_term_volatility",
        ),
    ),
    "action_scores": TensorSchema(
        "action_scores",
        "v1",
        ("buy_score", "sell_score", "hold_score", "reduce_score", "watch_score", "confidence", "uncertainty"),
    ),
}


def get_tensor_schema(name: str) -> TensorSchema:
    try:
        return SCHEMAS[name]
    except KeyError as exc:
        raise KeyError(f"Unknown tensor schema: {name}") from exc


def matrix_from_records(records: Sequence[Mapping[str, float | int | None]], schema: TensorSchema) -> np.ndarray:
    matrix = np.zeros((len(records), schema.feature_dim), dtype=np.float32)
    for row, record in enumerate(records):
        for col, name in enumerate(schema.feature_names):
            value = record.get(name, 0.0)
            matrix[row, col] = 0.0 if value is None else float(value)
    return np.nan_to_num(matrix, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32, copy=False)
