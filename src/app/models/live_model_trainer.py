from __future__ import annotations

import math
from datetime import datetime, timezone
from typing import Any

from app.features.feature_schema import LIVE_SHORT_HORIZON_SCHEMA
from app.models.model_artifact_registry import ModelArtifactRegistry
from app.models.model_validation import auc_like_score, validate_training_dataset


def train_live_short_horizon_model(
    rows: list[dict[str, Any]],
    *,
    registry: ModelArtifactRegistry | None = None,
    minimum_examples: int = 30,
    minimum_positive_labels: int = 5,
    minimum_negative_labels: int = 5,
    force_live_ineligible_reason: str | None = None,
) -> dict[str, Any]:
    registry = registry or ModelArtifactRegistry()
    ok, reasons = validate_training_dataset(
        rows,
        minimum_examples=minimum_examples,
        minimum_positive_labels=minimum_positive_labels,
        minimum_negative_labels=minimum_negative_labels,
    )
    feature_names = LIVE_SHORT_HORIZON_SCHEMA.feature_names
    if not ok:
        if force_live_ineligible_reason:
            reasons = (*reasons, force_live_ineligible_reason)
        metrics = _dataset_metrics(rows)
        artifact = _artifact_payload(feature_names, [0.0] * len(feature_names), 0.0, [0.0] * len(feature_names), 0.0, metrics, False, reasons)
        registry.save(artifact)
        return artifact
    x = [[float(row["features"][name]) for name in feature_names] for row in rows]
    y = [int(row["label"]) for row in rows]
    returns = [float(row.get("forward_net_return_bps", 0.0)) for row in rows]
    means, scales, x_scaled = _standardize(x)
    weights, bias = _fit_logistic(x_scaled, y)
    ret_weights, ret_bias = _fit_linear(x_scaled, returns)
    probs = [_sigmoid(_dot(row, weights) + bias) for row in x_scaled]
    auc = auc_like_score(y, probs)
    precision_at_k = _precision_at_k(y, probs, max(1, len(y) // 5))
    avg_return_top = _avg_return_top(returns, probs, max(1, len(y) // 5))
    live_eligible = auc >= 0.55 and precision_at_k >= 0.35 and avg_return_top > 0
    reason_codes = () if live_eligible else ("METRICS_BELOW_LIVE_THRESHOLDS",)
    if force_live_ineligible_reason:
        live_eligible = False
        reason_codes = (*reason_codes, force_live_ineligible_reason)
    metrics = {
        "auc": auc,
        "precision_at_k": precision_at_k,
        "avg_forward_net_return_bps_top_k": avg_return_top,
        "example_count": float(len(rows)),
        "positive_labels": float(sum(y)),
        "negative_labels": float(len(y) - sum(y)),
    }
    artifact = _artifact_payload(
        feature_names,
        _unscale_weights(weights, means, scales),
        _unscale_bias(weights, bias, means, scales),
        _unscale_weights(ret_weights, means, scales),
        _unscale_bias(ret_weights, ret_bias, means, scales),
        metrics,
        live_eligible,
        reason_codes,
    )
    registry.save(artifact)
    return artifact


def _artifact_payload(
    feature_names: tuple[str, ...],
    weights: list[float],
    bias: float,
    ret_weights: list[float],
    ret_bias: float,
    metrics: dict[str, float],
    live_eligible: bool,
    reason_codes: tuple[str, ...],
) -> dict[str, Any]:
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    return {
        "artifact_id": f"live_short_horizon.{stamp}",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "feature_schema_hash": LIVE_SHORT_HORIZON_SCHEMA.schema_hash,
        "feature_names": list(feature_names),
        "classification": {"family": "logistic_regression_sgd", "weights": weights, "bias": bias},
        "regression": {"family": "linear_regression_sgd", "weights": ret_weights, "bias": ret_bias},
        "thresholds": {
            "minimum_probability_success": 0.51,
            "minimum_expected_net_return_bps": 10.0,
            "maximum_uncertainty": 0.49,
        },
        "metrics": metrics,
        "live_eligible": live_eligible,
        "reason_codes": list(reason_codes),
        "label_definition": "label=1 when forward_net_return_bps > 20 after costs",
    }


def _dataset_metrics(rows: list[dict[str, Any]]) -> dict[str, float]:
    labels = [int(row.get("label", 0)) for row in rows]
    returns = [float(row.get("forward_net_return_bps", 0.0)) for row in rows]
    return {
        "example_count": float(len(rows)),
        "positive_labels": float(sum(labels)),
        "negative_labels": float(len(labels) - sum(labels)),
        "max_forward_net_return_bps": max(returns) if returns else 0.0,
        "min_forward_net_return_bps": min(returns) if returns else 0.0,
        "avg_forward_net_return_bps": sum(returns) / len(returns) if returns else 0.0,
    }


def _fit_logistic(x: list[list[float]], y: list[int]) -> tuple[list[float], float]:
    weights = [0.0] * len(x[0])
    bias = 0.0
    lr = 0.08
    for _ in range(250):
        for row, label in zip(x, y, strict=True):
            pred = _sigmoid(_dot(row, weights) + bias)
            err = pred - label
            weights = [w - lr * err * value for w, value in zip(weights, row, strict=True)]
            bias -= lr * err
    return weights, bias


def _fit_linear(x: list[list[float]], y: list[float]) -> tuple[list[float], float]:
    weights = [0.0] * len(x[0])
    bias = 0.0
    lr = 0.01
    for _ in range(180):
        for row, target in zip(x, y, strict=True):
            pred = _dot(row, weights) + bias
            err = pred - target
            weights = [w - lr * err * value for w, value in zip(weights, row, strict=True)]
            bias -= lr * err
    return weights, bias


def _standardize(x: list[list[float]]) -> tuple[list[float], list[float], list[list[float]]]:
    cols = list(zip(*x, strict=True))
    means = [sum(col) / len(col) for col in cols]
    scales = [max(1e-9, (sum((v - m) ** 2 for v in col) / len(col)) ** 0.5) for col, m in zip(cols, means, strict=True)]
    scaled = [[(value - means[i]) / scales[i] for i, value in enumerate(row)] for row in x]
    return means, scales, scaled


def _unscale_weights(weights: list[float], means: list[float], scales: list[float]) -> list[float]:
    del means
    return [w / scale for w, scale in zip(weights, scales, strict=True)]


def _unscale_bias(weights: list[float], bias: float, means: list[float], scales: list[float]) -> float:
    return bias - sum(w * mean / scale for w, mean, scale in zip(weights, means, scales, strict=True))


def _sigmoid(value: float) -> float:
    return 1.0 / (1.0 + math.exp(-max(-60.0, min(60.0, value))))


def _dot(row: list[float] | tuple[float, ...], weights: list[float] | tuple[float, ...]) -> float:
    return sum(value * weight for value, weight in zip(row, weights, strict=True))


def _precision_at_k(labels: list[int], scores: list[float], k: int) -> float:
    top = sorted(zip(scores, labels, strict=True), reverse=True)[:k]
    return sum(label for _, label in top) / max(1, len(top))


def _avg_return_top(returns: list[float], scores: list[float], k: int) -> float:
    top = sorted(zip(scores, returns, strict=True), reverse=True)[:k]
    return sum(value for _, value in top) / max(1, len(top))
