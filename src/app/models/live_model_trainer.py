from __future__ import annotations

import math
import os
from collections import defaultdict
from datetime import datetime, timedelta, timezone
from typing import Any

from app.features.feature_schema import LIVE_SHORT_HORIZON_SCHEMA
from app.models.model_artifact_registry import ModelArtifactRegistry
from app.models.model_validation import auc_like_score, validate_training_dataset


def _env_float(name: str, default: float) -> float:
    try:
        return float(os.getenv(name, str(default)))
    except (TypeError, ValueError):
        return default


def _env_int(name: str, default: int) -> int:
    try:
        return int(os.getenv(name, str(default)))
    except (TypeError, ValueError):
        return default


def _row_time(row: dict[str, Any]) -> datetime | None:
    value = str(row.get("as_of") or "")
    if not value:
        return None
    try:
        return datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError:
        return None


def train_live_short_horizon_model(
    rows: list[dict[str, Any]],
    *,
    registry: ModelArtifactRegistry | None = None,
    minimum_examples: int = 30,
    minimum_positive_labels: int = 5,
    minimum_negative_labels: int = 5,
    force_live_ineligible_reason: str | None = None,
    warm_start_artifact: dict[str, Any] | None = None,
    update_rows: list[dict[str, Any]] | None = None,
    training_state: dict[str, Any] | None = None,
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
        artifact = _artifact_payload(
            feature_names,
            [0.0] * len(feature_names),
            0.0,
            [0.0] * len(feature_names),
            0.0,
            metrics,
            False,
            reasons,
            training_state=training_state,
        )
        registry.save(artifact)
        return artifact
    # 시간 순서로 정렬해 "미래" 구간을 홀드아웃으로 떼어 일반화 성능을 정직하게 평가한다.
    # (이전 구현은 학습=평가(in-sample)라 지표가 낙관적으로 왜곡됐다.)
    rows = sorted(rows, key=lambda row: str(row.get("as_of") or ""))
    x = [[float(row["features"][name]) for name in feature_names] for row in rows]
    y = [int(row["label"]) for row in rows]
    # Winsorize before the fixed clip. A handful of extreme moves (a circuit-breaker
    # session produces several) otherwise dominate a squared-error regression and
    # drag the expected-net-return head with them; the fixed +/-500bps clip alone
    # does not help when the outliers sit inside it. Percentile bounds are computed
    # from the data, so a genuinely wide distribution is not artificially narrowed.
    returns = [
        _clip_return(value)
        for value in _winsorize(
            [float(row.get("forward_net_return_bps", 0.0)) for row in rows]
        )
    ]
    incremental = bool(warm_start_artifact and update_rows is not None)
    update_keys = {
        _observation_key(row)
        for row in (update_rows or ())
    }
    holdout_fraction = min(0.5, max(0.1, _env_float("LIVE_MODEL_HOLDOUT_FRACTION", 0.3)))
    horizon_s = _env_float("LIVE_LABEL_HORIZON_SECONDS", 600.0)
    embargo_s = _env_float("LIVE_MODEL_EMBARGO_SECONDS", 60.0)
    train_idx, val_idx, validation_symbol_count = _symbol_temporal_holdout(
        rows,
        holdout_fraction=holdout_fraction,
        horizon_seconds=horizon_s,
        embargo_seconds=embargo_s,
    )
    y_val = [y[index] for index in val_idx]
    y_train_purged = [y[index] for index in train_idx]
    holdout_evaluated = (
        bool(train_idx)
        and bool(val_idx)
        and sum(y_train_purged) >= 1
        and (len(y_train_purged) - sum(y_train_purged)) >= 1
        and sum(y_val) >= 1
        and (len(y_val) - sum(y_val)) >= 1
    )
    if holdout_evaluated:
        # Fit preprocessing on the training slice only. Using validation means/scales
        # leaks future regime information even when labels themselves are purged.
        train_means, train_scales, x_train_scaled = _standardize(
            [x[index] for index in train_idx]
        )
        x_val_scaled = _apply_standardization(
            [x[index] for index in val_idx],
            train_means,
            train_scales,
        )
        if incremental:
            initial_h = _warm_start_scaled_parameters(
                warm_start_artifact,
                "classification",
                feature_names,
                train_means,
                train_scales,
            )
            update_train_indices = [
                local_index
                for local_index, row_index in enumerate(train_idx)
                if _observation_key(rows[row_index]) in update_keys
            ]
            weights_h, bias_h = _fit_logistic(
                [x_train_scaled[index] for index in update_train_indices],
                [y_train_purged[index] for index in update_train_indices],
                initial_weights=initial_h[0],
                initial_bias=initial_h[1],
                epochs=_incremental_classification_epochs(),
            )
        else:
            weights_h, bias_h = _fit_logistic(x_train_scaled, y_train_purged)
        eval_labels = y_val
        eval_returns = [returns[index] for index in val_idx]
        eval_probs = [
            _sigmoid(_dot(row, weights_h) + bias_h)
            for row in x_val_scaled
        ]
    else:
        # 소규모/단일클래스 데이터는 홀드아웃이 불가능하므로 전체로 평가(합성 테스트 경로).
        means, scales, x_scaled = _standardize(x)
        if incremental:
            initial_preview = _warm_start_scaled_parameters(
                warm_start_artifact,
                "classification",
                feature_names,
                means,
                scales,
            )
            update_indices = [
                index
                for index, row in enumerate(rows)
                if _observation_key(row) in update_keys
            ]
            weights_preview, bias_preview = _fit_logistic(
                [x_scaled[index] for index in update_indices],
                [y[index] for index in update_indices],
                initial_weights=initial_preview[0],
                initial_bias=initial_preview[1],
                epochs=_incremental_classification_epochs(),
            )
        else:
            weights_preview, bias_preview = _fit_logistic(x_scaled, y)
        eval_labels = y
        eval_returns = returns
        eval_probs = [
            _sigmoid(_dot(row, weights_preview) + bias_preview)
            for row in x_scaled
        ]
    # Deployment parameters use all available rows only after holdout metrics have
    # been computed. This does not affect the eligibility score above.
    means, scales, x_scaled = _standardize(x)
    if incremental:
        update_indices = [
            index
            for index, row in enumerate(rows)
            if _observation_key(row) in update_keys
        ]
        initial_classification = _warm_start_scaled_parameters(
            warm_start_artifact,
            "classification",
            feature_names,
            means,
            scales,
        )
        initial_regression = _warm_start_scaled_parameters(
            warm_start_artifact,
            "regression",
            feature_names,
            means,
            scales,
        )
        weights, bias = _fit_logistic(
            [x_scaled[index] for index in update_indices],
            [y[index] for index in update_indices],
            initial_weights=initial_classification[0],
            initial_bias=initial_classification[1],
            epochs=_incremental_classification_epochs(),
        )
        ret_weights, ret_bias = _fit_linear(
            [x_scaled[index] for index in update_indices],
            [returns[index] for index in update_indices],
            initial_weights=initial_regression[0],
            initial_bias=initial_regression[1],
            epochs=_incremental_regression_epochs(),
        )
    else:
        weights, bias = _fit_logistic(x_scaled, y)
        ret_weights, ret_bias = _fit_linear(x_scaled, returns)
    auc = auc_like_score(eval_labels, eval_probs)
    top_k = _top_k_count(len(eval_labels))
    precision_at_k = _precision_at_k(eval_labels, eval_probs, top_k)
    avg_return_top = _avg_return_top(eval_returns, eval_probs, top_k)
    min_auc = _env_float("LIVE_MODEL_MIN_AUC", 0.55)
    min_precision = _env_float("LIVE_MODEL_MIN_PRECISION_AT_K", 0.35)
    min_avg_return = _env_float("LIVE_MODEL_MIN_AVG_RETURN_BPS", 0.0)
    live_eligible = auc >= min_auc and precision_at_k >= min_precision and avg_return_top > min_avg_return
    reason_codes = () if live_eligible else ("METRICS_BELOW_LIVE_THRESHOLDS",)
    if force_live_ineligible_reason:
        live_eligible = False
        reason_codes = (*reason_codes, force_live_ineligible_reason)
    metrics = {
        "auc": auc,
        "precision_at_k": precision_at_k,
        "avg_forward_net_return_bps_top_k": avg_return_top,
        "top_k_count": float(top_k),
        "top_k_fraction": top_k / max(1.0, float(len(eval_labels))),
        "example_count": float(len(rows)),
        "validation_example_count": float(len(val_idx)) if holdout_evaluated else 0.0,
        "validation_symbol_count": float(validation_symbol_count) if holdout_evaluated else 0.0,
        "holdout_train_count": float(len(train_idx)) if holdout_evaluated else 0.0,
        "holdout_evaluated": 1.0 if holdout_evaluated else 0.0,
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
        training_state=training_state,
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
    *,
    training_state: dict[str, Any] | None = None,
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
        "label_definition": "label=1 when forward_net_return_bps > LIVE_LABEL_MIN_NET_RETURN_BPS after costs",
        "training_state": dict(training_state or {}),
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


def _observation_key(row: dict[str, Any]) -> tuple[str, str]:
    return (
        str(row.get("ticker") or ""),
        str(row.get("as_of") or ""),
    )


def _warm_start_scaled_parameters(
    artifact: dict[str, Any] | None,
    section_name: str,
    feature_names: tuple[str, ...],
    means: list[float],
    scales: list[float],
) -> tuple[list[float], float]:
    section = dict((artifact or {}).get(section_name) or {})
    artifact_names = tuple((artifact or {}).get("feature_names") or ())
    raw_weights = [float(value) for value in section.get("weights") or ()]
    if artifact_names != feature_names or len(raw_weights) != len(feature_names):
        raise ValueError("INCOMPATIBLE_WARM_START_FEATURES")
    raw_bias = float(section.get("bias") or 0.0)
    scaled_weights = [
        weight * scale
        for weight, scale in zip(raw_weights, scales, strict=True)
    ]
    scaled_bias = raw_bias + sum(
        weight * mean
        for weight, mean in zip(raw_weights, means, strict=True)
    )
    return scaled_weights, scaled_bias


def _incremental_classification_epochs() -> int:
    return max(1, _env_int("LIVE_MODEL_INCREMENTAL_LOGISTIC_EPOCHS", 25))


def _incremental_regression_epochs() -> int:
    return max(1, _env_int("LIVE_MODEL_INCREMENTAL_LINEAR_EPOCHS", 18))


def _fit_logistic(
    x: list[list[float]],
    y: list[int],
    *,
    initial_weights: list[float] | None = None,
    initial_bias: float = 0.0,
    epochs: int | None = None,
) -> tuple[list[float], float]:
    if not x:
        return list(initial_weights or ()), float(initial_bias)
    if len(x) >= 1_000:
        try:
            return _fit_logistic_vectorized(
                x,
                y,
                initial_weights=initial_weights,
                initial_bias=initial_bias,
                epochs=epochs,
            )
        except (ImportError, ValueError, FloatingPointError):
            pass
    # 클래스 가중치로 불균형(소수 positive)에 대응 — 없으면 모델이 전부 음성(prob≈0)으로 붕괴한다.
    weights = list(initial_weights) if initial_weights is not None else [0.0] * len(x[0])
    bias = float(initial_bias)
    lr = 0.08
    l2 = _env_float("LIVE_MODEL_L2", 0.001)
    positives = sum(1 for label in y if label == 1)
    negatives = len(y) - positives
    total = max(1, len(y))
    weight_pos = total / (2.0 * positives) if positives else 1.0
    weight_neg = total / (2.0 * negatives) if negatives else 1.0
    for _ in range(250 if epochs is None else max(1, int(epochs))):
        for row, label in zip(x, y, strict=True):
            pred = _sigmoid(_dot(row, weights) + bias)
            class_weight = weight_pos if label == 1 else weight_neg
            err = (pred - label) * class_weight
            weights = [w - lr * (err * value + l2 * w) for w, value in zip(weights, row, strict=True)]
            bias -= lr * err
    return weights, bias


def _fit_linear(
    x: list[list[float]],
    y: list[float],
    *,
    initial_weights: list[float] | None = None,
    initial_bias: float = 0.0,
    epochs: int | None = None,
) -> tuple[list[float], float]:
    if not x:
        return list(initial_weights or ()), float(initial_bias)
    if len(x) >= 1_000:
        try:
            return _fit_linear_vectorized(
                x,
                y,
                initial_weights=initial_weights,
                initial_bias=initial_bias,
                epochs=epochs,
            )
        except (ImportError, ValueError, FloatingPointError):
            pass
    weights = list(initial_weights) if initial_weights is not None else [0.0] * len(x[0])
    bias = float(initial_bias)
    lr = 0.01
    l2 = _env_float("LIVE_MODEL_L2", 0.001)
    for _ in range(180 if epochs is None else max(1, int(epochs))):
        for row, target in zip(x, y, strict=True):
            pred = _dot(row, weights) + bias
            err = max(-250.0, min(250.0, pred - target))
            weights = [w - lr * (err * value + l2 * w) for w, value in zip(weights, row, strict=True)]
            bias -= lr * err
    return weights, bias


def _fit_logistic_vectorized(
    x: list[list[float]],
    y: list[int],
    *,
    initial_weights: list[float] | None = None,
    initial_bias: float = 0.0,
    epochs: int | None = None,
) -> tuple[list[float], float]:
    import numpy as np

    matrix = np.asarray(x, dtype=np.float64)
    labels = np.asarray(y, dtype=np.float64)
    weights = (
        np.asarray(initial_weights, dtype=np.float64).copy()
        if initial_weights is not None
        else np.zeros(matrix.shape[1], dtype=np.float64)
    )
    bias = float(initial_bias)
    positives = max(1, int(labels.sum()))
    negatives = max(1, len(labels) - positives)
    class_weights = np.where(
        labels > 0.5,
        len(labels) / (2.0 * positives),
        len(labels) / (2.0 * negatives),
    )
    lr = _env_float("LIVE_MODEL_BATCH_LOGISTIC_LR", 0.08)
    l2 = _env_float("LIVE_MODEL_L2", 0.001)
    epoch_count = (
        max(20, _env_int("LIVE_MODEL_BATCH_LOGISTIC_EPOCHS", 250))
        if epochs is None
        else max(1, int(epochs))
    )
    for _ in range(epoch_count):
        logits = np.clip(matrix @ weights + bias, -60.0, 60.0)
        predictions = 1.0 / (1.0 + np.exp(-logits))
        errors = (predictions - labels) * class_weights
        weights -= lr * ((matrix.T @ errors) / len(labels) + l2 * weights)
        bias -= lr * float(errors.mean())
    if not np.isfinite(weights).all() or not math.isfinite(bias):
        raise FloatingPointError("non-finite logistic parameters")
    return weights.tolist(), bias


def _fit_linear_vectorized(
    x: list[list[float]],
    y: list[float],
    *,
    initial_weights: list[float] | None = None,
    initial_bias: float = 0.0,
    epochs: int | None = None,
) -> tuple[list[float], float]:
    import numpy as np

    matrix = np.asarray(x, dtype=np.float64)
    targets = np.asarray(y, dtype=np.float64)
    weights = (
        np.asarray(initial_weights, dtype=np.float64).copy()
        if initial_weights is not None
        else np.zeros(matrix.shape[1], dtype=np.float64)
    )
    bias = float(initial_bias)
    lr = _env_float("LIVE_MODEL_BATCH_LINEAR_LR", 0.01)
    l2 = _env_float("LIVE_MODEL_L2", 0.001)
    epoch_count = (
        max(20, _env_int("LIVE_MODEL_BATCH_LINEAR_EPOCHS", 180))
        if epochs is None
        else max(1, int(epochs))
    )
    for _ in range(epoch_count):
        errors = np.clip(matrix @ weights + bias - targets, -250.0, 250.0)
        weights -= lr * ((matrix.T @ errors) / len(targets) + l2 * weights)
        bias -= lr * float(errors.mean())
    if not np.isfinite(weights).all() or not math.isfinite(bias):
        raise FloatingPointError("non-finite regression parameters")
    return weights.tolist(), bias


def _clip_return(value: float) -> float:
    limit = abs(_env_float("LIVE_MODEL_RETURN_CLIP_BPS", 500.0))
    if not math.isfinite(value):
        return 0.0
    return max(-limit, min(limit, value))


def _winsorize(values: list[float]) -> list[float]:
    """Clamp the tails to data-derived percentiles (identity when disabled).

    Distribution-aware rather than a fixed band: with the default 2% tails, a calm
    session is barely touched while a repricing session's handful of extreme moves
    stop dictating the regression slope.
    """
    fraction = min(0.2, max(0.0, _env_float("LIVE_MODEL_WINSORIZE_FRACTION", 0.02)))
    finite = [value for value in values if math.isfinite(value)]
    if fraction <= 0.0 or len(finite) < 20:
        return [value if math.isfinite(value) else 0.0 for value in values]
    ordered = sorted(finite)
    index = max(0, min(len(ordered) - 1, int(len(ordered) * fraction)))
    low = ordered[index]
    high = ordered[len(ordered) - 1 - index]
    if low > high:
        low, high = high, low
    return [
        max(low, min(high, value)) if math.isfinite(value) else 0.0 for value in values
    ]


def _top_k_count(row_count: int) -> int:
    fraction = max(0.001, min(0.2, _env_float("LIVE_MODEL_TOP_K_FRACTION", 0.01)))
    minimum = max(1, _env_int("LIVE_MODEL_TOP_K_MIN", 25))
    maximum = max(minimum, _env_int("LIVE_MODEL_TOP_K_MAX", 300))
    return max(1, min(row_count, maximum, max(minimum, int(row_count * fraction))))


def _standardize(x: list[list[float]]) -> tuple[list[float], list[float], list[list[float]]]:
    cols = list(zip(*x, strict=True))
    means = [sum(col) / len(col) for col in cols]
    scales = [max(1e-9, (sum((v - m) ** 2 for v in col) / len(col)) ** 0.5) for col, m in zip(cols, means, strict=True)]
    scaled = [[(value - means[i]) / scales[i] for i, value in enumerate(row)] for row in x]
    return means, scales, scaled


def _apply_standardization(
    x: list[list[float]],
    means: list[float],
    scales: list[float],
) -> list[list[float]]:
    return [
        [
            (value - means[index]) / scales[index]
            for index, value in enumerate(row)
        ]
        for row in x
    ]


def _symbol_temporal_holdout(
    rows: list[dict[str, Any]],
    *,
    holdout_fraction: float,
    horizon_seconds: float,
    embargo_seconds: float,
) -> tuple[list[int], list[int], int]:
    """Build a purged chronological holdout inside every sufficiently sized symbol."""
    grouped: dict[str, list[int]] = defaultdict(list)
    for index, row in enumerate(rows):
        grouped[str(row.get("ticker") or "__unknown__")].append(index)
    train_indices: list[int] = []
    validation_indices: list[int] = []
    validation_symbols = 0
    for indices in grouped.values():
        if len(indices) < 10:
            train_indices.extend(indices)
            continue
        split = max(1, min(len(indices) - 1, int(len(indices) * (1.0 - holdout_fraction))))
        local_train = indices[:split]
        local_validation = indices[split:]
        validation_start = _row_time(rows[local_validation[0]])
        if validation_start is not None:
            purged = [
                index
                for index in local_train
                if (moment := _row_time(rows[index])) is not None
                and moment + timedelta(seconds=horizon_seconds + embargo_seconds)
                <= validation_start
            ]
            if purged:
                local_train = purged
        train_indices.extend(local_train)
        validation_indices.extend(local_validation)
        validation_symbols += 1
    if not validation_indices:
        split = max(1, min(len(rows) - 1, int(len(rows) * (1.0 - holdout_fraction))))
        return list(range(split)), list(range(split, len(rows))), 1
    return sorted(train_indices), sorted(validation_indices), validation_symbols


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
