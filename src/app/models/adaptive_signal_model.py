"""Bounded online fitting of a small, portable nonlinear signal model.

Only the two output heads learn. The fixed ReLU projection captures interactions
without a deep-network training job and lowers to static MatMul/Add/ReLU on NPU.
The chronological holdout remains untouched, including after model promotion.
"""
from __future__ import annotations

import math
import os
from datetime import datetime, timedelta, timezone
from typing import Any

import numpy as np

from app.features.feature_schema import LIVE_SHORT_HORIZON_SCHEMA
from app.models.model_validation import auc_like_score, validate_training_dataset


def _number(name: str, default: float) -> float:
    try:
        value = float(os.getenv(name, str(default)))
        return value if math.isfinite(value) else default
    except (TypeError, ValueError):
        return default


def _time(value: Any) -> datetime | None:
    try:
        result = datetime.fromisoformat(str(value).replace("Z", "+00:00"))
        return result.replace(tzinfo=timezone.utc) if result.tzinfo is None else result.astimezone(timezone.utc)
    except (TypeError, ValueError):
        return None


def _row_horizon(row: dict[str, Any], fallback: float) -> float | None:
    """The purge must cover the label's actual exit contract, not a generic 10m."""
    recorded = row.get("label_horizon_seconds")
    if recorded is not None:
        try:
            seconds = float(recorded)
            return seconds if math.isfinite(seconds) and 0.0 < seconds <= 7 * 86400 else None
        except (TypeError, ValueError):
            return None
    basis = str(row.get("label_basis") or "")
    if basis.startswith("trade_plan:"):
        # Historical rows without the plan's actual holding horizon cannot be
        # certified from a plan id alone. Newly collected rows carry the value.
        return None
    if basis.startswith("strategy_exit_geometry:"):
        from app.strategy.exit_geometry import exit_geometry, resolve_exit_geometry

        strategy = basis.split(":", 1)[1].split("@", 1)[0]
        geometry = exit_geometry(strategy)
        if "@" in basis:
            try:
                cost = float(basis.rsplit(":", 1)[1].removesuffix("bps"))
                geometry = resolve_exit_geometry(strategy, round_trip_cost_bps=cost)
            except (TypeError, ValueError):
                return None
        return max(float(geometry.max_holding_seconds), fallback)
    return fallback


def network_parameters(payload: dict[str, Any]) -> dict[str, np.ndarray]:
    return {key: np.asarray(payload[key], dtype=np.float32) for key in (
        "mean", "scale", "hidden_weights", "hidden_bias", "output_weights", "output_bias"
    )}


def network_logits(features: np.ndarray, payload: dict[str, Any]) -> np.ndarray:
    p = network_parameters(payload)
    x = np.clip((np.asarray(features, dtype=np.float32) - p["mean"]) / p["scale"], -6.0, 6.0)
    return np.maximum(x @ p["hidden_weights"] + p["hidden_bias"], 0.0) @ p["output_weights"] + p["output_bias"]


def train_adaptive_signal_model(
    rows: list[dict[str, Any]], *, registry: Any, minimum_examples: int,
    minimum_positive_labels: int, minimum_negative_labels: int,
    force_live_ineligible_reason: str | None = None,
    warm_start_artifact: dict[str, Any] | None = None,
    training_state: dict[str, Any] | None = None,
) -> dict[str, Any]:
    from app.models.live_model_trainer import _artifact_payload, _top_k_count
    from app.models.live_signal_predictor import _prediction_thresholds

    names = LIVE_SHORT_HORIZON_SCHEMA.feature_names
    now = datetime.now(timezone.utc)
    horizon = min(7 * 86400.0, max(60.0, _number("LIVE_LABEL_HORIZON_SECONDS", 600.0)))
    embargo = min(7 * 86400.0, max(0.0, _number("LIVE_MODEL_EMBARGO_SECONDS", 60.0)))
    limit = max(256, min(32768, int(_number("LIVE_MODEL_REPLAY_MAX_ROWS", 8192))))
    valid = []
    rejected = 0
    for row in rows:
        moment = _time(row.get("as_of"))
        row_horizon = _row_horizon(row, horizon)
        try:
            vector = [float(row["features"][name]) for name in names]
            actual_net = float(row.get("raw_forward_net_return_bps", row["forward_net_return_bps"]))
            valid_row = (
                moment is not None and row_horizon is not None
                and moment + timedelta(seconds=row_horizon) <= now
                and row["label"] in (0, 1)
                and all(math.isfinite(value) and abs(value) <= np.finfo(np.float32).max for value in vector)
                and math.isfinite(actual_net) and abs(actual_net) <= np.finfo(np.float32).max
            )
        except (KeyError, TypeError, ValueError, OverflowError):
            valid_row = False
        if valid_row:
            # Alpha against an index is not cash P&L. A -30bps trade can have
            # +70bps alpha in a falling market and still lose deposited capital.
            normalized = dict(row, forward_net_return_bps=actual_net, label_horizon_seconds=row_horizon)
            if "raw_forward_net_return_bps" in row:
                normalized["label"] = int(actual_net > _number("LIVE_LABEL_MIN_NET_RETURN_BPS", 5.0))
            valid.append((moment, normalized))
        else:
            rejected += 1
    valid.sort(key=lambda item: item[0])
    # One observation per symbol/time, so replay duplicates cannot inflate evidence.
    unique = {(str(row.get("ticker") or ""), moment): (moment, row) for moment, row in valid}
    valid = sorted(unique.values(), key=lambda item: item[0])[-limit:]
    clean = [row for _, row in valid]
    ok, reasons = validate_training_dataset(
        clean, minimum_examples=minimum_examples,
        minimum_positive_labels=minimum_positive_labels,
        minimum_negative_labels=minimum_negative_labels,
    )
    metrics: dict[str, float] = {
        "example_count": float(len(clean)), "positive_labels": float(sum(row["label"] for row in clean)),
        "negative_labels": float(sum(1 - row["label"] for row in clean)),
        "rejected_rows": float(rejected), "holdout_evaluated": 0.0,
        "validation_example_count": 0.0, "runtime_policy_aligned_evaluation": 1.0,
    }
    reasons = list(reasons)
    nonlinear: dict[str, Any] = {}
    state = dict(training_state or {})
    if ok:
        fraction = min(0.5, max(0.1, _number("LIVE_MODEL_HOLDOUT_FRACTION", 0.3)))
        split = min(len(valid) - 1, max(1, int(len(valid) * (1.0 - fraction))))
        validation_start = valid[split][0]
        cutoff = validation_start - timedelta(seconds=horizon + embargo)
        train = [(moment, row) for moment, row in valid
                 if moment + timedelta(seconds=row["label_horizon_seconds"] + embargo) < validation_start]
        validation = [(moment, row) for moment, row in valid if moment >= validation_start]
        if (len(train) < minimum_examples or not validation
                or len({row["label"] for _, row in train}) < 2
                or len({row["label"] for _, row in validation}) < 2):
            reasons.append("INSUFFICIENT_PURGED_HOLDOUT")
        else:
            x = np.asarray([[row["features"][name] for name in names] for _, row in train], dtype=np.float64)
            labels = np.asarray([row["label"] for _, row in train], dtype=np.float64)
            # Holdout returns stay raw; outlier clipping is a training operation only.
            returns = np.asarray([row["forward_net_return_bps"] for _, row in train], dtype=np.float64)
            parent = (warm_start_artifact or {}).get("nonlinear") or {}
            parent_until = _time(parent.get("trained_through"))
            parent_label_end = _time(parent.get("training_label_end"))
            warm = bool(
                parent and parent.get("family") == "fixed_relu_two_head_v1"
                and (warm_start_artifact or {}).get("feature_schema_hash") == LIVE_SHORT_HORIZON_SCHEMA.schema_hash
                and parent_until is not None and parent_until < cutoff
                and parent_label_end is not None
                and parent_label_end + timedelta(seconds=embargo) < validation_start
            )
            if warm:
                p = {key: value.astype(np.float64) for key, value in network_parameters(parent).items()}
            else:
                hidden_dim = max(16, min(128, int(_number("LIVE_MODEL_HIDDEN_UNITS", 64))))
                rng = np.random.default_rng(731)
                active_dim = max(1, int(np.count_nonzero(x.std(axis=0) > 1e-6)))
                p = {
                    "mean": x.mean(axis=0),
                    "scale": np.maximum(x.std(axis=0), 1e-6),
                    "hidden_weights": rng.normal(0.0, 1.0 / math.sqrt(active_dim), (len(names), hidden_dim)),
                    "hidden_bias": rng.uniform(-0.5, 0.5, hidden_dim),
                    "output_weights": np.zeros((hidden_dim, 2)),
                    "output_bias": np.zeros(2),
                }
            hidden = np.maximum(np.clip((x - p["mean"]) / p["scale"], -6.0, 6.0) @ p["hidden_weights"] + p["hidden_bias"], 0.0)
            # Replay stays bounded; incremental fitting blends new evidence with the
            # newest history rather than forgetting a regime after one microbatch.
            replay = min(len(hidden), 1024 if warm else 4096)
            hidden, labels, returns = hidden[-replay:], labels[-replay:], returns[-replay:]
            steps = max(1, min(128, int(_number("LIVE_MODEL_ADAPTIVE_STEPS", 16 if warm else 96))))
            w = p["output_weights"][:, 0].copy()
            bias = float(p["output_bias"][0])
            l2 = min(1.0, max(0.001, _number("LIVE_MODEL_L2", 0.01)))
            # Unweighted log loss preserves the observed class prior. Balancing
            # classes changes probabilities and overstates success in a weak tape.
            for _ in range(steps):
                probability = 1.0 / (1.0 + np.exp(-np.clip(hidden @ w + bias, -60.0, 60.0)))
                error = probability - labels
                w -= 0.15 * (hidden.T @ error / len(labels) + l2 * w)
                bias -= 0.15 * float(error.mean())
            p["output_weights"][:, 0], p["output_bias"][0] = w, bias
            design = np.column_stack((hidden, np.ones(len(hidden))))
            penalty = np.eye(design.shape[1]) * max(1.0, l2 * len(design))
            penalty[-1, -1] = 0.001
            targets = np.clip(returns, -500.0, 500.0)
            solution = np.linalg.solve(design.T @ design + penalty, design.T @ targets)
            p["output_weights"][:, 1], p["output_bias"][1] = solution[:-1], solution[-1]
            nonlinear = {key: value.astype(np.float32).tolist() for key, value in p.items()}
            nonlinear.update({"family": "fixed_relu_two_head_v1", "trained_through": train[-1][0].isoformat(),
                "training_label_end": max(moment + timedelta(seconds=row["label_horizon_seconds"]) for moment, row in train).isoformat(),
                "validation_start": validation_start.isoformat()})
            val_x = np.asarray([[row["features"][name] for name in names] for _, row in validation], dtype=np.float32)
            logits = network_logits(val_x, nonlinear)
            probs = 1.0 / (1.0 + np.exp(-np.clip(logits[:, 0], -60.0, 60.0)))
            expected = logits[:, 1]
            val_labels = [row["label"] for _, row in validation]
            val_returns = np.asarray([row["forward_net_return_bps"] for _, row in validation], dtype=float)
            thresholds = _prediction_thresholds({"minimum_probability_success": 0.51, "minimum_expected_net_return_bps": 10.0, "maximum_uncertainty": 0.49})
            uncertainty = 1.0 - np.abs(probs - 0.5) * 2.0
            deployable = np.flatnonzero((probs >= thresholds["minimum_probability_success"]) & (expected >= thresholds["minimum_expected_net_return_bps"]) & (uncertainty <= thresholds["maximum_uncertainty"]))
            selected = sorted(deployable, key=lambda index: (expected[index], probs[index]), reverse=True)[:_top_k_count(len(validation))]
            net = val_returns[selected]
            avg = float(net.mean()) if len(net) else 0.0
            lower_bound = avg - 1.645 * float(net.std(ddof=1) / math.sqrt(len(net))) if len(net) > 1 else -math.inf
            auc = auc_like_score(val_labels, probs.tolist())
            precision = float(np.asarray(val_labels)[selected].mean()) if selected else 0.0
            metrics.update({
                "auc": auc, "precision_at_k": precision, "avg_forward_net_return_bps_top_k": avg,
                "top_k_count": float(len(selected)), "top_k_target_count": float(_top_k_count(len(validation))),
                "deployable_holdout_count": float(len(deployable)), "validation_example_count": float(len(validation)),
                "validation_symbol_count": float(len({row.get("ticker") for _, row in validation})),
                "holdout_train_count": float(len(train)), "holdout_evaluated": 1.0,
                "net_return_lower_bound_bps": lower_bound if math.isfinite(lower_bound) else -1e9,
                "brier_score": float(np.mean((probs - np.asarray(val_labels)) ** 2)),
                "training_steps": float(steps), "training_replay_count": float(replay),
                "parameter_count": float(sum(value.size for value in p.values())),
            })
            state.update({"mode": "incremental" if warm else "full", "model_family": "fixed_relu_two_head_v1",
                "purge_seconds": max(row["label_horizon_seconds"] for _, row in train) + embargo,
                "return_target": "actual_net_after_costs"})
            if (auc < _number("LIVE_MODEL_MIN_AUC", 0.55) or precision < _number("LIVE_MODEL_MIN_PRECISION_AT_K", 0.35)
                    or avg <= max(0.0, _number("LIVE_MODEL_MIN_AVG_RETURN_BPS", 0.0))):
                reasons.append("METRICS_BELOW_LIVE_THRESHOLDS")
            if lower_bound <= 0.0:
                reasons.append("NET_EDGE_NOT_STATISTICALLY_POSITIVE")
    if force_live_ineligible_reason:
        reasons.append(force_live_ineligible_reason)
    artifact = _artifact_payload(names, [0.0] * len(names), 0.0, [0.0] * len(names), 0.0,
        metrics, bool(nonlinear) and not reasons, tuple(reasons), training_state=state)
    if nonlinear:
        artifact["nonlinear"] = nonlinear
        artifact["classification"]["family"] = "fixed_relu_two_head_v1"
        artifact["regression"]["family"] = "fixed_relu_two_head_v1"
    registry.save(artifact)
    return artifact
