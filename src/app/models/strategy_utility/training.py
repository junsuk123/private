from __future__ import annotations

import json
from collections import defaultdict
from dataclasses import asdict
from pathlib import Path
from typing import Iterable

import numpy as np

from app.evaluation.stored_counterfactual import CounterfactualLabel
from app.models.strategy_utility.rgcn import (
    FixedShapeStrategyUtilityModel,
    StrategyUtilityModelConfig,
)
from app.routing.shadow_intelligence import STRATEGY_IDS


def train_counterfactual_checkpoint(
    labels: Iterable[CounterfactualLabel],
    output: str | Path,
    *,
    ridge: float = 1.0,
    input_feature_schema: str = "counterfactual_quantiles_v1",
    authorize_live_shadow: bool = False,
) -> dict[str, object]:
    """Calibrate strategy heads on causal stored counterfactual labels.

    The graph encoder remains deterministic; strategy and no-trade heads are
    ridge-fitted. This is a shadow checkpoint, never an authorization to trade.
    """
    rows = tuple(row for row in labels if len(row.features) == 12)
    if not rows:
        raise ValueError("no counterfactual rows with 12 causal features")
    config = StrategyUtilityModelConfig(
        batch_size=1,
        time_steps=1,
        max_nodes=1,
        feature_dim=12,
        relation_count=1,
        strategy_count=len(STRATEGY_IDS),
        hidden_dim=16,
        seed=17,
    )
    model = FixedShapeStrategyUtilityModel(config)
    features = np.asarray([row.features for row in rows], dtype=np.float32)
    encoder = model.relation_weights[0] + model.self_weight
    hidden = np.maximum(features @ encoder, 0.0)
    strategy_index = {name: index for index, name in enumerate(STRATEGY_IDS)}
    fitted: dict[str, int] = {}
    for strategy_id, index in strategy_index.items():
        selected = [position for position, row in enumerate(rows) if row.strategy_id == strategy_id]
        if not selected:
            continue
        h = hidden[selected]
        targets = np.asarray([_raw_target(rows[position]) for position in selected], dtype=np.float32)
        model.strategy_heads[index] = _ridge_fit(h, targets, ridge)
        fitted[strategy_id] = len(selected)

    grouped: dict[tuple[str, object], list[int]] = defaultdict(list)
    for position, row in enumerate(rows):
        grouped[(row.symbol, row.as_of)].append(position)
    group_positions = [positions[0] for positions in grouped.values()]
    no_trade_targets = []
    for positions in grouped.values():
        attractive = any(
            rows[position].triggered
            and rows[position].filled
            and rows[position].net_return_bps > 0
            for position in positions
        )
        no_trade_targets.append(-2.0 if attractive else 2.0)
    model.no_trade_head = _ridge_fit(
        hidden[group_positions],
        np.asarray(no_trade_targets, dtype=np.float32)[:, None],
        ridge,
    )[:, 0]
    checkpoint = model.save_checkpoint(output)
    minimum_rows = 10_000
    minimum_snapshots = 1_000
    strategy_coverage = all(fitted.get(strategy_id, 0) >= 500 for strategy_id in STRATEGY_IDS)
    live_shadow_authorized = bool(
        authorize_live_shadow
        and input_feature_schema == "realtime_microstructure_v1"
        and len(rows) >= minimum_rows
        and len(grouped) >= minimum_snapshots
        and strategy_coverage
    )
    report = {
        "checkpoint": str(checkpoint),
        "rows": len(rows),
        "snapshots": len(grouped),
        "strategies": fitted,
        "method": "causal_feature_encoder_plus_ridge_calibrated_heads",
        "input_feature_schema": input_feature_schema,
        "feature_provenance": (
            "causal_minute_bar_microstructure_proxy_v1"
            if input_feature_schema == "realtime_microstructure_v1"
            else "causal_counterfactual_quantiles_v1"
        ),
        "live_authorized": live_shadow_authorized,
        "authorization_scope": "shadow_inference_only",
        "authorization_checks": {
            "requested": bool(authorize_live_shadow),
            "minimum_rows": minimum_rows,
            "minimum_snapshots": minimum_snapshots,
            "strategy_minimum_rows": 500,
            "row_count_ok": len(rows) >= minimum_rows,
            "snapshot_count_ok": len(grouped) >= minimum_snapshots,
            "strategy_coverage_ok": strategy_coverage,
            "schema_matches_runtime": input_feature_schema == "realtime_microstructure_v1",
        },
        "config": asdict(config),
    }
    report_path = checkpoint.with_suffix(".json")
    report_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return report


def _ridge_fit(x: np.ndarray, y: np.ndarray, ridge: float) -> np.ndarray:
    regularizer = np.eye(x.shape[1], dtype=np.float32) * max(1e-6, float(ridge))
    return np.linalg.solve(x.T @ x + regularizer, x.T @ y).astype(np.float32)


def _raw_target(row: CounterfactualLabel) -> tuple[float, ...]:
    positive = row.filled and row.net_return_bps > 0
    gross = row.net_return_bps + row.cost_bps
    return (
        2.0 if positive else -2.0,
        float(np.clip(gross / 25.0, -5.0, 5.0)),
        _inverse_softplus(max(0.01, row.cost_bps / 10.0)),
        _inverse_softplus(max(0.01, max(-row.net_return_bps, 0.0) / 15.0)),
        _inverse_softplus(max(0.01, max(row.net_return_bps, 0.0) / 20.0)),
        2.0 if row.filled else -2.0,
        _inverse_softplus(15.0),
        _inverse_softplus(max(0.01, min(10.0, abs(row.net_return_bps) / 25.0))),
    )


def _inverse_softplus(value: float) -> float:
    value = max(1e-6, min(30.0, float(value)))
    return float(np.log(np.expm1(value)))
