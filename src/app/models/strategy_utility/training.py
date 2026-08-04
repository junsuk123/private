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
from app.models.strategy_utility.strategy_graph import (
    RELATION_NAMES,
    STRATEGY_NODE_COUNT,
    strategy_relation_adjacency,
)
from app.routing.shadow_intelligence import (
    COMPATIBILITY_UNAVAILABLE_REASONS,
    STRATEGY_IDS,
)


def train_counterfactual_checkpoint(
    labels: Iterable[CounterfactualLabel],
    output: str | Path,
    *,
    ridge: float = 1.0,
    input_feature_schema: str = "counterfactual_quantiles_v2_session_structure",
    authorize_live_shadow: bool = False,
) -> dict[str, object]:
    """Calibrate strategy heads on causal stored counterfactual labels.

    The graph encoder remains deterministic; strategy and no-trade heads are
    ridge-fitted. This is a shadow checkpoint, never an authorization to trade.
    """
    graph_schema = input_feature_schema in {
        "realtime_strategy_graph_v3",
        "realtime_strategy_graph_v4_market",
    }
    context_feature_dim = (
        28
        if input_feature_schema == "realtime_strategy_graph_v4_market"
        else (
            27
            if input_feature_schema
            in {"realtime_strategy_context_v2", "realtime_strategy_graph_v3"}
            else 12
        )
    )
    rows = tuple(row for row in labels if len(row.features) == context_feature_dim)
    if not rows:
        raise ValueError(
            f"no counterfactual rows with {context_feature_dim} causal features"
        )
    config = StrategyUtilityModelConfig(
        batch_size=1,
        time_steps=1,
        max_nodes=STRATEGY_NODE_COUNT if graph_schema else 1,
        feature_dim=(
            context_feature_dim + STRATEGY_NODE_COUNT
            if graph_schema
            else context_feature_dim
        ),
        relation_count=len(RELATION_NAMES) if graph_schema else 1,
        strategy_count=len(STRATEGY_IDS),
        hidden_dim=16,
        seed=17,
    )
    strategy_index = {name: index for index, name in enumerate(STRATEGY_IDS)}
    grouped: dict[tuple[str, object], list[int]] = defaultdict(list)
    for position, row in enumerate(rows):
        grouped[(row.symbol, row.as_of)].append(position)
    if graph_schema:
        model, fitted, validation_metrics = _fit_strategy_relation_graph(
            rows,
            grouped,
            config,
            ridge=ridge,
        )
        method = "ontology_strategy_graph_rgcn_joint_gradient_calibration"
    else:
        model = FixedShapeStrategyUtilityModel(config)
        features = np.asarray([row.features for row in rows], dtype=np.float32)
        encoder = model.relation_weights[0] + model.self_weight
        hidden = np.maximum(features @ encoder, 0.0)
        fitted = {}
        for strategy_id, index in strategy_index.items():
            selected = [
                position
                for position, row in enumerate(rows)
                if row.strategy_id == strategy_id
            ]
            if not selected:
                continue
            h = hidden[selected]
            targets = np.asarray(
                [_raw_target(rows[position]) for position in selected],
                dtype=np.float32,
            )
            masks = np.asarray(
                [_target_mask(rows[position]) for position in selected],
                dtype=np.float32,
            )
            model.strategy_heads[index] = _masked_ridge_fit(
                h,
                targets,
                masks,
                ridge,
            )
            fitted[strategy_id] = len(selected)
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
        validation_metrics = {}
        method = "causal_feature_encoder_plus_ridge_calibrated_heads"
    checkpoint = model.save_checkpoint(output)
    minimum_rows = 10_000
    minimum_snapshots = 1_000
    strategy_coverage = all(fitted.get(strategy_id, 0) >= 500 for strategy_id in STRATEGY_IDS)
    live_shadow_authorized = bool(
        authorize_live_shadow
        and input_feature_schema == "realtime_strategy_graph_v4_market"
        and len(rows) >= minimum_rows
        and len(grouped) >= minimum_snapshots
        and strategy_coverage
    )
    report = {
        "checkpoint": str(checkpoint),
        "rows": len(rows),
        "snapshots": len(grouped),
        "strategies": fitted,
        "label_outcomes": _label_outcome_summary(rows),
        "label_outcomes_by_market": {
            "KRX": _label_outcome_summary(
                tuple(
                    row
                    for row in rows
                    if row.symbol.isdigit() and len(row.symbol) == 6
                )
            ),
            "US": _label_outcome_summary(
                tuple(
                    row
                    for row in rows
                    if not (row.symbol.isdigit() and len(row.symbol) == 6)
                )
            ),
        },
        "method": method,
        "strategy_ids": list(STRATEGY_IDS),
        "strategy_count": len(STRATEGY_IDS),
        "feature_names": [
            *[
                f"causal_context_feature_{index}"
                for index in range(context_feature_dim)
            ],
            *(
                [f"strategy_identity:{strategy_id}" for strategy_id in STRATEGY_IDS]
                if graph_schema
                else []
            ),
        ],
        "relation_names": list(RELATION_NAMES) if graph_schema else ["context"],
        "training_data_range": {
            "start": min((row.as_of.isoformat() for row in rows), default=None),
            "end": max((row.label_end.isoformat() for row in rows), default=None),
        },
        "training_method": method,
        "validation_metrics": validation_metrics,
        "checkpoint_hash": _checkpoint_hash(checkpoint),
        "input_feature_schema": input_feature_schema,
        "feature_provenance": (
            "causal_minute_bar_microstructure_proxy_v1"
            if input_feature_schema == "realtime_strategy_graph_v4_market"
            else "causal_counterfactual_quantiles_v2_session_structure"
        ),
        "live_authorized": live_shadow_authorized,
        "authorization_scope": "ontology_gnn_realtime_trust_gated_execution",
        "authorization_checks": {
            "requested": bool(authorize_live_shadow),
            "minimum_rows": minimum_rows,
            "minimum_snapshots": minimum_snapshots,
            "strategy_minimum_rows": 500,
            "row_count_ok": len(rows) >= minimum_rows,
            "snapshot_count_ok": len(grouped) >= minimum_snapshots,
            "strategy_coverage_ok": strategy_coverage,
            "schema_matches_runtime": (
                input_feature_schema == "realtime_strategy_graph_v4_market"
            ),
        },
        "config": asdict(config),
    }
    report_path = checkpoint.with_suffix(".json")
    report_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return report


def _label_outcome_summary(
    rows: tuple[CounterfactualLabel, ...],
) -> dict[str, dict[str, float | int | str | None]]:
    summary: dict[str, dict[str, float | int | str | None]] = {}
    for strategy_id in STRATEGY_IDS:
        selected = [row for row in rows if row.strategy_id == strategy_id]
        simulated_filled = [row for row in selected if row.filled]
        filled = [row for row in selected if row.triggered and row.filled]
        positive = [row for row in filled if row.net_return_bps > 0]
        mean_net = (
            float(np.mean([row.net_return_bps for row in filled]))
            if filled
            else None
        )
        mean_cost = (
            float(np.mean([row.all_in_cost_bps for row in filled]))
            if filled
            else None
        )
        mean_gross = (
            float(
                np.mean(
                    [row.net_return_bps + row.all_in_cost_bps for row in filled]
                )
            )
            if filled
            else None
        )
        if not selected or not any(row.triggered for row in selected):
            # "No samples" has two very different causes and they demand
            # opposite responses: a strategy whose conditions merely never
            # occurred is waiting for a market, while a strategy the routing
            # layer cannot even instantiate is waiting for an ENGINEER. Eight of
            # sixteen ids were in the second group with nothing saying so, and
            # reading that as weak performance would retire strategies that were
            # never evaluated.
            unavailable = COMPATIBILITY_UNAVAILABLE_REASONS.get(strategy_id)
            diagnosis = (
                f"STRUCTURALLY_UNREACHABLE:{unavailable}"
                if unavailable
                else "NO_TRIGGERED_SAMPLES"
            )
        elif len(filled) < 20:
            diagnosis = "INSUFFICIENT_FILLED_SAMPLES"
        elif mean_gross is not None and mean_gross <= 0:
            diagnosis = "GROSS_EDGE_NON_POSITIVE"
        elif mean_net is not None and mean_net <= 0:
            diagnosis = "EXECUTION_COST_EXCEEDS_GROSS_EDGE"
        else:
            diagnosis = "POSITIVE_NET_EDGE_OBSERVED"
        summary[strategy_id] = {
            "labels": len(selected),
            "triggered": sum(row.triggered for row in selected),
            "trigger_rate": (
                sum(row.triggered for row in selected) / len(selected)
                if selected
                else None
            ),
            "simulated_filled": len(simulated_filled),
            "filled": len(filled),
            "positive_net": len(positive),
            "positive_net_rate_when_filled": (
                len(positive) / len(filled) if filled else None
            ),
            "mean_gross_return_bps_when_filled": mean_gross,
            "mean_net_return_bps_when_filled": mean_net,
            "mean_cost_bps_when_filled": mean_cost,
            "performance_diagnosis": diagnosis,
        }
    return summary


def _fit_strategy_relation_graph(
    rows: tuple[CounterfactualLabel, ...],
    grouped: dict[tuple[str, object], list[int]],
    config: StrategyUtilityModelConfig,
    *,
    ridge: float,
) -> tuple[
    FixedShapeStrategyUtilityModel,
    dict[str, int],
    dict[str, float | int],
]:
    strategy_index = {name: index for index, name in enumerate(STRATEGY_IDS)}
    snapshots: list[
        tuple[object, np.ndarray, np.ndarray, np.ndarray, bool]
    ] = []
    fitted = {strategy_id: 0 for strategy_id in STRATEGY_IDS}
    for positions in grouped.values():
        by_strategy = {rows[position].strategy_id: rows[position] for position in positions}
        if any(strategy_id not in by_strategy for strategy_id in STRATEGY_IDS):
            continue
        ordered = [by_strategy[strategy_id] for strategy_id in STRATEGY_IDS]
        context = np.asarray(ordered[0].features, dtype=np.float32)
        targets = np.asarray([_raw_target(row) for row in ordered], dtype=np.float32)
        target_masks = np.asarray(
            [_target_mask(row) for row in ordered],
            dtype=np.float32,
        )
        attractive = any(
            row.triggered and row.filled and row.net_return_bps > 0
            for row in ordered
        )
        snapshots.append(
            (ordered[0].as_of, context, targets, target_masks, attractive)
        )
        for strategy_id in STRATEGY_IDS:
            fitted[strategy_id] += 1
    if len(snapshots) < 2:
        raise ValueError("strategy relation graph requires complete multi-strategy snapshots")
    snapshots.sort(key=lambda item: item[0])
    split = max(1, min(len(snapshots) - 1, int(len(snapshots) * 0.8)))
    train = snapshots[:split]
    validation = snapshots[split:]
    train_context = np.asarray([item[1] for item in train], dtype=np.float32)
    train_x = np.concatenate(
        (
            np.repeat(train_context[:, None, :], STRATEGY_NODE_COUNT, axis=1),
            np.repeat(
                np.eye(STRATEGY_NODE_COUNT, dtype=np.float32)[None, :, :],
                len(train),
                axis=0,
            ),
        ),
        axis=2,
    )
    train_y = np.asarray([item[2] for item in train], dtype=np.float32)
    train_target_mask = np.asarray(
        [item[3] for item in train],
        dtype=np.float32,
    )
    train_loss_weight = np.ones_like(train_y, dtype=np.float32)
    # Profitable post-cost outcomes are intentionally rare.  Without
    # class-balanced weighting the success head minimizes loss by predicting
    # NO_TRADE for every snapshot, which looks accurate but cannot rank the
    # rare usable edges.  Balance only the success classification head; P&L
    # regression remains trained on realized fills through target_mask.
    for strategy_position in range(STRATEGY_NODE_COUNT):
        positive = train_y[:, strategy_position, 0] > 0.0
        positive_count = int(positive.sum())
        negative_count = int((~positive).sum())
        if positive_count and negative_count:
            total = positive_count + negative_count
            train_loss_weight[positive, strategy_position, 0] = min(
                20.0,
                total / (2.0 * positive_count),
            )
            train_loss_weight[~positive, strategy_position, 0] = (
                total / (2.0 * negative_count)
            )
    train_is_krx = train_context[:, -1] >= 0.5
    krx_count = int(train_is_krx.sum())
    us_count = int((~train_is_krx).sum())
    if krx_count and us_count:
        sample_weights = np.where(
            train_is_krx,
            len(train) * 0.5 / krx_count,
            len(train) * 0.5 / us_count,
        ).astype(np.float32)
    else:
        sample_weights = np.ones(len(train), dtype=np.float32)
    adjacency = strategy_relation_adjacency()
    model = FixedShapeStrategyUtilityModel(config)
    initial_relations = model.relation_weights.copy()
    learning_rate = 0.003
    epochs = 40
    batch_size = 256
    parameters = {
        "relation_weights": model.relation_weights,
        "self_weight": model.self_weight,
        "strategy_heads": model.strategy_heads,
    }
    first_moment = {name: np.zeros_like(value) for name, value in parameters.items()}
    second_moment = {name: np.zeros_like(value) for name, value in parameters.items()}
    step = 0
    rng = np.random.default_rng(config.seed)
    for _ in range(epochs):
        order = rng.permutation(len(train_x))
        for start in range(0, len(order), batch_size):
            indexes = order[start : start + batch_size]
            x = train_x[indexes]
            target = train_y[indexes]
            target_mask = train_target_mask[indexes]
            loss_weight = train_loss_weight[indexes]
            messages = np.einsum("rij,bjf->brif", adjacency, x, optimize=True)
            relational = np.einsum(
                "brnf,rfh->bnh",
                messages,
                model.relation_weights,
                optimize=True,
            )
            self_part = np.einsum(
                "bnf,fh->bnh",
                x,
                model.self_weight,
                optimize=True,
            )
            pre_activation = relational + self_part
            hidden = np.maximum(pre_activation, 0.0)
            prediction = np.einsum(
                "bnh,nhk->bnk",
                hidden,
                model.strategy_heads,
                optimize=True,
            )
            gradient_prediction = (
                2.0
                * (prediction - target)
                * target_mask
                * loss_weight
                / max(1.0, float(target_mask.sum()))
            )
            gradient_prediction *= sample_weights[indexes, None, None]
            gradients = {
                "strategy_heads": np.einsum(
                    "bnh,bnk->nhk",
                    hidden,
                    gradient_prediction,
                    optimize=True,
                ),
            }
            gradient_hidden = np.einsum(
                "bnk,nhk->bnh",
                gradient_prediction,
                model.strategy_heads,
                optimize=True,
            )
            gradient_pre = gradient_hidden * (pre_activation > 0)
            gradients["self_weight"] = np.einsum(
                "bnf,bnh->fh",
                x,
                gradient_pre,
                optimize=True,
            )
            gradients["relation_weights"] = np.einsum(
                "brnf,bnh->rfh",
                messages,
                gradient_pre,
                optimize=True,
            )
            step += 1
            for name, parameter in parameters.items():
                gradient = gradients[name] + float(ridge) * 1e-5 * parameter
                norm = float(np.linalg.norm(gradient))
                if norm > 5.0:
                    gradient = gradient * (5.0 / norm)
                first_moment[name] = 0.9 * first_moment[name] + 0.1 * gradient
                second_moment[name] = (
                    0.999 * second_moment[name] + 0.001 * gradient * gradient
                )
                corrected_first = first_moment[name] / (1.0 - 0.9**step)
                corrected_second = second_moment[name] / (1.0 - 0.999**step)
                parameter -= (
                    learning_rate
                    * corrected_first
                    / (np.sqrt(corrected_second) + 1e-8)
                )

    full_hidden = _graph_hidden(model, train_x, adjacency)
    no_trade_targets = np.repeat(
        np.asarray(
            [-2.0 if item[4] else 2.0 for item in train],
            dtype=np.float32,
        ),
        STRATEGY_NODE_COUNT,
    )
    model.no_trade_head = _ridge_fit(
        full_hidden.reshape(-1, config.hidden_dim),
        no_trade_targets[:, None],
        ridge,
    )[:, 0]

    validation_context = np.asarray([item[1] for item in validation], dtype=np.float32)
    validation_x = np.concatenate(
        (
            np.repeat(
                validation_context[:, None, :],
                STRATEGY_NODE_COUNT,
                axis=1,
            ),
            np.repeat(
                np.eye(STRATEGY_NODE_COUNT, dtype=np.float32)[None, :, :],
                len(validation),
                axis=0,
            ),
        ),
        axis=2,
    )
    validation_target = np.asarray([item[2] for item in validation], dtype=np.float32)
    validation_target_mask = np.asarray(
        [item[3] for item in validation],
        dtype=np.float32,
    )
    validation_hidden = _graph_hidden(model, validation_x, adjacency)
    validation_prediction = np.einsum(
        "bnh,nhk->bnk",
        validation_hidden,
        model.strategy_heads,
        optimize=True,
    )
    raw_mse = float(
        np.sum(
            (validation_prediction - validation_target) ** 2
            * validation_target_mask
        )
        / max(1.0, float(validation_target_mask.sum()))
    )
    success_direction_accuracy = float(
        np.mean(
            (validation_prediction[..., 0] > 0)
            == (validation_target[..., 0] > 0)
        )
    )
    validation_is_krx = validation_context[:, -1] >= 0.5

    def market_accuracy(mask: np.ndarray) -> float | None:
        if not bool(mask.any()):
            return None
        return float(
            np.mean(
                (validation_prediction[mask, ..., 0] > 0)
                == (validation_target[mask, ..., 0] > 0)
            )
        )

    return model, fitted, {
        "train_snapshots": len(train),
        "validation_snapshots": len(validation),
        "raw_head_mse": raw_mse,
        "success_direction_accuracy": success_direction_accuracy,
        "krx_success_direction_accuracy": market_accuracy(validation_is_krx),
        "us_success_direction_accuracy": market_accuracy(~validation_is_krx),
        "krx_training_weight": (
            float(sample_weights[train_is_krx][0]) if krx_count else None
        ),
        "us_training_weight": (
            float(sample_weights[~train_is_krx][0]) if us_count else None
        ),
        "relation_weight_update_l2": float(
            np.linalg.norm(model.relation_weights - initial_relations)
        ),
        "epochs": epochs,
    }


def _graph_hidden(
    model: FixedShapeStrategyUtilityModel,
    x: np.ndarray,
    adjacency: np.ndarray,
) -> np.ndarray:
    messages = np.einsum("rij,bjf->brif", adjacency, x, optimize=True)
    relational = np.einsum(
        "brnf,rfh->bnh",
        messages,
        model.relation_weights,
        optimize=True,
    )
    self_part = np.einsum("bnf,fh->bnh", x, model.self_weight, optimize=True)
    return np.maximum(relational + self_part, 0.0)


def _checkpoint_hash(path: Path) -> str:
    import hashlib

    return hashlib.sha256(path.read_bytes()).hexdigest()


def _ridge_fit(x: np.ndarray, y: np.ndarray, ridge: float) -> np.ndarray:
    regularizer = np.eye(x.shape[1], dtype=np.float32) * max(1e-6, float(ridge))
    return np.linalg.solve(x.T @ x + regularizer, x.T @ y).astype(np.float32)


def _masked_ridge_fit(
    x: np.ndarray,
    y: np.ndarray,
    mask: np.ndarray,
    ridge: float,
) -> np.ndarray:
    columns = []
    for index in range(y.shape[1]):
        selected = mask[:, index] > 0.0
        if not bool(selected.any()):
            columns.append(np.zeros(x.shape[1], dtype=np.float32))
            continue
        columns.append(
            _ridge_fit(
                x[selected],
                y[selected, index : index + 1],
                ridge,
            )[:, 0]
        )
    return np.stack(columns, axis=1).astype(np.float32)


def _raw_target(row: CounterfactualLabel) -> tuple[float, ...]:
    positive = row.triggered and row.filled and row.net_return_bps > 0
    gross = row.net_return_bps + row.cost_bps
    # Classification uncertainty is highest near the post-cost decision
    # boundary and lower for outcomes far from zero.  The old target used
    # absolute P&L magnitude directly, so a clearly negative -75 bp outcome
    # was mislabeled as ~3.0 uncertainty and could never satisfy the router's
    # <= 1.0 contract even when the forecast was directionally unambiguous.
    boundary_uncertainty = 0.15 + 0.85 * float(
        np.exp(-abs(row.net_return_bps) / 20.0)
    )
    # --- Borrow leg (channels 8-10) ---------------------------------------- #
    # The head grew from 8 to 11 channels when shorts arrived, but these targets
    # did not, so every retrain died on a (…,11) vs (…,8) broadcast and the only
    # possible checkpoint was one the runtime refuses to load. The decoder's
    # scaling is mirrored exactly: cost is softplus(raw)*10, locate probability
    # is sigmoid(raw), epistemic is softplus(raw).
    borrow_cost_target = _inverse_softplus(max(0.01, float(row.borrow_cost_bps or 0.0) / 10.0))
    borrow_available_target = 2.0 if row.borrow_available else -2.0
    # Epistemic uncertainty is model IGNORANCE, so the honest per-row proxy is
    # whether this row is evidence at all: an unfilled counterfactual is the
    # model extrapolating, a realized fill is the model being corrected. No
    # invented constant - a row that never traded carries the higher target.
    epistemic_target = _inverse_softplus(0.25 if (row.triggered and row.filled) else 1.0)
    return (
        2.0 if positive else -2.0,
        float(np.clip(gross / 25.0, -5.0, 5.0)),
        _inverse_softplus(max(0.01, row.cost_bps / 10.0)),
        _inverse_softplus(max(0.01, max(-row.net_return_bps, 0.0) / 15.0)),
        _inverse_softplus(max(0.01, max(row.net_return_bps, 0.0) / 20.0)),
        2.0 if row.filled else -2.0,
        _inverse_softplus(15.0),
        _inverse_softplus(boundary_uncertainty),
        borrow_cost_target,
        borrow_available_target,
        epistemic_target,
    )


def _target_mask(row: CounterfactualLabel) -> tuple[float, ...]:
    realized = 1.0 if row.triggered and row.filled else 0.0
    positive = 1.0 if realized and row.net_return_bps > 0.0 else 0.0
    negative = 1.0 if realized and row.net_return_bps <= 0.0 else 0.0
    # A cash long has no borrow leg, and the decoder already masks channels 8/9
    # to short strategies. Training them on long rows would fit noise into a
    # tensor the runtime then ignores - or, if the mask ever moved, teach the
    # model to charge a long an invented borrow cost. Unobserved locates are
    # equally excluded: "we did not ask the desk" is not "no inventory".
    is_short = 1.0 if row.is_short else 0.0
    borrow_cost_observed = is_short if row.borrow_cost_bps is not None else 0.0
    borrow_locate_observed = is_short if row.borrow_available is not None else 0.0
    return (
        1.0,
        realized,
        realized,
        negative,
        positive,
        1.0,
        realized,
        realized,
        borrow_cost_observed,
        borrow_locate_observed,
        1.0,
    )


def _inverse_softplus(value: float) -> float:
    value = max(1e-6, min(30.0, float(value)))
    return float(np.log(np.expm1(value)))
