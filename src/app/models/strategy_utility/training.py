from __future__ import annotations

import json
import os
from datetime import datetime, timezone, timedelta
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable

import numpy as np

from app.evaluation.stored_counterfactual import CounterfactualLabel
from app.features.strategy_graph_context import (
    STRATEGY_GRAPH_CONTEXT_DIM,
    STRATEGY_GRAPH_CONTEXT_FIELDS,
    STRATEGY_GRAPH_CONTEXT_SCHEMA,
)
from app.models.strategy_utility.rgcn import (
    FixedShapeStrategyUtilityModel,
    StrategyUtilityModelConfig,
)
from app.models.strategy_utility.strategy_graph import (
    RELATION_NAMES,
    STRATEGY_NODE_COUNT,
    strategy_relation_adjacency,
    strategy_ids_for_market,
    strategy_market_mask,
    relation_object_properties,
)
from app.routing.shadow_intelligence import (
    COMPATIBILITY_UNAVAILABLE_REASONS,
    STRATEGY_IDS,
)
from app.strategy.catalog import is_short_strategy
from app.models.strategy_utility.temporal_graph import (
    ARCHITECTURE, TIME_STEPS, CausalGraphHistory, GraphObservation,
)
from app.models.strategy_utility.label_contract import (
    LEGACY_BAR_POLICY, ENTRY_FROZEN_SHADOW_POLICY, BOUNDED_ADVISORY_SCOPE, bounded_advisory_markets,
)


def train_counterfactual_checkpoint(
    labels: Iterable[CounterfactualLabel],
    output: str | Path,
    *,
    ridge: float = 1.0,
    input_feature_schema: str = "counterfactual_quantiles_v2_session_structure",
    authorize_live_shadow: bool = False,
) -> dict[str, object]:
    """Fit the complete compact temporal R-GCN on causal, bounded replay.

    Relation transforms, self transform and all supervised heads are fitted;
    chronological holdouts are never refitted into the saved checkpoint.
    """
    graph_schema = input_feature_schema in {
        "realtime_strategy_graph_v3",
        "realtime_strategy_graph_v4_market",
        STRATEGY_GRAPH_CONTEXT_SCHEMA,
    }
    # The current schema takes its width from the contract module rather than a
    # literal, so the two can never disagree; the historical widths stay listed
    # so an old label set is still trainable for comparison.
    context_feature_dim = (
        STRATEGY_GRAPH_CONTEXT_DIM
        if input_feature_schema == STRATEGY_GRAPH_CONTEXT_SCHEMA
        else (
            28
            if input_feature_schema == "realtime_strategy_graph_v4_market"
            else (
                27
                if input_feature_schema
                in {"realtime_strategy_context_v2", "realtime_strategy_graph_v3"}
                else 12
            )
        )
    )
    rows = tuple(
        row
        for row in labels
        if len(row.features) == context_feature_dim and row.usable_for_training
    )
    if not rows:
        raise ValueError(
            f"no counterfactual rows with {context_feature_dim} causal features"
        )
    label_policies = {getattr(row, "label_execution_policy", LEGACY_BAR_POLICY) for row in rows}
    if len(label_policies) != 1:
        raise ValueError("mixed execution-policy label contracts cannot train one payoff model")
    label_policy = next(iter(label_policies))
    policy_matched = bool(label_policy == ENTRY_FROZEN_SHADOW_POLICY and any(row.outcome_observed for row in rows) and all(
        (getattr(row, "policy_id", "") and getattr(row, "feature_snapshot_id", ""))
        for row in rows if row.outcome_observed
    ))
    if label_policy == ENTRY_FROZEN_SHADOW_POLICY and not policy_matched:
        raise ValueError("policy shadow labels require frozen policy and feature provenance")
    config = StrategyUtilityModelConfig(
        batch_size=1,
        time_steps=TIME_STEPS if input_feature_schema == STRATEGY_GRAPH_CONTEXT_SCHEMA else 1,
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
        temporal_mode=1 if input_feature_schema == STRATEGY_GRAPH_CONTEXT_SCHEMA else 0,
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
        no_trade_masks = []
        for positions in grouped.values():
            attractive = any(
                rows[position].triggered
                and rows[position].filled
                and rows[position].net_return_bps > 0
                for position in positions
            )
            no_trade_targets.append(-2.0 if attractive else 2.0)
            applicable = set(strategy_ids_for_market(rows[positions[0]].symbol))
            observed = {rows[position].strategy_id for position in positions if rows[position].outcome_observed}
            no_trade_masks.append(float(attractive or applicable <= observed))
        model.no_trade_head = _masked_ridge_fit(
            hidden[group_positions],
            np.asarray(no_trade_targets, dtype=np.float32)[:, None],
            np.asarray(no_trade_masks, dtype=np.float32)[:, None],
            ridge,
        )[:, 0]
        validation_metrics = {}
        method = "causal_feature_encoder_plus_ridge_calibrated_heads"
    checkpoint = Path(output)
    trained_keys = set(validation_metrics.pop("_trained_snapshot_keys", ()))
    supervision_rows = tuple(row for row in rows if row.outcome_observed and (row.symbol, row.as_of) in trained_keys) if graph_schema else tuple(row for row in rows if row.outcome_observed)
    # Missing alternatives are no longer counted as observed training rows.
    # The bounded advisory source still needs 1000 separate training snapshots,
    # each with at least one actual completed walk, plus held-out market gates.
    minimum_rows = 1_000 if policy_matched else 10_000
    minimum_snapshots = 1_000
    # ``fitted`` counts SNAPSHOTS the strategy appeared in, which for the graph
    # path is every snapshot — 1,615 for all 16 ids, including the 6 whose
    # conditions never triggered even once. Gating authorization on that number
    # was vacuous: it certified "500+ rows" for heads with zero supervision.
    # Coverage has to be measured on realized outcomes, and specifically on the
    # UPSIDE ones, because those are the only rows that teach the model what a
    # profitable setup looks like (see ``_strategy_supervision``).
    supervision = _strategy_supervision(supervision_rows)
    supervised_strategy_ids = tuple(
        strategy_id
        for strategy_id in STRATEGY_IDS
        if supervision[strategy_id]["upside_supervised"]
    )
    # ``any``, not ``all``: several ids are structurally unreachable in this
    # deployment, so requiring every one of them to carry upside evidence can
    # never be satisfied. The per-strategy flags above are what the runtime
    # actually enforces; this only asks that the checkpoint taught at least one.
    strategy_coverage = bool(supervised_strategy_ids)
    skill_verdict = _skill_verdict(validation_metrics)
    base_authorization_ready = bool(
        authorize_live_shadow
        and policy_matched
        and input_feature_schema == STRATEGY_GRAPH_CONTEXT_SCHEMA
        and len(supervision_rows) >= minimum_rows
        and len({(row.symbol, row.as_of) for row in supervision_rows}) >= minimum_snapshots
        and strategy_coverage
    )
    market_authorization_checks, advisory_authorized_markets = (
        _market_authorization_verdicts(
            validation_metrics,
            base_ready=base_authorization_ready,
        )
    )
    market_authorization_checks = {market: {
        **checks, "bounded_advisory_authorized": checks["live_authorized"], "live_authorized": False,
    } for market, checks in market_authorization_checks.items()}
    # Frozen-entry shadow walks do not validate dynamic live exits. Their
    # independently validated downside/uncertainty may only tighten risk.
    live_authorized_markets = []
    live_shadow_authorized = False
    # Never replace a validated incumbent with a failing candidate or remove a
    # market authorization while adding the other market. Candidate artifacts
    # remain reviewable and can gather separate shadow evidence.
    incumbent_markets = set()
    incumbent_policy_matched = False
    if checkpoint.exists():
        try:
            old_report = json.loads(checkpoint.with_suffix(".json").read_text(encoding="utf-8"))
            if old_report.get("checkpoint_hash") == _checkpoint_hash(checkpoint):
                incumbent_markets = set(bounded_advisory_markets(old_report))
                incumbent_policy_matched = (old_report.get("label_execution_policy") == ENTRY_FROZEN_SHADOW_POLICY
                                           and old_report.get("label_policy_provenance_matched") is True)
        except (OSError, ValueError, TypeError):
            pass
    retained_incumbent = bool(incumbent_markets - set(advisory_authorized_markets)
                             or (incumbent_policy_matched and not policy_matched))
    if retained_incumbent:
        checkpoint = checkpoint.with_name(checkpoint.stem + ".candidate.npz")
    checkpoint = model.save_checkpoint(checkpoint)
    report = {
        "checkpoint": str(checkpoint),
        "retained_incumbent": retained_incumbent,
        "training_supervision_rows": len(supervision_rows),
        "rows": len(rows),
        "snapshots": len(grouped),
        "strategies": fitted,
        "label_outcomes": _label_outcome_summary(supervision_rows),
        "label_outcomes_by_market": {
            "KRX": _label_outcome_summary(
                tuple(
                    row
                    for row in supervision_rows
                    if row.symbol.isdigit() and len(row.symbol) == 6
                )
            ),
            "US": _label_outcome_summary(
                tuple(
                    row
                    for row in supervision_rows
                    if not (row.symbol.isdigit() and len(row.symbol) == 6)
                )
            ),
        },
        "method": method,
        "strategy_supervision": supervision,
        "upside_supervised_strategy_ids": list(supervised_strategy_ids),
        "minimum_upside_supervision_rows": _MINIMUM_UPSIDE_SUPERVISION_ROWS,
        "strategy_ids": list(STRATEGY_IDS),
        "strategy_count": len(STRATEGY_IDS),
        "feature_names": [
            # Real names for the aligned contract. The anonymous
            # ``causal_context_feature_{i}`` labels are what let slot 4 mean two
            # different things for a month without anyone reading the card
            # noticing.
            *(
                list(STRATEGY_GRAPH_CONTEXT_FIELDS)
                if input_feature_schema == STRATEGY_GRAPH_CONTEXT_SCHEMA
                else [
                    f"causal_context_feature_{index}"
                    for index in range(context_feature_dim)
                ]
            ),
            *(
                [f"strategy_identity:{strategy_id}" for strategy_id in STRATEGY_IDS]
                if graph_schema
                else []
            ),
        ],
        "relation_names": list(RELATION_NAMES) if graph_schema else ["context"],
        "relation_object_properties": list(relation_object_properties()) if graph_schema else [],
        "training_data_range": {
            "start": min((row.as_of.isoformat() for row in rows), default=None),
            "end": max((row.label_end.isoformat() for row in rows), default=None),
        },
        "training_method": method,
        "label_execution_policy": label_policy,
        "label_policy_provenance_matched": policy_matched,
        "payoff_semantics": "entry_frozen_shadow_net_after_cost" if policy_matched else "historical_strategy_geometry_counterfactual_net_after_cost",
        "dynamic_live_payoff_validated": False,
        "architecture": ARCHITECTURE if config.temporal_mode else "rgcn_v1",
        "temporal_contract": "three_causal_minute_observations_elapsed_time_decay" if config.temporal_mode else "legacy_single_snapshot",
        "validation_metrics": validation_metrics,
        "checkpoint_hash": _checkpoint_hash(checkpoint),
        "input_feature_schema": input_feature_schema,
        "feature_provenance": (
            # v5 reads the persisted per-bar spread / imbalance / liquidity
            # columns. v4's name said "proxy" because it derived those from the
            # bar's own geometry while serving supplied the real thing.
            "causal_minute_bar_microstructure_v2_aligned"
            if input_feature_schema == STRATEGY_GRAPH_CONTEXT_SCHEMA
            else (
                "causal_minute_bar_microstructure_proxy_v1"
                if input_feature_schema == "realtime_strategy_graph_v4_market"
                else "causal_counterfactual_quantiles_v2_session_structure"
            )
        ),
        "live_authorized": live_shadow_authorized,
        "live_authorized_markets": live_authorized_markets,
        "bounded_advisory_authorized_markets": advisory_authorized_markets,
        "authorization_scope": BOUNDED_ADVISORY_SCOPE if policy_matched else "research_shadow_only",
        "authorization_checks": {
            "requested": bool(authorize_live_shadow),
            "label_policy_provenance_matched": policy_matched,
            "dynamic_live_execution_authority": False,
            "minimum_rows": minimum_rows,
            "minimum_snapshots": minimum_snapshots,
            "strategy_minimum_upside_rows": _MINIMUM_UPSIDE_SUPERVISION_ROWS,
            "row_count_ok": len(supervision_rows) >= minimum_rows,
            "snapshot_count_ok": len({(row.symbol, row.as_of) for row in supervision_rows}) >= minimum_snapshots,
            "strategy_coverage_ok": strategy_coverage,
            "strategy_coverage_basis": "upside_supervision_rows_per_strategy",
            # The runtime schema is whatever the contract module currently
            # declares. Pinning a literal here is how an authorization check
            # keeps passing for a schema the serving path no longer emits.
            "schema_matches_runtime": (
                input_feature_schema == STRATEGY_GRAPH_CONTEXT_SCHEMA
            ),
            # These are now blocking authorization checks. The runtime may still
            # use an unproven checkpoint as a shadow estimator, but never as an
            # order authority.
            **skill_verdict,
            "by_market": market_authorization_checks,
            **_context_field_coverage(rows, input_feature_schema),
        },
        "config": asdict(config),
    }
    report_path = checkpoint.with_suffix(".json")
    temporary_report = report_path.with_name(f".{report_path.stem}.writing.json")
    temporary_report.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    temporary_report.replace(report_path)
    return report


#: Minimum realized POSITIVE outcomes before a strategy's upside head counts as
#: taught. Same bar ``_label_outcome_summary`` already applies to fills, so the
#: report and the authorization agree on what "enough samples" means.
_MINIMUM_UPSIDE_SUPERVISION_ROWS = 20


def _strategy_supervision(
    rows: tuple[CounterfactualLabel, ...],
) -> dict[str, dict[str, int | bool]]:
    """Rows that actually carry gradient for each strategy head, per channel side.

    ``_target_mask`` masks the P&L channels to realized fills, and splits them:
    channel 3 (adverse excursion, MAE) is masked to ``negative`` outcomes and
    channel 4 (favourable excursion, MFE) to ``positive`` ones. The decoder then
    forms the whole expectation as ``probability * mfe - (1 - probability) * mae``,
    so **MFE is the only upside term in every net-edge forecast the model emits.**

    Measured on the 2026-08-03 checkpoint (25,840 rows): MAE heads received 5-112
    rows while MFE heads received 0-21, six of them exactly zero. An MFE head with
    no rows keeps its random initialization, and one with a single row reproduces
    that row — ``rvgi_box_breakout`` learned +132bps from one lucky fill and went
    on to forecast positive edges whose realized mean was -168bps. The forecasts
    were not mismodelled, they were unsupervised.

    Publishing the counts lets the runtime drop the upside term it has no evidence
    for instead of shipping noise as a prediction.
    """
    supervision: dict[str, dict[str, int | bool]] = {}
    for strategy_id in STRATEGY_IDS:
        selected = [row for row in rows if row.strategy_id == strategy_id]
        realized = [
            row
            for row in selected
            if row.outcome_observed and row.triggered and row.filled
        ]
        upside = [row for row in realized if row.net_return_bps > 0]
        supervision[strategy_id] = {
            "labels": len(selected),
            "realized_rows": len(realized),
            "upside_rows": len(upside),
            "downside_rows": len(realized) - len(upside),
            "upside_supervised": len(upside) >= _MINIMUM_UPSIDE_SUPERVISION_ROWS,
        }
    return supervision


def _label_outcome_summary(
    rows: tuple[CounterfactualLabel, ...],
) -> dict[str, dict[str, float | int | str | None]]:
    summary: dict[str, dict[str, float | int | str | None]] = {}
    for strategy_id in STRATEGY_IDS:
        selected = [row for row in rows if row.strategy_id == strategy_id]
        simulated_filled = [row for row in selected if row.outcome_observed and row.filled]
        filled = [
            row
            for row in selected
            if row.outcome_observed and row.triggered and row.filled
        ]
        positive = [row for row in filled if row.net_return_bps > 0]
        negative = [row for row in filled if row.net_return_bps < 0]
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
            "negative_net": len(negative),
            "positive_net_rate_when_filled": (
                len(positive) / len(filled) if filled else None
            ),
            "mean_gross_return_bps_when_filled": mean_gross,
            "mean_net_return_bps_when_filled": mean_net,
            "mean_cost_bps_when_filled": mean_cost,
            "mean_holding_seconds_when_filled": (
                float(np.mean([row.holding_seconds for row in filled]))
                if filled
                else None
            ),
            "mean_mae_bps_when_filled": (
                float(np.mean([row.mae_bps for row in filled])) if filled else None
            ),
            "mean_mfe_bps_when_filled": (
                float(np.mean([row.mfe_bps for row in filled])) if filled else None
            ),
            "profit_factor_when_filled": (
                float(
                    sum(row.net_return_bps for row in positive)
                    / max(1e-9, abs(sum(row.net_return_bps for row in negative)))
                )
                if positive and negative
                else None
            ),
            "cost_floor_dominated_rate": (
                sum(row.cost_floor_dominated for row in filled) / len(filled)
                if filled
                else None
            ),
            "exit_reason_counts": dict(
                sorted(Counter(row.exit_reason for row in filled).items())
            ),
            "performance_diagnosis": diagnosis,
        }
    return summary


#: A binary context flag needs at least this share of minority observations
#: before its weight means anything. Below it the head has seen essentially one
#: value and the other side is an untrained regime.
_MINIMUM_FLAG_SUPPORT = 0.01


def _context_field_coverage(
    rows: tuple[CounterfactualLabel, ...],
    input_feature_schema: str,
) -> dict[str, object]:
    """Which context fields did this training window actually vary?

    ``rvgi_available`` and ``box_context_available`` are constant at 1.0 across
    all 3,597 snapshots: every bar in the window had a warmed-up indicator. Their
    weights are therefore fitted on one value only, and the moment serving hits
    the condition they exist to signal — an unavailable RVGI, where the rvgi
    columns go to 0.0 — the model runs on an untrained regime with nothing
    reporting it.

    Deleting them is the wrong fix: without the flag, "0.0 because unavailable"
    and "0.0 because that is the value" become the same input, which is the
    silent-default failure the context contract exists to prevent. Reporting is
    the fix, so coverage is a property an operator can read rather than a
    property they have to assume.
    """
    if input_feature_schema != STRATEGY_GRAPH_CONTEXT_SCHEMA:
        return {}
    by_snapshot: dict[tuple[str, object], tuple[float, ...]] = {}
    for row in rows:
        by_snapshot[(row.symbol, row.as_of)] = row.features
    if not by_snapshot:
        return {}
    contexts = np.asarray(list(by_snapshot.values()), dtype=np.float64)
    constant: list[str] = []
    weak_flags: dict[str, float] = {}
    for index, name in enumerate(STRATEGY_GRAPH_CONTEXT_FIELDS):
        column = contexts[:, index]
        distinct = np.unique(column)
        if distinct.size == 1:
            constant.append(name)
        elif distinct.size == 2:
            support = float(min((column == value).mean() for value in distinct))
            if support < _MINIMUM_FLAG_SUPPORT:
                weak_flags[name] = support
    return {
        "context_fields_constant_in_training": constant,
        "context_flags_below_minimum_support": weak_flags,
        "context_flag_minimum_support": _MINIMUM_FLAG_SUPPORT,
        "context_coverage_note": (
            "constant fields have single-valued weights; serving may still vary "
            "them, and that regime is untrained rather than merely rare"
        ),
    }


def _skill_verdict(metrics: dict[str, object]) -> dict[str, object]:
    """Two plain answers, so nobody has to infer them from raw percentiles.

    They are separate questions and the current data answers them differently:
    the head ranks trades better than chance, and it does NOT have a
    demonstrated positive net edge. Collapsing those into one "authorized" flag
    is what let a below-baseline accuracy read as "no predictive power" while an
    AUC of 0.74 sat unmeasured.
    """
    auc_low = metrics.get("selection_auc_ci_low")
    null = metrics.get("selection_auc_within_symbol_null")
    p_value = metrics.get("selection_auc_permutation_p")
    non_positive = metrics.get("selection_top_decile_net_p_nonpositive")
    ranking_established = bool(
        isinstance(auc_low, float)
        and isinstance(null, float)
        and isinstance(p_value, float)
        # Beats chance with 95% of the clustered bootstrap AND clears the
        # within-symbol null, so the separation is not just "prefers symbols
        # that happen to win more often".
        and auc_low > 0.5
        and p_value < 0.05
    )
    edge_established = bool(
        isinstance(non_positive, float) and non_positive < 0.05
    )
    return {
        "selection_ranking_skill_established": ranking_established,
        "selection_ranking_clears_within_symbol_null": bool(
            isinstance(auc_low, float) and isinstance(null, float) and auc_low > null
        ),
        "selection_net_edge_established": edge_established,
        "selection_skill_note": (
            "ranking skill and a profitable edge are different claims; "
            "promotion to live election needs the second, not just the first"
        ),
    }


def _market_authorization_verdicts(
    validation_metrics: dict[str, object],
    *,
    base_ready: bool,
) -> tuple[dict[str, dict[str, object]], list[str]]:
    """Resolve checkpoint authority independently for KRX and US evidence."""
    checks: dict[str, dict[str, object]] = {}
    authorized_markets: list[str] = []
    for market, prefix in (("KRX", "krx"), ("US", "us")):
        market_metrics = {
            key.removeprefix(f"{prefix}_"): value
            for key, value in validation_metrics.items()
            if key.startswith(f"{prefix}_selection_")
        }
        verdict = _skill_verdict(market_metrics)
        validation_rows = int(
            validation_metrics.get(f"{prefix}_selection_rows") or 0
        )
        validation_symbols = int(
            validation_metrics.get(f"{prefix}_selection_symbols") or 0
        )
        authorized = bool(
            base_ready
            and validation_rows >= 20
            and validation_symbols >= 5
            and verdict["selection_ranking_skill_established"]
            and verdict["selection_net_edge_established"]
        )
        checks[market] = {
            "selection_rows": validation_rows,
            "minimum_selection_rows": 20,
            "selection_symbols": validation_symbols,
            "minimum_selection_symbols": 5,
            **verdict,
            "live_authorized": authorized,
        }
        if authorized:
            authorized_markets.append(market)
    return checks, authorized_markets


@dataclass(frozen=True)
class _Snapshot:
    """One (symbol, timestamp) row of the fixed-shape training grid."""

    as_of: object
    label_end: object
    symbol: str
    context: np.ndarray
    targets: np.ndarray
    target_masks: np.ndarray
    attractive: bool
    nets: np.ndarray
    filled: np.ndarray
    no_trade_observed: bool = True


def _market_purged_split(
    snapshots: list[_Snapshot],
    *,
    train_fraction: float = 0.8,
) -> tuple[list[_Snapshot], list[_Snapshot], int]:
    """Build independently purged chronological KR and US holdouts.

    A single global boundary let the market with the newest tape occupy the
    entire validation set. In production that yielded 280 US selection rows and
    zero KR rows. Each market now keeps its own chronological boundary, so
    collection time in one market cannot erase validation for the other.
    """
    by_market: dict[str, list[_Snapshot]] = {"KRX": [], "US": []}
    for item in snapshots:
        market = "KRX" if item.symbol.isdigit() and len(item.symbol) == 6 else "US"
        by_market[market].append(item)

    train: list[_Snapshot] = []
    validation: list[_Snapshot] = []
    purged = 0
    for market_rows in by_market.values():
        market_rows.sort(key=lambda item: item.as_of)
        if len(market_rows) < 2:
            train.extend(market_rows)
            continue
        split = max(
            1,
            min(len(market_rows) - 1, int(len(market_rows) * train_fraction)),
        )
        market_validation = market_rows[split:]
        boundary = market_validation[0].as_of
        market_train = [
            item for item in market_rows[:split] if item.label_end + timedelta(seconds=60) < boundary
        ]
        purged += split - len(market_train)
        train.extend(market_train)
        validation.extend(market_validation)
    train.sort(key=lambda item: item.as_of)
    validation.sort(key=lambda item: item.as_of)
    if validation:
        # All markets share learned parameters. A later US training label must
        # not teach the encoder while an earlier KR window is called holdout.
        boundary = validation[0].as_of
        shared_causal_train = [item for item in train if item.label_end + timedelta(seconds=60) < boundary]
        purged += len(train) - len(shared_causal_train)
        train = shared_causal_train
    return train, validation, max(0, purged)


def _rank_auc(scores: np.ndarray, outcomes: np.ndarray) -> float | None:
    """Probability a positive outranks a negative, ties counted as half."""
    positives = int(outcomes.sum())
    negatives = int(outcomes.size - positives)
    if positives == 0 or negatives == 0:
        return None
    order = np.argsort(scores, kind="mergesort")
    ranks = np.empty(scores.size, dtype=np.float64)
    ordered_scores = scores[order]
    start = 0
    while start < order.size:
        stop = start
        while stop + 1 < order.size and ordered_scores[stop + 1] == ordered_scores[start]:
            stop += 1
        ranks[order[start : stop + 1]] = (start + stop) / 2.0 + 1.0
        start = stop + 1
    rank_sum = float(ranks[outcomes].sum())
    return (rank_sum - positives * (positives + 1) / 2.0) / (positives * negatives)


def _selection_metrics(
    scores: np.ndarray,
    nets: np.ndarray,
    symbols: np.ndarray,
    *,
    draws: int = 1000,
    seed: int = 17,
) -> dict[str, float | int | None]:
    """Can the head RANK the trades it would actually take?

    Accuracy cannot answer that. Channel 0's grid is ~97% trivial negatives, so
    a constant "not a success" predictor beats a model that ranks well but sits
    on the wrong side of a threshold — which is exactly what the v4/v5 cards
    showed, and it read as "no predictive power" when the ranking AUC was 0.72.

    Two things here are not optional:

    - **Symbol-clustered resampling.** Rows from one symbol share a price path
      and 12x-overlapping forward windows, so bootstrapping ROWS would treat
      hundreds of dependent observations as independent and produce a CI several
      times too narrow.
    - **A within-symbol permutation null instead of 0.5.** Shuffling outcomes
      inside each symbol preserves per-symbol win rates, so the null keeps
      whatever separation comes from the model merely preferring symbols that
      win more often. Measured at 0.647 on the current data — a pooled AUC of
      0.72 is a far weaker claim against that null than against 0.5, and only
      the excess over it is evidence of timing skill.
    """
    outcomes = nets > 0.0
    observed = _rank_auc(scores, outcomes)
    result: dict[str, float | int | None] = {
        "selection_rows": int(scores.size),
        "selection_symbols": int(np.unique(symbols).size),
        "selection_positive_rate": (
            float(outcomes.mean()) if outcomes.size else None
        ),
        "selection_mean_net_bps": float(nets.mean()) if nets.size else None,
        "selection_auc": observed,
    }
    if observed is None or scores.size < 20:
        return result

    rng = np.random.default_rng(seed)
    unique_symbols = np.unique(symbols)
    index_by_symbol = [np.flatnonzero(symbols == symbol) for symbol in unique_symbols]

    def top_decile_net(selected: np.ndarray) -> float:
        count = max(1, int(selected.size * 0.1))
        best = selected[np.argsort(-scores[selected], kind="mergesort")[:count]]
        return float(nets[best].mean())

    boot_auc: list[float] = []
    boot_net: list[float] = []
    for _ in range(draws):
        drawn = rng.integers(0, len(index_by_symbol), size=len(index_by_symbol))
        selected = np.concatenate([index_by_symbol[position] for position in drawn])
        value = _rank_auc(scores[selected], outcomes[selected])
        if value is not None:
            boot_auc.append(value)
        boot_net.append(top_decile_net(selected))

    null_auc: list[float] = []
    for _ in range(draws):
        shuffled = outcomes.copy()
        for indices in index_by_symbol:
            shuffled[indices] = rng.permutation(shuffled[indices])
        value = _rank_auc(scores, shuffled)
        if value is not None:
            null_auc.append(value)

    everything = np.arange(scores.size)
    result.update(
        {
            "selection_auc_ci_low": float(np.percentile(boot_auc, 2.5)) if boot_auc else None,
            "selection_auc_ci_high": float(np.percentile(boot_auc, 97.5)) if boot_auc else None,
            "selection_auc_within_symbol_null": (
                float(np.mean(null_auc)) if null_auc else None
            ),
            # Fraction of within-symbol permutations reaching the observed AUC.
            "selection_auc_permutation_p": (
                float(np.mean(np.asarray(null_auc) >= observed)) if null_auc else None
            ),
            "selection_top_decile_net_bps": top_decile_net(everything),
            "selection_top_decile_net_ci_low": (
                float(np.percentile(boot_net, 2.5)) if boot_net else None
            ),
            "selection_top_decile_net_ci_high": (
                float(np.percentile(boot_net, 97.5)) if boot_net else None
            ),
            # The number that decides whether this model may take a trade: how
            # often the selected decile fails to make money at all.
            "selection_top_decile_net_p_nonpositive": (
                float(np.mean(np.asarray(boot_net) <= 0.0)) if boot_net else None
            ),
        }
    )
    return result


def _krx_flag_index(context_width: int) -> int:
    """Column carrying the KRX flag, for the market-balanced sample weights.

    Read by NAME on the aligned contract. The v4 layout happened to put the flag
    last, and this code took ``context[:, -1]`` -- so the moment the contract
    reordered, the market split silently started reading
    ``box_context_available`` and every US row was classified as KRX. Exactly the
    positional-assumption failure the contract module exists to stop, which is
    why the index is derived rather than assumed.
    """
    if context_width == STRATEGY_GRAPH_CONTEXT_DIM:
        return STRATEGY_GRAPH_CONTEXT_FIELDS.index("is_krx")
    return context_width - 1


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
    snapshots: list[_Snapshot] = []
    fitted = {strategy_id: 0 for strategy_id in STRATEGY_IDS}
    for (symbol, _key_as_of), positions in grouped.items():
        by_strategy = {rows[position].strategy_id: rows[position] for position in positions}
        if any(strategy_id not in by_strategy for strategy_id in STRATEGY_IDS):
            continue
        ordered = [by_strategy[strategy_id] for strategy_id in STRATEGY_IDS]
        now = datetime.now(timezone.utc)
        if any(row.as_of.tzinfo is None or row.label_end.tzinfo is None
               or row.label_end < row.as_of or row.label_end > now
               or not np.isfinite(row.features).all()
               or not np.isfinite([row.net_return_bps, row.cost_bps, row.mae_bps, row.mfe_bps, row.holding_seconds]).all()
               or row.cost_bps < 0 for row in ordered):
            continue
        context = np.asarray(ordered[0].features, dtype=np.float32)
        if any(not np.array_equal(np.asarray(row.features, dtype=np.float32), context) for row in ordered):
            continue
        targets = np.asarray([_raw_target(row) for row in ordered], dtype=np.float32)
        target_masks = np.asarray(
            [_target_mask(row) for row in ordered],
            dtype=np.float32,
        )
        market_strategy_ids = set(strategy_ids_for_market(str(symbol)))
        attractive = any(
            row.outcome_observed
            and row.triggered
            and row.filled
            and row.net_return_bps > 0
            for row in ordered
            if row.strategy_id in market_strategy_ids
        )
        snapshots.append(
            _Snapshot(
                as_of=ordered[0].as_of,
                label_end=max(row.label_end for row in ordered),
                symbol=str(symbol),
                context=context,
                targets=targets,
                target_masks=target_masks,
                attractive=attractive,
                no_trade_observed=attractive or all(row.outcome_observed for row in ordered
                                                   if row.strategy_id in market_strategy_ids),
                # Carried so validation can measure SELECTION quality, which
                # needs the realised P&L and the symbol each row came from.
                nets=np.asarray(
                    [float(row.net_return_bps) for row in ordered], dtype=np.float32
                ),
                filled=np.asarray(
                    [
                        bool(row.outcome_observed and row.triggered and row.filled)
                        for row in ordered
                    ],
                    dtype=bool,
                ),
            )
        )
        for strategy_id in STRATEGY_IDS:
            fitted[strategy_id] += 1
    if len(snapshots) < 2:
        raise ValueError("strategy relation graph requires complete multi-strategy snapshots")
    # Bound training independently per market; a newer US tape cannot evict KR.
    replay_limit = _bounded_env_int("GNN_REPLAY_MAX_SNAPSHOTS_PER_MARKET", 2048, 64, 4096)
    snapshots = sorted(snapshots, key=lambda item: item.as_of)
    kr = [item for item in snapshots if item.symbol.isdigit() and len(item.symbol) == 6][-replay_limit:]
    us = [item for item in snapshots if not (item.symbol.isdigit() and len(item.symbol) == 6)][-replay_limit:]
    snapshots = sorted((*kr, *us), key=lambda item: item.as_of)
    temporal_inputs = {}
    if config.temporal_mode:
        history = CausalGraphHistory(max_symbols=max(1, len({item.symbol for item in snapshots})))
        for item in snapshots:
            temporal_inputs[(item.symbol, item.as_of)] = history.inputs(
                GraphObservation(item.symbol, item.as_of, tuple(item.context)))
    train, validation, purged_rows = _market_purged_split(snapshots)
    if not validation or not train:
        raise ValueError("strategy relation graph requires non-overlapping purged train and validation")
    train_context = np.asarray([item.context for item in train], dtype=np.float32)
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
    train_y = np.asarray([item.targets for item in train], dtype=np.float32)
    train_target_mask = np.asarray(
        [item.target_masks for item in train],
        dtype=np.float32,
    )
    train_market_mask = np.asarray(
        [strategy_market_mask(item.symbol) for item in train],
        dtype=np.float32,
    )
    train_x *= train_market_mask[:, :, None]
    train_target_mask *= train_market_mask[:, :, None]
    train_loss_weight = np.ones_like(train_y, dtype=np.float32)
    # Unweighted Bernoulli loss keeps success probabilities on the observed
    # class prior. Class balancing improved ranking while overstating cash edge.
    krx_flag = _krx_flag_index(train_context.shape[1])
    train_is_krx = train_context[:, krx_flag] >= 0.5
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
    train_adjacency = np.asarray(
        [strategy_relation_adjacency(market=item.symbol) for item in train],
        dtype=np.float32,
    )
    if config.temporal_mode:
        train_x = np.asarray([temporal_inputs[(item.symbol, item.as_of)][0] for item in train])
        train_adjacency = np.asarray([temporal_inputs[(item.symbol, item.as_of)][1] for item in train])
        train_temporal_mask = np.asarray([temporal_inputs[(item.symbol, item.as_of)][2] for item in train])
    else:
        train_x = train_x[:, None]
        train_adjacency = train_adjacency[:, None]
        train_temporal_mask = train_market_mask[:, None]
    model = FixedShapeStrategyUtilityModel(config)
    initial_relations = model.relation_weights.copy()
    learning_rate = 0.003
    epochs = _bounded_env_int("GNN_TRAINING_EPOCHS", 12, 1, 40)
    maximum_steps = _bounded_env_int("GNN_TRAINING_MAX_STEPS", 128, 1, 256)
    batch_size = 128
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
            if step >= maximum_steps:
                break
            indexes = order[start : start + batch_size]
            x = train_x[indexes]
            target = train_y[indexes]
            target_mask = train_target_mask[indexes]
            loss_weight = train_loss_weight[indexes]
            adjacency = train_adjacency[indexes]
            time_mask = train_temporal_mask[indexes]
            messages = np.einsum("btrij,btjf->btrif", adjacency, x, optimize=True)
            relational = np.einsum("btrnf,rfh->btnh", messages, model.relation_weights, optimize=True)
            self_part = np.einsum("btnf,fh->btnh", x, model.self_weight, optimize=True)
            pre_activation = relational + self_part
            hidden_by_time = np.maximum(pre_activation, 0.0)
            hidden = np.einsum("t,btn,btnh->bnh", model.temporal_weights, time_mask, hidden_by_time, optimize=True)
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
            probability = 1.0 / (1.0 + np.exp(-np.clip(prediction[..., 0], -30, 30)))
            gradient_prediction[..., 0] = ((probability - (target[..., 0] > 0))
                                          * target_mask[..., 0] / max(1.0, float(target_mask[..., 0].sum())))
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
            gradient_pre = (gradient_hidden[:, None] * time_mask[..., None]
                            * model.temporal_weights[None, :, None, None] * (pre_activation > 0))
            gradients["self_weight"] = np.einsum("btnf,btnh->fh", x, gradient_pre, optimize=True)
            gradients["relation_weights"] = np.einsum("btrnf,btnh->rfh", messages, gradient_pre, optimize=True)
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

    full_hidden = _temporal_graph_hidden(model, train_x, train_adjacency, train_temporal_mask)
    no_trade_targets = np.repeat(
        np.asarray(
            [-2.0 if item.attractive else 2.0 for item in train],
            dtype=np.float32,
        ),
        STRATEGY_NODE_COUNT,
    )
    no_trade_mask = np.repeat(np.asarray([item.no_trade_observed for item in train], dtype=np.float32), STRATEGY_NODE_COUNT)
    model.no_trade_head = _masked_ridge_fit(
        full_hidden.reshape(-1, config.hidden_dim),
        no_trade_targets[:, None],
        no_trade_mask[:, None],
        ridge,
    )[:, 0]

    validation_context = np.asarray([item.context for item in validation], dtype=np.float32)
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
    validation_target = np.asarray([item.targets for item in validation], dtype=np.float32)
    validation_target_mask = np.asarray(
        [item.target_masks for item in validation],
        dtype=np.float32,
    )
    validation_market_mask = np.asarray(
        [strategy_market_mask(item.symbol) for item in validation],
        dtype=np.float32,
    )
    validation_x *= validation_market_mask[:, :, None]
    validation_target_mask *= validation_market_mask[:, :, None]
    validation_adjacency = np.asarray(
        [strategy_relation_adjacency(market=item.symbol) for item in validation],
        dtype=np.float32,
    )
    if config.temporal_mode:
        validation_x = np.asarray([temporal_inputs[(item.symbol, item.as_of)][0] for item in validation])
        validation_adjacency = np.asarray([temporal_inputs[(item.symbol, item.as_of)][1] for item in validation])
        validation_temporal_mask = np.asarray([temporal_inputs[(item.symbol, item.as_of)][2] for item in validation])
    else:
        validation_x = validation_x[:, None]
        validation_adjacency = validation_adjacency[:, None]
        validation_temporal_mask = validation_market_mask[:, None]
    validation_hidden = _temporal_graph_hidden(model, validation_x, validation_adjacency, validation_temporal_mask)
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
    # Direction accuracy is meaningless without its majority-class baseline.
    #
    # Channel 0 is conditional on a realized fill, matching the decoder's hurdle
    # model: channel 5 predicts whether the strategy fills, while channel 0 asks
    # whether that filled trade wins. Treating every non-trigger as a loss made
    # those two heads contradictory and overwhelmed the rare payoff evidence.
    #
    # Channel 5's target is the filled indicator (+2 filled / -2 not), so the
    # realized mask is read straight off the validation targets.
    predicted_success = validation_prediction[..., 0] > 0
    actual_success = validation_target[..., 0] > 0
    realized_cells = (validation_target[..., 5] > 0) & (
        validation_market_mask > 0
    )
    observed_success_cells = validation_target_mask[..., 0] > 0

    def _cells(row_mask: np.ndarray | None) -> np.ndarray:
        return (
            observed_success_cells
            if row_mask is None
            else observed_success_cells & row_mask[:, None]
        )

    def _accuracy(cells: np.ndarray) -> float | None:
        if not bool(cells.any()):
            return None
        return float(np.mean(predicted_success[cells] == actual_success[cells]))

    def _majority_baseline(cells: np.ndarray) -> float | None:
        """Score of the best constant predictor -- the bar accuracy must clear."""
        if not bool(cells.any()):
            return None
        positive_rate = float(np.mean(actual_success[cells]))
        return max(positive_rate, 1.0 - positive_rate)

    success_direction_accuracy = _accuracy(_cells(None))
    validation_is_krx = (
        validation_context[:, _krx_flag_index(validation_context.shape[1])] >= 0.5
    )

    # SELECTION quality on the cells that became real trades. This is the
    # measurement the accuracy figures above cannot make: a head that ranks
    # correctly but sits on the wrong side of its threshold scores below the
    # majority baseline while still separating winners from losers.
    filled_mask = np.asarray([item.filled for item in validation], dtype=bool) & (
        validation_market_mask > 0
    )
    # Authorization must rank the value the live selector trades, not just the
    # win logit. Payoff magnitude matters: a high chance of a tiny win is not a
    # better trade than a slightly lower chance of a much larger net payoff.
    selection_scores = _expected_net_from_raw(validation_prediction)[
        filled_mask
    ].astype(np.float64)
    selection_nets = np.asarray(
        [item.nets for item in validation], dtype=np.float64
    )[filled_mask]
    selection_symbols = np.repeat(
        np.asarray([item.symbol for item in validation]),
        STRATEGY_NODE_COUNT,
    ).reshape(len(validation), STRATEGY_NODE_COUNT)[filled_mask]
    selection_market_is_krx = np.repeat(
        validation_is_krx[:, None], STRATEGY_NODE_COUNT, axis=1
    )[filled_mask]
    selection = _selection_metrics(
        selection_scores,
        selection_nets,
        selection_symbols,
    )

    def market_selection(prefix: str, mask: np.ndarray) -> dict[str, object]:
        values = _selection_metrics(
            selection_scores[mask],
            selection_nets[mask],
            selection_symbols[mask],
            seed=17 if prefix == "krx" else 29,
        )
        return {f"{prefix}_{key}": value for key, value in values.items()}

    return model, fitted, {
        "_trained_snapshot_keys": [(item.symbol, item.as_of) for item in train],
        "train_snapshots": len(train),
        "validation_snapshots": len(validation),
        "krx_train_snapshots": sum(
            item.symbol.isdigit() and len(item.symbol) == 6 for item in train
        ),
        "us_train_snapshots": sum(
            not (item.symbol.isdigit() and len(item.symbol) == 6) for item in train
        ),
        "krx_validation_snapshots": int(validation_is_krx.sum()),
        "us_validation_snapshots": int((~validation_is_krx).sum()),
        # Training rows dropped because their label resolved after the split.
        "purged_train_snapshots": purged_rows,
        "raw_head_mse": raw_mse,
        **selection,
        **market_selection("krx", selection_market_is_krx),
        **market_selection("us", ~selection_market_is_krx),
        "success_direction_accuracy": success_direction_accuracy,
        # ALWAYS read the accuracy against this. At or below it, the metric is
        # reporting class imbalance rather than skill and must not be used to
        # justify promotion.
        "success_direction_majority_baseline": _majority_baseline(_cells(None)),
        # Restricted to triggered-and-filled cells, where "success" vs "failure"
        # is a real outcome instead of "the strategy never fired".
        "success_direction_accuracy_realized": _accuracy(realized_cells),
        "success_direction_majority_baseline_realized": _majority_baseline(
            realized_cells
        ),
        "success_direction_realized_cells": int(realized_cells.sum()),
        "success_direction_total_cells": int(observed_success_cells.sum()),
        "krx_success_direction_accuracy": _accuracy(_cells(validation_is_krx)),
        "us_success_direction_accuracy": _accuracy(_cells(~validation_is_krx)),
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
        "gradient_steps": step,
        "maximum_gradient_steps": maximum_steps,
        "replay_snapshots": len(snapshots),
        "time_steps": config.time_steps,
    }


def _bounded_env_int(name: str, default: int, low: int, high: int) -> int:
    try:
        return min(high, max(low, int(os.getenv(name, str(default)))))
    except (TypeError, ValueError):
        return default


def _temporal_graph_hidden(model, x, adjacency, masks):
    messages = np.einsum("btrij,btjf->btrif", adjacency, x, optimize=True)
    pre = (np.einsum("btrnf,rfh->btnh", messages, model.relation_weights, optimize=True)
           + np.einsum("btnf,fh->btnh", x, model.self_weight, optimize=True))
    return np.einsum("t,btn,btnh->bnh", model.temporal_weights, masks, np.maximum(pre, 0.0), optimize=True)


def _graph_hidden(
    model: FixedShapeStrategyUtilityModel,
    x: np.ndarray,
    adjacency: np.ndarray,
) -> np.ndarray:
    if adjacency.ndim == 3:
        messages = np.einsum("rij,bjf->brif", adjacency, x, optimize=True)
    elif adjacency.ndim == 4:
        messages = np.einsum("brij,bjf->brif", adjacency, x, optimize=True)
    else:
        raise ValueError(f"unexpected strategy adjacency shape: {adjacency.shape}")
    relational = np.einsum(
        "brnf,rfh->bnh",
        messages,
        model.relation_weights,
        optimize=True,
    )
    self_part = np.einsum("bnf,fh->bnh", x, model.self_weight, optimize=True)
    return np.maximum(relational + self_part, 0.0)


def _expected_net_from_raw(raw: np.ndarray) -> np.ndarray:
    """Decode the payoff heads used by live utility, excluding uncertainty."""
    probability = 1.0 / (1.0 + np.exp(-np.clip(raw[..., 0], -60.0, 60.0)))
    downside = np.logaddexp(0.0, raw[..., 3]) * 15.0
    upside = np.logaddexp(0.0, raw[..., 4]) * 20.0
    expected_net = probability * upside - (1.0 - probability) * downside
    for position, strategy_id in enumerate(STRATEGY_IDS):
        if is_short_strategy(strategy_id):
            expected_net[..., position] -= (
                np.logaddexp(0.0, raw[..., position, 8]) * 10.0
            )
    return expected_net


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
    if not row.outcome_observed:
        # A missing arm or unfinished walk is neither a failed fill nor evidence
        # of uncertainty. Keep the graph node, but supervise no output channel.
        return (0.0,) * 11
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
        # Win probability is conditional on a realized fill. Reachability has
        # its own supervised head at channel 5; an untriggered strategy is
        # unknown payoff, not a realized loss.
        realized,
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
