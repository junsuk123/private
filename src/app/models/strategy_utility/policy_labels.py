"""Read causal labels from forward shadow journals without modifying the database.

These observations measure entry-frozen target/stop/time barriers, not the
subsequently tightened exits of an owned live position. Legacy bar backtests,
missing contexts and policy-mismatched outcomes cannot acquire this contract.
"""
from __future__ import annotations

import json
import math
import sqlite3
import time
from collections import defaultdict
from contextlib import closing
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

from app.evaluation.stored_counterfactual import CounterfactualLabel
from app.features.strategy_graph_context import (
    STRATEGY_GRAPH_CONTEXT_DIM, STRATEGY_GRAPH_CONTEXT_SCHEMA, context_index,
)
from app.models.strategy_utility.policy_context import validate_graph_training_context
from app.risk.ontology_thresholds import POLICY_FAMILY_VERSION
from app.strategy.catalog import STRATEGY_IDS, is_short_strategy

LABEL_EXECUTION_POLICY = f"{POLICY_FAMILY_VERSION}-entry-frozen-shadow"
_IDENTITY = ("plan_id", "strategy_key", "strategy_id", "direction", "market",
             "execution_product", "symbol", "signal_at")
_OUTCOME_NUMBERS = ("net_return_bps", "gross_return_bps", "trading_cost_bps",
                    "borrow_cost_bps", "holding_seconds")
_MAX_ROWS = 4096


def _time(value: Any) -> datetime:
    parsed = value if isinstance(value, datetime) else datetime.fromisoformat(str(value).replace("Z", "+00:00"))
    if parsed.tzinfo is None or parsed.utcoffset() is None:
        raise ValueError("a journal timestamp must include its timezone")
    return parsed.astimezone(timezone.utc)


def _number(value: Any) -> float:
    if isinstance(value, bool):
        raise ValueError("boolean is not a measurement")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError("non-finite measurement")
    return result


def _same_number(left: Any, right: Any) -> bool:
    return math.isclose(_number(left), _number(right), rel_tol=1e-9, abs_tol=1e-7)


def _joined_label(row: sqlite3.Row, now: datetime) -> CounterfactualLabel | None:
    plan, outcome = json.loads(row["plan_json"]), json.loads(row["outcome_json"])
    if not isinstance(plan, dict) or not isinstance(outcome, dict):
        return None
    # Compare the JSON identities with both relational rows. Joining only on the
    # outer plan_id would otherwise permit a replaced/tampered JSON payload.
    for field in _IDENTITY:
        values = (plan[field], outcome[field], row[f"p_{field}"], row[f"o_{field}"])
        if field == "signal_at":
            values = tuple(_time(value) for value in values)
        if len(set(values)) != 1:
            return None
    symbol, strategy, market = plan["symbol"], plan["strategy_id"], plan["market"]
    if (market not in {"KR", "US"} or strategy not in STRATEGY_IDS or is_short_strategy(strategy)
            or plan["direction"] != "LONG" or plan["execution_product"] != "CASH"):
        return None
    signal, resolved = _time(plan["signal_at"]), _time(outcome["resolved_at"])
    if not signal < resolved <= now or resolved != _time(row["o_resolved_at"]):
        return None
    if (outcome.get("outcome") not in {"TARGET", "STOP", "MAX_HOLDING_TIME"}
            or outcome["outcome"] != row["o_outcome"]
            or outcome.get("scored") is not True or outcome.get("executable") is not True
            or outcome.get("signal_admissible") is not True
            or row["o_scored"] != 1 or row["o_executable"] != 1):
        return None
    diagnostics = plan["diagnostics"]
    if not isinstance(diagnostics, Mapping):
        return None
    contract = diagnostics["exit_contract"]
    context = diagnostics["graph_training_context"]
    if not isinstance(contract, Mapping) or not isinstance(context, Mapping) or not validate_graph_training_context(
            context, symbol=symbol, market=market, as_of=signal):
        return None
    features = tuple(_number(value) for value in context["features"])
    if (context.get("schema") != STRATEGY_GRAPH_CONTEXT_SCHEMA
            or len(features) != STRATEGY_GRAPH_CONTEXT_DIM
            or features[context_index("is_krx")] != (1.0 if market == "KR" else 0.0)
            or plan.get("feature_snapshot_id") != context.get("feature_snapshot_id")):
        return None
    snapshot_at = _time(context["as_of"])
    if snapshot_at > signal:
        return None
    policy = contract["ontology_risk_policy"]
    if not isinstance(policy, Mapping):
        return None
    policy_id = contract.get("policy_id")
    if (not isinstance(policy_id, str) or not policy_id
            or contract.get("risk_policy_family") != POLICY_FAMILY_VERSION
            or policy.get("policy_family") != POLICY_FAMILY_VERSION
            or outcome.get("risk_policy_family") != POLICY_FAMILY_VERSION
            or policy.get("policy_id") != policy_id or outcome.get("risk_policy_id") != policy_id
            or policy.get("symbol") != symbol or policy.get("market") != market
            or contract.get("ontology_entry_permitted") is not True
            or policy.get("valid_for_entry") is not True or policy.get("reason_codes")
            or not _time(policy["as_of"]) <= _time(contract["resolved_at"]) <= signal <= _time(policy["expires_at"])):
        return None
    if (contract.get("forecast_basis") != "original_strategy_forecast"
            or _number(contract["forecast_gross_bps"]) <= 0
            or _number(contract["policy_requested_horizon_seconds"]) <= 0
            or not _same_number(contract["forecast_gross_bps"], plan["predicted_gross_edge_bps"])):
        return None
    for plan_key, policy_key in (("target_rate", "target_return_rate"),
                                  ("stop_rate", "soft_stop_rate"),
                                  ("max_holding_seconds", "maximum_holding_seconds")):
        if _number(plan[plan_key]) <= 0 or not _same_number(plan[plan_key], policy[policy_key]):
            return None
    cost = _number(outcome["trading_cost_bps"])
    borrow = _number(outcome["borrow_cost_bps"])
    if (cost < 0 or borrow != 0 or not _same_number(cost, plan["expected_trading_cost_bps"])
            or not _same_number(cost, _number(policy["all_in_cost_rate"]) * 10000)):
        return None
    for field in _OUTCOME_NUMBERS:
        if not _same_number(outcome[field], row[f"o_{field}"]):
            return None
    entry, exit_price = _number(outcome["entry_price"]), _number(outcome["exit_price"])
    holding, fill = _number(outcome["holding_seconds"]), _number(outcome["fill_ratio"])
    gross, net = _number(outcome["gross_return_bps"]), _number(outcome["net_return_bps"])
    mae, mfe = _number(outcome["max_adverse_excursion_bps"]), _number(outcome["max_favorable_excursion_bps"])
    if (entry <= 0 or exit_price <= 0 or not 0 < fill <= 1 or holding <= 0
            or holding >= (resolved - signal).total_seconds() or mae < 0 or mfe < 0
            or not _same_number(gross, (exit_price / entry - 1) * 10000)
            or not _same_number(net, gross - cost)):
        return None
    return CounterfactualLabel(
        as_of=signal, label_end=resolved, symbol=symbol, strategy_id=strategy,
        triggered=True, filled=True, net_return_bps=net, cost_bps=cost,
        exit_reason=outcome["outcome"], features=features, direction="LONG",
        execution_product="CASH", feature_snapshot_at=snapshot_at,
        fill_ratio=fill, gross_return_bps=gross, mae_bps=mae, mfe_bps=mfe,
        holding_seconds=holding, cost_floor_dominated=gross <= cost,
        label_execution_policy=LABEL_EXECUTION_POLICY, policy_id=policy_id,
        feature_snapshot_id=context["feature_snapshot_id"],
    )


def load_policy_shadow_labels(
    database: Path, *, now: datetime, limit: int = 2048,
) -> tuple[CounterfactualLabel, ...]:
    """Read at most 4096 joined records, with masked placeholders for absent arms.

    Opens SQLite with ``mode=ro`` and does not instantiate a migrating store.
    Missing/incompatible/corrupt journals yield no labels. A limit bounds joined
    observations; output has at most that many times the catalogue size rows.
    """
    moment = _time(now)
    cap = max(0, min(_MAX_ROWS, int(limit)))
    database = Path(database)
    if not cap or not database.is_file():
        return ()
    columns = [f"{alias}.{field} AS {alias}_{field}" for alias in ("p", "o") for field in _IDENTITY]
    columns += [f"o.{field} AS o_{field}" for field in
                ("resolved_at", "outcome", "scored", "executable", *_OUTCOME_NUMBERS)]
    query = (
        "SELECT p.plan_json, o.outcome_json, " + ", ".join(columns)
        + " FROM shadow_plans p JOIN shadow_outcomes o ON o.plan_id=p.plan_id"
        " WHERE julianday(o.resolved_at)<=julianday(?) AND o.scored=1"
        " AND p.direction='LONG' AND p.execution_product='CASH'"
        " AND length(p.plan_json)<=131072 AND length(o.outcome_json)<=32768"
        " ORDER BY o.resolved_at DESC, p.plan_id DESC LIMIT ?"
    )
    try:
        with closing(sqlite3.connect(database.resolve().as_uri() + "?mode=ro", uri=True, timeout=.5)) as conn:
            conn.row_factory = sqlite3.Row
            deadline = time.monotonic() + 3.0
            conn.set_progress_handler(lambda: int(time.monotonic() > deadline), 10000)
            rows = conn.execute(query, (moment.isoformat(), cap)).fetchall()
    except (OSError, sqlite3.Error):
        return ()
    groups: dict[tuple[str, datetime], list[CounterfactualLabel]] = defaultdict(list)
    for row in rows:
        try:
            label = _joined_label(row, moment)
        except (KeyError, TypeError, ValueError, OverflowError):
            label = None
        if label is not None:
            groups[(label.symbol, label.as_of)].append(label)
    result: list[CounterfactualLabel] = []
    for _, observed in sorted(groups.items()):
        # One decision must have one unchanged feature snapshot; never choose
        # whichever conflicting duplicate happens to give the best return.
        first = observed[0]
        by_strategy = {row.strategy_id: row for row in observed}
        if (len(by_strategy) != len(observed) or any(
                row.features != first.features or row.feature_snapshot_id != first.feature_snapshot_id
                or row.feature_snapshot_at != first.feature_snapshot_at for row in observed)):
            continue
        label_end = max(row.label_end for row in observed)
        for strategy in STRATEGY_IDS:
            result.append(by_strategy.get(strategy) or CounterfactualLabel(
                as_of=first.as_of, label_end=label_end, symbol=first.symbol,
                strategy_id=strategy, triggered=False, filled=False,
                net_return_bps=0., cost_bps=0., exit_reason="FUTURE_WINDOW_CENSORED",
                features=first.features, feature_snapshot_at=first.feature_snapshot_at,
                fill_ratio=0., label_execution_policy=LABEL_EXECUTION_POLICY,
                feature_snapshot_id=first.feature_snapshot_id,
            ))
    return tuple(result)
