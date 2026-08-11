"""Read-only projection of the trained strategy R-GCN for the account UI.

This module deliberately does not use the market-fact ontology graph.  It reads
the tensors that were actually saved by strategy-utility training, reconstructs
that checkpoint's fixed strategy topology, and overlays recorded CPU-GNN
inference telemetry.  A stale checkpoint therefore remains visible as stale
instead of being presented as a successful live inference.
"""

from __future__ import annotations

from collections import deque
from datetime import datetime, timezone
import json
import math
import os
from pathlib import Path
from typing import Any

import numpy as np

from app.models.strategy_utility.rgcn import _HEAD_CHANNELS
from app.strategy.catalog import STRATEGY_IDS


DEFAULT_METADATA_PATH = Path("data/models/strategy_utility/rgcn_shadow.json")
DEFAULT_INFERENCE_PATH = Path("logs/refactor-shadow-comparison.jsonl")

_MOMENTUM = {
    "intraday_momentum",
    "event_momentum",
    "cross_sectional_relative_strength",
    "market_intraday_momentum",
    # A carry is a continuation thesis held across a session boundary.
    "overnight_gap_carry",
}
_BREAKOUT = {
    "breakout_volume",
    "gap_context",
    "rvgi_box_breakout",
    "opening_range_breakout",
}
_REVERSION = {
    "vwap_mean_reversion",
    "liquidity_shock_reversal",
    "adaptive_anchored_vwap_reversion",
    "ofi_microprice_exhaustion_reversal",
}
_RELATION_NAMES = (
    "same_methodology_family",
    "confirming_methodology",
    "contrasting_methodology",
)
_CHECKPOINT_OUTPUT_NAMES = (
    "probability_success",
    "gross_return_bps",
    "cost_bps",
    "mae_bps",
    "mfe_bps",
    "fill_probability",
    "holding_seconds",
    "aleatoric_uncertainty",
)


def build_strategy_gnn_visualization(
    *,
    metadata_path: str | Path = DEFAULT_METADATA_PATH,
    inference_path: str | Path = DEFAULT_INFERENCE_PATH,
    inference_line_limit: int = 1200,
) -> dict[str, Any]:
    """Return the learned checkpoint graph plus honest live-inference status."""

    metadata_file = Path(metadata_path)
    inference_file = Path(inference_path)
    generated_at = datetime.now(timezone.utc).isoformat()
    if not metadata_file.exists():
        return _empty_payload(generated_at, "GNN_CHECKPOINT_METADATA_MISSING")

    try:
        metadata = json.loads(metadata_file.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return _empty_payload(generated_at, "GNN_CHECKPOINT_METADATA_INVALID")

    checkpoint_value = metadata.get("checkpoint") or metadata_file.with_suffix(".npz")
    checkpoint_file = Path(str(checkpoint_value).replace("\\", "/"))
    if not checkpoint_file.is_absolute():
        direct = checkpoint_file
        relative_to_metadata = metadata_file.parent / checkpoint_file.name
        checkpoint_file = direct if direct.exists() else relative_to_metadata
    if not checkpoint_file.exists():
        return _empty_payload(generated_at, "GNN_CHECKPOINT_MISSING", metadata=metadata)

    try:
        with np.load(checkpoint_file, allow_pickle=False) as archive:
            relation_weights = np.asarray(archive["relation_weights"], dtype=np.float32)
            self_weight = np.asarray(archive["self_weight"], dtype=np.float32)
            strategy_heads = np.asarray(archive["strategy_heads"], dtype=np.float32)
            raw_config = tuple(int(value) for value in archive["config"].tolist())
    except (OSError, KeyError, ValueError):
        return _empty_payload(generated_at, "GNN_CHECKPOINT_TENSORS_INVALID", metadata=metadata)

    config = dict(metadata.get("config") or {})
    trained_strategy_ids = tuple(
        str(value)
        for value in (metadata.get("strategy_ids") or ())
        if str(value).strip()
    )
    strategy_count = int(config.get("strategy_count") or strategy_heads.shape[0])
    if not trained_strategy_ids:
        trained_strategy_ids = tuple(STRATEGY_IDS[:strategy_count])
    trained_strategy_ids = trained_strategy_ids[:strategy_count]

    relation_names = tuple(metadata.get("relation_names") or _RELATION_NAMES)
    relation_norms = np.linalg.norm(relation_weights, axis=(1, 2))
    relation_scale = float(max(float(relation_norms.max(initial=0.0)), 1e-12))
    relation_strength = {
        str(name): round(float(relation_norms[index] / relation_scale), 6)
        for index, name in enumerate(relation_names[: len(relation_norms)])
    }
    head_norms = np.linalg.norm(strategy_heads, axis=(1, 2))
    head_scale = float(max(float(head_norms.max(initial=0.0)), 1e-12))
    label_outcomes = dict(metadata.get("label_outcomes") or {})
    # ``training_labels`` counts every snapshot the strategy appeared in, which for
    # the graph path is all of them — the same number that used to certify "500+
    # rows" for heads with zero supervision. Carry the rows that actually trained
    # the upside alongside it so the graph cannot imply evidence it does not have.
    strategy_supervision = dict(metadata.get("strategy_supervision") or {})
    minimum_upside_rows = int(
        metadata.get("minimum_upside_supervision_rows")
        or os.getenv("GNN_MIN_UPSIDE_SUPERVISION_ROWS", "20")
    )

    def _upside_rows(strategy_id: str) -> int | None:
        row = strategy_supervision.get(strategy_id)
        if isinstance(row, dict):
            return int(row.get("upside_rows") or 0)
        outcome = label_outcomes.get(strategy_id)
        if isinstance(outcome, dict) and "positive_net" in outcome:
            return int(outcome.get("positive_net") or 0)
        return None

    inference = _read_inference_overlay(
        inference_file,
        trained_strategy_ids,
        max_lines=max(1, int(inference_line_limit)),
    )
    inference_by_strategy = inference.pop("by_strategy")

    nodes: list[dict[str, Any]] = []
    for index, strategy_id in enumerate(trained_strategy_ids):
        outcome = dict(label_outcomes.get(strategy_id) or {})
        overlay = dict(inference_by_strategy.get(strategy_id) or {})
        nodes.append(
            {
                "id": strategy_id,
                "label": strategy_id.replace("_", " "),
                "kind": "strategy",
                "layer": "strategy_topology",
                "cluster": _strategy_cluster(strategy_id),
                "checkpoint_index": index,
                "learned_strength": round(float(head_norms[index] / head_scale), 6),
                "training_labels": int(outcome.get("labels") or 0),
                "training_filled_rows": int(outcome.get("filled") or 0),
                "training_upside_rows": _upside_rows(strategy_id),
                "minimum_upside_rows": minimum_upside_rows,
                "upside_supervised": (
                    None
                    if _upside_rows(strategy_id) is None
                    else _upside_rows(strategy_id) >= minimum_upside_rows
                ),
                "training_positive_net_rate": _finite_or_none(
                    outcome.get("positive_net_rate_when_filled")
                ),
                "inference_count": int(overlay.get("count") or 0),
                "latest_utility": _finite_or_none(overlay.get("utility")),
                "latest_probability": _finite_or_none(overlay.get("probability_success")),
                "latest_expected_net_bps": _finite_or_none(
                    overlay.get("expected_net_return_bps")
                ),
                "active": bool(overlay),
            }
        )

    topology_links = _checkpoint_links(trained_strategy_ids, relation_strength)
    parameter_nodes, parameter_links = _checkpoint_parameter_graph(
        metadata=metadata,
        trained_strategy_ids=trained_strategy_ids,
        relation_names=relation_names,
        relation_weights=relation_weights,
        self_weight=self_weight,
        strategy_heads=strategy_heads,
    )
    nodes.extend(parameter_nodes)
    links = [*topology_links, *parameter_links]
    runtime_reasons: list[str] = []
    checkpoint_head_channels = int(strategy_heads.shape[-1])
    if strategy_count != len(STRATEGY_IDS):
        runtime_reasons.append("GNN_STRATEGY_COUNT_MISMATCH")
    if checkpoint_head_channels != _HEAD_CHANNELS:
        runtime_reasons.append("GNN_HEAD_SCHEMA_MISMATCH")
    if len(raw_config) != 8:
        runtime_reasons.append("GNN_CONFIG_SCHEMA_MISMATCH")

    validation = dict(metadata.get("validation_metrics") or {})
    cluster_counts: dict[str, int] = {}
    for node in nodes:
        cluster_counts[node["cluster"]] = cluster_counts.get(node["cluster"], 0) + 1
    return {
        "schema": "strategy_rgcn_visualization_v1",
        "generated_at": generated_at,
        "source": {
            "kind": "trained_strategy_rgcn_checkpoint",
            "metadata": str(metadata_file),
            "checkpoint": str(checkpoint_file),
            "inference_log": str(inference_file),
            "not_ontology_fact_graph": True,
        },
        "model": {
            "available": True,
            "checkpoint_hash": metadata.get("checkpoint_hash"),
            "method": metadata.get("training_method") or metadata.get("method"),
            "input_feature_schema": metadata.get("input_feature_schema"),
            "feature_dim": int(self_weight.shape[0]),
            "hidden_dim": int(self_weight.shape[1]),
            "strategy_count": strategy_count,
            "relation_count": int(relation_weights.shape[0]),
            "head_channels": checkpoint_head_channels,
            "training_rows": int(metadata.get("rows") or 0),
            "training_snapshots": int(metadata.get("snapshots") or 0),
            "validation_accuracy": _finite_or_none(
                validation.get("success_direction_accuracy")
            ),
            "relation_strength": relation_strength,
            "runtime_compatible": not runtime_reasons,
            "runtime_reasons": runtime_reasons,
            "runtime_strategy_count": len(STRATEGY_IDS),
            "runtime_head_channels": _HEAD_CHANNELS,
        },
        "inference": inference,
        "clusters": [
            {"id": cluster, "node_count": count}
            for cluster, count in sorted(cluster_counts.items())
        ],
        "nodes": nodes,
        "links": links,
        "counts": {
            "nodes": len(nodes),
            "links": len(links),
            "strategy_nodes": len(trained_strategy_ids),
            "strategy_links": len(topology_links),
            "parameter_nodes": len(parameter_nodes),
            "parameter_links": len(parameter_links),
        },
    }


def build_strategy_gnn_state(
    *,
    inference_path: str | Path = DEFAULT_INFERENCE_PATH,
    active_window_seconds: float = 6.0,
) -> dict[str, Any]:
    """Lightweight liveness signal polled independently from the large graph."""

    path = Path(inference_path)
    now = datetime.now(timezone.utc)
    empty = {
        "state": "OFFLINE",
        "active": False,
        "phase": "idle",
        "updated_at": None,
        "age_seconds": None,
        "symbol": None,
        "action": None,
        "strategy_id": None,
    }
    if not path.exists():
        return {**empty, "reason_codes": ["GNN_INFERENCE_LOG_MISSING"]}
    try:
        modified_at = datetime.fromtimestamp(path.stat().st_mtime, tz=timezone.utc)
        age_seconds = max(0.0, (now - modified_at).total_seconds())
        rows = _tail_lines(path, max_lines=12)
    except OSError:
        return {**empty, "reason_codes": ["GNN_INFERENCE_LOG_UNREADABLE"]}
    latest: dict[str, Any] = {}
    for raw in reversed(rows):
        try:
            latest = json.loads(raw)
            break
        except json.JSONDecodeError:
            continue
    decisions = [
        item for item in (latest.get("decisions") or ())
        if item.get("path") in {"cpu_gnn", "npu_gnn"}
    ]
    decision = decisions[-1] if decisions else {}
    active = age_seconds <= max(1.0, float(active_window_seconds))
    blocked = bool(decision) and not decision.get("strategy_id")
    return {
        "state": "INFERENCE_RUNNING" if active else "BLOCKED" if blocked else "IDLE",
        "active": active,
        "phase": "message_passing" if active else "blocked" if blocked else "idle",
        "updated_at": modified_at.isoformat(),
        "age_seconds": round(age_seconds, 3),
        "observed_at": latest.get("as_of"),
        "symbol": latest.get("symbol"),
        "action": decision.get("action"),
        "strategy_id": decision.get("strategy_id"),
        "path": decision.get("path"),
        "utility": _finite_or_none(decision.get("utility")),
        "reason_codes": list(decision.get("reason_codes") or ()),
        "activation": _activation_map(decision, active=active),
    }


# Per-strategy verdicts the router records for every evaluated arm. The reason
# code carries the strategy id, which is what makes a real per-node activation
# possible: ``ONTOLOGY_BLOCKED:vwap_mean_reversion`` says that node's gate was
# shut on this pass, not that some animation frame came around.
_BLOCKED_PREFIXES = ("ONTOLOGY_BLOCKED:", "STRATEGY_NOT_ALLOWED:", "COMPAT_ZERO:")
_EVALUATED_PREFIXES = ("NON_POSITIVE_NET_EDGE:", "NEGATIVE_UTILITY:", "BELOW_THRESHOLD:")


def _activation_map(decision: dict[str, Any], *, active: bool) -> dict[str, Any]:
    """Real activation per graph node, or an explicit "not instrumented".

    The UI used to sweep the four layers on a 3.6-second wall-clock carousel,
    which looked like staged inference while carrying no information at all: the
    only live input was one boolean. This returns what the inference record
    actually knows, so a node's glow can be a measurement.

    Layers the record does not carry (the input encoder and the hidden message
    vector are not logged) are reported as ``observed: false`` rather than given
    a synthesised value — an un-instrumented layer must look un-instrumented.
    """

    reason_codes = tuple(str(code) for code in (decision.get("reason_codes") or ()))
    selected = str(decision.get("strategy_id") or "")
    action = str(decision.get("action") or "")
    strategies: dict[str, dict[str, Any]] = {}

    def classify(strategy_id: str) -> tuple[str, float]:
        if strategy_id and strategy_id == selected:
            # A selected arm that still resolved to NO_TRADE is election-level
            # activity, not an order, and is shown as evaluated rather than won.
            if action and action != "NO_TRADE":
                return "SELECTED", 1.0
            return "ELECTED_NO_TRADE", 0.72
        for code in reason_codes:
            if not code.endswith(f":{strategy_id}"):
                continue
            if code.startswith(_BLOCKED_PREFIXES):
                return "GATE_BLOCKED", 0.18
            if code.startswith(_EVALUATED_PREFIXES):
                return "EVALUATED_NON_POSITIVE", 0.45
        return "UNEVALUATED", 0.0

    for strategy_id in STRATEGY_IDS:
        state, intensity = classify(strategy_id)
        strategies[strategy_id] = {
            "state": state,
            # Intensity is what the renderer scales glow and pulse amplitude by.
            # Zero means "do not animate this node", which is the honest state for
            # an arm this pass never looked at.
            "intensity": round(intensity if active else min(intensity, 0.12), 4),
        }

    # Head channels the record does carry, mapped onto the same names the graph
    # payload gives its output nodes so the client needs no translation table.
    channels = {
        "probability_success": _finite_or_none(decision.get("probability_success")),
        "cost_bps": _finite_or_none(decision.get("expected_cost_bps")),
        "aleatoric_uncertainty": _finite_or_none(decision.get("total_uncertainty")),
    }
    net_bps = _finite_or_none(decision.get("expected_net_return_bps"))
    cost_bps = channels["cost_bps"]
    if net_bps is not None and cost_bps is not None:
        channels["gross_return_bps"] = net_bps + cost_bps
    # Drop the unmeasured ones BEFORE anything counts them: a NO_TRADE decision
    # logs all three as null, and counting the keys rather than the values had the
    # layer reporting "3 channels observed" with nothing in it.
    measured = {key: value for key, value in channels.items() if value is not None}
    return {
        "strategies": strategies,
        "channels": measured,
        "selected_strategy_id": selected or None,
        # Which layers carry measured values this pass. The renderer reads this
        # instead of inventing a sequence.
        "layers": {
            "input": {"observed": False, "reason": "ENCODER_INPUT_NOT_LOGGED"},
            "message_passing": {"observed": False, "reason": "HIDDEN_STATE_NOT_LOGGED"},
            "strategy_election": {
                "observed": bool(reason_codes) or bool(selected),
                "evaluated": sum(
                    1
                    for item in strategies.values()
                    if item["state"] != "UNEVALUATED"
                ),
            },
            "output_decode": {"observed": bool(measured), "channels": len(measured)},
        },
    }


def _checkpoint_parameter_graph(
    *,
    metadata: dict[str, Any],
    trained_strategy_ids: tuple[str, ...],
    relation_names: tuple[str, ...],
    relation_weights: np.ndarray,
    self_weight: np.ndarray,
    strategy_heads: np.ndarray,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Expand every saved non-zero tensor parameter into a computation graph.

    The strategy topology above shows which nodes exchange messages.  This view
    adds the shared feature encoder and every per-strategy output-head weight, so
    "all connections" means all learned checkpoint parameters rather than only
    the 13 high-level strategy nodes.
    """

    feature_dim, hidden_dim = self_weight.shape
    head_channels = int(strategy_heads.shape[-1])
    raw_feature_names = tuple(str(value) for value in (metadata.get("feature_names") or ()))
    feature_names = tuple(
        raw_feature_names[index] if index < len(raw_feature_names) else f"feature_{index:02d}"
        for index in range(feature_dim)
    )
    output_names = tuple(
        _CHECKPOINT_OUTPUT_NAMES[index]
        if index < len(_CHECKPOINT_OUTPUT_NAMES)
        else f"checkpoint_head_{index}"
        for index in range(head_channels)
    )
    nodes: list[dict[str, Any]] = []
    for index, name in enumerate(feature_names):
        identity = name.startswith("strategy_identity:")
        nodes.append(
            {
                "id": f"feature:{index}",
                "label": name.replace("causal_context_feature_", "context ").replace("strategy_identity:", "identity ").replace("_", " "),
                "kind": "feature",
                "layer": "input",
                "cluster": "input_identity" if identity else "input_context",
                "feature_index": index,
            }
        )
    for index in range(hidden_dim):
        nodes.append(
            {
                "id": f"hidden:{index}",
                "label": f"message hidden {index:02d}",
                "kind": "hidden",
                "layer": "message_passing",
                "cluster": "hidden",
                "hidden_index": index,
            }
        )
    for strategy_index, strategy_id in enumerate(trained_strategy_ids):
        family = _strategy_cluster(strategy_id)
        for channel_index, channel_name in enumerate(output_names):
            nodes.append(
                {
                    "id": f"output:{strategy_index}:{channel_index}",
                    "label": f"{strategy_id.replace('_', ' ')} · {channel_name.replace('_', ' ')}",
                    "kind": "output",
                    "layer": "strategy_head",
                    "cluster": "output",
                    "family": family,
                    "strategy_id": strategy_id,
                    "channel": channel_name,
                    "channel_index": channel_index,
                }
            )

    encoder_max = max(
        float(np.abs(self_weight).max(initial=0.0)),
        float(np.abs(relation_weights).max(initial=0.0)),
        1e-12,
    )
    head_max = max(float(np.abs(strategy_heads).max(initial=0.0)), 1e-12)
    links: list[dict[str, Any]] = []
    for feature_index in range(feature_dim):
        for hidden_index in range(hidden_dim):
            value = float(self_weight[feature_index, hidden_index])
            if value != 0.0:
                links.append(
                    {
                        "source": f"feature:{feature_index}",
                        "target": f"hidden:{hidden_index}",
                        "relation": "self_encoder_weight",
                        "kind": "learned_parameter",
                        "weight": round(value, 7),
                        "learned_strength": round(abs(value) / encoder_max, 6),
                    }
                )
            for relation_index in range(min(len(relation_names), relation_weights.shape[0])):
                value = float(relation_weights[relation_index, feature_index, hidden_index])
                if value == 0.0:
                    continue
                links.append(
                    {
                        "source": f"feature:{feature_index}",
                        "target": f"hidden:{hidden_index}",
                        "relation": f"relation_encoder:{relation_names[relation_index]}",
                        "kind": "learned_parameter",
                        "weight": round(value, 7),
                        "learned_strength": round(abs(value) / encoder_max, 6),
                    }
                )
    for strategy_index, strategy_id in enumerate(trained_strategy_ids):
        for hidden_index in range(hidden_dim):
            for channel_index in range(head_channels):
                value = float(strategy_heads[strategy_index, hidden_index, channel_index])
                if value == 0.0:
                    continue
                links.append(
                    {
                        "source": f"hidden:{hidden_index}",
                        "target": f"output:{strategy_index}:{channel_index}",
                        "relation": "strategy_head_weight",
                        "kind": "learned_parameter",
                        "weight": round(value, 7),
                        "learned_strength": round(abs(value) / head_max, 6),
                        "strategy_id": strategy_id,
                    }
                )
        for channel_index in range(head_channels):
            links.append(
                {
                    "source": strategy_id,
                    "target": f"output:{strategy_index}:{channel_index}",
                    "relation": "owns_output_head",
                    "kind": "structural",
                    "learned_strength": 1.0,
                    "strategy_id": strategy_id,
                }
            )
    return nodes, links


def _checkpoint_links(
    strategy_ids: tuple[str, ...],
    relation_strength: dict[str, float],
) -> list[dict[str, Any]]:
    raw: list[tuple[str, str, str]] = []
    for source in strategy_ids:
        for target in strategy_ids:
            if source == target:
                continue
            if _same_family(source, target):
                raw.append((source, target, "same_methodology_family"))
            if (
                (source in _MOMENTUM and target in _BREAKOUT)
                or (source in _BREAKOUT and target in _MOMENTUM)
            ):
                raw.append((source, target, "confirming_methodology"))
            if (
                (source in (_MOMENTUM | _BREAKOUT) and target in _REVERSION)
                or (source in _REVERSION and target in (_MOMENTUM | _BREAKOUT))
            ):
                raw.append((source, target, "contrasting_methodology"))

    degrees: dict[tuple[str, str], int] = {}
    for _source, target, relation in raw:
        degrees[(target, relation)] = degrees.get((target, relation), 0) + 1
    return [
        {
            "source": source,
            "target": target,
            "relation": relation,
            "prior_weight": round(1.0 / max(1, degrees[(target, relation)]), 6),
            "learned_strength": relation_strength.get(relation, 0.0),
        }
        for source, target, relation in raw
    ]


def _read_inference_overlay(
    path: Path,
    strategy_ids: tuple[str, ...],
    *,
    max_lines: int,
) -> dict[str, Any]:
    by_strategy: dict[str, dict[str, Any]] = {}
    latest_at: str | None = None
    symbols: set[str] = set()
    successful = 0
    blocked = 0
    reason_counts: dict[str, int] = {}
    if not path.exists():
        return {
            "available": False,
            "latest_at": None,
            "symbols": 0,
            "successful_decisions": 0,
            "blocked_decisions": 0,
            "latest_reason_codes": ["GNN_INFERENCE_LOG_MISSING"],
            "by_strategy": by_strategy,
        }

    for raw in _tail_lines(path, max_lines=max_lines):
        try:
            row = json.loads(raw)
        except json.JSONDecodeError:
            continue
        observed = str(row.get("as_of") or "")
        latest_at = max(latest_at or observed, observed) if observed else latest_at
        symbol = str(row.get("symbol") or "").strip()
        if symbol:
            symbols.add(symbol)
        decisions = [
            item
            for item in (row.get("decisions") or ())
            if item.get("path") in {"cpu_gnn", "npu_gnn"}
        ]
        for decision in decisions:
            strategy_id = str(decision.get("strategy_id") or "")
            utility = _finite_or_none(decision.get("utility"))
            if strategy_id in strategy_ids and utility is not None:
                successful += 1
                current = by_strategy.setdefault(strategy_id, {"count": 0})
                current.update(decision)
                current["count"] = int(current.get("count") or 0) + 1
                current["symbol"] = symbol
                current["as_of"] = observed
            else:
                blocked += 1
            for reason in decision.get("reason_codes") or ():
                text = str(reason)
                reason_counts[text] = reason_counts.get(text, 0) + 1

    latest_reasons = [
        key for key, _count in sorted(reason_counts.items(), key=lambda item: (-item[1], item[0]))[:8]
    ]
    return {
        "available": True,
        "latest_at": latest_at,
        "symbols": len(symbols),
        "successful_decisions": successful,
        "blocked_decisions": blocked,
        "latest_reason_codes": latest_reasons,
        "by_strategy": by_strategy,
    }


def _tail_lines(path: Path, *, max_lines: int) -> deque[str]:
    block_size = 128 * 1024
    with path.open("rb") as handle:
        handle.seek(0, 2)
        position = handle.tell()
        chunks: list[bytes] = []
        newline_count = 0
        while position > 0 and newline_count <= max_lines:
            read_size = min(block_size, position)
            position -= read_size
            handle.seek(position)
            chunk = handle.read(read_size)
            chunks.append(chunk)
            newline_count += chunk.count(b"\n")
    raw_lines = b"".join(reversed(chunks)).splitlines()[-max_lines:]
    return deque(
        (line.decode("utf-8", errors="ignore") for line in raw_lines),
        maxlen=max_lines,
    )


def _same_family(source: str, target: str) -> bool:
    return any(source in family and target in family for family in (_MOMENTUM, _BREAKOUT, _REVERSION))


def _strategy_cluster(strategy_id: str) -> str:
    if strategy_id in _MOMENTUM:
        return "momentum"
    if strategy_id in _BREAKOUT:
        return "breakout"
    if strategy_id in _REVERSION:
        return "reversion"
    if "relative" in strategy_id:
        return "relative_strength"
    return "specialist"


def _finite_or_none(value: Any) -> float | None:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    return result if math.isfinite(result) else None


def _empty_payload(
    generated_at: str,
    reason: str,
    *,
    metadata: dict[str, Any] | None = None,
) -> dict[str, Any]:
    return {
        "schema": "strategy_rgcn_visualization_v1",
        "generated_at": generated_at,
        "source": {"kind": "trained_strategy_rgcn_checkpoint", "not_ontology_fact_graph": True},
        "model": {
            "available": False,
            "checkpoint_hash": (metadata or {}).get("checkpoint_hash"),
            "runtime_compatible": False,
            "runtime_reasons": [reason],
        },
        "inference": {
            "available": False,
            "latest_at": None,
            "symbols": 0,
            "successful_decisions": 0,
            "blocked_decisions": 0,
            "latest_reason_codes": [reason],
        },
        "clusters": [],
        "nodes": [],
        "links": [],
        "counts": {"nodes": 0, "links": 0},
    }
