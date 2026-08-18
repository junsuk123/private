"""Reconstruct training examples from stored decisions and their resolved outcomes.

A decision row already carries everything the model saw: the contexts, the regime
probabilities, the micro features and the graph the snapshot was built from. This module
replays those rows into :class:`TrainingExample` objects so the model is trained on
exactly the inputs it will be served, rather than on a parallel feature pipeline that can
drift away from the live one.

Two rules make the labels honest
--------------------------------
* **Resolved only.** A decision whose outcome window has not closed is excluded, never
  imputed. ``horizon_minutes`` defines the window and the cutoff is applied against the
  decision's own timestamp.
* **Realised, not intended.** ``trade_quality`` comes from the fill, not from what the
  strategy hoped for. A decision that was gated and never filled has no trade label and
  contributes only its regime label.
"""

from __future__ import annotations

import json
from datetime import datetime, timedelta, timezone
from typing import Any, Mapping

from app.context.domestic_context import DomesticContext
from app.context.global_context import GlobalContext
from app.models.graph_snapshot import (
    GraphSnapshotBuilder,
    StockNodeObservation,
)
from app.models.temporal_hetero_gnn import REGIME_LABELS, TemporalHeteroGnnConfig
from app.models.temporal_hetero_gnn_training import TrainingExample
from app.storage.trading_state_store import (
    TradingStateStore,
    default_trading_state_store,
    iso_column,
)

__all__ = [
    "MINIMUM_TRAINING_EXAMPLES",
    "build_training_examples",
]

#: Below this, a checkpoint would be fitted to noise. Chosen as the point at which the
#: 13-label regime head has at least ~10 observations per label; publishing under it would
#: flip the runtime from OFFLINE to HEALTHY without the evidence to justify it.
MINIMUM_TRAINING_EXAMPLES = 150


def _loads(value: Any) -> dict[str, Any]:
    try:
        payload = json.loads(str(value or "{}"))
    except ValueError:
        return {}
    return payload if isinstance(payload, dict) else {}


def _parse(value: Any) -> datetime | None:
    if not value:
        return None
    try:
        moment = datetime.fromisoformat(str(value))
    except ValueError:
        return None
    return (moment if moment.tzinfo else moment.replace(tzinfo=timezone.utc)).astimezone(
        timezone.utc
    )


def build_training_examples(
    store: TradingStateStore | None = None,
    *,
    config: TemporalHeteroGnnConfig,
    since: datetime,
    horizon_minutes: int = 30,
    now: datetime | None = None,
    limit: int = 5000,
) -> tuple[TrainingExample, ...]:
    """Replay stored decisions into training examples.

    Rebuilds each decision's graph snapshot from its persisted contexts. The snapshot is
    reconstructed rather than stored as a tensor because the tensor layout is versioned
    with the model: a stored tensor would silently mismatch after a feature is appended,
    whereas a stored context reconstructs correctly against whatever the current layout is.
    """
    target = store or default_trading_state_store()
    moment = (now or datetime.now(timezone.utc)).astimezone(timezone.utc)
    cutoff = moment - timedelta(minutes=max(1, int(horizon_minutes)))
    builder = GraphSnapshotBuilder(
        max_nodes=config.max_nodes, time_steps=config.time_steps
    )

    rows = target.fetch_all(
        "select d.decision_id, d.decided_at, d.ticker, d.trace_json,"
        "       o.filled_quantity, o.average_fill_price, o.limit_price, o.side"
        " from strategy_decision d"
        " left join order_intent o on o.decision_id = d.decision_id"
        " where d.decided_at >= ? and d.decided_at <= ?"
        " order by d.decided_at limit ?",
        (iso_column(since), iso_column(cutoff), int(limit)),
    )

    examples: list[TrainingExample] = []
    for row in rows:
        trace = _loads(row["trace_json"])
        decided_at = _parse(row["decided_at"])
        if decided_at is None or not trace:
            continue
        snapshot = _rebuild_snapshot(builder, trace, decided_at)
        if snapshot is None:
            continue
        node_id = f"STOCK::{str(row['ticker']).upper()}"
        if snapshot.index_of(node_id) is None:
            continue
        regime_labels = {
            label: float(value)
            for label, value in (trace.get("regime_probabilities") or {}).items()
            if label in REGIME_LABELS
        }
        quality, realised = _outcome(row)
        examples.append(
            TrainingExample(
                snapshot=snapshot,
                node_id=node_id,
                regime_labels=regime_labels,
                trade_quality=quality,
                realised_return_bps=realised,
            )
        )
    return tuple(examples)


def _outcome(row: Mapping[str, Any]) -> tuple[float | None, float | None]:
    """Realised quality and return, or ``(None, None)`` for an unfilled decision."""
    filled = int(row["filled_quantity"] or 0)
    average = row["average_fill_price"]
    limit = row["limit_price"]
    if filled <= 0 or average is None or limit is None or float(average) <= 0.0:
        return None, None
    side = str(row["side"] or "BUY").upper()
    direction = -1.0 if side in {"SELL", "SHORT"} else 1.0
    edge_bps = direction * (float(limit) - float(average)) / float(average) * 10_000.0
    return (1.0 if edge_bps > 0.0 else 0.0), edge_bps


def _rebuild_snapshot(
    builder: GraphSnapshotBuilder, trace: Mapping[str, Any], decided_at: datetime
):
    """Rebuild the graph the decision was made on, from its stored contexts."""
    from app.context.sector_context import SectorContext
    from app.context.temporal_context import build_temporal_snapshot

    micro = trace.get("micro_context") or {}
    if not isinstance(micro, Mapping):
        return None
    ticker = str(trace.get("ticker") or "")
    if not ticker:
        return None
    sector_payload = trace.get("sector_context") or {}
    sector = str(sector_payload.get("sector") or "") or None

    observation = StockNodeObservation(
        ticker=ticker,
        sector=sector,
        venue="KRX",
        session_return=micro.get("session_return"),
        vwap_distance_bps=micro.get("vwap_distance_bps"),
        ema_gap_bps=micro.get("ema_gap_bps"),
        momentum=micro.get("momentum"),
        realized_volatility=micro.get("realized_volatility"),
        volume_intensity=micro.get("volume_intensity"),
        trade_intensity=micro.get("trade_intensity"),
        spread_bps=micro.get("spread_bps"),
        depth=micro.get("depth"),
        orderbook_imbalance=micro.get("orderbook_imbalance"),
        trade_imbalance=micro.get("trade_imbalance"),
        relative_strength=micro.get("relative_strength"),
        breakout_state=micro.get("breakout_state"),
    )
    temporal_payload = trace.get("temporal_context") or {}
    market_group = str(temporal_payload.get("market_group") or "KR")
    temporal = build_temporal_snapshot(market_group, decided_at)

    sectors: list[SectorContext] = []
    if sector:
        sectors.append(
            SectorContext(
                captured_at=decided_at,
                context_id=str(sector_payload.get("context_id") or f"replay-{sector}"),
                sector=sector,
                market_group=str(sector_payload.get("market_group") or market_group),
                sector_return=sector_payload.get("return"),
                breadth=sector_payload.get("breadth"),
                volume_z=sector_payload.get("volume_z"),
                volatility=sector_payload.get("volatility"),
                relative_strength=sector_payload.get("relative_strength"),
                foreign_flow=sector_payload.get("foreign_flow"),
                leader_strength=sector_payload.get("leader_strength"),
                leader_concentration=sector_payload.get("leader_concentration"),
                global_alignment=sector_payload.get("global_alignment"),
                confidence=float(sector_payload.get("confidence") or 0.0),
                member_count=int(sector_payload.get("member_count") or 0),
            )
        )

    return builder.build(
        captured_at=decided_at,
        temporal=temporal,
        global_context=_global_from(trace.get("global_context")),
        domestic_context=_domestic_from(trace.get("domestic_context")),
        sector_contexts=sectors,
        stocks=[observation],
        regime_prior=trace.get("regime_probabilities") or {},
    )


def _global_from(payload: Any) -> GlobalContext | None:
    if not isinstance(payload, Mapping) or not payload.get("context_id"):
        return None
    from app.context.global_context import GlobalGroupScore

    groups = {
        name: GlobalGroupScore(
            group=str(name),
            score=value.get("score"),
            raw_score=value.get("raw_score"),
            momentum=value.get("momentum"),
            level=value.get("level"),
            observed_members=tuple(value.get("observed_members") or ()),
            stale_members=tuple(value.get("stale_members") or ()),
            freshness=float(value.get("freshness") or 0.0),
            weight=float(value.get("weight") or 1.0),
        )
        for name, value in (payload.get("groups") or {}).items()
        if isinstance(value, Mapping)
    }
    captured = _parse(payload.get("captured_at"))
    if captured is None:
        return None
    return GlobalContext(
        captured_at=captured,
        context_id=str(payload["context_id"]),
        direction=payload.get("direction"),
        momentum=payload.get("momentum"),
        risk_sentiment=payload.get("risk_sentiment"),
        volatility=payload.get("volatility"),
        rates_pressure=payload.get("rates_pressure"),
        fx_pressure=payload.get("fx_pressure"),
        global_alignment=payload.get("global_alignment"),
        confidence=float(payload.get("confidence") or 0.0),
        groups=groups,
        coverage=float(payload.get("coverage") or 0.0),
    )


def _domestic_from(payload: Any) -> DomesticContext | None:
    if not isinstance(payload, Mapping) or not payload.get("context_id"):
        return None
    captured = _parse(payload.get("captured_at"))
    if captured is None:
        return None
    return DomesticContext(
        captured_at=captured,
        context_id=str(payload["context_id"]),
        global_context_id=payload.get("global_context_id"),
        direction=payload.get("direction"),
        breadth=payload.get("breadth"),
        liquidity=payload.get("liquidity"),
        volatility=payload.get("volatility"),
        flow=payload.get("flow"),
        leadership=payload.get("leadership"),
        venue_divergence=payload.get("venue_divergence"),
        confidence=float(payload.get("confidence") or 0.0),
        global_agreement=payload.get("global_agreement"),
        components=dict(payload.get("components") or {}),
    )
