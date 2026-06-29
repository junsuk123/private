from __future__ import annotations

import json
from collections import defaultdict
from contextlib import closing
from pathlib import Path
from typing import Any

from app.data.realtime_store import RealtimeMarketDataStore
from app.features.feature_schema import LIVE_SHORT_HORIZON_SCHEMA
from app.features.live_feature_frame import FeatureFrameError, LiveFeatureFrameBuilder
from app.models.live_model_trainer import train_live_short_horizon_model
from app.models.model_artifact_registry import ModelArtifactRegistry


def collect_live_feature_frames_from_realtime_store(
    *,
    db_path: str | Path = "data/store/realtime_market_data.sqlite3",
    journal_path: str | Path = "logs/live-feature-frames.jsonl",
    symbols: tuple[str, ...] | None = None,
) -> dict[str, Any]:
    db_path = Path(db_path)
    if not db_path.exists():
        return {"built": 0, "symbols": (), "errors": {"store": "REALTIME_STORE_MISSING"}}
    store = RealtimeMarketDataStore(db_path)
    target_symbols = symbols or _symbols_in_realtime_store(store)
    builder = LiveFeatureFrameBuilder(store, journal_path=journal_path)
    built = 0
    errors: dict[str, str] = {}
    for symbol in target_symbols:
        try:
            builder.build(symbol)
            built += 1
        except (FeatureFrameError, RuntimeError, ValueError) as exc:
            errors[symbol] = str(exc)
    return {"built": built, "symbols": tuple(target_symbols), "errors": errors}


def train_live_short_horizon_from_collected_features(
    *,
    journal_path: str | Path = "logs/live-feature-frames.jsonl",
    registry: ModelArtifactRegistry | None = None,
    minimum_examples: int = 30,
    minimum_positive_labels: int = 5,
    minimum_negative_labels: int = 5,
) -> dict[str, Any]:
    rows = build_live_training_rows_from_feature_journal(journal_path)
    artifact = train_live_short_horizon_model(
        rows,
        registry=registry,
        minimum_examples=minimum_examples,
        minimum_positive_labels=minimum_positive_labels,
        minimum_negative_labels=minimum_negative_labels,
        force_live_ineligible_reason=None if rows else "NO_COLLECTED_LIVE_FEATURE_FRAMES",
    )
    _annotate_saved_artifact(
        artifact,
        registry or ModelArtifactRegistry(),
        {
            "source": str(journal_path),
            "source_type": "collected_live_feature_frames",
            "row_count": len(rows),
            "label_rule": "next_collected_return_1m_after_costs_bps > 20",
            "schema_hash": LIVE_SHORT_HORIZON_SCHEMA.schema_hash,
        },
    )
    return artifact


def build_live_training_rows_from_feature_journal(journal_path: str | Path) -> list[dict[str, Any]]:
    frames = _load_feature_frames(journal_path)
    by_symbol: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for frame in frames:
        if frame.get("feature_schema_hash") != LIVE_SHORT_HORIZON_SCHEMA.schema_hash:
            continue
        values = frame.get("values")
        if not isinstance(values, dict):
            continue
        by_symbol[str(frame.get("symbol") or "")].append(frame)

    rows: list[dict[str, Any]] = []
    for symbol, symbol_frames in by_symbol.items():
        ordered = _dedupe_sorted_frames(symbol_frames)
        for current, nxt in zip(ordered, ordered[1:], strict=False):
            features = {
                name: float(current["values"].get(name, 0.0))
                for name in LIVE_SHORT_HORIZON_SCHEMA.feature_names
            }
            next_return_bps = float(nxt["values"].get("return_1m", 0.0)) * 10_000.0
            observed_cost_bps = max(0.0, float(current["values"].get("spread_bps", 0.0))) + 10.0
            forward_net_return_bps = next_return_bps - observed_cost_bps
            rows.append(
                {
                    "features": features,
                    "label": int(forward_net_return_bps > 20.0),
                    "forward_net_return_bps": forward_net_return_bps,
                    "ticker": symbol,
                    "as_of": str(current.get("decision_time") or ""),
                    "source": str(journal_path),
                }
            )
    return rows


def _load_feature_frames(journal_path: str | Path) -> list[dict[str, Any]]:
    path = Path(journal_path)
    if not path.exists():
        return []
    frames: list[dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        try:
            payload = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(payload, dict):
            frames.append(payload)
    return frames


def _symbols_in_realtime_store(store: RealtimeMarketDataStore) -> tuple[str, ...]:
    with closing(store._connect()) as conn:  # noqa: SLF001 - narrow internal query for pipeline orchestration.
        rows = conn.execute(
            """
            select symbol from realtime_ticks
            union
            select symbol from realtime_orderbook
            order by symbol
            """
        ).fetchall()
    return tuple(str(row[0]) for row in rows)


def _dedupe_sorted_frames(frames: list[dict[str, Any]]) -> list[dict[str, Any]]:
    seen: set[tuple[str, str]] = set()
    deduped: list[dict[str, Any]] = []
    for frame in sorted(frames, key=lambda item: str(item.get("decision_time") or "")):
        key = (
            str(frame.get("decision_time") or ""),
            json.dumps(frame.get("values") or {}, sort_keys=True, separators=(",", ":")),
        )
        if key in seen:
            continue
        seen.add(key)
        deduped.append(frame)
    return deduped


def _annotate_saved_artifact(artifact: dict[str, Any], registry: ModelArtifactRegistry, training_data: dict[str, Any]) -> None:
    paths = [registry.root / f"{artifact['artifact_id']}.json"]
    if artifact.get("live_eligible") is True:
        paths.append(registry.latest_path)
    for path in paths:
        if not path.exists():
            continue
        payload = json.loads(path.read_text(encoding="utf-8"))
        payload["training_data"] = training_data
        path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
