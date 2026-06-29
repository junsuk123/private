from __future__ import annotations

import json
from collections import defaultdict
from contextlib import closing
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

from app.data.realtime_store import RealtimeMarketDataStore
from app.features.feature_schema import LIVE_SHORT_HORIZON_SCHEMA
from app.features.live_feature_frame import FeatureFrameError, LiveFeatureFrameBuilder
from app.models.live_model_trainer import train_live_short_horizon_model
from app.models.model_artifact_registry import ModelArtifactRegistry


DEFAULT_REALTIME_STORE_PATH = Path("data/store/realtime_market_data.sqlite3")
DEFAULT_FEATURE_JOURNAL_PATH = Path("logs/live-feature-frames.jsonl")
DEFAULT_LABEL_MIN_FORWARD_SECONDS = 30.0


def collect_live_feature_frames_from_realtime_store(
    *,
    db_path: str | Path = DEFAULT_REALTIME_STORE_PATH,
    journal_path: str | Path = DEFAULT_FEATURE_JOURNAL_PATH,
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
    journal_path: str | Path = DEFAULT_FEATURE_JOURNAL_PATH,
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
            "label_rule": "first_collected_return_1m_after_30s_after_costs_bps > 20",
            "schema_hash": LIVE_SHORT_HORIZON_SCHEMA.schema_hash,
        },
    )
    return artifact


def live_training_status(
    *,
    db_path: str | Path = DEFAULT_REALTIME_STORE_PATH,
    journal_path: str | Path = DEFAULT_FEATURE_JOURNAL_PATH,
    registry: ModelArtifactRegistry | None = None,
) -> dict[str, Any]:
    db_path = Path(db_path)
    journal_path = Path(journal_path)
    registry = registry or ModelArtifactRegistry()
    rows = build_live_training_rows_from_feature_journal(journal_path)
    latest_ineligible = _latest_saved_artifact(registry)
    return {
        "realtime_store_exists": db_path.exists(),
        "realtime_store_path": str(db_path),
        "feature_journal_exists": journal_path.exists(),
        "feature_journal_path": str(journal_path),
        "feature_frame_lines": _line_count(journal_path),
        "training_rows": len(rows),
        "latest_live_eligible_exists": registry.latest_path.exists(),
        "latest_ineligible_artifact": latest_ineligible,
    }


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
        for index, current in enumerate(ordered):
            nxt = _next_frame_after_minimum_horizon(
                ordered,
                index,
                minimum_forward_seconds=DEFAULT_LABEL_MIN_FORWARD_SECONDS,
            )
            if nxt is None:
                continue
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


def _next_frame_after_minimum_horizon(
    frames: list[dict[str, Any]],
    index: int,
    *,
    minimum_forward_seconds: float,
) -> dict[str, Any] | None:
    current_time = _parse_frame_time(frames[index])
    if current_time is None:
        return frames[index + 1] if index + 1 < len(frames) else None
    cutoff = current_time + timedelta(seconds=minimum_forward_seconds)
    for candidate in frames[index + 1 :]:
        candidate_time = _parse_frame_time(candidate)
        if candidate_time is None or candidate_time >= cutoff:
            return candidate
    return None


def _parse_frame_time(frame: dict[str, Any]) -> datetime | None:
    value = frame.get("decision_time")
    if not value:
        return None
    try:
        return datetime.fromisoformat(str(value).replace("Z", "+00:00"))
    except ValueError:
        return None


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


def _line_count(path: Path) -> int:
    if not path.exists():
        return 0
    return sum(1 for line in path.read_text(encoding="utf-8").splitlines() if line.strip())


def _latest_saved_artifact(registry: ModelArtifactRegistry) -> dict[str, Any] | None:
    candidates = sorted(
        (path for path in registry.root.glob("live_short_horizon.*.json") if path.is_file()),
        key=lambda path: path.stat().st_mtime,
        reverse=True,
    )
    if not candidates:
        return None
    try:
        payload = json.loads(candidates[0].read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {"path": str(candidates[0]), "readable": False}
    return {
        "artifact_id": str(payload.get("artifact_id") or candidates[0].stem),
        "path": str(candidates[0]),
        "live_eligible": bool(payload.get("live_eligible")),
        "reason_codes": tuple(str(item) for item in payload.get("reason_codes") or ()),
        "example_count": int(float((payload.get("metrics") or {}).get("example_count") or 0)),
        "training_rows": int((payload.get("training_data") or {}).get("row_count") or 0),
    }
