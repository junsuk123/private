from __future__ import annotations

import json
import math
import os
import sqlite3
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


def _label_min_net_return_bps() -> float:
    # 단타용으로 라벨을 완화: 비용 차감 후 순수익이 이 값(bps) 초과면 positive.
    # 기존 20bps는 너무 빡빡해 positive가 ~1%뿐이라 모델이 붕괴했다.
    try:
        return float(os.getenv("LIVE_LABEL_MIN_NET_RETURN_BPS", "5.0"))
    except (TypeError, ValueError):
        return 5.0


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
    target_symbols = _symbols_in_realtime_store(store) if symbols is None else tuple(symbols)
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
    rows = build_live_training_rows_from_feature_journal(journal_path, db_path=DEFAULT_REALTIME_STORE_PATH)
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
            "label_rule": f"forward_mark_price_return_after_30s_after_costs_bps > {_label_min_net_return_bps()}",
            "schema_hash": LIVE_SHORT_HORIZON_SCHEMA.schema_hash,
            "row_quality": _row_quality_summary(rows),
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
    rows = build_live_training_rows_from_feature_journal(journal_path, db_path=db_path)
    latest_saved = _latest_saved_artifact(registry)
    latest_live_eligible = _live_eligible_artifact(registry)
    return {
        "realtime_store_exists": db_path.exists(),
        "realtime_store_path": str(db_path),
        "feature_journal_exists": journal_path.exists(),
        "feature_journal_path": str(journal_path),
        "feature_frame_lines": _line_count(journal_path),
        "training_rows": len(rows),
        "latest_live_eligible_exists": registry.latest_path.exists(),
        "latest_live_eligible_artifact": latest_live_eligible,
        "latest_saved_artifact": latest_saved,
        "latest_ineligible_artifact": latest_saved if latest_saved and not latest_saved.get("live_eligible") else None,
    }


def build_live_training_rows_from_feature_journal(
    journal_path: str | Path,
    *,
    db_path: str | Path = DEFAULT_REALTIME_STORE_PATH,
) -> list[dict[str, Any]]:
    frames = _load_feature_frames(journal_path)
    price_lookup = _FramePriceLookup(db_path, frames)
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
            if not _frame_passes_training_quality(current):
                continue
            try:
                features = {
                    name: float(current["values"].get(name, 0.0))
                    for name in LIVE_SHORT_HORIZON_SCHEMA.feature_names
                }
            except (TypeError, ValueError):
                continue
            if any(not math.isfinite(value) for value in features.values()):
                continue
            current_price = _frame_mark_price(current, price_lookup)
            future_price = _frame_mark_price(nxt, price_lookup)
            if current_price is not None and future_price is not None and current_price > 0:
                gross_forward_return_bps = (future_price / current_price - 1.0) * 10_000.0
                label_source = "forward_mark_price"
            else:
                gross_forward_return_bps = float(nxt["values"].get("return_1m", 0.0)) * 10_000.0
                label_source = "fallback_next_return_1m"
            observed_cost_bps = max(0.0, float(current["values"].get("spread_bps", 0.0))) + 10.0
            forward_net_return_bps = gross_forward_return_bps - observed_cost_bps
            rows.append(
                {
                    "features": features,
                    "label": int(forward_net_return_bps > _label_min_net_return_bps()),
                    "forward_net_return_bps": forward_net_return_bps,
                    "gross_forward_return_bps": gross_forward_return_bps,
                    "label_source": label_source,
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


def _frame_passes_training_quality(frame: dict[str, Any]) -> bool:
    values = frame.get("values")
    if not isinstance(values, dict):
        return False
    max_spread_bps = _env_float("LIVE_TRAINING_MAX_SPREAD_BPS", 80.0)
    max_cost_to_vol = _env_float("LIVE_TRAINING_MAX_COST_TO_VOLATILITY_RATIO", 5_000.0)
    try:
        spread_bps = float(values.get("spread_bps", 0.0))
        bid_depth = float(values.get("bid_depth", 0.0))
        ask_depth = float(values.get("ask_depth", 0.0))
        cost_to_vol = float(values.get("cost_to_volatility_ratio", 0.0))
    except (TypeError, ValueError):
        return False
    if not all(math.isfinite(value) for value in (spread_bps, bid_depth, ask_depth, cost_to_vol)):
        return False
    return spread_bps <= max_spread_bps and bid_depth > 0 and ask_depth > 0 and cost_to_vol <= max_cost_to_vol


def _env_float(name: str, default: float) -> float:
    try:
        return float(os.getenv(name, str(default)))
    except (TypeError, ValueError):
        return default


def _frame_mark_price(frame: dict[str, Any], lookup: "_FramePriceLookup") -> float | None:
    direct = frame.get("mark_price")
    try:
        price = float(direct)
    except (TypeError, ValueError):
        price = 0.0
    if math.isfinite(price) and price > 0:
        return price
    record_ids = frame.get("source_record_ids")
    if not isinstance(record_ids, list | tuple):
        return None
    for record_id in reversed(record_ids):
        price = lookup.price_for(str(record_id))
        if price is not None:
            return price
    return None


class _FramePriceLookup:
    def __init__(self, db_path: str | Path, frames: list[dict[str, Any]]) -> None:
        self._prices: dict[str, float] = {}
        path = Path(db_path)
        if not path.exists():
            return
        record_ids: set[str] = set()
        for frame in frames:
            ids = frame.get("source_record_ids")
            if isinstance(ids, list | tuple):
                record_ids.update(str(record_id) for record_id in ids if record_id)
        if not record_ids:
            return
        try:
            with sqlite3.connect(path) as conn:
                self._load_tick_prices(conn, record_ids)
                missing = record_ids - set(self._prices)
                if missing:
                    self._load_orderbook_mid_prices(conn, missing)
        except sqlite3.Error:
            self._prices = {}

    def price_for(self, record_id: str) -> float | None:
        return self._prices.get(record_id)

    def _load_tick_prices(self, conn: sqlite3.Connection, record_ids: set[str]) -> None:
        for chunk in _chunks(tuple(record_ids), 800):
            placeholders = ",".join("?" for _ in chunk)
            rows = conn.execute(
                f"select record_id, price from realtime_ticks where record_id in ({placeholders})",
                chunk,
            ).fetchall()
            for record_id, price in rows:
                value = float(price)
                if math.isfinite(value) and value > 0:
                    self._prices[str(record_id)] = value

    def _load_orderbook_mid_prices(self, conn: sqlite3.Connection, record_ids: set[str]) -> None:
        for chunk in _chunks(tuple(record_ids), 800):
            placeholders = ",".join("?" for _ in chunk)
            rows = conn.execute(
                f"select record_id, best_bid, best_ask from realtime_orderbook where record_id in ({placeholders})",
                chunk,
            ).fetchall()
            for record_id, bid, ask in rows:
                bid_value = float(bid)
                ask_value = float(ask)
                if math.isfinite(bid_value) and math.isfinite(ask_value) and bid_value > 0 and ask_value > 0:
                    self._prices[str(record_id)] = (bid_value + ask_value) / 2.0


def _chunks(values: tuple[str, ...], size: int) -> list[tuple[str, ...]]:
    return [values[index : index + size] for index in range(0, len(values), size)]


def _row_quality_summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    by_source: dict[str, int] = defaultdict(int)
    for row in rows:
        by_source[str(row.get("label_source") or "unknown")] += 1
    returns = [float(row.get("forward_net_return_bps", 0.0)) for row in rows]
    return {
        "label_sources": dict(sorted(by_source.items())),
        "avg_forward_net_return_bps": sum(returns) / len(returns) if returns else 0.0,
        "max_forward_net_return_bps": max(returns) if returns else 0.0,
        "min_forward_net_return_bps": min(returns) if returns else 0.0,
    }


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
        "metrics": payload.get("metrics") or {},
    }


def _live_eligible_artifact(registry: ModelArtifactRegistry) -> dict[str, Any] | None:
    if not registry.latest_path.exists():
        return None
    try:
        payload = json.loads(registry.latest_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {"path": str(registry.latest_path), "readable": False}
    return {
        "artifact_id": str(payload.get("artifact_id") or registry.latest_path.stem),
        "path": str(registry.latest_path),
        "live_eligible": bool(payload.get("live_eligible")),
        "reason_codes": tuple(str(item) for item in payload.get("reason_codes") or ()),
        "example_count": int(float((payload.get("metrics") or {}).get("example_count") or 0)),
        "training_rows": int((payload.get("training_data") or {}).get("row_count") or 0),
        "metrics": payload.get("metrics") or {},
    }
