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
DEFAULT_ACCOUNT_DASHBOARD_STORE_PATH = Path("data/store/account_dashboard.sqlite3")
DEFAULT_LABEL_MIN_FORWARD_SECONDS = 30.0
# Triple-barrier 라벨 기본값: 전략 청산 기준과 정렬(TP=take_profit 25bps, SL=stop_loss 100bps,
# 지평=장중 보유창 10분). "30초 뒤 첫 프레임 수익률" 단일 라벨은 노이즈(std~120bps)에 압도돼
# 모델이 붕괴했다. 경로가 TP/SL 중 무엇을 먼저 터치하는지로 라벨링하면 신호가 살아난다.
# 지평은 실데이터 시간기반 홀드아웃 스윕으로 선정: 600s에서 AUC 0.563/상위픽 +51bps로 적격,
# 900s 이상에서는 노이즈가 커져 부적격이 됐다.
DEFAULT_LABEL_HORIZON_SECONDS = 600.0
DEFAULT_LABEL_TAKE_PROFIT_BPS = 25.0
DEFAULT_LABEL_STOP_LOSS_BPS = 100.0
MIN_TRIPLE_BARRIER_PATH_POINTS = 2


def _label_min_net_return_bps() -> float:
    # 가격 경로가 없을 때(테스트/스토어 미비) 쓰는 폴백 단일-전방 라벨의 임계값(bps).
    try:
        return float(os.getenv("LIVE_LABEL_MIN_NET_RETURN_BPS", "5.0"))
    except (TypeError, ValueError):
        return 5.0


def _label_horizon_seconds() -> float:
    try:
        return max(60.0, float(os.getenv("LIVE_LABEL_HORIZON_SECONDS", str(DEFAULT_LABEL_HORIZON_SECONDS))))
    except (TypeError, ValueError):
        return DEFAULT_LABEL_HORIZON_SECONDS


def _label_take_profit_bps() -> float:
    try:
        return abs(float(os.getenv("LIVE_LABEL_TAKE_PROFIT_BPS", str(DEFAULT_LABEL_TAKE_PROFIT_BPS))))
    except (TypeError, ValueError):
        return DEFAULT_LABEL_TAKE_PROFIT_BPS


def _label_stop_loss_bps() -> float:
    try:
        return abs(float(os.getenv("LIVE_LABEL_STOP_LOSS_BPS", str(DEFAULT_LABEL_STOP_LOSS_BPS))))
    except (TypeError, ValueError):
        return DEFAULT_LABEL_STOP_LOSS_BPS


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
            "trade_feedback": _trade_event_stats(DEFAULT_ACCOUNT_DASHBOARD_STORE_PATH),
            "label_rule": (
                f"triple_barrier tp={_label_take_profit_bps()}bps sl={_label_stop_loss_bps()}bps "
                f"horizon={_label_horizon_seconds()}s net_of_costs "
                f"(fallback: forward_return_after_30s > {_label_min_net_return_bps()}bps)"
            ),
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
        "trade_feedback": _trade_event_stats(DEFAULT_ACCOUNT_DASHBOARD_STORE_PATH),
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

    horizon_seconds = _label_horizon_seconds()
    take_profit_bps = _label_take_profit_bps()
    stop_loss_bps = _label_stop_loss_bps()

    rows: list[dict[str, Any]] = []
    for symbol, symbol_frames in by_symbol.items():
        ordered = _dedupe_sorted_frames(symbol_frames)
        times = [_parse_frame_time(frame) for frame in ordered]
        prices = [_frame_mark_price(frame, price_lookup) for frame in ordered]
        for index, current in enumerate(ordered):
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
            labelled = _triple_barrier_label(
                ordered,
                times,
                prices,
                index,
                horizon_seconds=horizon_seconds,
                take_profit_bps=take_profit_bps,
                stop_loss_bps=stop_loss_bps,
            )
            if labelled is None:
                # 가격 경로가 없으면(테스트/스토어 미비) 기존 단일-전방 라벨로 후퇴한다.
                labelled = _legacy_forward_label(ordered, index, price_lookup)
            if labelled is None:
                continue
            label, forward_net_return_bps, gross_forward_return_bps, label_source = labelled
            rows.append(
                {
                    "features": features,
                    "label": label,
                    "forward_net_return_bps": forward_net_return_bps,
                    "gross_forward_return_bps": gross_forward_return_bps,
                    "label_source": label_source,
                    "ticker": symbol,
                    "as_of": str(current.get("decision_time") or ""),
                    "source": str(journal_path),
                }
            )
    return rows


def _trade_event_stats(db_path: str | Path) -> dict[str, Any]:
    path = Path(db_path)
    if not path.exists():
        return {
            "store_exists": False,
            "store_path": str(path),
            "events_total": 0,
            "filled_events": 0,
            "latest_event_at": None,
            "latest_filled_at": None,
        }
    try:
        with closing(sqlite3.connect(path, timeout=5.0)) as conn:
            conn.execute("pragma busy_timeout = 5000")
            total_row = conn.execute("select count(*) from trade_events").fetchone()
            filled_row = conn.execute(
                """
                select count(*), max(occurred_at)
                from trade_events
                where filled_quantity > 0
                   or upper(coalesce(order_status, '')) in ('FILLED', 'PARTIALLY_FILLED')
                """
            ).fetchone()
            latest_row = conn.execute("select max(occurred_at) from trade_events").fetchone()
    except sqlite3.Error as exc:
        return {
            "store_exists": True,
            "store_path": str(path),
            "events_total": 0,
            "filled_events": 0,
            "latest_event_at": None,
            "latest_filled_at": None,
            "error": str(exc),
        }
    return {
        "store_exists": True,
        "store_path": str(path),
        "events_total": int((total_row or (0,))[0] or 0),
        "filled_events": int((filled_row or (0, None))[0] or 0),
        "latest_event_at": (latest_row or (None,))[0],
        "latest_filled_at": (filled_row or (0, None))[1],
    }


def _observed_cost_bps(frame: dict[str, Any]) -> float:
    try:
        spread_bps = max(0.0, float(frame["values"].get("spread_bps", 0.0)))
    except (TypeError, ValueError, KeyError):
        spread_bps = 0.0
    return spread_bps + 10.0


def _triple_barrier_label(
    ordered: list[dict[str, Any]],
    times: list[datetime | None],
    prices: list[float | None],
    index: int,
    *,
    horizon_seconds: float,
    take_profit_bps: float,
    stop_loss_bps: float,
) -> tuple[int, float, float, str] | None:
    """전방 지평(horizon) 안에서 순수익 경로가 +TP를 먼저 터치하면 1, -SL을 먼저 터치하면 0.

    어느 배리어도 터치하지 않으면 지평 종단 순수익의 부호로 라벨링한다. 라벨은 미래 가격을
    사용하지만 피처는 as-of 시점 값만 쓰므로 룩어헤드가 아니다.
    """
    entry_time = times[index]
    entry_price = prices[index]
    if entry_time is None or entry_price is None or entry_price <= 0:
        return None
    cost_bps = _observed_cost_bps(ordered[index])
    horizon_cutoff = entry_time + timedelta(seconds=horizon_seconds)
    forward_prices: list[float] = []
    for j in range(index + 1, len(ordered)):
        candidate_time = times[j]
        if candidate_time is None:
            continue
        if candidate_time > horizon_cutoff:
            break
        candidate_price = prices[j]
        if candidate_price is not None and candidate_price > 0:
            forward_prices.append(candidate_price)
    if len(forward_prices) < MIN_TRIPLE_BARRIER_PATH_POINTS:
        return None
    for price in forward_prices:
        gross_bps = (price / entry_price - 1.0) * 10_000.0
        net_bps = gross_bps - cost_bps
        if net_bps >= take_profit_bps:
            return (1, net_bps, gross_bps, "triple_barrier_take_profit")
        if net_bps <= -stop_loss_bps:
            return (0, net_bps, gross_bps, "triple_barrier_stop_loss")
    terminal_gross_bps = (forward_prices[-1] / entry_price - 1.0) * 10_000.0
    terminal_net_bps = terminal_gross_bps - cost_bps
    return (int(terminal_net_bps > 0.0), terminal_net_bps, terminal_gross_bps, "triple_barrier_terminal")


def _legacy_forward_label(
    ordered: list[dict[str, Any]],
    index: int,
    price_lookup: "_FramePriceLookup",
) -> tuple[int, float, float, str] | None:
    nxt = _next_frame_after_minimum_horizon(
        ordered,
        index,
        minimum_forward_seconds=DEFAULT_LABEL_MIN_FORWARD_SECONDS,
    )
    if nxt is None:
        return None
    current_price = _frame_mark_price(ordered[index], price_lookup)
    future_price = _frame_mark_price(nxt, price_lookup)
    if current_price is not None and future_price is not None and current_price > 0:
        gross_forward_return_bps = (future_price / current_price - 1.0) * 10_000.0
        label_source = "forward_mark_price"
    else:
        gross_forward_return_bps = float(nxt["values"].get("return_1m", 0.0)) * 10_000.0
        label_source = "fallback_next_return_1m"
    forward_net_return_bps = gross_forward_return_bps - _observed_cost_bps(ordered[index])
    return (
        int(forward_net_return_bps > _label_min_net_return_bps()),
        forward_net_return_bps,
        gross_forward_return_bps,
        label_source,
    )


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
