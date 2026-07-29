from __future__ import annotations

import json
import hashlib
import math
import os
import sqlite3
import threading
from collections import defaultdict
from contextlib import closing
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

from app.data.realtime_store import RealtimeMarketDataStore
from app.features.feature_schema import LIVE_SHORT_HORIZON_SCHEMA
from app.features.live_feature_frame import FeatureFrameError, LiveFeatureFrameBuilder
from app.models.live_model_trainer import train_live_short_horizon_model
from app.models.model_artifact_registry import ModelArtifactRegistry, _atomic_write_text


DEFAULT_REALTIME_STORE_PATH = Path("data/store/realtime_market_data.sqlite3")
DEFAULT_FEATURE_JOURNAL_PATH = Path("logs/live-feature-frames.jsonl")
DEFAULT_ACCOUNT_DASHBOARD_STORE_PATH = Path("data/store/account_dashboard.sqlite3")
DEFAULT_TRAINING_ROW_STORE_PATH = Path("data/store/live_training_rows.sqlite3")
DEFAULT_NEWS_TRUST_PATH = Path("data/store/news_trust.json")
DEFAULT_LABEL_MIN_FORWARD_SECONDS = 30.0
# Triple-barrier 라벨 기본값: 전략 청산 기준과 정렬(TP=take_profit 25bps, SL=stop_loss 100bps,
# 지평=장중 보유창 10분). "30초 뒤 첫 프레임 수익률" 단일 라벨은 노이즈(std~120bps)에 압도돼
# 모델이 붕괴했다. 경로가 TP/SL 중 무엇을 먼저 터치하는지로 라벨링하면 신호가 살아난다.
# 지평은 실데이터 시간기반 홀드아웃 스윕으로 선정: 600s에서 AUC 0.563/상위픽 +51bps로 적격,
# 900s 이상에서는 노이즈가 커져 부적격이 됐다.
DEFAULT_LABEL_HORIZON_SECONDS = 600.0
DEFAULT_LABEL_TAKE_PROFIT_BPS = 25.0
DEFAULT_LABEL_STOP_LOSS_BPS = 100.0
# The prediction path still consumes every live tick. Only labelled training
# observations are thinned: with a 600-second label horizon, sub-15-second rows
# overlap almost completely and let one bursty symbol dominate the fit.
DEFAULT_TRAINING_MIN_ROW_SPACING_SECONDS = 15.0
MIN_TRIPLE_BARRIER_PATH_POINTS = 2
TRAINING_RECIPE_VERSION = "canonical_observation_thinned_symbol_temporal_holdout_v3"
INCREMENTAL_TRAINING_STATE_VERSION = 1
_LIVE_TRAINING_LOCK = threading.Lock()
_FEATURE_FRAME_CACHE_LOCK = threading.Lock()
_FEATURE_FRAME_CACHE: dict[str, Any] = {
    "path": None,
    "offset": 0,
    "mtime_ns": None,
    "head_digest": None,
    "frames": [],
}


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


def backfill_live_feature_frames_from_realtime_store(
    *,
    db_path: str | Path = DEFAULT_REALTIME_STORE_PATH,
    journal_path: str | Path = DEFAULT_FEATURE_JOURNAL_PATH,
    stride_seconds: int = 5,
    maximum_frames: int = 1_200,
    lookback_hours: int = 8,
) -> dict[str, Any]:
    """Materialize causal as-of frames from retained KIS ticks and books.

    This closes the five-minute sampling bottleneck without inventing observations:
    every frame is rebuilt only from records received by its decision timestamp.
    """
    database = Path(db_path)
    if not database.exists():
        return {"built": 0, "attempted": 0, "errors": {"store": "REALTIME_STORE_MISSING"}}
    stride_seconds = max(
        1,
        int(_env_float("LIVE_TRAINING_BACKFILL_STRIDE_SECONDS", stride_seconds)),
    )
    maximum_frames = max(
        1,
        int(_env_float("LIVE_TRAINING_BACKFILL_MAX_FRAMES", maximum_frames)),
    )
    lookback_hours = max(
        1,
        int(_env_float("LIVE_TRAINING_BACKFILL_LOOKBACK_HOURS", lookback_hours)),
    )
    journal = Path(journal_path)
    existing = {
        (str(frame.get("symbol") or ""), str(frame.get("decision_time") or ""))
        for frame in _load_feature_frames(journal)
        if frame.get("feature_schema_hash") == LIVE_SHORT_HORIZON_SCHEMA.schema_hash
    }
    if len(existing) >= maximum_frames:
        return {
            "built": 0,
            "attempted": 0,
            "symbols": (),
            "errors": {},
            "reason": "BACKFILL_TARGET_REACHED",
            "compatible_frames": len(existing),
        }
    remaining_frames = maximum_frames - len(existing)
    cutoff = (datetime.now(timezone.utc) - timedelta(hours=max(1, lookback_hours))).isoformat()
    candidates: list[tuple[str, str]] = []
    with closing(sqlite3.connect(database)) as conn:
        rows = conn.execute(
            """
            select symbol, received_at
            from realtime_orderbook
            where received_at >= ?
              and source = 'kis_realtime_websocket'
              and length(symbol) = 6
              and symbol not glob '*[^0-9]*'
            order by received_at desc
            limit ?
            """,
            (cutoff, max(10_000, maximum_frames * max(3, stride_seconds))),
        ).fetchall()
    seen_buckets: set[tuple[str, int]] = set()
    for symbol, raw_time in reversed(rows):
        moment = _parse_iso_time(str(raw_time))
        if moment is None:
            continue
        bucket = int(moment.timestamp() // max(1, stride_seconds))
        key = (str(symbol), bucket)
        if key in seen_buckets:
            continue
        seen_buckets.add(key)
        decision_time = moment.isoformat()
        if (str(symbol), decision_time) not in existing:
            candidates.append((str(symbol), decision_time))
    candidates = candidates[-max(1, remaining_frames):]
    builder = LiveFeatureFrameBuilder(
        RealtimeMarketDataStore(database),
        journal_path=journal,
    )
    built = 0
    errors: dict[str, int] = defaultdict(int)
    for symbol, raw_time in candidates:
        moment = _parse_iso_time(raw_time)
        if moment is None:
            continue
        try:
            builder.build(symbol, decision_time=moment)
            built += 1
        except (FeatureFrameError, RuntimeError, ValueError) as exc:
            errors[str(exc).split(":", 1)[0]] += 1
    return {
        "built": built,
        "attempted": len(candidates),
        "symbols": tuple(sorted({symbol for symbol, _ in candidates})),
        "errors": dict(errors),
    }


def train_live_short_horizon_from_collected_features(
    *,
    journal_path: str | Path = DEFAULT_FEATURE_JOURNAL_PATH,
    registry: ModelArtifactRegistry | None = None,
    minimum_examples: int = 30,
    minimum_positive_labels: int = 5,
    minimum_negative_labels: int = 5,
    training_row_store_path: str | Path | None = None,
) -> dict[str, Any]:
    # The collection worker and the dedicated periodic trainer can reach this
    # function at the same time. Serialize the expensive fit so they cannot train
    # duplicate challengers concurrently; the follower will then hit the dataset
    # fingerprint fast path.
    with _LIVE_TRAINING_LOCK:
        return _train_live_short_horizon_from_collected_features_unlocked(
            journal_path=journal_path,
            registry=registry,
            minimum_examples=minimum_examples,
            minimum_positive_labels=minimum_positive_labels,
            minimum_negative_labels=minimum_negative_labels,
            training_row_store_path=training_row_store_path,
        )


def _train_live_short_horizon_from_collected_features_unlocked(
    *,
    journal_path: str | Path,
    registry: ModelArtifactRegistry | None,
    minimum_examples: int,
    minimum_positive_labels: int,
    minimum_negative_labels: int,
    training_row_store_path: str | Path | None,
) -> dict[str, Any]:
    _prune_news_sentiment_store()
    fresh_rows = build_live_training_rows_from_feature_journal(
        journal_path,
        db_path=DEFAULT_REALTIME_STORE_PATH,
    )
    row_store_path = _training_row_store_path(journal_path, training_row_store_path)
    materialized_rows, row_merge = _merge_materialized_training_rows(
        row_store_path,
        fresh_rows,
    )
    rows = _thin_training_rows(materialized_rows)
    registry = registry or ModelArtifactRegistry()
    dataset_fingerprint = _training_rows_fingerprint(rows)
    previous = _latest_saved_payload(registry)
    previous_training = (previous or {}).get("training_data") or {}
    data_format_signature = _incremental_data_format_signature()
    previous_state = dict((previous or {}).get("training_state") or {})
    incremental_compatible = bool(
        previous
        and int(previous_state.get("format_version") or 0)
        == INCREMENTAL_TRAINING_STATE_VERSION
        and previous_state.get("data_format_signature") == data_format_signature
        and previous.get("feature_schema_hash") == LIVE_SHORT_HORIZON_SCHEMA.schema_hash
        and tuple(previous.get("feature_names") or ())
        == LIVE_SHORT_HORIZON_SCHEMA.feature_names
        and (previous.get("classification") or {}).get("family")
        == "logistic_regression_sgd"
        and (previous.get("regression") or {}).get("family")
        == "linear_regression_sgd"
    )
    if (
        incremental_compatible
        and dataset_fingerprint
        and previous_training.get("dataset_fingerprint") == dataset_fingerprint
    ):
        return {
            **previous,
            "training_skipped": True,
            "skip_reason": "UNCHANGED_LABELLED_DATASET",
        }
    changed_keys = set(row_merge.get("changed_keys") or ())
    incremental_rows = [
        row for row in rows if _training_row_key(row) in changed_keys
    ]
    training_mode = "incremental" if incremental_compatible else "full"
    full_retrain_reason = None
    if not incremental_compatible:
        full_retrain_reason = (
            "NO_COMPATIBLE_PARENT_STATE"
            if previous
            else "FIRST_MODEL"
        )
    elif not incremental_rows:
        # A changed thinned dataset without a directly selected changed row can
        # occur when a newer observation replaces the representative of a time
        # bucket. Continue from the parent with that bucket representative.
        previous_ids = set(previous_state.get("trained_observation_ids") or ())
        incremental_rows = [
            row
            for row in rows
            if _training_row_key(row) not in previous_ids
        ]
    if training_mode == "incremental" and not incremental_rows:
        return {
            **previous,
            "training_skipped": True,
            "skip_reason": "NO_EFFECTIVE_INCREMENTAL_ROWS",
        }
    update_news_trust_from_rows(rows)
    artifact = train_live_short_horizon_model(
        rows,
        registry=registry,
        minimum_examples=minimum_examples,
        minimum_positive_labels=minimum_positive_labels,
        minimum_negative_labels=minimum_negative_labels,
        force_live_ineligible_reason=None if rows else "NO_COLLECTED_LIVE_FEATURE_FRAMES",
        warm_start_artifact=previous if training_mode == "incremental" else None,
        update_rows=incremental_rows if training_mode == "incremental" else None,
        training_state={
            "format_version": INCREMENTAL_TRAINING_STATE_VERSION,
            "data_format_signature": data_format_signature,
            "mode": training_mode,
            "full_retrain_reason": full_retrain_reason,
            "parent_artifact_id": (
                previous.get("artifact_id") if training_mode == "incremental" else None
            ),
            "incremental_example_count": (
                len(incremental_rows) if training_mode == "incremental" else len(rows)
            ),
            "cumulative_example_count": len(rows),
            "trained_observation_ids": [
                _training_row_key(row) for row in rows
            ],
        },
    )
    _annotate_saved_artifact(
        artifact,
        registry,
        {
            "source": str(journal_path),
            "source_type": "collected_live_feature_frames",
            "row_count": len(rows),
            "materialized_row_count": len(materialized_rows),
            "fresh_row_count": len(fresh_rows),
            "new_materialized_row_count": row_merge["new_rows"],
            "duplicate_rows_removed": row_merge["duplicate_rows_removed"],
            "invalid_rows_removed": row_merge["invalid_rows_removed"],
            "minimum_row_spacing_seconds": _training_min_row_spacing_seconds(),
            "training_mode": training_mode,
            "full_retrain_reason": full_retrain_reason,
            "parent_artifact_id": (
                previous.get("artifact_id") if training_mode == "incremental" else None
            ),
            "incremental_row_count": (
                len(incremental_rows) if training_mode == "incremental" else len(rows)
            ),
            "materialized_row_store": str(row_store_path),
            "dataset_fingerprint": dataset_fingerprint,
            "trade_feedback": _trade_event_stats(DEFAULT_ACCOUNT_DASHBOARD_STORE_PATH),
            "label_rule": (
                f"triple_barrier tp={_label_take_profit_bps()}bps sl={_label_stop_loss_bps()}bps "
                f"horizon={_label_horizon_seconds()}s net_of_costs "
                f"(fallback: forward_return_after_30s > {_label_min_net_return_bps()}bps)"
            ),
            "schema_hash": LIVE_SHORT_HORIZON_SCHEMA.schema_hash,
            "training_recipe_version": TRAINING_RECIPE_VERSION,
            "row_quality": _row_quality_summary(rows),
        },
    )
    return artifact


def live_training_status(
    *,
    db_path: str | Path = DEFAULT_REALTIME_STORE_PATH,
    journal_path: str | Path = DEFAULT_FEATURE_JOURNAL_PATH,
    registry: ModelArtifactRegistry | None = None,
    training_row_store_path: str | Path | None = None,
) -> dict[str, Any]:
    db_path = Path(db_path)
    journal_path = Path(journal_path)
    registry = registry or ModelArtifactRegistry()
    row_store_path = _training_row_store_path(journal_path, training_row_store_path)
    rows = _load_materialized_training_rows(row_store_path)
    if not rows:
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
        "training_row_store_path": str(row_store_path),
        "training_row_store_exists": row_store_path.exists(),
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
                    "feature_schema_hash": LIVE_SHORT_HORIZON_SCHEMA.schema_hash,
                }
            )
    return _market_adjust_rows(rows)


def _market_adjust_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """R1 — cross-sectional market/sector adjustment of the label (confounding fix).

    Raw forward return mixes the common market move with the stock-specific effect,
    so a naive label credits (or blames) news for market beta. Here each frame's
    forward net return is demeaned against other symbols that entered in the same
    short time bucket, yielding an ABNORMAL return; the binary label is the sign of
    that abnormal return. When a bucket has <2 distinct symbols the market factor is
    unidentifiable, so it degrades gracefully to the raw (absolute) label.
    """
    if os.getenv("LIVE_LABEL_MARKET_ADJUST", "true").strip().lower() not in {"1", "true", "yes", "on"}:
        return rows
    bucket_seconds = max(30.0, _env_float("LIVE_LABEL_MARKET_BUCKET_SECONDS", 300.0))
    buckets: dict[int, list[float]] = defaultdict(list)
    bucket_symbols: dict[int, set[str]] = defaultdict(set)
    parsed: list[tuple[int, float] | None] = []
    for row in rows:
        entry_time = _parse_iso_time(str(row.get("as_of") or ""))
        raw = float(row.get("forward_net_return_bps", 0.0))
        if entry_time is None:
            parsed.append(None)
            continue
        key = int(entry_time.timestamp() // bucket_seconds)
        buckets[key].append(raw)
        bucket_symbols[key].add(str(row.get("ticker") or ""))
        parsed.append((key, raw))
    for row, item in zip(rows, parsed, strict=True):
        raw = float(row.get("forward_net_return_bps", 0.0))
        row["raw_forward_net_return_bps"] = raw
        if item is None or len(bucket_symbols[item[0]]) < 2:
            row["market_bps"] = 0.0
            continue  # cannot identify a market factor from a single name; keep absolute.
        key, _ = item
        market_bps = sum(buckets[key]) / len(buckets[key])
        abnormal_bps = raw - market_bps
        row["market_bps"] = market_bps
        row["forward_net_return_bps"] = abnormal_bps
        row["label"] = int(abnormal_bps > 0.0)
        row["label_source"] = f"{row.get('label_source', 'unknown')}+market_adjusted"
    return rows


def _parse_iso_time(value: str) -> datetime | None:
    if not value:
        return None
    try:
        return datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError:
        return None


def _prune_news_sentiment_store() -> None:
    """Keep the news-sentiment store bounded (memory-safe). Best-effort.

    Live scoring only needs recent news (TTL ~6h) and training reads the value
    journaled at frame-build time, so the store can be pruned aggressively. Runs
    on the periodic training loop, so no extra scheduler is needed.
    """
    try:
        from app.features.news_sentiment_store import NewsSentimentStore

        NewsSentimentStore().prune(older_than_days=_env_float("NEWS_SENTIMENT_RETENTION_DAYS", 2.0))
    except Exception:  # noqa: BLE001 - housekeeping must never break training.
        pass


def update_news_trust_from_rows(
    rows: list[dict[str, Any]],
    *,
    path: str | Path = DEFAULT_NEWS_TRUST_PATH,
) -> dict[str, Any]:
    """Outcome-calibrated news trust (approach A, redefined on ABNORMAL returns).

    Runs on the market-adjusted rows from `_market_adjust_rows`, so `forward_net_return_bps`
    here is already demeaned against contemporaneous names — the edge measures how much
    positive-news candidates OUT-PERFORMED the market, not raw beta. Still only a mild,
    well-gated multiplier on the advisory confirm bonus (never a trade trigger). Pure
    arithmetic over floats already in the rows — KB-scale output.
    """
    threshold = _env_float("NEWS_TRUST_MIN_SENTIMENT", 0.1)
    min_samples = int(_env_float("NEWS_TRUST_MIN_SAMPLES", 30))
    all_returns = [float(row.get("forward_net_return_bps", 0.0)) for row in rows]
    positive_returns = [
        float(row.get("forward_net_return_bps", 0.0))
        for row in rows
        if float((row.get("features") or {}).get("news_sentiment", 0.0)) > threshold
    ]
    if len(positive_returns) >= min_samples and all_returns:
        baseline = sum(all_returns) / len(all_returns)
        positive_mean = sum(positive_returns) / len(positive_returns)
        edge_bps = positive_mean - baseline
        # Conservative: +50bps of abnormal edge → 1.5x; -50bps → 0.5x (clamped tight).
        scale = max(0.5, min(1.5, 1.0 + edge_bps / 100.0))
        calibrated = True
    else:
        edge_bps = 0.0
        scale = 1.0
        calibrated = False
    payload = {
        "news_confirm_scale": round(scale, 4),
        "edge_bps": round(edge_bps, 3),
        "positive_samples": len(positive_returns),
        "total_samples": len(all_returns),
        "calibrated": calibrated,
    }
    try:
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    except OSError:
        pass
    return payload


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
    max_frames = max(1_000, int(_env_float("LIVE_TRAINING_MAX_FEATURE_FRAMES", 25_000)))
    resolved = str(path.resolve())
    stat = path.stat()
    size = stat.st_size
    mtime_ns = stat.st_mtime_ns
    with path.open("rb") as handle:
        head_digest = hashlib.sha256(handle.read(4096)).hexdigest()
    with _FEATURE_FRAME_CACHE_LOCK:
        cached_path = _FEATURE_FRAME_CACHE.get("path")
        cached_offset = int(_FEATURE_FRAME_CACHE.get("offset") or 0)
        cached_mtime_ns = _FEATURE_FRAME_CACHE.get("mtime_ns")
        cached_head_digest = _FEATURE_FRAME_CACHE.get("head_digest")
        cached_frames = list(_FEATURE_FRAME_CACHE.get("frames") or ())
        incremental_cache_valid = (
            cached_path == resolved
            and 0 <= cached_offset <= size
            and cached_head_digest == head_digest
            and (cached_offset < size or cached_mtime_ns == mtime_ns)
        )
        if incremental_cache_valid:
            if cached_offset == size:
                return cached_frames
            with path.open("rb") as handle:
                handle.seek(cached_offset)
                lines = handle.read().decode("utf-8", errors="replace").splitlines()
            frames = [*cached_frames, *_parse_feature_frame_lines(lines)]
        else:
            lines = _read_recent_lines(path, max_frames)
            frames = _parse_feature_frame_lines(lines)
        if len(frames) > max_frames:
            frames = frames[-max_frames:]
        _FEATURE_FRAME_CACHE.update(
            {
                "path": resolved,
                "offset": size,
                "mtime_ns": mtime_ns,
                "head_digest": head_digest,
                "frames": frames,
            }
        )
        return list(frames)


def _parse_feature_frame_lines(lines: list[str]) -> list[dict[str, Any]]:
    frames: list[dict[str, Any]] = []
    for line in lines:
        if not line.strip():
            continue
        try:
            payload = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(payload, dict):
            frames.append(payload)
    return frames


def _read_recent_lines(path: Path, maximum: int) -> list[str]:
    block_size = 1024 * 1024
    chunks: list[bytes] = []
    newline_count = 0
    with path.open("rb") as handle:
        position = handle.seek(0, os.SEEK_END)
        while position > 0 and newline_count <= maximum:
            read_size = min(block_size, position)
            position -= read_size
            handle.seek(position)
            chunk = handle.read(read_size)
            chunks.append(chunk)
            newline_count += chunk.count(b"\n")
    data = b"".join(reversed(chunks)).decode("utf-8", errors="replace")
    return data.splitlines()[-maximum:]


def _frame_passes_training_quality(frame: dict[str, Any]) -> bool:
    values = frame.get("values")
    if not isinstance(values, dict):
        return False
    max_spread_bps = _env_float("LIVE_TRAINING_MAX_SPREAD_BPS", 80.0)
    max_cost_to_vol = _env_float("LIVE_TRAINING_MAX_COST_TO_VOLATILITY_RATIO", 5_000.0)
    drop_flat_frames = os.getenv("LIVE_TRAINING_DROP_FLAT_FRAMES", "true").strip().lower() in {
        "1",
        "true",
        "yes",
        "on",
    }
    try:
        spread_bps = float(values.get("spread_bps", 0.0))
        bid_depth = float(values.get("bid_depth", 0.0))
        ask_depth = float(values.get("ask_depth", 0.0))
        cost_to_vol = float(values.get("cost_to_volatility_ratio", 0.0))
    except (TypeError, ValueError):
        return False
    if not all(math.isfinite(value) for value in (spread_bps, bid_depth, ask_depth, cost_to_vol)):
        return False
    if spread_bps > max_spread_bps or bid_depth <= 0 or ask_depth <= 0 or cost_to_vol > max_cost_to_vol:
        return False
    if drop_flat_frames and not _frame_has_training_signal(values):
        return False
    return True


def _frame_has_training_signal(values: dict[str, Any]) -> bool:
    """Reject quote-refresh duplicates that carry no usable short-horizon signal."""
    min_abs_return = _env_float("LIVE_TRAINING_MIN_ABS_RETURN", 0.0001)
    min_volatility = _env_float("LIVE_TRAINING_MIN_REALIZED_VOLATILITY", 0.000001)
    min_volume_spike_delta = _env_float("LIVE_TRAINING_MIN_VOLUME_SPIKE_DELTA", 0.05)
    return_keys = (
        "return_1s",
        "return_5s",
        "return_10s",
        "return_30s",
        "return_1m",
        "return_3m",
        "distance_from_vwap",
        "max_drop_3m",
    )
    try:
        return_signal = max(abs(float(values.get(name, 0.0) or 0.0)) for name in return_keys)
        volatility = abs(float(values.get("realized_volatility_3m", 0.0) or 0.0))
        volume_spike = abs(float(values.get("volume_spike_ratio", 1.0) or 1.0) - 1.0)
    except (TypeError, ValueError):
        return False
    if not all(math.isfinite(value) for value in (return_signal, volatility, volume_spike)):
        return False
    return (
        return_signal >= min_abs_return
        or volatility >= min_volatility
        or volume_spike >= min_volume_spike_delta
    )


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


def _training_row_store_path(
    journal_path: str | Path,
    configured: str | Path | None,
) -> Path:
    if configured is not None:
        return Path(configured)
    journal = Path(journal_path)
    if journal == DEFAULT_FEATURE_JOURNAL_PATH:
        return DEFAULT_TRAINING_ROW_STORE_PATH
    return journal.with_suffix(".training_rows.sqlite3")


def _merge_materialized_training_rows(
    path: Path,
    fresh_rows: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], dict[str, int]]:
    path.parent.mkdir(parents=True, exist_ok=True)
    with closing(sqlite3.connect(path)) as conn:
        _ensure_training_row_schema(conn)
        compacted = _compact_materialized_training_rows(conn)
        before_keys = {
            str(row[0])
            for row in conn.execute("select row_key from live_training_rows").fetchall()
        }
        existing_payloads = {
            str(key): str(payload)
            for key, payload in conn.execute(
                "select row_key, payload from live_training_rows"
            ).fetchall()
        }
        before_count = len(before_keys)
        now = datetime.now(timezone.utc).isoformat()
        payloads = [
            (
                _training_row_key(row),
                str(row.get("as_of") or ""),
                json.dumps(row, sort_keys=True, separators=(",", ":")),
                now,
            )
            for row in fresh_rows
        ]
        changed_keys = {
            key
            for key, _as_of, payload, _updated_at in payloads
            if existing_payloads.get(key) != payload
        }
        if payloads:
            conn.executemany(
                """
                insert into live_training_rows(row_key, as_of, payload, updated_at)
                values (?, ?, ?, ?)
                on conflict(row_key) do update set
                  as_of=excluded.as_of,
                  payload=excluded.payload,
                  updated_at=excluded.updated_at
                where live_training_rows.payload <> excluded.payload
                """,
                payloads,
            )
        maximum = max(1_000, int(_env_float("LIVE_TRAINING_MAX_MATERIALIZED_ROWS", 100_000)))
        count = int(conn.execute("select count(*) from live_training_rows").fetchone()[0])
        if count > maximum:
            conn.execute(
                """
                delete from live_training_rows
                where row_key in (
                  select row_key from live_training_rows
                  order by as_of asc, row_key asc
                  limit ?
                )
                """,
                (count - maximum,),
            )
        conn.commit()
        after_count = int(
            conn.execute("select count(*) from live_training_rows").fetchone()[0]
        )
    return _load_materialized_training_rows(path), {
        "before_rows": before_count,
        "after_rows": after_count,
        "new_rows": sum(
            1
            for row in fresh_rows
            if _training_row_key(row) not in before_keys
        ),
        "duplicate_rows_removed": int(compacted["duplicate_rows_removed"]),
        "invalid_rows_removed": int(compacted["invalid_rows_removed"]),
        "changed_keys": tuple(sorted(changed_keys)),
    }


def _load_materialized_training_rows(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    try:
        with closing(sqlite3.connect(path)) as conn:
            _ensure_training_row_schema(conn)
            rows = conn.execute(
                "select payload from live_training_rows order by as_of asc, row_key asc"
            ).fetchall()
    except sqlite3.Error:
        return []
    loaded: list[dict[str, Any]] = []
    for (payload,) in rows:
        try:
            row = json.loads(str(payload))
        except json.JSONDecodeError:
            continue
        if isinstance(row, dict) and _row_matches_live_schema(row):
            loaded.append(row)
    return loaded


def _row_matches_live_schema(row: dict[str, Any]) -> bool:
    row_hash = str(row.get("feature_schema_hash") or "")
    if row_hash and row_hash != LIVE_SHORT_HORIZON_SCHEMA.schema_hash:
        return False
    features = row.get("features")
    return isinstance(features, dict) and all(
        name in features for name in LIVE_SHORT_HORIZON_SCHEMA.feature_names
    )


def _ensure_training_row_schema(conn: sqlite3.Connection) -> None:
    conn.execute(
        """
        create table if not exists live_training_rows (
          row_key text primary key,
          as_of text not null,
          payload text not null,
          updated_at text not null
        )
        """
    )
    conn.execute(
        "create index if not exists idx_live_training_rows_as_of on live_training_rows(as_of)"
    )


def _training_row_key(row: dict[str, Any]) -> str:
    # A labelled observation has one identity. label_source is deliberately not
    # part of the key: as more future price path arrives, a provisional terminal
    # label may mature into TP/SL and must replace the old label, not coexist with it.
    source = "|".join(
        (
            str(row.get("ticker") or ""),
            str(row.get("as_of") or ""),
            str(row.get("feature_schema_hash") or LIVE_SHORT_HORIZON_SCHEMA.schema_hash),
        )
    )
    return hashlib.sha256(source.encode("utf-8")).hexdigest()


def _compact_materialized_training_rows(
    conn: sqlite3.Connection,
) -> dict[str, int]:
    stored = conn.execute(
        "select payload, updated_at from live_training_rows"
    ).fetchall()
    canonical: dict[str, tuple[dict[str, Any], str]] = {}
    invalid_rows = 0
    for raw_payload, raw_updated_at in stored:
        try:
            row = json.loads(str(raw_payload))
        except json.JSONDecodeError:
            invalid_rows += 1
            continue
        if not isinstance(row, dict) or not _row_matches_live_schema(row):
            invalid_rows += 1
            continue
        key = _training_row_key(row)
        updated_at = str(raw_updated_at or "")
        incumbent = canonical.get(key)
        if incumbent is None or updated_at >= incumbent[1]:
            canonical[key] = (row, updated_at)
    duplicate_rows = max(0, len(stored) - invalid_rows - len(canonical))
    expected_keys = {
        str(row[0])
        for row in conn.execute("select row_key from live_training_rows").fetchall()
    }
    if (
        invalid_rows
        or duplicate_rows
        or expected_keys != set(canonical)
    ):
        conn.execute("delete from live_training_rows")
        conn.executemany(
            """
            insert into live_training_rows(row_key, as_of, payload, updated_at)
            values (?, ?, ?, ?)
            """,
            (
                (
                    key,
                    str(row.get("as_of") or ""),
                    json.dumps(row, sort_keys=True, separators=(",", ":")),
                    updated_at,
                )
                for key, (row, updated_at) in canonical.items()
            ),
        )
        conn.commit()
    return {
        "duplicate_rows_removed": duplicate_rows,
        "invalid_rows_removed": invalid_rows,
        "remaining_rows": len(canonical),
    }


def _training_min_row_spacing_seconds() -> float:
    try:
        return max(
            1.0,
            float(
                os.getenv(
                    "LIVE_TRAINING_MIN_ROW_SPACING_SECONDS",
                    str(DEFAULT_TRAINING_MIN_ROW_SPACING_SECONDS),
                )
            ),
        )
    except (TypeError, ValueError):
        return DEFAULT_TRAINING_MIN_ROW_SPACING_SECONDS


def _thin_training_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Keep second-level data without letting bursty one-symbol frames dominate."""
    spacing = _training_min_row_spacing_seconds()
    buckets: dict[tuple[str, int], dict[str, Any]] = {}
    unparsed: list[dict[str, Any]] = []
    for row in sorted(rows, key=lambda item: str(item.get("as_of") or "")):
        moment = _row_time_for_thinning(row)
        if moment is None:
            unparsed.append(row)
            continue
        bucket = int(moment.timestamp() // spacing)
        # Keep the latest observation in each per-symbol bucket. This preserves
        # genuine 5-second market evolution while collapsing sub-second refreshes.
        buckets[(str(row.get("ticker") or ""), bucket)] = row
    return sorted(
        [*buckets.values(), *unparsed],
        key=lambda item: (str(item.get("as_of") or ""), str(item.get("ticker") or "")),
    )


def _row_time_for_thinning(row: dict[str, Any]) -> datetime | None:
    try:
        value = datetime.fromisoformat(
            str(row.get("as_of") or "").replace("Z", "+00:00")
        )
    except ValueError:
        return None
    if value.tzinfo is None:
        value = value.replace(tzinfo=timezone.utc)
    return value


def _training_rows_fingerprint(rows: list[dict[str, Any]]) -> str:
    if not rows:
        return ""
    digest = hashlib.sha256()
    digest.update(TRAINING_RECIPE_VERSION.encode("utf-8"))
    digest.update(b"\n")
    for row in rows:
        digest.update(str(row.get("ticker") or "").encode("utf-8"))
        digest.update(b"|")
        digest.update(str(row.get("as_of") or "").encode("utf-8"))
        digest.update(b"|")
        digest.update(str(int(row.get("label") or 0)).encode("ascii"))
        digest.update(b"|")
        try:
            forward_return = float(row.get("forward_net_return_bps") or 0.0)
        except (TypeError, ValueError):
            forward_return = 0.0
        digest.update(f"{forward_return:.8f}".encode("ascii"))
        digest.update(b"|")
        features = row.get("features") if isinstance(row.get("features"), dict) else {}
        for name in LIVE_SHORT_HORIZON_SCHEMA.feature_names:
            try:
                value = float(features.get(name, 0.0))
            except (TypeError, ValueError):
                value = 0.0
            digest.update(name.encode("utf-8"))
            digest.update(b"=")
            digest.update(f"{value:.12g}".encode("ascii"))
            digest.update(b";")
        digest.update(b"\n")
    return digest.hexdigest()


def _incremental_data_format_signature() -> str:
    payload = {
        "feature_schema_hash": LIVE_SHORT_HORIZON_SCHEMA.schema_hash,
        "feature_names": LIVE_SHORT_HORIZON_SCHEMA.feature_names,
        "training_recipe_version": TRAINING_RECIPE_VERSION,
        "minimum_row_spacing_seconds": _training_min_row_spacing_seconds(),
        "classification_family": "logistic_regression_sgd",
        "regression_family": "linear_regression_sgd",
    }
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()


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
    artifact["training_data"] = training_data
    paths = [registry.root / f"{artifact['artifact_id']}.json"]
    if bool((artifact.get("deployment") or {}).get("promoted")):
        paths.append(registry.latest_path)
    for path in paths:
        if not path.exists():
            continue
        payload = json.loads(path.read_text(encoding="utf-8"))
        payload["training_data"] = training_data
        # Keep annotations atomic too: reliability readers must never observe a
        # briefly empty/partial active-model JSON document.
        _atomic_write_text(path, json.dumps(payload, indent=2, sort_keys=True))


def _latest_saved_payload(registry: ModelArtifactRegistry) -> dict[str, Any] | None:
    candidates = sorted(
        (path for path in registry.root.glob("live_short_horizon.*.json") if path.is_file()),
        key=lambda path: path.stat().st_mtime,
        reverse=True,
    )
    for path in candidates:
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        if isinstance(payload, dict):
            return payload
    return None


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
        "deployment": payload.get("deployment") or {},
        "training_data": payload.get("training_data") or {},
        "created_at": payload.get("created_at"),
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
        "deployment": payload.get("deployment") or {},
        "training_data": payload.get("training_data") or {},
        "created_at": payload.get("created_at"),
    }
