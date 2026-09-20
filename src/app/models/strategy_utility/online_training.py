"""Bounded local-DB graph updates, scheduled outside the quote/decision threads."""
from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timedelta, timezone
import json
import os
from pathlib import Path
import sqlite3
import tempfile
import threading
import time

from app.paths import realtime_market_database_path, runtime_database_path


def _limit(name, default, low, high):
    try:
        return min(high, max(low, int(os.getenv(name, default))))
    except (TypeError, ValueError):
        return default


def _snapshot_database(source: Path, target: Path, now: datetime) -> dict:
    """Copy bounded recent completed bars into an isolated training snapshot.

    Stream provenance is retained; the existing preferred-feed loader chooses
    one stream per symbol/day identically for OHLCV and book features.
    """
    symbols_per_market = _limit("GNN_TRAIN_SYMBOLS_PER_MARKET", 8, 5, 16)
    bars_per_symbol = _limit("GNN_TRAIN_BARS_PER_SYMBOL", 720, 120, 2048)
    cutoff = (now - timedelta(minutes=1)).isoformat()
    with sqlite3.connect(source.resolve().as_uri() + "?mode=ro", uri=True, timeout=1.0) as conn:
        schema = conn.execute("SELECT sql FROM sqlite_master WHERE type='table' AND name='realtime_minute_bars'").fetchone()
        if not schema:
            return {"rows": 0, "latest": None}
        symbols = []
        for domestic in (True, False):
            predicate = "symbol GLOB '[0-9][0-9][0-9][0-9][0-9][0-9]'"
            if not domestic:
                predicate = "NOT (" + predicate + ")"
            symbols.extend(row[0] for row in conn.execute(
                "SELECT symbol FROM realtime_minute_bars WHERE " + predicate
                + " AND minute_start <= ? GROUP BY symbol ORDER BY MAX(minute_start) DESC LIMIT ?",
                (cutoff, symbols_per_market)))
        with sqlite3.connect(target) as out:
            out.execute(schema[0])
            count, latest, symbol_progress = 0, None, {}
            for symbol in symbols:
                rows = conn.execute("SELECT * FROM realtime_minute_bars WHERE symbol=? AND minute_start<=? "
                                    "ORDER BY minute_start DESC LIMIT ?", (symbol, cutoff, bars_per_symbol)).fetchall()
                if not rows:
                    continue
                placeholders = ",".join("?" for _ in rows[0])
                out.executemany("INSERT INTO realtime_minute_bars VALUES (" + placeholders + ")", rows)
                count += len(rows)
                moment = conn.execute("SELECT MAX(minute_start) FROM realtime_minute_bars WHERE symbol=? AND minute_start<=?", (symbol, cutoff)).fetchone()[0]
                latest = max(latest or moment, moment)
                symbol_progress[symbol] = {"rows": len(rows), "latest": moment}
    return {"rows": count, "latest": latest, "symbol_progress": symbol_progress}


def run_graph_update(*, database: Path | None = None, checkpoint: Path | None = None,
                     now: datetime | None = None, policy_database: Path | None = None) -> dict:
    from app.evaluation.stored_counterfactual import EvaluationConfig, build_labels, load_minute_bars, load_minute_microstructure
    from app.features.strategy_graph_context import STRATEGY_GRAPH_CONTEXT_SCHEMA
    from app.models.strategy_utility.training import train_counterfactual_checkpoint
    from app.models.strategy_utility.policy_labels import load_policy_shadow_labels
    from app.models.strategy_utility.label_contract import LEGACY_BAR_POLICY, ENTRY_FROZEN_SHADOW_POLICY
    from app.trading.directional_shadow import DEFAULT_SHADOW_STORE_PATH

    database = database or realtime_market_database_path()
    checkpoint = checkpoint or runtime_database_path("models/strategy_utility/temporal_rgcn.npz", env_var="REFACTOR_GNN_CHECKPOINT")
    now = now or datetime.now(timezone.utc)
    policy_database = policy_database or Path(os.getenv("DIRECTIONAL_SHADOW_STORE_PATH", DEFAULT_SHADOW_STORE_PATH))
    state_path = checkpoint.with_suffix(".training-state.json")
    try:
        state = json.loads(state_path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        state = {}
    minimum_new = _limit("GNN_MIN_NEW_LABELLED_SNAPSHOTS", 64, 16, 512)
    forward = load_policy_shadow_labels(policy_database, now=now,
        limit=_limit("GNN_FORWARD_LABEL_REPLAY_LIMIT", 4096, 64, 8192))
    matured_forward = [row for row in forward if row.outcome_observed and row.label_end <= now]
    use_forward = len({(row.symbol, row.as_of) for row in matured_forward}) >= minimum_new
    if use_forward:
        labels, matured = forward, matured_forward
        progress = {"rows": len(forward), "latest": max(row.label_end.isoformat() for row in matured)}
        label_policy = ENTRY_FROZEN_SHADOW_POLICY
    else:
        if not database.exists():
            return {"status": "waiting", "reason": "GRAPH_LOCAL_DATABASE_MISSING", "matched_forward_snapshots": len({(row.symbol, row.as_of) for row in matured_forward})}
        with tempfile.TemporaryDirectory(prefix="obaits-graph-replay-") as temporary:
            snapshot = Path(temporary) / "replay.sqlite3"
            progress = _snapshot_database(database, snapshot, now)
            if not progress["rows"] or progress["symbol_progress"] == state.get("symbol_progress"):
                return {"status": "waiting", "reason": "GRAPH_NO_NEW_COMPLETED_BARS", **progress}
            labels = build_labels(load_minute_bars(snapshot), EvaluationConfig(
                feature_schema_name=STRATEGY_GRAPH_CONTEXT_SCHEMA, align_strategy_horizons=True),
                microstructure_by_symbol=load_minute_microstructure(snapshot))
        matured = [row for row in labels if row.outcome_observed and row.label_end <= now]
        label_policy = LEGACY_BAR_POLICY
    def market_of(row):
        return "KR" if row.symbol.isdigit() and len(row.symbol) == 6 else "US"
    prior_by_market = state.get("latest_label_time_by_market", {})
    if use_forward:
        def label_key(row):
            return "|".join((row.symbol, row.strategy_id, row.as_of.isoformat(), row.label_end.isoformat(), row.policy_id, row.feature_snapshot_id))
        previous_keys = set(state.get("forward_completed_label_keys", ()))
        new_snapshots = {(row.symbol, row.as_of) for row in matured if label_key(row) not in previous_keys}
    else:
        new_snapshots = {(row.symbol, row.as_of) for row in matured
                         if not prior_by_market.get(market_of(row))
                         or row.as_of.isoformat() > prior_by_market[market_of(row)]}
    if len(new_snapshots) < minimum_new:
        return {"status": "waiting", "reason": "GRAPH_INSUFFICIENT_NEW_LABELS", "new_snapshots": len(new_snapshots)}
    # Censored strategy rows remain in complete snapshots with masked outcomes;
    # the trainer enforces label-end purge and training-only supervision counts.
    report = train_counterfactual_checkpoint(labels, checkpoint,
        input_feature_schema=STRATEGY_GRAPH_CONTEXT_SCHEMA, authorize_live_shadow=use_forward)
    state_path.parent.mkdir(parents=True, exist_ok=True)
    temporary_state = state_path.with_suffix(".writing.json")
    latest_by_market = {market: max((row.as_of.isoformat() for row in matured if market_of(row) == market),
                                  default=prior_by_market.get(market))
                        for market in ("KR", "US")}
    update = ({"forward_completed_label_keys": [label_key(row) for row in matured]}
              if use_forward else {"latest_label_time_by_market": latest_by_market})
    temporary_state.write_text(json.dumps({**state, **progress, **update,
        "label_execution_policy": label_policy, "finished_at": now.isoformat()}), encoding="utf-8")
    os.replace(temporary_state, state_path)
    return {"status": "complete", "checkpoint": report["checkpoint"], "live_authorized": report["live_authorized"],
            "label_execution_policy": label_policy,
            "bounded_advisory_authorized_markets": report.get("bounded_advisory_authorized_markets", []),
            "retained_incumbent": report["retained_incumbent"], "new_snapshots": len(new_snapshots),
            "gradient_steps": report["validation_metrics"].get("gradient_steps", 0)}


class GraphTrainingScheduler:
    def __init__(self, worker=None, *, clock=time.monotonic):
        self.worker = worker or run_graph_update
        self.clock = clock
        self._pool = ThreadPoolExecutor(max_workers=1, thread_name_prefix="obaits-graph-update")
        self._lock = threading.Lock()
        self._future = None
        self._last_start = float("-inf")
        self._status = {"status": "waiting"}

    def tick(self) -> dict:
        if os.getenv("AUTO_TRAIN_TEMPORAL_RGCN", "true").lower() not in {"true", "1", "yes", "on"}:
            return {"status": "disabled"}
        with self._lock:
            if self._future is not None and self._future.done():
                try:
                    self._status = self._future.result()
                except Exception as exc:
                    self._status = {"status": "error", "reason": f"{type(exc).__name__}: {exc}"}
                self._future = None
            interval = _limit("GNN_TRAIN_INTERVAL_SECONDS", 900, 300, 86400)
            if self._future is None and self.clock() - self._last_start >= interval:
                self._last_start = self.clock()
                self._future = self._pool.submit(self.worker)
                return {**self._status, "running": True}
            return {**self._status, "running": self._future is not None}

    def close(self):
        self._pool.shutdown(wait=False, cancel_futures=True)


_scheduler = None


def maybe_schedule_graph_training() -> dict:
    global _scheduler
    if _scheduler is None:
        _scheduler = GraphTrainingScheduler()
    return _scheduler.tick()
