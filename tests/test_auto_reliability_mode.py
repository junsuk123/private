from __future__ import annotations

import json
import sys
import tempfile
from datetime import datetime, timedelta, timezone
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from app import web as web_module
from app.features.feature_schema import LIVE_SHORT_HORIZON_SCHEMA


def _ready_snapshot() -> dict:
    return {
        "score": 1.0,
        "threshold": 0.9,
        "ready": True,
        "reasons": [],
        "components": {},
        "active_markets": ["US"],
        "evaluated_at": datetime.now(timezone.utc).isoformat(),
    }


def test_reliability_evaluator_requires_every_hard_gate() -> None:
    policy = SimpleNamespace(conflicts=lambda: [])
    with (
        patch.object(
            web_module,
            "_cached_kis_connection_probe",
            return_value={"ok": True, "account_checked": True, "actual_equity": 200_000},
        ),
        patch.object(
            web_module,
            "evaluate_live_runtime_gates",
            return_value=SimpleNamespace(ok=True, failures=()),
        ),
        patch.object(
            web_module,
            "load_short_horizon_strategy_config",
            return_value={"execution": {"live_trading_enabled": True}},
        ),
        patch.object(web_module, "_env_flag", return_value=True),
        patch.object(web_module.TradingPolicySnapshot, "from_environment", return_value=policy),
        patch.object(web_module, "_latest_model_reliability", return_value={"ok": True}),
        patch.object(web_module, "_auto_market_health", return_value={"ok": True}),
        patch.object(web_module, "_active_live_market_groups", return_value=("US",)),
    ):
        result = web_module._evaluate_auto_reliability()

    assert result["ready"] is True
    assert result["score"] == 1.0


def test_market_health_is_vacuously_ready_when_no_core_feed_is_required() -> None:
    result = web_module._auto_market_health(datetime.now(timezone.utc), ())

    assert result["ok"] is True
    assert result["ready_markets"] == []
    assert result["missing_markets"] == []


def test_model_reliability_uses_active_champion_not_rejected_challenger() -> None:
    now = datetime.now(timezone.utc)
    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        active = {
            "artifact_id": "active-champion",
            "created_at": now.isoformat(),
            "feature_schema_hash": LIVE_SHORT_HORIZON_SCHEMA.schema_hash,
            "live_eligible": True,
            "metrics": {"auc": 0.72, "precision_at_k": 0.43},
            "reason_codes": [],
        }
        challenger = {
            "artifact_id": "rejected-challenger",
            "feature_schema_hash": LIVE_SHORT_HORIZON_SCHEMA.schema_hash,
            "live_eligible": False,
            "metrics": {"auc": 0.60, "precision_at_k": 0.22},
            "reason_codes": ["METRICS_BELOW_LIVE_THRESHOLDS"],
        }
        (root / "latest.json").write_text(json.dumps(active), encoding="utf-8")
        (root / "live_short_horizon.rejected.json").write_text(
            json.dumps(challenger),
            encoding="utf-8",
        )
        with (
            patch.object(web_module, "Path", return_value=root),
            patch.dict(
                "os.environ",
                {
                    "AUTO_RELIABILITY_MODEL_MAX_AGE_SECONDS": "1800",
                    "AUTO_RELIABILITY_ACTIVE_MODEL_MAX_AGE_SECONDS": "2592000",
                },
            ),
        ):
            result = web_module._latest_model_reliability(now)

    assert result["ok"] is True
    assert result["artifact_id"] == "active-champion"
    assert result["metrics"]["auc"] == 0.72
    assert result["latest_challenger"]["artifact_id"] == "rejected-challenger"
    assert result["latest_challenger"]["live_eligible"] is False


def test_successful_unchanged_training_heartbeat_keeps_incumbent_fresh() -> None:
    now = datetime.now(timezone.utc)
    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        active = {
            "artifact_id": "active-champion",
            "created_at": now.isoformat(),
            "feature_schema_hash": LIVE_SHORT_HORIZON_SCHEMA.schema_hash,
            "live_eligible": True,
            "metrics": {"auc": 0.72, "precision_at_k": 0.43},
            "reason_codes": [],
        }
        challenger = {
            **active,
            "artifact_id": "unchanged-dataset-challenger",
            "live_eligible": False,
        }
        (root / "latest.json").write_text(json.dumps(active), encoding="utf-8")
        challenger_path = root / "live_short_horizon.old.json"
        challenger_path.write_text(json.dumps(challenger), encoding="utf-8")
        old_timestamp = now.timestamp() - 3_600
        challenger_path.touch()
        import os

        os.utime(challenger_path, (old_timestamp, old_timestamp))
        with web_module._live_lock:
            previous = dict(web_module._live_training_heartbeat)
            web_module._live_training_heartbeat.update(
                {
                    "finished_at": now.isoformat(),
                    "ok": True,
                    "skipped": True,
                }
            )
        try:
            with (
                patch.object(web_module, "Path", return_value=root),
                patch.dict(
                    "os.environ",
                    {"AUTO_RELIABILITY_MODEL_MAX_AGE_SECONDS": "1800"},
                ),
            ):
                result = web_module._latest_model_reliability(now)
        finally:
            with web_module._live_lock:
                web_module._live_training_heartbeat.clear()
                web_module._live_training_heartbeat.update(previous)

    assert result["ok"] is True
    assert result["training_heartbeat_ok"] is True
    assert result["training_age_seconds"] < 1


def test_recent_running_training_cycle_is_a_healthy_heartbeat() -> None:
    now = datetime.now(timezone.utc)
    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        artifact = {
            "artifact_id": "active-champion",
            "created_at": now.isoformat(),
            "feature_schema_hash": LIVE_SHORT_HORIZON_SCHEMA.schema_hash,
            "live_eligible": True,
            "metrics": {"auc": 0.72, "precision_at_k": 0.43},
            "reason_codes": [],
        }
        (root / "latest.json").write_text(json.dumps(artifact), encoding="utf-8")
        (root / "live_short_horizon.running.json").write_text(
            json.dumps(artifact),
            encoding="utf-8",
        )
        with web_module._live_lock:
            previous = dict(web_module._live_training_heartbeat)
            web_module._live_training_heartbeat.update(
                {
                    "started_at": now.isoformat(),
                    "finished_at": None,
                    "ok": False,
                    "error": None,
                }
            )
        try:
            with (
                patch.object(web_module, "Path", return_value=root),
                patch.dict(
                    "os.environ",
                    {"LIVE_TRAINING_RUNNING_HEARTBEAT_MAX_SECONDS": "900"},
                ),
            ):
                result = web_module._latest_model_reliability(now)
        finally:
            with web_module._live_lock:
                web_module._live_training_heartbeat.clear()
                web_module._live_training_heartbeat.update(previous)

    assert result["training_heartbeat_ok"] is True
    assert result["training_in_progress"] is True
    assert result["training_heartbeat_at"] == now.isoformat()


def test_model_reliability_rejects_canonically_stale_incumbent() -> None:
    now = datetime.now(timezone.utc)
    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        artifact = {
            "artifact_id": "stale-champion",
            "created_at": (now.replace(microsecond=0) - timedelta(hours=7)).isoformat(),
            "feature_schema_hash": LIVE_SHORT_HORIZON_SCHEMA.schema_hash,
            "live_eligible": True,
            "metrics": {"auc": 0.72, "precision_at_k": 0.43},
            "reason_codes": [],
        }
        (root / "latest.json").write_text(json.dumps(artifact), encoding="utf-8")
        (root / "live_short_horizon.stale.json").write_text(
            json.dumps(artifact), encoding="utf-8"
        )
        with patch.object(web_module, "Path", return_value=root):
            result = web_module._latest_model_reliability(now)

    assert result["ok"] is False
    assert result["trust_level"] == "SHADOW_ONLY"
    assert "MODEL_AGE_EXCEEDED" in result["reason_codes"]


def test_reliability_step_promotes_only_after_sustained_passes() -> None:
    now = datetime.now(timezone.utc)
    with web_module._live_lock:
        web_module._auto_reliability_state["ready_streak"] = 0
        web_module._auto_reliability_state["unready_streak"] = 0
        web_module._auto_reliability_state["manual_hold"] = False
    with (
        patch.dict(
            "os.environ",
            {
                "AUTO_RELIABILITY_PROMOTE_CONSECUTIVE": "2",
                "AUTO_RELIABILITY_DEMOTE_CONSECUTIVE": "2",
            },
        ),
        patch.object(web_module, "_evaluate_auto_reliability", return_value=_ready_snapshot()),
        patch.object(web_module, "_active_operation_mode", side_effect=["learning", "learning"]),
        patch.object(web_module, "_auto_reliability_enter_learning"),
        patch.object(web_module, "_auto_reliability_learning_maintenance"),
        patch.object(
            web_module,
            "_auto_reliability_transition_to_live",
            return_value={"ok": True, "status": "started", "live_trading_status": "armed"},
        ) as promote,
    ):
        first = web_module._auto_reliability_step(now)
        second = web_module._auto_reliability_step(now)

    assert first["mode"] == "learning"
    assert second["mode"] == "live_trading"
    promote.assert_called_once()


def test_manual_learning_hold_is_never_auto_promoted() -> None:
    now = datetime.now(timezone.utc)
    with web_module._live_lock:
        previous_hold = web_module._auto_reliability_state.get("manual_hold", False)
        web_module._auto_reliability_state["ready_streak"] = 20
        web_module._auto_reliability_state["unready_streak"] = 0
        web_module._auto_reliability_state["manual_hold"] = True
    try:
        with (
            patch.object(web_module, "_evaluate_auto_reliability", return_value=_ready_snapshot()),
            patch.object(web_module, "_active_operation_mode", return_value="learning"),
            patch.object(web_module, "_auto_reliability_learning_maintenance"),
            patch.object(web_module, "_auto_reliability_transition_to_live") as promote,
        ):
            result = web_module._auto_reliability_step(now)
    finally:
        with web_module._live_lock:
            web_module._auto_reliability_state["manual_hold"] = previous_hold

    assert result["mode"] == "learning"
    assert result["promotion_suppressed_reason"] == "MANUAL_LEARNING_HOLD"
    promote.assert_not_called()


def test_failed_auto_promotion_rolls_back_to_learning() -> None:
    blocked = {
        "ok": True,
        "status": "started",
        "live_trading_status": "blocked",
    }
    with (
        patch.object(web_module, "_operation_mode_start_response", return_value=blocked),
        patch.object(web_module, "_auto_reliability_enter_learning") as rollback,
    ):
        result = web_module._auto_reliability_transition_to_live()

    assert result == blocked
    rollback.assert_called_once_with()


def test_live_mode_is_demoted_immediately_on_broker_failure() -> None:
    failed = {
        **_ready_snapshot(),
        "score": 0.8,
        "ready": False,
        "reasons": ["BROKER_NOT_READY"],
    }
    with (
        patch.object(web_module, "_evaluate_auto_reliability", return_value=failed),
        patch.object(web_module, "_active_operation_mode", return_value="live_trading"),
        patch.object(web_module, "_auto_reliability_enter_learning"),
        patch.object(web_module, "_auto_reliability_transition_to_learning") as demote,
        patch.object(web_module, "_auto_reliability_learning_maintenance"),
    ):
        result = web_module._auto_reliability_step()

    assert result["mode"] == "learning"
    demote.assert_called_once()


def _model_snapshot(reasons, trust="SHADOW_ONLY") -> dict:
    return {
        **_ready_snapshot(),
        "score": 0.8,
        "ready": False,
        "reasons": list(reasons),
        "components": {"model": {"ok": False, "trust_level": trust}},
    }


def _run_step(snapshot):
    with (
        patch.object(web_module, "_evaluate_auto_reliability", return_value=snapshot),
        patch.object(web_module, "_active_operation_mode", return_value="learning"),
        patch.object(web_module, "_auto_reliability_enter_learning"),
        patch.object(web_module, "_auto_reliability_enforce_sell_only") as enforce,
        patch.object(web_module, "_auto_reliability_enter_model_degraded") as degraded,
        patch.object(web_module, "_auto_reliability_learning_maintenance"),
    ):
        result = web_module._auto_reliability_step()
    return result, enforce, degraded


def test_learning_mode_fails_closed_when_reliability_is_low() -> None:
    # A component other than the model must still stop entries outright.
    result, enforce, degraded = _run_step(_model_snapshot(["BROKER_NOT_READY"]))

    assert result["mode"] == "learning"
    enforce.assert_called_once_with("BROKER_NOT_READY")
    degraded.assert_not_called()


def test_model_only_demotion_falls_back_instead_of_blocking_entries() -> None:
    # model_staleness documents trained_model -> shadow_only -> ontology/bandit.
    # The registry already refuses to price entries with a stale artifact, so the
    # controller must let that fallback run rather than disabling buys.
    result, enforce, degraded = _run_step(_model_snapshot(["MODEL_NOT_READY"]))

    assert result["mode"] == "learning"
    degraded.assert_called_once_with("MODEL_NOT_READY")
    enforce.assert_not_called()


def test_model_only_degradation_does_not_leave_live_mode() -> None:
    with web_module._live_lock:
        web_module._auto_reliability_state["unready_streak"] = 0
    with (
        patch.object(
            web_module,
            "_evaluate_auto_reliability",
            return_value=_model_snapshot(["MODEL_NOT_READY"]),
        ),
        patch.object(web_module, "_active_operation_mode", return_value="live_trading"),
        patch.object(web_module, "_auto_reliability_enter_model_degraded") as degraded,
        patch.object(web_module, "_auto_reliability_transition_to_learning") as demote,
        patch.object(web_module, "_live_market_data_startup_grace_active", return_value=False),
    ):
        result = web_module._auto_reliability_step()

    assert result["mode"] == "live_trading"
    assert result["last_transition_reason"] == "MODEL_DEGRADED_FALLBACK"
    degraded.assert_called_once_with("MODEL_NOT_READY")
    demote.assert_not_called()


def test_brief_market_data_gap_requires_longer_live_mode_hysteresis() -> None:
    failed = {
        **_ready_snapshot(),
        "score": 0.8,
        "ready": False,
        "reasons": ["MARKET_DATA_NOT_READY"],
    }
    with web_module._live_lock:
        web_module._auto_reliability_state["unready_streak"] = 0
    with (
        patch.dict(
            "os.environ",
            {"AUTO_RELIABILITY_MARKET_DATA_DEMOTE_CONSECUTIVE": "3"},
        ),
        patch.object(web_module, "_evaluate_auto_reliability", return_value=failed),
        patch.object(web_module, "_active_operation_mode", return_value="live_trading"),
        patch.object(web_module, "_auto_reliability_transition_to_learning") as demote,
        patch.object(web_module, "_auto_reliability_learning_maintenance"),
        patch.object(web_module, "_live_market_data_startup_grace_active", return_value=False),
    ):
        first = web_module._auto_reliability_step()
        second = web_module._auto_reliability_step()
        third = web_module._auto_reliability_step()

    assert first["mode"] == second["mode"] == "live_trading"
    assert third["mode"] == "learning"
    demote.assert_called_once()


def test_unusable_model_still_fails_closed() -> None:
    # UNUSABLE means there is no artifact to reason about at all.
    result, enforce, degraded = _run_step(
        _model_snapshot(["MODEL_NOT_READY"], trust="UNUSABLE")
    )

    enforce.assert_called_once_with("MODEL_NOT_READY")
    degraded.assert_not_called()


def test_model_degradation_combined_with_another_failure_fails_closed() -> None:
    result, enforce, degraded = _run_step(
        _model_snapshot(["MODEL_NOT_READY", "MARKET_DATA_NOT_READY"])
    )

    assert enforce.call_count == 1
    degraded.assert_not_called()


def test_model_degraded_fallback_can_be_switched_off() -> None:
    with patch.dict(
        "os.environ", {"AUTO_RELIABILITY_MODEL_DEGRADED_FALLBACK": "false"}
    ):
        result, enforce, degraded = _run_step(_model_snapshot(["MODEL_NOT_READY"]))

    enforce.assert_called_once_with("MODEL_NOT_READY")
    degraded.assert_not_called()


def test_model_degraded_lifts_a_block_stamped_with_a_stale_reason() -> None:
    """The block records why it was placed, which may be several cycles old.

    A block stamped MARKET_DATA_NOT_READY stayed in force after the feed
    recovered, because the caller had already established that the only remaining
    failure is the model. The live snapshot wins over the stale string.
    """
    calls = []
    engine = SimpleNamespace(
        get_status=lambda: {
            "buy_enabled": False,
            "buy_disabled_reason": "AUTO_RELIABILITY_DEMOTION:MARKET_DATA_NOT_READY",
        },
        enable_buys=lambda reason: calls.append(reason),
    )
    with patch.object(web_module, "_realtime_trading_engine", engine):
        web_module._auto_reliability_enter_model_degraded("MODEL_NOT_READY")
    assert len(calls) == 1


def test_model_degraded_only_lifts_a_block_this_controller_placed() -> None:
    # A manual disable, a liquidation, or REALTIME_BUY_ENABLED=false must survive.
    for disabled_reason in (
        "MANUAL_OPERATOR_DISABLE",
        "REALTIME_BUY_ENABLED=false",
        "LIQUIDATION_REQUESTED",
    ):
        engine = SimpleNamespace(
            get_status=lambda reason=disabled_reason: {
                "buy_enabled": False,
                "buy_disabled_reason": reason,
            },
            enable_buys=lambda *a, **k: (_ for _ in ()).throw(
                AssertionError(f"must not re-enable after {disabled_reason}")
            ),
        )
        with patch.object(web_module, "_realtime_trading_engine", engine):
            web_module._auto_reliability_enter_model_degraded("MODEL_NOT_READY")

    calls = []
    engine = SimpleNamespace(
        get_status=lambda: {
            "buy_enabled": False,
            "buy_disabled_reason": "AUTO_RELIABILITY_DEMOTION:MODEL_NOT_READY",
        },
        enable_buys=lambda reason: calls.append(reason),
    )
    with patch.object(web_module, "_realtime_trading_engine", engine):
        web_module._auto_reliability_enter_model_degraded("MODEL_NOT_READY")
    assert len(calls) == 1 and calls[0].startswith("AUTO_RELIABILITY_MODEL_DEGRADED:")


# --- KRX feed readiness vs. a market that is simply not printing --------------- #
#
# The defect these guard, measured on 2026-08-11 (09:00-15:20 KST):
#
#   * The KRX quote window was 20s while the stream's own inter-arrival p99.9 is
#     16.1s and its maximum in a healthy session is 102.6s. Ordinary quiet tripped
#     it 153 times covering 86.8 minutes — 22.8% of the session — and each trip
#     demoted the whole process to learning mode and disabled buys under
#     "MARKET_DATA_NOT_READY". Six such demotions were recorded that day, at 09:24,
#     10:00, 10:29, 10:51, 12:45 and 15:24 KST.
#   * The 15:20-15:30 closing auction stops trade prints by design, which emptied
#     the tick set and collapsed the healthy list even though 11 symbols had
#     orderbook updates inside 20 seconds.
#
# Neither is a data outage, and per-order quote freshness is gated separately and far
# more tightly, so calling either one unhealthy blocked trading for no safety gain.


def _market_data_db(path: Path, *, book_age_s: float, tick_age_s: float | None) -> Path:
    """A store with one KRX symbol whose book and tick ages are controlled."""
    import sqlite3
    from contextlib import closing

    now = datetime.now(timezone.utc)
    database = path / "market.sqlite3"
    # ``closing`` for the same reason the production call sites now use it: the
    # connection's context manager ends the transaction, not the connection, and the
    # open handle is what stops the temp directory from being removed on Windows.
    with closing(sqlite3.connect(database)) as conn:
        conn.execute(
            "create table realtime_orderbook (symbol text, received_at text, source text)"
        )
        conn.execute(
            "create table realtime_ticks (symbol text, received_at text, source text)"
        )
        conn.execute(
            "insert into realtime_orderbook values (?,?,?)",
            ("005930", (now - timedelta(seconds=book_age_s)).isoformat(), "kis_realtime_websocket"),
        )
        if tick_age_s is not None:
            conn.execute(
                "insert into realtime_ticks values (?,?,?)",
                ("005930", (now - timedelta(seconds=tick_age_s)).isoformat(), "kis_realtime_websocket"),
            )
        # ``closing`` does NOT commit -- that is the other half of the gotcha. The
        # connection's context manager commits without closing; ``closing`` closes
        # without committing. A writer needs both.
        conn.commit()
    return database


def _health(database: Path, *, continuous: bool, kr_age_s: int = 120) -> dict:
    """Health verdict with the window pinned, so this tests logic not deployment.

    ``AUTO_RELIABILITY_KRX_MAX_AGE_SECONDS`` is set in ``config/secrets/kis_api_keys.env``,
    which product code copies into ``os.environ`` as soon as a KIS client is built —
    so leaving it ambient would make these assertions depend on which other test ran
    first, and on the operator's deployed value.
    """
    env = {
        "REALTIME_MARKET_DATA_DB": str(database),
        "AUTO_RELIABILITY_KRX_MAX_AGE_SECONDS": str(kr_age_s),
        "AUTO_RELIABILITY_KRX_TRADE_ACTIVITY_MAX_AGE_SECONDS": "120",
        "AUTO_RELIABILITY_KRX_MIN_HEALTHY_SYMBOLS": "1",
    }
    with patch.dict("os.environ", env), patch.object(
        web_module, "_kr_trading_is_continuous", return_value=continuous
    ):
        return web_module._auto_market_health(datetime.now(timezone.utc), ("KRX",))


def test_ordinary_quote_quiet_no_longer_reads_as_a_data_outage() -> None:
    """A 45s-old book is inside the stream's normal cadence, not an outage."""
    with tempfile.TemporaryDirectory() as tmp:
        database = _market_data_db(Path(tmp), book_age_s=45.0, tick_age_s=45.0)
        assert _health(database, continuous=True)["ok"] is True


def test_a_genuinely_silent_feed_is_still_unhealthy() -> None:
    """The window was widened, not removed: 5 minutes of silence must still fail."""
    with tempfile.TemporaryDirectory() as tmp:
        database = _market_data_db(Path(tmp), book_age_s=300.0, tick_age_s=300.0)
        assert _health(database, continuous=True)["ok"] is False


def test_call_auction_silence_on_the_trade_channel_is_not_an_outage() -> None:
    """15:20-15:30 collects orders and prints once; the quote feed proves liveness."""
    with tempfile.TemporaryDirectory() as tmp:
        database = _market_data_db(Path(tmp), book_age_s=5.0, tick_age_s=None)
        auction = _health(database, continuous=False)
        assert auction["ok"] is True
        assert auction["kr_continuous_session"] is False
        # ...and during continuous trading the same silence IS a fault worth reporting.
        assert _health(database, continuous=True)["ok"] is False


def test_auction_waiver_still_requires_a_live_quote_feed() -> None:
    """The waiver drops the trade requirement, not the connection requirement."""
    with tempfile.TemporaryDirectory() as tmp:
        database = _market_data_db(Path(tmp), book_age_s=600.0, tick_age_s=None)
        assert _health(database, continuous=False)["ok"] is False


def test_continuous_session_predicate_knows_the_auction_windows() -> None:
    def at(iso: str) -> datetime:
        return datetime.fromisoformat(iso)

    # 09:30, 15:25, 08:50, 16:30 KST respectively.
    assert web_module._kr_trading_is_continuous(at("2026-08-11T00:30:00+00:00")) is True
    assert web_module._kr_trading_is_continuous(at("2026-08-11T06:25:00+00:00")) is False
    assert web_module._kr_trading_is_continuous(at("2026-08-10T23:50:00+00:00")) is False
    assert web_module._kr_trading_is_continuous(at("2026-08-11T07:30:00+00:00")) is False


def test_no_recent_trades_is_not_reported_as_minute_bar_warmup() -> None:
    """"0/20 bars, ~20 min to go" for names that have been warm since the open.

    The warm-up panel measures "active" over a 120s trade window, so during a call
    auction it saw nothing and rendered a shortfall against a symbol it never found.
    """
    store = SimpleNamespace(
        active_symbols=lambda *args, **kwargs: (),
        recent_minute_bars=lambda *args, **kwargs: (),
        latest_orderbook=lambda *args, **kwargs: None,
    )
    with patch.object(web_module, "RealtimeMarketDataStore", lambda *a, **k: store), patch.object(
        web_module, "_realtime_engine_account_snapshot", return_value=None
    ):
        detail = web_module._buy_candidate_warmup_detail()

    assert detail["no_recent_trade_activity"] is True
    assert detail["warming_up"] is False
    assert detail["eta_minutes"] == 0
