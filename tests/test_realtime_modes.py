from __future__ import annotations

import sys
import unittest
from datetime import datetime, timedelta, timezone
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from fastapi.testclient import TestClient

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from app.realtime import OperationMode, OperationModeManager, ShortHorizonRiskPolicy
from app.schemas.domain import OrderAction, OrderIntent
from app import web as web_module
from app.web import app


class RealtimeModesTest(unittest.TestCase):
    def test_learning_uses_unified_realtime_data_and_disallows_orders(self) -> None:
        state = OperationModeManager().start(OperationMode.LEARNING)

        self.assertEqual(state.data_environment, "realtime")
        self.assertFalse(state.synthetic_data_allowed)
        self.assertFalse(state.live_orders_allowed)
        self.assertTrue(state.training_allowed)
        self.assertIn("Use one unified realtime data store only: data/store.", state.guardrails)

    def test_legacy_paper_trading_replay_uses_unified_realtime_data_without_live_orders(self) -> None:
        state = OperationModeManager().start(OperationMode.TESTING)

        self.assertEqual(state.data_environment, "realtime")
        self.assertFalse(state.synthetic_data_allowed)
        self.assertFalse(state.live_orders_allowed)
        self.assertFalse(state.training_allowed)
        self.assertTrue(state.paper_trading_allowed)
        self.assertFalse(state.live_readiness_allowed)
        self.assertIn("Legacy paper trading replay must not submit live broker orders.", state.guardrails)

    def test_kis_paper_and_readiness_modes_use_realtime_data_and_keep_live_orders_blocked(self) -> None:
        paper = OperationModeManager().start(OperationMode.PAPER_TRADING)
        live_readiness = OperationModeManager().start(OperationMode.LIVE_READINESS)

        self.assertTrue(paper.paper_trading_allowed)
        self.assertFalse(paper.live_orders_allowed)
        self.assertEqual(paper.execution_label, "KIS paper trading API")
        self.assertTrue(live_readiness.live_readiness_allowed)
        self.assertFalse(live_readiness.live_orders_allowed)
        self.assertEqual(live_readiness.execution_label, "KIS live readiness check")

    def test_short_horizon_policy_reduces_before_large_loss(self) -> None:
        policy = ShortHorizonRiskPolicy()

        signal = policy.classify("TEST", 30, expected_return=-0.01, downside_risk=0.02, confidence=0.8)

        self.assertEqual(signal.action, OrderAction.REDUCE)
        self.assertEqual(signal.reason, "short_horizon_drawdown_guard")

    def test_short_horizon_policy_caps_buy_intent_weight(self) -> None:
        policy = ShortHorizonRiskPolicy(max_position_weight_intraday=0.02)
        intent = OrderIntent(
            ticker="TEST",
            market="SIM",
            action=OrderAction.BUY,
            suggested_weight=0.20,
            confidence=0.7,
            valid_until=datetime.now(timezone.utc) + timedelta(minutes=1),
            reasoning_summary=("fast edge",),
            supporting_factors=("edge",),
            contradicting_factors=(),
            source_data_ids=("unit",),
        )

        capped = policy.cap_intent(intent)

        self.assertEqual(capped.suggested_weight, 0.02)
        self.assertIn("Intraday position capped", capped.reasoning_summary[-1])

    def test_realtime_runtime_endpoint_reports_low_latency_policy(self) -> None:
        client = TestClient(app)
        with patch.dict("os.environ", {"ONTOLOGY_ACCELERATOR": "NPU"}):
            data = client.get("/api/realtime/runtime").json()

        self.assertIn("acceleration", data)
        self.assertEqual(data["acceleration"]["latency_profile"], "low_latency")
        self.assertIn(5, data["acceleration"]["prediction_horizons_seconds"])
        self.assertIn("short_horizon_policy", data)

    def test_training_mode_starts_continuous_collection_until_stop(self) -> None:
        client = TestClient(app)
        with patch("app.web._start_live_worker") as start_worker:
            data = client.post("/api/operation-mode/start", json={"mode": "learning"}).json()

        self.assertTrue(data["ok"])
        self.assertEqual(data["training_status"], "continuous_collection_started")
        self.assertEqual(data["data_policy"]["analysis_input_stores"], ["data/store"])
        start_worker.assert_called_once_with("learning")

    def test_paper_api_test_starts_background_collection_without_blocking(self) -> None:
        client = TestClient(app)
        with (
            patch("app.web._start_live_worker") as start_worker,
            patch("app.web._start_streaming_demo", return_value="demo-test") as start_demo,
            patch("app.web._start_kis_connection_probe_background") as kis_probe_background,
            patch("app.web._get_or_refresh_live") as refresh_live,
        ):
            data = client.post("/api/operation-mode/start", json={"mode": "paper_trading"}).json()

        self.assertTrue(data["ok"])
        self.assertEqual(data["paper_trading_status"], "background_collection_started")
        self.assertEqual(data["paper_trading_kind"], "kis_paper_api")
        self.assertEqual(data["demo_id"], "demo-test")
        self.assertEqual(data["demo_status"], "initialized")
        self.assertEqual(data["kis_connection"]["status"], "checking")
        self.assertEqual(data["kis_connection"]["mode"], "paper")
        self.assertEqual(data["data_policy"]["analysis_input_stores"], ["data/store"])
        start_demo.assert_called_once()
        start_worker.assert_called_once_with("learning")
        kis_probe_background.assert_called_once_with(paper=True, include_account=True)
        refresh_live.assert_not_called()

    def test_live_api_test_checks_readiness_without_streaming_orders(self) -> None:
        client = TestClient(app)
        with (
            patch("app.web._start_live_worker") as start_worker,
            patch("app.web._start_streaming_demo") as start_demo,
            patch(
                "app.web._kis_connection_probe",
                return_value={"ok": True, "mode": "live", "account_checked": True, "actual_deposit": 1000000},
            ) as kis_probe,
            patch("app.web._get_or_refresh_live") as refresh_live,
        ):
            data = client.post("/api/operation-mode/start", json={"mode": "live_readiness"}).json()

        self.assertTrue(data["ok"])
        self.assertEqual(data["live_readiness_status"], "checked")
        self.assertEqual(data["live_readiness_kind"], "kis_live_readiness")
        self.assertEqual(data["kis_connection"]["mode"], "live")
        start_worker.assert_called_once_with("learning")
        start_demo.assert_not_called()
        kis_probe.assert_called_once_with(paper=False, include_account=True)
        refresh_live.assert_not_called()

    def test_live_readiness_status_keeps_checked_broker_account(self) -> None:
        client = TestClient(app)
        broker_state = {
            "ok": True,
            "mode": "live",
            "account_checked": True,
            "actual_deposit": 1000000,
            "holdings_count": 2,
            "account_suffix": "...28",
        }
        with (
            patch("app.web._start_live_worker"),
            patch("app.web._kis_connection_probe", return_value=broker_state),
            patch("app.web._get_or_refresh_live"),
        ):
            started = client.post("/api/operation-mode/start", json={"mode": "live_readiness"}).json()
            status = client.get("/api/operation-mode/status").json()

        self.assertTrue(started["ok"])
        self.assertEqual(status["active"]["mode"], "live_readiness")
        self.assertEqual(status["active"]["kis_connection"]["actual_deposit"], 1000000)
        self.assertEqual(status["active"]["kis_connection"]["holdings_count"], 2)
        self.assertEqual(status["kis_connection"]["account_suffix"], "...28")

    def test_status_uses_checked_live_account_as_operating_basis(self) -> None:
        client = TestClient(app)
        broker_state = {
            "ok": True,
            "mode": "live",
            "account_checked": True,
            "actual_deposit": 800000,
            "invested_value": 200000,
            "actual_equity": 1000000,
            "account_suffix": "...28",
        }
        fallback_snapshot = {
            "context": SimpleNamespace(
                account=SimpleNamespace(cash=123),
                report=SimpleNamespace(equity=456, cash_weight=0.27, daily_pnl_ratio=0.0),
                risk_results=(),
            ),
            "last_updated": None,
            "last_error": None,
        }
        with (
            patch("app.web._start_live_worker"),
            patch("app.web._kis_connection_probe", return_value=broker_state),
            patch("app.web._get_or_refresh_live", return_value=fallback_snapshot),
        ):
            client.post("/api/operation-mode/start", json={"mode": "live_readiness"})
            status = client.get("/api/status").json()

        self.assertEqual(status["basis_source"], "kis_live_account")
        self.assertEqual(status["cash"], 800000)
        self.assertEqual(status["equity"], 1000000)
        self.assertEqual(status["cash_weight"], 0.8)

    def test_paper_mode_auto_uses_checked_live_account_basis_as_initial_cash(self) -> None:
        client = TestClient(app)
        with web_module._live_lock:
            web_module._operation_mode_state["last_kis_connection"] = {
                "ok": True,
                "mode": "live",
                "account_checked": True,
                "actual_deposit": 700000,
                "invested_value": 300000,
                "actual_equity": 1000000,
                "account_suffix": "...28",
            }
        with (
            patch("app.web._start_live_worker"),
            patch("app.web._start_streaming_demo", return_value="demo-live-basis") as start_demo,
            patch("app.web._kis_connection_probe", return_value={"ok": True, "mode": "paper"}),
            patch("app.web._get_or_refresh_live"),
        ):
            data = client.post(
                "/api/operation-mode/start",
                json={
                    "mode": "paper_trading",
                    "target_return_rate": 0.02,
                    "period_minutes": 20,
                    "initial_cash": 10000000,
                    "initial_cash_source": "live_account",
                },
            ).json()

        self.assertTrue(data["ok"])
        self.assertEqual(data["initial_cash"], 1000000)
        self.assertEqual(data["initial_cash_source"], "kis_live_account")
        self.assertEqual(data["profit_gain_source"], "auto_goal_account_liquidity")
        self.assertGreater(data["profit_gain"], 0.25)
        self.assertEqual(start_demo.call_args.kwargs["initial_cash"], 1000000)
        self.assertEqual(start_demo.call_args.kwargs["profit_gain"], data["profit_gain"])

    def test_paper_trading_start_defaults_to_live_account_and_auto_gain(self) -> None:
        client = TestClient(app)
        with web_module._live_lock:
            web_module._operation_mode_state["last_kis_connection"] = {
                "ok": True,
                "mode": "live",
                "account_checked": True,
                "actual_deposit": 450000,
                "invested_value": 550000,
                "actual_equity": 1000000,
                "account_suffix": "...28",
            }
        with patch("app.web._start_streaming_demo", return_value="demo-auto") as start_demo:
            data = client.post(
                "/api/paper-trading/start",
                json={
                    "target_return_rate": 3,
                    "period_minutes": 120,
                    "initial_cash_source": "auto",
                },
            ).json()

        self.assertEqual(data["initial_cash"], 1000000)
        self.assertEqual(data["initial_cash_source"], "kis_live_account")
        self.assertEqual(data["profit_gain_source"], "auto_goal_account_liquidity")
        self.assertEqual(data["target_return_rate"], 0.03)
        self.assertEqual(start_demo.call_args.kwargs["initial_cash"], 1000000)
        self.assertEqual(start_demo.call_args.kwargs["profit_gain"], data["profit_gain"])

    def test_auto_initial_cash_uses_default_without_blocking_when_no_basis_is_cached(self) -> None:
        client = TestClient(app)
        with web_module._live_lock:
            web_module._operation_mode_state["last_kis_connection"] = None
        with (
            patch("app.web._start_auto_live_readiness_check") as auto_readiness,
            patch("app.web._start_streaming_demo", return_value="demo-auto-refresh") as start_demo,
        ):
            data = client.post(
                "/api/paper-trading/start",
                json={
                    "target_return_rate": 2,
                    "period_minutes": 390,
                    "initial_cash_source": "auto",
                },
            ).json()

        auto_readiness.assert_called_once_with()
        self.assertEqual(data["initial_cash"], 10000000)
        self.assertEqual(data["initial_cash_source"], "default_auto")
        self.assertEqual(start_demo.call_args.kwargs["initial_cash"], 10000000)

    def test_startup_live_readiness_runs_read_only_account_probe(self) -> None:
        class ImmediateThread:
            def __init__(self, target, name=None, daemon=None):
                self.target = target

            def start(self) -> None:
                self.target()

        live_connection = {
            "ok": True,
            "mode": "live",
            "account_checked": True,
            "actual_deposit": 600000,
            "invested_value": 400000,
            "actual_equity": 1000000,
            "account_suffix": "...28",
        }
        with web_module._live_lock:
            web_module._auto_live_readiness_started = False
            web_module._operation_mode_state["last_kis_connection"] = None

        with (
            patch("app.web.threading.Thread", ImmediateThread),
            patch("app.web._kis_connection_probe", return_value=live_connection) as probe,
        ):
            web_module._start_auto_live_readiness_check()

        probe.assert_called_once_with(paper=False, include_account=True)
        basis = web_module._last_live_account_basis()
        self.assertIsNotNone(basis)
        self.assertEqual(basis["equity"], 1000000)

    def test_research_diagnostics_uses_lightweight_cached_volume(self) -> None:
        client = TestClient(app)

        class Graph:
            nodes = {"A": object(), "B": object()}

            def triples(self):
                return (("A", "supports", "B"),)

        snapshot = {
            "research_result": SimpleNamespace(
                diagnostics={"events_count": 1, "live_data_present": True},
                skipped_sources=(),
                events=("event",),
                market_snapshots=tuple(range(50)),
            ),
            "context": SimpleNamespace(
                graph=Graph(),
                reasoning_paths=tuple(range(50)),
                ontology_runtime=SimpleNamespace(as_dict=lambda: {"uses_npu": False}),
            ),
            "stored_new_records": {"events": 1},
            "store_summary": {"events": 10, "market_snapshots": 20},
            "last_updated": None,
            "last_error": None,
            "is_refreshing": False,
        }
        with (
            patch("app.web._live_snapshot", return_value=snapshot),
            patch("app.web.LocalResearchStore.data_volume", side_effect=AssertionError("full scan should not run")),
        ):
            data = client.get("/api/research/diagnostics").json()

        self.assertEqual(data["data_volume"]["by_kind"]["events"], 10)
        self.assertEqual(len(data["market_snapshots"]), 25)
        self.assertEqual(len(data["reasoning_paths"]), 25)

    def test_live_snapshot_default_refresh_omits_full_graph_payload(self) -> None:
        client = TestClient(app)

        class Graph:
            nodes = {"A": object(), "B": object()}

            def triples(self):
                return (("A", "supports", "B"),)

        snapshot = {
            "research_result": SimpleNamespace(
                diagnostics={"events_count": 1},
                skipped_sources=(),
            ),
            "context": SimpleNamespace(
                account=SimpleNamespace(cash=1000),
                report=SimpleNamespace(equity=1000, cash_weight=1.0, daily_pnl_ratio=0.0),
                graph=Graph(),
                reasoning_paths=tuple(range(50)),
                ontology_runtime=SimpleNamespace(as_dict=lambda: {"uses_npu": False}),
            ),
            "store_summary": {"events": 10},
            "stored_new_records": {},
            "last_updated": None,
            "last_error": None,
            "is_refreshing": False,
        }
        with (
            patch("app.web._live_snapshot", return_value=snapshot),
            patch("app.web._get_or_refresh_live", side_effect=AssertionError("default refresh should use cache only")),
            patch("app.web._graph_payload", side_effect=AssertionError("full graph should not be built")),
        ):
            data = client.post("/api/live-snapshot", json={"force_refresh": False}).json()

        self.assertTrue(data["graph"]["summary_only"])
        self.assertEqual(data["graph"]["counts"]["links"], 1)
        self.assertNotIn("nodes", data["graph"])

    def test_force_refresh_schedules_collection_without_blocking_request(self) -> None:
        snapshot = {
            "research_result": SimpleNamespace(),
            "context": SimpleNamespace(),
            "context_mode": "paper_trading",
            "store_summary": {},
            "stored_new_records": {},
            "last_updated": datetime.now(),
            "last_error": None,
            "is_refreshing": False,
            "progress": {},
            "learning": {},
            "collection_log": [],
            "graph_payload": None,
            "graph_payload_context_id": None,
        }
        with (
            patch("app.web._active_operation_mode", return_value="paper_trading"),
            patch("app.web._live_snapshot", return_value=snapshot),
            patch("app.web._ensure_background_refresh") as ensure_refresh,
            patch("app.web._refresh_live_cache", side_effect=AssertionError("collection must run in the background")),
        ):
            result = web_module._get_or_refresh_live(force_refresh=True)

        self.assertIs(result, snapshot)
        ensure_refresh.assert_called_once()

    def test_mode_cache_clear_preserves_store_summary_for_diagnostics(self) -> None:
        with web_module._live_lock:
            previous = dict(web_module._live_state.get("store_summary") or {})
            try:
                web_module._live_state["store_summary"] = {
                    "events": 7,
                    "raw_records": 8,
                    "market_snapshots": 9,
                    "macro_metrics": 1,
                }
                web_module._clear_live_analysis_cache_unlocked()

                self.assertEqual(web_module._live_state["store_summary"]["events"], 7)
                self.assertEqual(web_module._live_state["store_summary"]["market_snapshots"], 9)
            finally:
                web_module._live_state["store_summary"] = previous

    def test_live_trading_button_mode_is_blocked_by_default(self) -> None:
        client = TestClient(app)
        with (
            patch("app.web._start_live_worker") as start_worker,
            patch(
                "app.web._kis_connection_probe",
                return_value={"ok": True, "mode": "live", "account_checked": True, "actual_deposit": 1000000},
            ) as kis_probe,
            patch("app.web.load_short_horizon_strategy_config", return_value={"execution": {"live_trading_enabled": False}}),
            patch.dict("os.environ", {"LIVE_TRADING_ENABLED": "false", "KIS_LIVE_ENABLED": "false"}),
        ):
            data = client.post("/api/operation-mode/start", json={"mode": "live_trading"}).json()

        self.assertTrue(data["ok"])
        self.assertEqual(data["live_trading_status"], "blocked")
        self.assertFalse(data["live_trading_enabled_by_config"])
        self.assertFalse(data["live_trading_enabled_by_env"])
        self.assertIn("blocked", data["live_trading_message"])
        start_worker.assert_called_once_with("learning")
        kis_probe.assert_called_once_with(paper=False, include_account=True)

    def test_live_trading_gate_can_be_armed_only_when_config_and_env_allow(self) -> None:
        client = TestClient(app)
        with (
            patch("app.web._start_live_worker"),
            patch(
                "app.web._kis_connection_probe",
                return_value={"ok": True, "mode": "live", "account_checked": True, "actual_deposit": 1000000},
            ),
            patch("app.web.load_short_horizon_strategy_config", return_value={"execution": {"live_trading_enabled": True}}),
            patch.dict("os.environ", {"LIVE_TRADING_ENABLED": "true", "KIS_LIVE_ENABLED": "true"}),
        ):
            data = client.post("/api/operation-mode/start", json={"mode": "live_trading"}).json()

        self.assertTrue(data["ok"])
        self.assertEqual(data["live_trading_status"], "armed")
        self.assertTrue(data["live_trading_enabled_by_config"])
        self.assertTrue(data["live_trading_enabled_by_env"])
        self.assertIn("RiskManager", data["live_trading_message"])

    def test_stop_learning_endpoint_keeps_continuous_collection_alive(self) -> None:
        client = TestClient(app)
        with patch("app.web._start_live_worker") as start_worker:
            data = client.post("/api/operation-mode/stop-learning").json()

        self.assertTrue(data["ok"])
        self.assertEqual(data["status"], "continuous")
        self.assertIn("collection_log", data)
        start_worker.assert_called_once_with("learning")

    def test_live_progress_reports_learning_schedule_and_collection_log(self) -> None:
        client = TestClient(app)
        data = client.get("/api/live-progress").json()

        self.assertIn("learning", data)
        self.assertIn("collection_log", data)
        self.assertIn("refresh_interval_seconds", data["learning"])


if __name__ == "__main__":
    unittest.main()
