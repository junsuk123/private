from __future__ import annotations

import json
import sys
import tempfile
import time
import unittest
from datetime import datetime, timedelta, timezone
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from fastapi.testclient import TestClient

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from app.realtime import OperationMode, OperationModeManager, ShortHorizonRiskPolicy
from app.schemas.domain import AccountSnapshot, FinalOrder, Holding, MarketSnapshot, OrderAction, OrderIntent, OrderSide, OrderType, SourceMetadata
from app.storage import StoredResearch
from app import web as web_module
from app.web import app


class RealtimeModesTest(unittest.TestCase):
    def setUp(self) -> None:
        with web_module._live_lock:
            web_module._operation_mode_state["last_kis_connection"] = None
            web_module._operation_mode_state["live_trading_baseline_equity"] = None

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

    def test_live_trading_allows_background_training_and_live_orders(self) -> None:
        state = OperationModeManager().start(OperationMode.LIVE_TRADING)

        self.assertTrue(state.live_orders_allowed)
        self.assertTrue(state.training_allowed)
        self.assertFalse(state.synthetic_data_allowed)

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
        self.assertIn("live_training", data)

    def test_cash_fit_keeps_sell_orders_without_cash_requirement(self) -> None:
        account = AccountSnapshot(cash=0.0, holdings=(), cash_by_currency={"KRW": 0.0, "USD": 0.0})
        sell_order = FinalOrder(
            ticker="SOXX",
            market="NASD",
            order_type=OrderType.LIMIT,
            side=OrderSide.SELL,
            quantity=1,
            limit_price=500.0,
        )
        buy_order = FinalOrder(
            ticker="QQQ",
            market="NASD",
            order_type=OrderType.LIMIT,
            side=OrderSide.BUY,
            quantity=1,
            limit_price=100.0,
        )

        kept, skipped = web_module._cash_fit_executable_orders([sell_order, buy_order], account)

        self.assertEqual(kept, [sell_order])
        self.assertEqual(len(skipped), 1)
        self.assertEqual(skipped[0]["ticker"], "QQQ")

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
        # 학습과 거래 플로우는 독립이므로 거래 시작은 학습 워커를 건드리지 않는다.
        self.assertEqual(data["paper_trading_status"], "trading_loop_started_independently")
        self.assertEqual(data["paper_trading_kind"], "kis_paper_api")
        self.assertEqual(data["demo_id"], "demo-test")
        self.assertEqual(data["demo_status"], "initialized")
        self.assertEqual(data["kis_connection"]["status"], "checking")
        self.assertEqual(data["kis_connection"]["mode"], "paper")
        self.assertEqual(data["data_policy"]["analysis_input_stores"], ["data/store"])
        start_demo.assert_called_once()
        start_worker.assert_not_called()
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
        # 점검/거래 플로우는 학습 워커와 독립적으로 동작한다.
        start_worker.assert_not_called()
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
            "krw_cash": 800000,
            "cash_by_currency": {"KRW": 800000, "USD": 12.34},
            "foreign_cash_by_currency": {"USD": 12.34},
            "base_currency": "KRW",
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
        self.assertEqual(status["krw_cash"], 800000)
        self.assertEqual(status["cash_by_currency"], {"KRW": 800000.0, "USD": 12.34})
        self.assertEqual(status["foreign_cash_by_currency"], {"USD": 12.34})
        self.assertEqual(status["equity"], 1000000)
        self.assertEqual(status["cash_weight"], 0.8)

    def test_status_keeps_orderable_krw_cash_separate_from_foreign_cash(self) -> None:
        client = TestClient(app)
        broker_state = {
            "ok": True,
            "mode": "live",
            "account_checked": True,
            "actual_deposit": 800000,
            "krw_cash": 800000,
            "foreign_cash_krw": 12000,
            "cash": 812000,
            "cash_by_currency": {"KRW": 800000, "USD": 10.0},
            "foreign_cash_by_currency": {"USD": 10.0},
            "base_currency": "KRW",
            "invested_value": 188000,
            "actual_equity": 1000000,
            "account_suffix": "...28",
        }
        with web_module._live_lock:
            web_module._operation_mode_state["last_kis_connection"] = broker_state

        status = client.get("/api/status").json()

        self.assertEqual(status["cash"], 800000)
        self.assertEqual(status["cash_equivalent_krw"], 812000)
        self.assertEqual(status["krw_cash"], 800000)
        self.assertEqual(status["foreign_cash_krw"], 12000)
        self.assertEqual(status["cash_weight"], 0.8)

    def test_status_and_live_snapshot_use_same_live_account_cash_basis(self) -> None:
        client = TestClient(app)
        broker_state = {
            "ok": True,
            "mode": "live",
            "account_checked": True,
            "actual_deposit": 2401,
            "krw_cash": 2401,
            "foreign_cash_krw": 4963.63,
            "cash": 2401,
            "cash_equivalent_krw": 7364.63,
            "cash_by_currency": {"KRW": 2401, "USD": 3.22},
            "foreign_cash_by_currency": {"USD": 3.22},
            "base_currency": "KRW",
            "invested_value": 2580,
            "actual_equity": 9944.63,
            "account_suffix": "...28",
        }
        snapshot = {
            "research_result": SimpleNamespace(diagnostics={}, skipped_sources=()),
            "context": SimpleNamespace(
                account=SimpleNamespace(cash=10_000_000),
                report=SimpleNamespace(equity=10_000_000, cash_weight=1.0, daily_pnl_ratio=0.0),
                graph=SimpleNamespace(nodes={}, triples=lambda: ()),
                reasoning_paths=(),
                ontology_runtime=SimpleNamespace(as_dict=lambda: {"uses_npu": False}),
                risk_results=(),
            ),
            "is_refreshing": False,
            "store_summary": {},
            "stored_new_records": {},
            "last_updated": None,
            "last_error": None,
        }
        with web_module._live_lock:
            web_module._operation_mode_state["last_kis_connection"] = broker_state
        with patch("app.web._live_snapshot", return_value=snapshot):
            status = client.get("/api/status").json()
            live_snapshot = client.post("/api/live-snapshot", json={"force_refresh": False, "include_graph": False}).json()

        self.assertEqual(status["cash"], 2401)
        self.assertEqual(live_snapshot["status"]["cash"], 2401)
        self.assertEqual(status["cash_equivalent_krw"], 7364.63)
        self.assertEqual(live_snapshot["status"]["cash_equivalent_krw"], 7364.63)
        self.assertEqual(status["equity"], live_snapshot["status"]["equity"])

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
        self.assertEqual(data["initial_cash"], 700000)
        self.assertEqual(data["initial_cash_source"], "kis_live_account")
        self.assertEqual(data["profit_gain_source"], "auto_goal_account_liquidity")
        self.assertGreater(data["profit_gain"], 0.25)
        self.assertEqual(start_demo.call_args.kwargs["initial_cash"], 700000)
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

        self.assertEqual(data["initial_cash"], 450000)
        self.assertEqual(data["initial_cash_source"], "kis_live_account")
        self.assertEqual(data["profit_gain_source"], "auto_goal_account_liquidity")
        self.assertEqual(data["target_return_rate"], 0.03)
        self.assertEqual(start_demo.call_args.kwargs["initial_cash"], 450000)
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
            patch("app.web._start_kis_realtime_collector"),
            patch("app.web._start_realtime_trading_engine") as start_engine,
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
        self.assertFalse(data["runtime_gate"]["ok"])
        # 학습 워커는 거래와 독립이므로 호출되지 않고, 실시간 거래 엔진이 대신 가동된다.
        start_worker.assert_not_called()
        start_engine.assert_called_once()
        kis_probe.assert_called_once_with(paper=False, include_account=True)

    def test_live_trading_gate_can_be_armed_only_when_config_and_env_allow(self) -> None:
        client = TestClient(app)
        with (
            patch("app.web._start_live_worker"),
            patch("app.web._start_kis_realtime_collector"),
            patch("app.web._start_realtime_trading_engine"),
            patch(
                "app.web._kis_connection_probe",
                return_value={"ok": True, "mode": "live", "account_checked": True, "actual_deposit": 1000000},
            ),
            patch("app.web.load_short_horizon_strategy_config", return_value={"execution": {"live_trading_enabled": True}}),
            patch("app.web.evaluate_live_runtime_gates", return_value=SimpleNamespace(ok=True, failures=())),
            patch.dict("os.environ", {"LIVE_TRADING_ENABLED": "true", "KIS_LIVE_ENABLED": "true"}),
        ):
            data = client.post("/api/operation-mode/start", json={"mode": "live_trading"}).json()

        self.assertTrue(data["ok"])
        self.assertEqual(data["live_trading_status"], "armed")
        self.assertTrue(data["live_trading_enabled_by_config"])
        self.assertTrue(data["live_trading_enabled_by_env"])
        self.assertTrue(data["runtime_gate"]["ok"])
        self.assertIn("RiskManager", data["live_trading_message"])

    def test_live_trading_start_preserves_cached_analysis_context(self) -> None:
        client = TestClient(app)
        cached_context = SimpleNamespace(
            markets=(),
            reasoning_paths=(),
            candidate_selection=None,
        )
        with web_module._live_lock:
            previous_context = web_module._live_state.get("context")
            previous_mode = web_module._live_state.get("context_mode")
            web_module._live_state["context"] = cached_context
            web_module._live_state["context_mode"] = "learning"
        try:
            with (
                patch("app.web._start_kis_realtime_collector"),
                patch("app.web._start_realtime_trading_engine"),
                patch("app.web._ensure_background_refresh") as ensure_refresh,
                patch(
                    "app.web._kis_connection_probe",
                    return_value={"ok": True, "mode": "live", "account_checked": True, "actual_deposit": 1000000},
                ),
                patch("app.web.load_short_horizon_strategy_config", return_value={"execution": {"live_trading_enabled": True}}),
                patch("app.web.evaluate_live_runtime_gates", return_value=SimpleNamespace(ok=True, failures=())),
                patch.dict("os.environ", {"LIVE_TRADING_ENABLED": "true", "KIS_LIVE_ENABLED": "true"}),
            ):
                data = client.post("/api/operation-mode/start", json={"mode": "live_trading"}).json()

            self.assertEqual(data["live_trading_status"], "armed")
            with web_module._live_lock:
                self.assertIs(web_module._live_state["context"], cached_context)
                self.assertEqual(web_module._live_state["context_mode"], "learning")
            ensure_refresh.assert_called_once()
        finally:
            with web_module._live_lock:
                web_module._live_state["context"] = previous_context
                web_module._live_state["context_mode"] = previous_mode

    def test_realtime_buy_candidates_include_cached_context_candidates_first(self) -> None:
        account = AccountSnapshot(cash=100000.0, holdings=(), cash_by_currency={"KRW": 100000.0})
        cached_context = SimpleNamespace(
            markets=(
                MarketSnapshot(
                    "005930",
                    "KOSPI",
                    "Samsung",
                    "Technology",
                    70000.0,
                    10_000_000,
                    0.02,
                    SourceMetadata("unit", datetime.now(timezone.utc)),
                ),
                MarketSnapshot(
                    "000660",
                    "KOSPI",
                    "SK hynix",
                    "Technology",
                    90000.0,
                    10_000_000,
                    0.02,
                    SourceMetadata("unit", datetime.now(timezone.utc)),
                ),
            ),
            reasoning_paths=(SimpleNamespace(ticker="005930", conclusion="BuyCandidate"),),
            candidate_selection=SimpleNamespace(candidate_stocks=("000660",)),
        )
        with web_module._live_lock:
            previous_context = web_module._live_state.get("context")
            web_module._live_state["context"] = cached_context
        try:
            with (
                patch("app.web._active_live_market_groups", return_value=("KRX",)),
                patch("app.web._load_realtime_collection_symbols", return_value=("111111",)),
                patch("app.web._live_account_snapshot_for_analysis", return_value=account),
                patch("app.web._live_affordable_buy_candidate_symbols", return_value=()),
                patch("app.web.RealtimeMarketDataStore") as store_cls,
                patch("app.web._cached_volume_surge_symbols", return_value=()),
                patch.dict("os.environ", {"REALTIME_BUY_CANDIDATE_LIMIT": "5"}),
            ):
                store_cls.return_value.active_symbols.return_value = ("222222",)
                store_cls.return_value.latest_tick.side_effect = lambda symbol: SimpleNamespace(price=5000.0) if symbol in {"111111", "222222"} else None
                candidates = web_module._realtime_buy_candidates()

            self.assertEqual(candidates[:2], ("005930", "000660"))
            self.assertIn("111111", candidates)
            self.assertIn("222222", candidates)
        finally:
            with web_module._live_lock:
                web_module._live_state["context"] = previous_context

    def test_cached_context_buy_candidates_exclude_symbols_above_orderable_cash(self) -> None:
        account = AccountSnapshot(cash=100000.0, holdings=(), cash_by_currency={"KRW": 100000.0})
        cached_context = SimpleNamespace(
            markets=(
                MarketSnapshot(
                    "005930",
                    "KOSPI",
                    "Samsung",
                    "Technology",
                    450780.0,
                    10_000_000,
                    0.02,
                    SourceMetadata("unit", datetime.now(timezone.utc)),
                ),
                MarketSnapshot(
                    "000001",
                    "KOSPI",
                    "Affordable KR",
                    "Technology",
                    4000.0,
                    10_000_000,
                    0.02,
                    SourceMetadata("unit", datetime.now(timezone.utc)),
                ),
            ),
            reasoning_paths=(
                SimpleNamespace(ticker="005930", conclusion="BuyCandidate"),
                SimpleNamespace(ticker="000001", conclusion="BuyCandidate"),
            ),
            candidate_selection=SimpleNamespace(candidate_stocks=()),
        )
        with web_module._live_lock:
            previous_context = web_module._live_state.get("context")
            web_module._live_state["context"] = cached_context
        try:
            with (
                patch("app.web._active_live_market_groups", return_value=("KRX",)),
                patch("app.web._live_account_snapshot_for_analysis", return_value=account),
            ):
                candidates = web_module._cached_context_buy_candidates()

            self.assertEqual(candidates, ("000001",))
        finally:
            with web_module._live_lock:
                web_module._live_state["context"] = previous_context

    def test_realtime_buy_candidates_include_affordable_discovery_when_context_empty(self) -> None:
        class FakeKisClient:
            prices = {"005930": 70_000.0, "000660": 150_000.0, "AAPL": 300.0, "MSFT": 20.0}

            def __init__(self, *args, **kwargs) -> None:
                pass

            def get_market_snapshot(self, symbol, market, company_name=None, sector=None):
                return MarketSnapshot(
                    symbol,
                    market,
                    company_name or symbol,
                    sector or "Unknown",
                    self.prices[symbol],
                    10_000_000,
                    0.02,
                    SourceMetadata(
                        "KIS broker quote",
                        datetime.now(timezone.utc),
                        source_type="broker_api",
                        trust_level=5,
                        is_realtime=True,
                        quality_score=1.0,
                    ),
                )

        account = AccountSnapshot(
            cash=100000.0,
            holdings=(),
            cash_by_currency={"KRW": 100000.0, "USD": 20.0},
            cash_equivalent_krw=130000.0,
        )
        with web_module._live_lock:
            previous_context = web_module._live_state.get("context")
            web_module._live_state["context"] = None
        try:
            with (
                patch("app.web._active_live_market_groups", return_value=("KRX", "US")),
                patch("app.web._live_account_snapshot_for_analysis", return_value=account),
                patch("app.web._held_or_recent_buy_tickers", return_value=set()),
                patch("app.web.KisDevelopersApiClient", FakeKisClient),
                patch("app.web.load_krx_listed_universe", return_value=("005930.KS", "000660.KS")),
                patch("app.web.load_us_listed_universe", return_value=("AAPL", "MSFT")),
                patch("app.web._load_realtime_collection_symbols", return_value=()),
                patch("app.web.RealtimeMarketDataStore") as store_cls,
                patch("app.web._cached_volume_surge_symbols", return_value=()),
                patch.dict("os.environ", {"REALTIME_BUY_CANDIDATE_LIMIT": "4"}),
            ):
                store_cls.return_value.active_symbols.return_value = ()
                store_cls.return_value.latest_tick.return_value = None
                candidates = web_module._realtime_buy_candidates()

            self.assertIn("005930", candidates)
            self.assertNotIn("000660", candidates)
            self.assertNotIn("AAPL", candidates)
            self.assertIn("MSFT", candidates)
        finally:
            with web_module._live_lock:
                web_module._live_state["context"] = previous_context

    def test_live_order_journal_snapshot_reports_submitted_and_blocked_orders(self) -> None:
        events = [
            {
                "event_type": "live_order_blocked",
                "recorded_at": "2026-06-30T00:00:00+00:00",
                "payload": {
                    "order": {"ticker": "005930", "market": "KR", "side": "BUY", "quantity": 1, "limit_price": 70000},
                    "reason_codes": ["LIVE_ORDER_SUBMIT_ENABLED_NOT_TRUE"],
                },
            },
            {
                "event_type": "live_order_submitted",
                "recorded_at": "2026-06-30T00:01:00+00:00",
                "payload": {
                    "order": {"ticker": "SOXX", "market": "US-LISTED", "side": "BUY", "quantity": 1, "limit_price": 624.71},
                    "broker_order_id": "OVRS000010",
                    "status": "submitted",
                },
            },
        ]
        with tempfile.TemporaryDirectory() as tmp:
            journal_path = Path(tmp) / "live-orders.jsonl"
            journal_path.write_text("\n".join(json.dumps(event) for event in events), encoding="utf-8")

            snapshot = web_module._live_order_journal_snapshot(journal_path)

        self.assertEqual(snapshot["orders_count"], 2)
        self.assertEqual(snapshot["submitted_count"], 1)
        self.assertEqual(snapshot["blocked_count"], 1)
        self.assertEqual(snapshot["recent_orders"][0]["ticker"], "005930")
        self.assertEqual(snapshot["recent_executions"][0]["broker_order_id"], "OVRS000010")

    def test_live_trading_progress_exposes_runtime_gate_and_order_journal(self) -> None:
        client = TestClient(app)
        journal = {
            "path": "logs/live-orders.jsonl",
            "orders_count": 1,
            "submitted_count": 0,
            "blocked_count": 1,
            "error_count": 0,
            "recent_orders": [
                {
                    "event_type": "live_order_blocked",
                    "ticker": "005930",
                    "market": "KR",
                    "side": "BUY",
                    "quantity": 1,
                    "limit_price": 70000,
                    "reason_codes": ("MANUAL_ARMING_FILE_MISSING",),
                }
            ],
            "recent_executions": [],
        }
        connection = {
            "ok": True,
            "mode": "live",
            "account_checked": True,
            "actual_deposit": 800000,
            "krw_cash": 800000,
            "cash": 800000,
            "invested_value": 200000,
            "actual_equity": 1000000,
            "holdings": 1,
            "holdings_count": 1,
            "positions": [
                {
                    "ticker": "005930",
                    "market": "KR",
                    "quantity": 2,
                    "average_price": 50000,
                    "last_price": 100000,
                    "market_value": 200000,
                    "unrealized_pnl": 100000,
                    "currency": "KRW",
                }
            ],
        }
        with (
            patch("app.web._kis_connection_probe", return_value=connection),
            patch("app.web._live_snapshot", return_value={"live_execution_summary": {"submitted": 0}}),
            patch("app.web.evaluate_live_runtime_gates", return_value=SimpleNamespace(ok=False, failures=("MANUAL_ARMING_FILE_MISSING",))),
            patch("app.web._live_order_journal_snapshot", return_value=journal),
        ):
            data = client.get("/api/live-trading/progress").json()

        self.assertFalse(data["runtime_gate"]["ok"])
        self.assertEqual(data["orders_count"], 1)
        self.assertEqual(data["live_order_journal"]["blocked_count"], 1)
        self.assertEqual(data["recent_orders"][0]["ticker"], "005930")
        self.assertEqual(data["positions"][0]["ticker"], "005930")
        self.assertEqual(data["connection"]["positions"][0]["quantity"], 2)
        self.assertIn("MANUAL_ARMING_FILE_MISSING", data["message"])
        with web_module._live_lock:
            cached = web_module._operation_mode_state["last_kis_connection"]
        self.assertEqual(cached["positions"][0]["ticker"], "005930")

    def test_live_trading_progress_reconciles_filled_buy_before_balance_refresh(self) -> None:
        client = TestClient(app)
        journal = {
            "path": "logs/live-orders.jsonl",
            "orders_count": 1,
            "submitted_count": 1,
            "blocked_count": 0,
            "error_count": 0,
            "recent_orders": [],
            "recent_executions": [],
            "submitted_orders": [
                {
                    "event_type": "live_order_submitted",
                    "recorded_at": datetime.now(timezone.utc).isoformat(),
                    "ticker": "288180",
                    "market": "KR",
                    "side": "BUY",
                    "quantity": 1,
                    "limit_price": 8700,
                    "broker_order_id": "0032617900",
                    "status": "ACCEPTED",
                }
            ],
        }
        connection = {
            "ok": True,
            "mode": "live",
            "account_checked": True,
            "actual_deposit": 63303,
            "krw_cash": 63303,
            "cash": 63303,
            "invested_value": 2600,
            "actual_equity": 209555,
            "holdings": 1,
            "holdings_count": 1,
            "positions": [
                {
                    "ticker": "012860",
                    "market": "KR",
                    "quantity": 1,
                    "average_price": 2610,
                    "last_price": 2600,
                    "market_value": 2600,
                    "market_value_krw": 2600,
                    "unrealized_pnl": -10,
                    "currency": "KRW",
                }
            ],
        }

        class FilledOrderBroker:
            def __init__(self, *args, **kwargs) -> None:
                pass

            def get_order_status(self, order_id: str) -> SimpleNamespace:
                return SimpleNamespace(
                    order_id=order_id,
                    ticker="288180",
                    side=OrderSide.BUY,
                    quantity=1,
                    price=8700,
                    executed_value=8700,
                    status="FILLED",
                    message="filled",
                    executed_at=datetime.now(timezone.utc),
                )

        with (
            patch("app.web._kis_connection_probe", return_value=connection),
            patch("app.web.KisDevelopersApiClient", FilledOrderBroker),
            patch("app.web._live_snapshot", return_value={"live_execution_summary": {"submitted": 1}}),
            patch("app.web.evaluate_live_runtime_gates", return_value=SimpleNamespace(ok=False, failures=("LIVE_TRADING_ENABLED_NOT_TRUE",))),
            patch("app.web._live_order_journal_snapshot", return_value=journal),
        ):
            data = client.get("/api/live-trading/progress").json()
            status = client.get("/api/status").json()

        self.assertEqual([item["ticker"] for item in data["positions"]], ["012860", "288180"])
        self.assertEqual(data["pending_positions"][0]["ticker"], "288180")
        self.assertEqual(data["pending_positions"][0]["position_state"], "pending_balance")
        self.assertEqual(data["connection"]["holdings_count"], 2)
        self.assertEqual([item["ticker"] for item in status["positions"]], ["012860", "288180"])

    def test_status_exposes_live_positions_for_gui_refresh(self) -> None:
        client = TestClient(app)
        connection = {
            "ok": True,
            "mode": "live",
            "account_checked": True,
            "actual_deposit": 63303,
            "krw_cash": 63303,
            "cash": 63303,
            "invested_value": 90000,
            "actual_equity": 153303,
            "account_suffix": "...28",
            "positions": [
                {
                    "ticker": "012860",
                    "market": "KR",
                    "quantity": 1,
                    "average_price": 2610,
                    "last_price": 2600,
                    "market_value": 2600,
                    "unrealized_pnl": -10,
                    "currency": "KRW",
                },
                {
                    "ticker": "LAUR",
                    "market": "NASD",
                    "quantity": 1,
                    "average_price": 36.295,
                    "last_price": 36.32,
                    "market_value": 36.32,
                    "unrealized_pnl": 0.025,
                    "currency": "USD",
                },
            ],
        }
        with web_module._live_lock:
            previous = web_module._operation_mode_state.get("last_kis_connection")
            previous_at = web_module._operation_mode_state.get("last_kis_connection_checked_at")
            web_module._operation_mode_state["last_kis_connection"] = connection
            web_module._operation_mode_state["last_kis_connection_checked_at"] = time.time()
        try:
            data = client.get("/api/status").json()
        finally:
            with web_module._live_lock:
                web_module._operation_mode_state["last_kis_connection"] = previous
                web_module._operation_mode_state["last_kis_connection_checked_at"] = previous_at

        self.assertEqual(data["basis_source"], "kis_live_account")
        self.assertTrue(data["account_checked"])
        self.assertEqual(data["holdings_count"], 2)
        self.assertEqual([item["ticker"] for item in data["positions"]], ["012860", "LAUR"])

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

    def test_live_broker_research_keeps_only_affordable_markets_for_account(self) -> None:
        now = datetime.now(timezone.utc)
        source = SourceMetadata(
            source_name="KIS broker quote",
            retrieved_at=now,
            source_type="broker_api",
            trust_level=5,
            observed_at=now,
            is_realtime=True,
            quality_score=1.0,
        )
        stored = StoredResearch(
            events=(),
            raw_records=(),
            market_snapshots=(
                MarketSnapshot("MSFT", "NASDAQ", "Microsoft", "Technology", 367.6, 10_000_000, 0.02, source),
                MarketSnapshot("PENNY", "NASDAQ", "Affordable US", "Technology", 2.5, 10_000_000, 0.02, source),
                MarketSnapshot("005930", "KOSPI", "Samsung", "Technology", 323000.0, 10_000_000, 0.02, source),
                MarketSnapshot("000001", "KOSPI", "Affordable KR", "Technology", 4000.0, 10_000_000, 0.02, source),
            ),
            macro_metrics=(),
            realtime_quotes=(),
            realtime_executions=(),
            graph_triples=(),
            reasoning_paths=(),
        )
        account = AccountSnapshot(
            cash=5011.0,
            holdings=(),
            cash_by_currency={"KRW": 5011.0, "USD": 3.22},
            cash_equivalent_krw=9983.0,
        )

        with patch("app.web._active_live_market_groups", return_value=("US", "KRX")):
            filtered = web_module._live_broker_only_research(stored, account=account)

        self.assertEqual(tuple(market.ticker for market in filtered.market_snapshots), ("PENNY", "000001"))

    def test_live_broker_research_keeps_held_overseas_positions_even_when_not_buy_affordable(self) -> None:
        now = datetime.now(timezone.utc)
        source = SourceMetadata(
            source_name="KIS broker quote",
            retrieved_at=now,
            source_type="broker_api",
            trust_level=5,
            observed_at=now,
            is_realtime=True,
            quality_score=1.0,
        )
        stored = StoredResearch(
            events=(),
            raw_records=(),
            market_snapshots=(
                MarketSnapshot("AAME", "NASDAQ", "Held US", "Technology", 1.72, 10_000_000, 0.02, source),
                MarketSnapshot("AAPL", "NASDAQ", "Apple", "Technology", 287.5, 10_000_000, 0.02, source),
            ),
            macro_metrics=(),
            realtime_quotes=(),
            realtime_executions=(),
            graph_triples=(),
            reasoning_paths=(),
        )
        account = AccountSnapshot(
            cash=2401.0,
            holdings=(Holding("AAME", "NASDAQ", "Held US", "Technology", 1, 1.71, 1.72),),
            cash_by_currency={"KRW": 2401.0, "USD": 0.49},
            cash_equivalent_krw=7376.28,
        )

        with patch("app.web._active_live_market_groups", return_value=("US",)):
            filtered = web_module._live_broker_only_research(stored, account=account)

        self.assertEqual(tuple(market.ticker for market in filtered.market_snapshots), ("AAME",))

    def test_live_broker_targets_include_domestic_and_overseas_candidates(self) -> None:
        now = datetime.now(timezone.utc)
        source = SourceMetadata(
            source_name="listed_universe_reference",
            retrieved_at=now,
        )
        stored = StoredResearch(
            events=(),
            raw_records=(),
            market_snapshots=(
                MarketSnapshot("005930", "KOSPI", "Samsung", "Technology", 70_000.0, 10_000_000, 0.02, source),
                MarketSnapshot("MSFT", "NASDAQ", "Microsoft", "Technology", 367.6, 10_000_000, 0.02, source),
            ),
            macro_metrics=(),
            realtime_quotes=(),
            realtime_executions=(),
            graph_triples=(),
            reasoning_paths=(
                SimpleNamespace(ticker="005930", conclusion="BuyCandidate"),
                SimpleNamespace(ticker="MSFT", conclusion="BuyCandidate"),
            ),
        )

        with patch("app.web._active_live_market_groups", return_value=("US", "KRX")):
            targets = web_module._live_broker_targets_for_active_session(stored)

        self.assertEqual(targets, ("005930", "MSFT"))

    def test_live_affordable_us_discovery_adds_symbols_for_small_usd_balance(self) -> None:
        now = datetime.now(timezone.utc)
        source = SourceMetadata(source_name="listed_universe_reference", retrieved_at=now)
        stored = StoredResearch(
            events=(),
            raw_records=(),
            market_snapshots=(),
            macro_metrics=(),
            realtime_quotes=(),
            realtime_executions=(),
            graph_triples=(),
            reasoning_paths=(),
        )
        account = AccountSnapshot(
            cash=2401.0,
            holdings=(),
            cash_by_currency={"KRW": 2401.0, "USD": 3.22},
            cash_equivalent_krw=7364.63,
        )

        with (
            patch("app.web._is_live_market_extended_open", return_value=True),
            patch("app.web.load_us_listed_universe", return_value=("PENNY", "MICRO", "BIG")),
            patch("app.web._rotated_symbols", side_effect=lambda symbols: symbols),
            patch.dict("os.environ", {"LIVE_US_AFFORDABLE_DISCOVERY_LIMIT": "2"}),
        ):
            targets = web_module._live_affordable_us_discovery_targets(stored, account)

        self.assertEqual(tuple(target.ticker for target in targets), ("PENNY", "MICRO"))
        self.assertTrue(all(target.market == "NASDAQ" for target in targets))

    def test_live_affordable_us_discovery_excludes_held_and_recent_buy_tickers(self) -> None:
        now = datetime.now(timezone.utc)
        source = SourceMetadata(source_name="listed_universe_reference", retrieved_at=now)
        stored = StoredResearch(
            events=(),
            raw_records=(),
            market_snapshots=(),
            macro_metrics=(),
            realtime_quotes=(),
            realtime_executions=(),
            graph_triples=(),
            reasoning_paths=(),
        )
        account = AccountSnapshot(
            cash=2401.0,
            holdings=(Holding("PENNY", "NASDAQ", "PENNY", "Unknown", 1, 1.1, 1.1),),
            cash_by_currency={"KRW": 2401.0, "USD": 3.22},
            cash_equivalent_krw=7364.63,
        )

        with (
            patch("app.web._is_live_market_extended_open", return_value=True),
            patch("app.web.load_us_listed_universe", return_value=("PENNY", "MICRO", "BIG")),
            patch("app.web._recent_live_buy_tickers", return_value={"MICRO"}),
            patch("app.web._rotated_symbols", side_effect=lambda symbols: symbols),
            patch.dict("os.environ", {"LIVE_US_AFFORDABLE_DISCOVERY_LIMIT": "2"}),
        ):
            targets = web_module._live_affordable_us_discovery_targets(stored, account)

        self.assertEqual(tuple(target.ticker for target in targets), ("BIG",))

    def test_live_trading_worker_uses_live_refresh_interval_even_when_learning_is_active(self) -> None:
        self.assertEqual(
            web_module._live_worker_interval_seconds(True, "live_trading"),
            web_module.LIVE_REFRESH_SECONDS,
        )
        self.assertEqual(
            web_module._live_worker_interval_seconds(True, "learning"),
            web_module.LEARNING_COLLECTION_INTERVAL_SECONDS,
        )

    def test_live_probe_uses_orderable_usd_without_inflating_us_position_krw(self) -> None:
        class FakeKisClient:
            class Endpoints:
                base_url = "https://openapi.koreainvestment.com:9443"

            class Credentials:
                account_no = "12345678"

            endpoints = Endpoints()
            credentials = Credentials()
            token_source = "cache"

            def __init__(self, *args, **kwargs) -> None:
                pass

            def issue_access_token(self) -> str:
                return "token"

            def get_portfolio(self):
                account = AccountSnapshot(
                    cash=2401.0,
                    holdings=(
                        Holding("AACG", "NASD", "AACG", "Unknown", 1, 1.01, 1.01),
                        Holding("AAME", "NASD", "AAME", "Unknown", 1, 1.71, 1.71),
                    ),
                    cash_by_currency={"KRW": 2401.0, "USD": 0.49},
                    cash_equivalent_krw=7376.28,
                )
                return SimpleNamespace(account=account)

        with patch("app.web.KisDevelopersApiClient", FakeKisClient):
            connection = web_module._kis_connection_probe(paper=False, include_account=True)

        self.assertEqual(connection["cash_by_currency"]["USD"], 0.49)
        self.assertAlmostEqual(connection["positions"][0]["market_value_krw"], 1565.0, delta=5.0)
        self.assertLess(connection["invested_value"], 5000.0)

    def test_live_execution_reports_market_closed_when_no_trading_session_is_open(self) -> None:
        context = SimpleNamespace(
            intents=(),
            risk_results=(),
            signals=(),
            markets=(),
            account=AccountSnapshot(cash=2401.0, holdings=(), cash_by_currency={"KRW": 2401.0}),
        )

        with patch("app.web._active_live_market_groups", return_value=()):
            summary = web_module._run_live_trading_execution_cycle(context)

        self.assertEqual(summary["reason"], "MARKET_SESSION_CLOSED")
        self.assertEqual(summary["diagnostics"]["message"], "No supported KIS live trading session is open.")

    def test_live_market_extended_session_includes_us_premarket(self) -> None:
        premarket = datetime(2026, 6, 30, 10, 30, tzinfo=timezone.utc)

        self.assertTrue(web_module._is_live_market_extended_open("US", premarket))
        self.assertFalse(web_module._is_live_market_core_open("US", premarket))

    def test_live_market_extended_session_includes_krx_after_hours(self) -> None:
        after_hours = datetime(2026, 6, 30, 7, 30, tzinfo=timezone.utc)

        self.assertTrue(web_module._is_live_market_extended_open("KRX", after_hours))
        self.assertFalse(web_module._is_live_market_core_open("KRX", after_hours))

    def test_live_market_extended_session_includes_krx_opening_auction(self) -> None:
        opening_auction = datetime(2026, 6, 30, 23, 45, tzinfo=timezone.utc)

        self.assertTrue(web_module._is_live_market_extended_open("KRX", opening_auction))
        self.assertFalse(web_module._is_live_market_core_open("KRX", opening_auction))

    def test_live_affordable_krx_discovery_default_limit_is_broader_for_small_cash(self) -> None:
        stored = StoredResearch(
            events=(),
            raw_records=(),
            market_snapshots=(),
            macro_metrics=(),
            realtime_quotes=(),
            realtime_executions=(),
            graph_triples=(),
            reasoning_paths=(),
        )
        account = AccountSnapshot(cash=102413.0, holdings=(), cash_by_currency={"KRW": 102413.0})
        universe = tuple(f"{index:06d}.KS" for index in range(1, 321))

        with (
            patch("app.web._is_live_market_extended_open", return_value=True),
            patch("app.web.load_krx_listed_universe", return_value=universe),
        ):
            targets = web_module._live_affordable_krx_discovery_targets(stored, account)

        self.assertEqual(len(targets), 300)


if __name__ == "__main__":
    unittest.main()
