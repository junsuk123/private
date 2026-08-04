from __future__ import annotations

import json
import sys
import tempfile
import time
import unittest
from contextlib import closing
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
            web_module._operation_mode_state["stable_account_basis"] = None
            web_module._operation_mode_state["account_degraded_since"] = None
            web_module._affordable_candidate_cache.update({"key": None, "at": 0.0, "symbols": ()})
            web_module._live_buy_candidate_backoff_until.clear()
            web_module._volume_surge_warm_cache.update({"at": 0.0, "symbols": ()})
            web_module._us_learning_watchlist_cache.update(
                {
                    "at": 0.0,
                    "cash_usd": None,
                    "symbols": (),
                    "pool": (),
                    "rotation_index": 0,
                }
            )
        from app.graph import macro_micro_feed

        macro_micro_feed.clear()

    def _isolate_live_account_refresh(self) -> None:
        # Hermetic isolation for the status/basis tests: on a machine with working live
        # KIS credentials the status path fetches the REAL account via
        # _refresh_live_account_basis_for_auto, overriding the mocked last_kis_connection.
        # Force it to None so the basis is driven only by the test's explicit mock (as in
        # CI where live KIS is unavailable). Applied per-test (not in setUp) because the
        # live-readiness startup test needs the real refresh path with a mocked probe.
        patcher = patch("app.web._refresh_live_account_basis_for_auto", return_value=None)
        patcher.start()
        self.addCleanup(patcher.stop)

    def test_account_stabilizer_rejects_missing_position_without_sale_cash(self) -> None:
        stable = {
            "equity": 200_000.0,
            "cash_equivalent_krw": 150_000.0,
            "krw_cash": 100_000.0,
            "foreign_cash_krw": 50_000.0,
            "positions": [{"ticker": "F", "quantity": 2, "currency": "USD"}],
        }
        partial = {
            **stable,
            "cash_equivalent_krw": 100_000.0,
            "krw_cash": 50_000.0,
            "positions": [],
        }

        self.assertIs(web_module._stabilize_account_basis(stable), stable)
        self.assertIs(web_module._stabilize_account_basis(partial), stable)

    def test_account_basis_merge_preserves_transiently_missing_orderable_cash(self) -> None:
        previous = {
            "equity": 200_000.0,
            "krw_cash": 100_000.0,
            "foreign_cash_krw": 50_000.0,
            "orderable_cash_by_currency": {"KRW": 99_000.0, "USD": 30.0},
            "positions": [],
        }
        current = {
            **previous,
            "equity": 199_500.0,
            "orderable_cash_by_currency": {"KRW": 0.0, "USD": 30.0},
        }

        merged = web_module._merge_live_account_basis_with_previous(current, previous)

        self.assertEqual(merged["orderable_cash_by_currency"]["KRW"], 99_000.0)

    def test_account_basis_caps_cash_components_to_broker_equity(self) -> None:
        basis = web_module._account_basis_from_kis_connection(
            {
                "ok": True,
                "account_checked": True,
                "actual_equity": 200_000.0,
                "invested_value": 0.0,
                "krw_cash": 156_000.0,
                "cash_equivalent_krw": 200_000.0,
                "foreign_cash_krw": 92_000.0,
                "cash_by_currency": {"KRW": 156_000.0, "USD": 67.2},
                "orderable_cash_by_currency": {"KRW": 99_000.0, "USD": 67.2},
                "positions": [],
            }
        )

        self.assertEqual(basis["equity"], 200_000.0)
        self.assertEqual(basis["cash_equivalent_krw"], 200_000.0)
        self.assertEqual(basis["foreign_cash_krw"], 44_000.0)

    def test_learning_uses_unified_realtime_data_and_disallows_orders(self) -> None:
        state = OperationModeManager().start(OperationMode.LEARNING)

        self.assertEqual(state.data_environment, "realtime")
        self.assertFalse(state.synthetic_data_allowed)
        self.assertFalse(state.live_orders_allowed)
        self.assertTrue(state.training_allowed)
        self.assertIn("Use one unified realtime data store only: data/store.", state.guardrails)

    def test_us_market_holiday_closes_us_group_but_not_krx(self) -> None:
        holiday_kst = datetime(2026, 7, 3, 5, 0, tzinfo=timezone.utc)  # 14:00 KST, Jul 3 ET holiday

        self.assertFalse(web_module._is_open_live_market_ticker("LABT", "NASDAQ", holiday_kst))
        self.assertTrue(web_module._is_open_live_market_ticker("005930", "KR", holiday_kst))
        self.assertEqual(web_module._active_live_market_groups(holiday_kst), ("KRX",))

    def test_deprecated_testing_mode_is_normalized_to_live_trading(self) -> None:
        state = OperationModeManager().start(OperationMode.TESTING)

        self.assertEqual(state.mode, OperationMode.LIVE_TRADING)
        self.assertEqual(state.data_environment, "realtime")
        self.assertFalse(state.synthetic_data_allowed)
        self.assertTrue(state.live_orders_allowed)
        self.assertTrue(state.training_allowed)
        self.assertFalse(state.paper_trading_allowed)
        self.assertFalse(state.live_readiness_allowed)
        self.assertIn("Paper trading modes are removed and normalized to live trading.", state.guardrails)

    def test_kis_paper_mode_is_normalized_to_live_trading(self) -> None:
        paper = OperationModeManager().start(OperationMode.PAPER_TRADING)
        live_readiness = OperationModeManager().start(OperationMode.LIVE_READINESS)

        self.assertEqual(paper.mode, OperationMode.LIVE_TRADING)
        self.assertFalse(paper.paper_trading_allowed)
        self.assertTrue(paper.live_orders_allowed)
        self.assertEqual(paper.execution_label, "Realtime trading gate")
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

    def test_paper_api_request_starts_live_trading_mode(self) -> None:
        client = TestClient(app)
        with (
            patch("app.web._start_live_worker") as start_worker,
            patch("app.web._start_streaming_demo", return_value="demo-test") as start_demo,
            patch("app.web._start_kis_realtime_collector") as realtime_collector,
            patch("app.web._start_realtime_trading_engine") as trading_engine,
            patch(
                "app.web._kis_connection_probe",
                return_value={"ok": True, "mode": "live", "account_checked": True, "actual_deposit": 1000000},
            ) as kis_probe,
            patch("app.web._get_or_refresh_live") as refresh_live,
            patch("app.web.evaluate_live_runtime_gates", return_value=SimpleNamespace(ok=True, failures=())),
        ):
            data = client.post("/api/operation-mode/start", json={"mode": "paper_trading"}).json()

        self.assertTrue(data["ok"])
        self.assertEqual(data["mode"], "live_trading")
        self.assertEqual(data["requested_mode"], "paper_trading")
        self.assertEqual(data["mode_normalized_from"], "paper_trading")
        self.assertEqual(data["kis_connection"]["mode"], "live")
        self.assertEqual(data["data_policy"]["analysis_input_stores"], ["data/store"])
        start_demo.assert_not_called()
        start_worker.assert_not_called()
        realtime_collector.assert_called_once()
        trading_engine.assert_called_once()
        kis_probe.assert_any_call(paper=False, include_account=True)
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
        kis_probe.assert_any_call(paper=False, include_account=True)
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
        self._isolate_live_account_refresh()
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
        self._isolate_live_account_refresh()
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
        self._isolate_live_account_refresh()
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

    def test_deprecated_paper_operation_request_does_not_start_streaming_demo(self) -> None:
        client = TestClient(app)
        with (
            patch("app.web._start_live_worker"),
            patch("app.web._start_streaming_demo", return_value="demo-live-basis") as start_demo,
            patch("app.web._start_kis_realtime_collector"),
            patch("app.web._start_realtime_trading_engine"),
            patch("app.web._kis_connection_probe", return_value={"ok": True, "mode": "live"}),
            patch("app.web._get_or_refresh_live"),
            patch("app.web.evaluate_live_runtime_gates", return_value=SimpleNamespace(ok=True, failures=())),
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
        self.assertEqual(data["mode"], "live_trading")
        self.assertEqual(data["mode_normalized_from"], "paper_trading")
        start_demo.assert_not_called()

    def test_paper_trading_start_endpoint_is_removed(self) -> None:
        client = TestClient(app)
        with patch("app.web._start_streaming_demo", return_value="demo-auto") as start_demo:
            response = client.post(
                "/api/paper-trading/start",
                json={
                    "target_return_rate": 3,
                    "period_minutes": 120,
                    "initial_cash_source": "auto",
                },
            )

        data = response.json()
        self.assertEqual(response.status_code, 410)
        self.assertFalse(data["ok"])
        self.assertEqual(data["status"], "removed")
        self.assertEqual(data["mode"], "live_trading")
        start_demo.assert_not_called()

    def test_paper_trading_step_endpoint_is_removed(self) -> None:
        client = TestClient(app)
        response = client.post("/api/paper-trading/step", json={"demo_id": "demo-auto-refresh"})

        self.assertEqual(response.status_code, 410)
        self.assertEqual(response.json()["status"], "removed")

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
                patch("app.web._cached_domestic_ranking_symbols", return_value=()),
                patch.dict("os.environ", {"REALTIME_BUY_CANDIDATE_LIMIT": "5"}),
            ):
                store_cls.return_value.active_symbols.return_value = ("222222",)
                # The affordability filter requires a FRESH (timestamped) tick — a bare
                # price is treated as stale (price 0) and the candidate is dropped.
                _now = datetime.now(timezone.utc)
                store_cls.return_value.latest_tick.side_effect = (
                    lambda symbol: SimpleNamespace(price=5000.0, received_at=_now, exchange_timestamp=_now)
                    if symbol in {"111111", "222222"}
                    else None
                )
                store_cls.return_value.latest_orderbook.side_effect = (
                    lambda symbol: SimpleNamespace(
                        best_bid=4990.0,
                        best_ask=5010.0,
                        total_bid_volume=500000.0,
                        total_ask_volume=500000.0,
                        received_at=_now,
                        source="kis_realtime_websocket",
                    )
                    if symbol in {"005930", "000660", "111111", "222222"}
                    else None
                )
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

    def test_realtime_buy_candidates_prioritize_macro_micro_ranked_intents(self) -> None:
        from app.graph import macro_micro_feed

        account = AccountSnapshot(cash=100000.0, holdings=(), cash_by_currency={"KRW": 100000.0})
        macro_micro_feed.record_bundle(
            {
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "macro_result": {"blocks_buy": False},
                "ranked_trade_intents": [
                    {"side": "BUY", "symbol": "000660", "rank": 0},
                    {"side": "BUY", "symbol": "005930", "rank": 1},
                ],
                "blocked_candidates": ["111111"],
            }
        )
        try:
            with (
                patch("app.web._active_live_market_groups", return_value=("KRX",)),
                patch("app.web._load_realtime_collection_symbols", return_value=("111111",)),
                patch("app.web._live_account_snapshot_for_analysis", return_value=account),
                patch("app.web._live_affordable_buy_candidate_symbols", return_value=()),
                patch("app.web._cached_context_buy_candidates", return_value=()),
                patch("app.web._cached_volume_surge_symbols", return_value=()),
                patch("app.web._cached_domestic_ranking_symbols", return_value=()),
                patch("app.web.RealtimeMarketDataStore") as store_cls,
                patch.dict("os.environ", {"REALTIME_BUY_CANDIDATE_LIMIT": "5", "REALTIME_MACRO_MICRO_ENFORCE": "true"}),
            ):
                _now = datetime.now(timezone.utc)
                store_cls.return_value.active_symbols.return_value = ()
                store_cls.return_value.latest_tick.return_value = None
                store_cls.return_value.latest_orderbook.return_value = SimpleNamespace(
                    best_bid=4990.0,
                    best_ask=5010.0,
                    total_bid_volume=500000.0,
                    total_ask_volume=500000.0,
                    received_at=_now,
                    source="kis_realtime_websocket",
                )
                candidates = web_module._realtime_buy_candidates()

            self.assertEqual(candidates[:2], ("000660", "005930"))
            self.assertNotIn("111111", candidates)
        finally:
            macro_micro_feed.clear()

    def test_macro_micro_block_buy_removes_new_buy_candidates(self) -> None:
        from app.graph import macro_micro_feed

        macro_micro_feed.record_bundle(
            {
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "macro_result": {"blocks_buy": True},
                "ranked_trade_intents": [{"side": "BUY", "symbol": "005930", "rank": 0}],
            }
        )
        try:
            with patch.dict("os.environ", {"REALTIME_MACRO_MICRO_ENFORCE": "true"}):
                candidates = web_module._realtime_buy_candidates()

            self.assertEqual(candidates, ())
        finally:
            macro_micro_feed.clear()

    def test_macro_micro_insufficient_data_does_not_remove_fallback_candidates(self) -> None:
        from app.graph import macro_micro_feed

        account = AccountSnapshot(cash=100000.0, holdings=(), cash_by_currency={"KRW": 100000.0})
        macro_micro_feed.record_bundle(
            {
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "macro_result": {"blocks_buy": True, "reason_codes": ["MACRO_INSUFFICIENT_DATA"]},
                "ranked_trade_intents": [],
                "blocked_candidates": ["005930"],
            }
        )
        try:
            with (
                patch("app.web._active_live_market_groups", return_value=("KRX",)),
                patch("app.web._load_realtime_collection_symbols", return_value=("005930",)),
                patch("app.web._live_account_snapshot_for_analysis", return_value=account),
                patch("app.web._live_affordable_buy_candidate_symbols", return_value=()),
                patch("app.web._cached_context_buy_candidates", return_value=()),
                patch("app.web._cached_volume_surge_symbols", return_value=()),
                patch("app.web._cached_domestic_ranking_symbols", return_value=()),
                patch("app.web.RealtimeMarketDataStore") as store_cls,
                patch.dict("os.environ", {"REALTIME_MACRO_MICRO_ENFORCE": "true"}),
            ):
                _now = datetime.now(timezone.utc)
                store_cls.return_value.active_symbols.return_value = ()
                store_cls.return_value.latest_tick.return_value = None
                store_cls.return_value.latest_orderbook.return_value = SimpleNamespace(
                    best_bid=69900.0,
                    best_ask=70000.0,
                    total_bid_volume=500000.0,
                    total_ask_volume=500000.0,
                    received_at=_now,
                    source="kis_realtime_websocket",
                )
                candidates = web_module._realtime_buy_candidates()

            self.assertEqual(candidates, ("005930",))
        finally:
            macro_micro_feed.clear()

    def test_realtime_buy_candidates_include_domestic_ranking_candidates(self) -> None:
        account = AccountSnapshot(cash=100000.0, holdings=(), cash_by_currency={"KRW": 100000.0})
        market = MarketSnapshot(
            "035420",
            "KOSPI",
            "NAVER",
            "Technology",
            50000.0,
            10_000_000,
            0.02,
            SourceMetadata("unit", datetime.now(timezone.utc)),
        )
        with web_module._live_lock:
            previous_context = web_module._live_state.get("context")
            web_module._live_state["context"] = None
        try:
            with (
                patch("app.web._active_live_market_groups", return_value=("KRX",)),
                patch("app.web._live_account_snapshot_for_analysis", return_value=account),
                patch("app.web._load_realtime_collection_symbols", return_value=()),
                patch("app.web._live_affordable_buy_candidate_symbols", return_value=()),
                patch("app.web._cached_volume_surge_symbols", return_value=()),
                patch("app.web._cached_domestic_ranking_symbols", return_value=("035420",)),
                patch("app.web._candidate_affordability_market", return_value=market),
                patch("app.web.RealtimeMarketDataStore") as store_cls,
                patch.dict("os.environ", {"REALTIME_BUY_CANDIDATE_LIMIT": "4", "REALTIME_US_SCAN_FALLBACK_LIMIT": "0"}),
            ):
                _now = datetime.now(timezone.utc)
                store_cls.return_value.active_symbols.return_value = ()
                store_cls.return_value.latest_orderbook.return_value = SimpleNamespace(
                    best_bid=49950.0,
                    best_ask=50000.0,
                    total_bid_volume=500000.0,
                    total_ask_volume=500000.0,
                    received_at=_now,
                    source="kis_realtime_websocket",
                )
                candidates = web_module._realtime_buy_candidates()

            self.assertEqual(candidates, ("035420",))
        finally:
            with web_module._live_lock:
                web_module._live_state["context"] = previous_context

    def test_realtime_buy_candidates_price_domestic_ranking_from_orderbook_without_tick(self) -> None:
        account = AccountSnapshot(cash=100000.0, holdings=(), cash_by_currency={"KRW": 100000.0})
        with web_module._live_lock:
            previous_context = web_module._live_state.get("context")
            web_module._live_state["context"] = None
        try:
            with (
                patch("app.web._active_live_market_groups", return_value=("KRX",)),
                patch("app.web._live_account_snapshot_for_analysis", return_value=account),
                patch("app.web._load_realtime_collection_symbols", return_value=()),
                patch("app.web._live_affordable_buy_candidate_symbols", return_value=()),
                patch("app.web._cached_volume_surge_symbols", return_value=()),
                patch("app.web._cached_domestic_ranking_symbols", return_value=("066430",)),
                patch("app.web.RealtimeMarketDataStore") as store_cls,
                patch.dict("os.environ", {"REALTIME_BUY_CANDIDATE_LIMIT": "4", "REALTIME_US_SCAN_FALLBACK_LIMIT": "0"}),
            ):
                _now = datetime.now(timezone.utc)
                store_cls.return_value.active_symbols.return_value = ()
                store_cls.return_value.latest_tick.return_value = None
                store_cls.return_value.latest_orderbook.return_value = SimpleNamespace(
                    best_bid=2005.0,
                    best_ask=2015.0,
                    total_bid_volume=500000.0,
                    total_ask_volume=500000.0,
                    received_at=_now,
                    source="kis_realtime_websocket",
                )
                candidates = web_module._realtime_buy_candidates()

            self.assertEqual(candidates, ("066430",))
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
            cash=0.0,
            holdings=(),
            cash_by_currency={"KRW": 0.0, "USD": 20.0},
            orderable_cash_by_currency={"KRW": 100000.0},
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
                patch("app.web._load_us_listed_exchange_map", return_value={"AAPL": "NASD", "MSFT": "NASD"}),
                patch("app.web._load_realtime_collection_symbols", return_value=()),
                patch("app.web.RealtimeMarketDataStore") as store_cls,
                patch("app.web._cached_volume_surge_symbols", return_value=()),
                patch("app.web._cached_domestic_ranking_symbols", return_value=()),
                patch.dict("os.environ", {"REALTIME_BUY_CANDIDATE_LIMIT": "4", "REALTIME_US_SCAN_FALLBACK_LIMIT": "0"}),
            ):
                _now = datetime.now(timezone.utc)
                store_cls.return_value.active_symbols.return_value = ()
                store_cls.return_value.latest_tick.return_value = None
                store_cls.return_value.latest_orderbook.side_effect = (
                    lambda symbol: SimpleNamespace(
                        best_bid=69900.0,
                        best_ask=70100.0,
                        received_at=_now,
                        source="kis_realtime_websocket",
                    )
                    if symbol == "005930"
                    else None
                )
                candidates = web_module._realtime_buy_candidates()

            self.assertIn("005930", candidates)
            self.assertNotIn("000660", candidates)
            self.assertNotIn("AAPL", candidates)
            self.assertIn("MSFT", candidates)
        finally:
            with web_module._live_lock:
                web_module._live_state["context"] = previous_context

    def test_us_buy_candidate_with_no_orderbook_is_temporarily_backed_off(self) -> None:
        account = AccountSnapshot(cash=1000.0, holdings=(), cash_by_currency={"USD": 1000.0})
        with web_module._live_lock:
            previous_backoff = dict(web_module._live_buy_candidate_backoff_until)
            web_module._live_buy_candidate_backoff_until.clear()
        try:
            web_module._apply_live_buy_candidate_backoff(
                {
                    "rejections": [
                        {
                            "symbol": "WBI",
                            "side": "BUY",
                            "reason_codes": ["EXEC_NO_ORDERBOOK_BLOCKED"],
                        }
                    ]
                }
            )

            with (
                patch("app.web._active_live_market_groups", return_value=("US",)),
                patch("app.web._live_account_snapshot_for_analysis", return_value=account),
                patch("app.web._load_realtime_collection_symbols", return_value=()),
                patch("app.web._live_affordable_buy_candidate_symbols", return_value=("WBI", "MSFT")),
                patch("app.web._cached_volume_surge_symbols", return_value=()),
                patch("app.web._cached_domestic_ranking_symbols", return_value=()),
                patch("app.web.RealtimeMarketDataStore") as store_cls,
                patch.dict("os.environ", {"REALTIME_BUY_CANDIDATE_LIMIT": "4", "REALTIME_US_SCAN_FALLBACK_LIMIT": "0"}),
            ):
                _now = datetime.now(timezone.utc)
                store_cls.return_value.active_symbols.return_value = ()
                store_cls.return_value.latest_tick.side_effect = (
                    lambda symbol: SimpleNamespace(price=100.0, received_at=_now, exchange_timestamp=_now)
                    if symbol == "MSFT"
                    else None
                )
                store_cls.return_value.latest_orderbook.return_value = None
                candidates = web_module._realtime_buy_candidates()

            self.assertNotIn("WBI", candidates)
            self.assertIn("MSFT", candidates)
        finally:
            with web_module._live_lock:
                web_module._live_buy_candidate_backoff_until.clear()
                web_module._live_buy_candidate_backoff_until.update(previous_backoff)

    def test_us_buy_candidate_with_wide_spread_is_temporarily_rotated_out(self) -> None:
        with web_module._live_lock:
            previous_backoff = dict(web_module._live_buy_candidate_backoff_until)
            web_module._live_buy_candidate_backoff_until.clear()
        try:
            web_module._apply_live_buy_candidate_backoff(
                {
                    "rejections": [
                        {
                            "symbol": "DEFT",
                            "side": "BUY",
                            "reason_codes": ["WIDE_SPREAD:159.9>70.0bps", "LOW_LIQUIDITY"],
                        }
                    ]
                }
            )

            self.assertTrue(web_module._live_buy_candidate_in_backoff("DEFT"))
        finally:
            with web_module._live_lock:
                web_module._live_buy_candidate_backoff_until.clear()
                web_module._live_buy_candidate_backoff_until.update(previous_backoff)

    def test_realtime_buy_candidates_warm_volume_surge_before_filtering(self) -> None:
        account = AccountSnapshot(cash=1000.0, holdings=(), cash_by_currency={"USD": 1000.0})
        with web_module._live_lock:
            previous_context = web_module._live_state.get("context")
            previous_warm = dict(web_module._volume_surge_warm_cache)
            web_module._live_state["context"] = None
            web_module._volume_surge_warm_cache.update({"at": 0.0, "symbols": ()})
        calls = []

        def fake_refresh(context, *, symbols):
            calls.append(tuple(symbols))
            return {"ok": True, "symbols": tuple(symbols), "saved": {"realtime_ticks": 0, "orderbooks": 0}}

        try:
            with (
                patch("app.web._active_live_market_groups", return_value=("US",)),
                patch("app.web._live_account_snapshot_for_analysis", return_value=account),
                patch("app.web._load_realtime_collection_symbols", return_value=()),
                patch("app.web._live_affordable_buy_candidate_symbols", return_value=()),
                patch("app.web._cached_volume_surge_symbols", return_value=("DEFT", "TTGT", "ABCWS")),
                patch("app.web._cached_domestic_ranking_symbols", return_value=()),
                patch("app.trading.us_realtime_bridge.refresh_us_realtime_for_context_buy_candidates", side_effect=fake_refresh),
                patch("app.web.RealtimeMarketDataStore") as store_cls,
                patch.dict(
                    "os.environ",
                    {"REALTIME_BUY_CANDIDATE_LIMIT": "4", "REALTIME_US_VOLUME_SURGE_WARM_LIMIT": "2"},
                ),
            ):
                store_cls.return_value.active_symbols.return_value = ()
                store_cls.return_value.latest_tick.return_value = None
                store_cls.return_value.latest_orderbook.return_value = None
                web_module._realtime_buy_candidates()

            self.assertEqual(calls, [("DEFT", "TTGT")])
        finally:
            with web_module._live_lock:
                web_module._live_state["context"] = previous_context
                web_module._volume_surge_warm_cache.clear()
                web_module._volume_surge_warm_cache.update(previous_warm)

    def test_krx_buy_candidate_without_orderbook_is_warmed_before_evaluation(self) -> None:
        account = AccountSnapshot(cash=100000.0, holdings=(), cash_by_currency={"KRW": 100000.0})
        with web_module._live_lock:
            previous_context = web_module._live_state.get("context")
            previous_pending = dict(web_module._pending_krx_buy_candidate_warmup)
            web_module._live_state["context"] = None
            web_module._pending_krx_buy_candidate_warmup.clear()
        try:
            with (
                patch("app.web._active_live_market_groups", return_value=("KRX",)),
                patch("app.web._live_account_snapshot_for_analysis", return_value=account),
                patch("app.web._held_or_recent_buy_tickers", return_value=set()),
                patch("app.web._load_realtime_collection_symbols", return_value=()),
                patch("app.web._live_affordable_buy_candidate_symbols", return_value=("005930",)),
                patch("app.web.RealtimeMarketDataStore") as store_cls,
                patch("app.web._cached_volume_surge_symbols", return_value=()),
                patch("app.web._cached_domestic_ranking_symbols", return_value=()),
                patch.dict("os.environ", {"REALTIME_BUY_CANDIDATE_LIMIT": "4"}),
            ):
                store_cls.return_value.active_symbols.return_value = ()
                store_cls.return_value.latest_tick.return_value = None
                store_cls.return_value.latest_orderbook.return_value = None
                candidates = web_module._realtime_buy_candidates()
                collector_symbols = web_module._kis_realtime_collector_symbols()

            self.assertIn("005930", candidates)
            self.assertIn("005930", collector_symbols)
            self.assertEqual(collector_symbols[0], "005930")
            with web_module._live_lock:
                self.assertIn("005930", web_module._pending_krx_buy_candidate_warmup)

            fresh_orderbook = SimpleNamespace(
                best_bid=69900.0,
                best_ask=70000.0,
                total_bid_volume=500000.0,
                total_ask_volume=500000.0,
                source="kis_realtime_websocket",
                received_at=datetime.now(timezone.utc),
            )
            market = MarketSnapshot(
                ticker="005930",
                market="KOSPI",
                company_name="Samsung",
                sector="Tech",
                last_price=70000.0,
                average_daily_trading_value=1_000_000_000.0,
                volatility_20d=0.02,
                source=SourceMetadata("kis", datetime.now(timezone.utc), source_type="broker_api", trust_level=5),
            )
            with (
                patch("app.web._active_live_market_groups", return_value=("KRX",)),
                patch("app.web._live_account_snapshot_for_analysis", return_value=account),
                patch("app.web._held_or_recent_buy_tickers", return_value=set()),
                patch("app.web._load_realtime_collection_symbols", return_value=()),
                patch("app.web._live_affordable_buy_candidate_symbols", return_value=()),
                patch("app.web.RealtimeMarketDataStore") as store_cls,
                patch("app.web._cached_volume_surge_symbols", return_value=()),
                patch("app.web._cached_domestic_ranking_symbols", return_value=()),
                patch("app.web._candidate_affordability_market", return_value=market),
                patch.dict("os.environ", {"REALTIME_BUY_CANDIDATE_LIMIT": "4"}),
            ):
                store_cls.return_value.active_symbols.return_value = ()
                store_cls.return_value.latest_orderbook.return_value = fresh_orderbook
                candidates_after_warmup = web_module._realtime_buy_candidates()

            self.assertIn("005930", candidates_after_warmup)
        finally:
            with web_module._live_lock:
                web_module._live_state["context"] = previous_context
                web_module._pending_krx_buy_candidate_warmup.clear()
                web_module._pending_krx_buy_candidate_warmup.update(previous_pending)

    def test_pending_krx_buy_candidate_is_cleared_after_fresh_orderbook(self) -> None:
        with web_module._live_lock:
            previous_pending = dict(web_module._pending_krx_buy_candidate_warmup)
            web_module._pending_krx_buy_candidate_warmup.clear()
            web_module._pending_krx_buy_candidate_warmup["005930"] = time.monotonic() + 120.0
        try:
            fresh_orderbook = SimpleNamespace(
                best_bid=69900.0,
                best_ask=70000.0,
                source="kis_realtime_websocket",
                received_at=datetime.now(timezone.utc),
            )
            with patch("app.web.RealtimeMarketDataStore") as store_cls:
                store_cls.return_value.latest_orderbook.return_value = fresh_orderbook
                pending = web_module._pending_krx_buy_candidate_warmup_symbols()

            self.assertNotIn("005930", pending)
            with web_module._live_lock:
                self.assertNotIn("005930", web_module._pending_krx_buy_candidate_warmup)
        finally:
            with web_module._live_lock:
                web_module._pending_krx_buy_candidate_warmup.clear()
                web_module._pending_krx_buy_candidate_warmup.update(previous_pending)

    def test_dashboard_stream_symbol_stays_in_collector_priority(self) -> None:
        from app.data.market_session import MarketPhase

        account = AccountSnapshot(cash=100000.0, holdings=(), cash_by_currency={"KRW": 100000.0})
        with web_module._live_lock:
            previous_watch = dict(web_module._dashboard_krx_watch)
            web_module._dashboard_krx_watch.clear()
        try:
            with (
                patch("app.data.market_session.market_phase", return_value=MarketPhase.REGULAR),
                patch("app.web._request_kis_realtime_collector_resubscribe") as request,
            ):
                web_module._observe_dashboard_market_stream("396500")
                web_module._observe_dashboard_market_stream("396500")
            request.assert_called_once_with("dashboard_stream", ("396500",))

            with (
                patch("app.web._live_account_snapshot_for_analysis", return_value=account),
                patch("app.web._load_realtime_collection_symbols", return_value=("005930",)),
                patch("app.web._live_affordable_buy_candidate_symbols", return_value=("413630",)),
                patch("app.web._cached_domestic_ranking_symbols", return_value=()),
                patch("app.web._pending_krx_buy_candidate_warmup_symbols", return_value=()),
                patch.dict("os.environ", {"REALTIME_COLLECTOR_MAX_SYMBOLS": "2"}),
            ):
                symbols = web_module._kis_realtime_collector_symbols()

            self.assertEqual(symbols[0], "396500")
        finally:
            with web_module._live_lock:
                web_module._dashboard_krx_watch.clear()
                web_module._dashboard_krx_watch.update(previous_watch)

    def test_dashboard_stream_does_not_reconnect_regular_feed_after_hours(self) -> None:
        from app.data.market_session import MarketPhase

        with web_module._live_lock:
            previous_watch = dict(web_module._dashboard_krx_watch)
            web_module._dashboard_krx_watch.clear()
        try:
            with (
                patch("app.data.market_session.market_phase", return_value=MarketPhase.AFTER),
                patch("app.web.RealtimeMarketDataStore") as store_cls,
                patch("app.web._request_kis_realtime_collector_resubscribe") as request,
            ):
                store_cls.return_value.latest_tick.return_value = None
                store_cls.return_value.latest_orderbook.return_value = None
                web_module._observe_dashboard_market_stream("396500")

            request.assert_not_called()
            self.assertEqual(web_module._dashboard_krx_watch_symbols(), ("396500",))
        finally:
            with web_module._live_lock:
                web_module._dashboard_krx_watch.clear()
                web_module._dashboard_krx_watch.update(previous_watch)

    def test_kis_realtime_collector_symbols_include_held_and_context_krx_names(self) -> None:
        account = AccountSnapshot(
            cash=100000.0,
            holdings=(Holding("006880", "KR", "Unit", "Unknown", 1, 4500.0, 4500.0),),
            cash_by_currency={"KRW": 100000.0},
        )
        cached_context = SimpleNamespace(
            reasoning_paths=(SimpleNamespace(ticker="232680", conclusion="BuyCandidate"),),
            candidate_selection=SimpleNamespace(candidate_stocks=("012210", "AAPL")),
        )
        with web_module._live_lock:
            previous_context = web_module._live_state.get("context")
            previous_pending = dict(web_module._pending_krx_buy_candidate_warmup)
            web_module._live_state["context"] = cached_context
            web_module._pending_krx_buy_candidate_warmup.clear()
        try:
            with (
                patch("app.web._live_account_snapshot_for_analysis", return_value=account),
                patch("app.web._load_realtime_collection_symbols", return_value=("005930",)),
                patch("app.web._live_affordable_buy_candidate_symbols", return_value=("279570", "MSFT")),
                patch("app.web._cached_domestic_ranking_symbols", return_value=()),
                patch.dict("os.environ", {"REALTIME_COLLECTOR_MAX_SYMBOLS": "8"}),
            ):
                symbols = web_module._kis_realtime_collector_symbols()

            # A HELD position must always come first: without a live feed it cannot
            # be priced to exit, so it outranks every research subscription.
            self.assertEqual(symbols[0], "006880")
            # Session anchors sit above the rotating pool (they must survive the
            # 300s rotation for session-structure strategies) but below held names.
            self.assertIn("005930", symbols)
            # Context / affordable KRX candidates are still subscribed.
            for expected in ("279570", "232680", "012210"):
                self.assertIn(expected, symbols)
            # The domestic websocket speaks domestic TR_IDs only.
            self.assertNotIn("AAPL", symbols)
            self.assertNotIn("MSFT", symbols)
        finally:
            with web_module._live_lock:
                web_module._live_state["context"] = previous_context
                web_module._pending_krx_buy_candidate_warmup.clear()
                web_module._pending_krx_buy_candidate_warmup.update(previous_pending)

    def test_realtime_buy_candidates_reserve_slots_for_both_open_markets(self) -> None:
        account = AccountSnapshot(
            cash=100000.0,
            holdings=(),
            cash_by_currency={"KRW": 100000.0, "USD": 1000.0},
        )
        with (
            patch("app.web._active_live_market_groups", return_value=("KRX", "US")),
            patch("app.web._live_account_snapshot_for_analysis", return_value=account),
            patch("app.web._is_live_buy_candidate_symbol", return_value=True),
            patch("app.web._ticker_market_group_for_live_trading", side_effect=lambda symbol, _market="": "KRX" if str(symbol).isdigit() else "US"),
        ):
            ordered = web_module._prioritize_realtime_buy_candidates(("AAPL", "005930", "MSFT", "066310"))

        self.assertEqual(ordered, ("005930", "AAPL", "066310", "MSFT"))

    def test_realtime_buy_candidates_place_us_scan_before_broad_affordable_scan(self) -> None:
        account = AccountSnapshot(cash=0.0, holdings=(), cash_by_currency={"USD": 1000.0})
        captured: list[tuple[str, ...]] = []

        def capture_filter(symbols, **kwargs):
            captured.append(tuple(symbols))
            return tuple(symbols)

        with (
            patch("app.web._active_live_market_groups", return_value=("US",)),
            patch("app.web._live_account_snapshot_for_analysis", return_value=account),
            patch("app.web._load_realtime_collection_symbols", return_value=()),
            patch("app.web._cached_context_buy_candidates", return_value=()),
            patch("app.web._live_affordable_buy_candidate_symbols", return_value=("ACET", "ACFN")),
            patch("app.web._pending_krx_buy_candidate_warmup_symbols", return_value=()),
            patch("app.web._cached_domestic_ranking_symbols", return_value=()),
            patch("app.web._cached_volume_surge_symbols", return_value=()),
            patch("app.web._fresh_macro_micro_bundle", return_value=None),
            patch("app.web._warm_us_volume_surge_candidates_for_buy_filter", return_value=()),
            patch("app.web._us_open_cash_scan_fallback_candidates", return_value=("RBBN", "BZ")),
            patch("app.web._filter_realtime_buy_candidates_by_affordability", side_effect=capture_filter),
            patch("app.web.RealtimeMarketDataStore") as store_cls,
            patch.dict("os.environ", {"REALTIME_BUY_CANDIDATE_LIMIT": "8"}),
        ):
            store_cls.return_value.active_symbols.return_value = ()
            candidates = web_module._realtime_buy_candidates()

        self.assertEqual(candidates[:4], ("RBBN", "BZ", "ACET", "ACFN"))
        self.assertEqual(captured[0][:4], ("RBBN", "BZ", "ACET", "ACFN"))

    def test_realtime_engine_candidates_do_not_run_synchronous_broker_discovery(self) -> None:
        with (
            patch("app.web.RealtimeMarketDataStore") as store_cls,
            patch("app.web._cached_context_buy_candidates", return_value=("HOWL",)),
            patch("app.web._cached_domestic_ranking_symbols", return_value=()),
            patch(
                "app.web._prioritize_realtime_buy_candidates",
                side_effect=lambda symbols, **_: tuple(symbols),
            ),
            patch("app.web._is_live_buy_candidate_symbol", return_value=True),
            patch("app.web._is_open_live_market_ticker", return_value=True),
            patch("app.web._live_affordable_buy_candidate_symbols") as broker_discovery,
        ):
            store_cls.return_value.active_symbols.return_value = ("HST", "HUYA")
            with web_module._live_lock:
                web_module._us_learning_watchlist_cache["symbols"] = ("HYZD",)
            candidates = web_module._realtime_engine_buy_candidates()

        # Cached ranking/watchlist symbols are ordering hints only. The engine
        # may evaluate a symbol only when a fresh realtime tick proves that it
        # is actually streaming.
        self.assertEqual(candidates, ("HST", "HUYA"))
        broker_discovery.assert_not_called()

    def test_web_macro_observer_uses_live_us_market_context(self) -> None:
        now = datetime.now(timezone.utc)

        class Store:
            def recent_ticks(self, symbol, since):
                return tuple(
                    SimpleNamespace(
                        price=100.0 + index,
                        volume=10,
                        received_at=now,
                        exchange_timestamp=now,
                    )
                    for index in range(8)
                )

            def recent_orderbooks(self, symbol, since):
                return ()

            def latest_tick(self, symbol):
                return SimpleNamespace(received_at=now, price=107.0)

        feature_builder = SimpleNamespace(
            build=lambda *args, **kwargs: (_ for _ in ()).throw(RuntimeError("no technical frame"))
        )
        decision_engine = SimpleNamespace(
            store=Store(),
            feature_builder=feature_builder,
            market_refresher=None,
        )
        observer = web_module._build_macro_micro_observer(decision_engine)

        self.assertIsNotNone(observer)
        bundle = observer(
            AccountSnapshot(cash=1_000.0, holdings=()),
            (),
            ("AAPL",),
            now,
        )

        self.assertEqual(bundle.macro_result.candidate_symbols, ("AAPL",))
        self.assertNotIn("MACRO_INSUFFICIENT_DATA", bundle.macro_result.reason_codes)

    def test_us_learning_watchlist_stays_fixed_during_cache_ttl(self) -> None:
        account = AccountSnapshot(
            cash=0.0,
            holdings=(),
            cash_by_currency={"USD": 143.25},
        )
        with (
            patch("app.web._account_snapshot_from_live_basis", return_value=account),
            patch("app.web._last_live_account_basis", return_value={}),
            patch(
                "app.web._recent_affordable_us_watchlist",
                return_value=("AAPL", "MSFT"),
            ) as recent,
            patch(
                "app.web._live_affordable_buy_candidate_symbols",
                return_value=("NVDA",),
            ) as discover,
            patch(
                "app.web._liquid_affordable_us_seed_symbols",
                return_value=(),
            ),
            patch.dict("os.environ", {"REALTIME_US_WATCHLIST_TTL_SEC": "1800"}),
        ):
            first = web_module._sticky_us_learning_symbols(2)
            second = web_module._sticky_us_learning_symbols(2)

        self.assertEqual(first, ("AAPL", "MSFT"))
        self.assertEqual(second, first)
        recent.assert_called_once()
        discover.assert_not_called()

    def test_us_learning_watchlist_fills_only_missing_slots(self) -> None:
        account = AccountSnapshot(
            cash=0.0,
            holdings=(),
            cash_by_currency={"USD": 143.25},
        )
        with (
            patch("app.web._account_snapshot_from_live_basis", return_value=account),
            patch("app.web._last_live_account_basis", return_value={}),
            patch(
                "app.web._recent_affordable_us_watchlist",
                return_value=("AAPL",),
            ),
            patch(
                "app.web._live_affordable_buy_candidate_symbols",
                return_value=("AAPL", "MSFT", "005930"),
            ),
            patch(
                "app.web._liquid_affordable_us_seed_symbols",
                return_value=(),
            ),
        ):
            symbols = web_module._sticky_us_learning_symbols(2)

        self.assertEqual(symbols, ("AAPL", "MSFT"))

    def test_us_learning_watchlist_prefers_liquid_seed_before_random_discovery(self) -> None:
        account = AccountSnapshot(
            cash=0.0,
            holdings=(),
            cash_by_currency={"USD": 67.57},
        )
        with (
            patch("app.web._account_snapshot_from_live_basis", return_value=account),
            patch("app.web._last_live_account_basis", return_value={}),
            patch("app.web._recent_affordable_us_watchlist", return_value=("T",)),
            patch(
                "app.web._liquid_affordable_us_seed_symbols",
                return_value=("SOFI",),
            ) as seed,
            patch(
                "app.web._live_affordable_buy_candidate_symbols",
                return_value=("STHO",),
            ) as discover,
        ):
            symbols = web_module._sticky_us_learning_symbols(2)

        self.assertEqual(symbols, ("T", "SOFI"))
        seed.assert_called_once_with(account, limit=1)
        discover.assert_not_called()

    def test_us_learning_watchlist_rotates_broker_verified_pool(self) -> None:
        account = AccountSnapshot(
            cash=0.0,
            holdings=(),
            cash_by_currency={"USD": 200.0},
        )
        with (
            patch("app.web._account_snapshot_from_live_basis", return_value=account),
            patch("app.web._last_live_account_basis", return_value={}),
            patch(
                "app.web._recent_affordable_us_watchlist",
                return_value=("AAPL", "MSFT", "SOFI", "PFE"),
            ),
            patch(
                "app.web._liquid_affordable_us_seed_symbols",
                return_value=(),
            ),
            patch.dict(
                "os.environ",
                {
                    "REALTIME_US_ROTATION_POOL_MULTIPLIER": "2",
                    "REALTIME_US_WATCHLIST_RECHECK_SEC": "180",
                },
            ),
        ):
            first = web_module._sticky_us_learning_symbols(2)
            with web_module._live_lock:
                web_module._us_learning_watchlist_cache["at"] = 0.0
            second = web_module._sticky_us_learning_symbols(2)

        self.assertEqual(first, ("AAPL", "MSFT"))
        self.assertEqual(second, ("SOFI", "PFE"))

    def test_recent_us_watchlist_ranks_sustained_fresh_ticks(self) -> None:
        account = AccountSnapshot(
            cash=0.0,
            holdings=(),
            cash_by_currency={"USD": 67.57},
        )
        now = datetime.now(timezone.utc)
        with tempfile.TemporaryDirectory() as temp_dir:
            database = Path(temp_dir) / "realtime.sqlite3"
            import sqlite3

            with closing(sqlite3.connect(database)) as connection:
                connection.executescript(
                    """
                    create table realtime_ticks (
                      record_id text primary key,
                      symbol text,
                      received_at text,
                      price real,
                      volume integer
                    );
                    create table realtime_orderbook (
                      record_id text primary key,
                      symbol text,
                      received_at text,
                      spread_bps real
                    );
                    """
                )
                for index in range(6):
                    connection.execute(
                        "insert into realtime_ticks values (?, ?, ?, ?, ?)",
                        (
                            f"active-{index}",
                            "SOFI",
                            (now - timedelta(seconds=20 + index)).isoformat(),
                            12.0,
                            100,
                        ),
                    )
                connection.execute(
                    "insert into realtime_ticks values (?, ?, ?, ?, ?)",
                    ("one-off", "STHO", (now - timedelta(seconds=2)).isoformat(), 4.0, 1),
                )
                for index in range(3):
                    connection.execute(
                        "insert into realtime_orderbook values (?, ?, ?, ?)",
                        (
                            f"book-{index}",
                            "SOFI",
                            (now - timedelta(seconds=index)).isoformat(),
                            8.0,
                        ),
                    )
                connection.commit()

            with patch.dict(
                "os.environ",
                {
                    "REALTIME_US_WATCHLIST_MIN_TICKS": "3",
                    "REALTIME_US_WATCHLIST_MAX_TICK_AGE_SEC": "180",
                },
            ):
                symbols = web_module._recent_affordable_us_watchlist(
                    account,
                    limit=2,
                    database=database,
                )

        self.assertEqual(symbols, ("SOFI",))

    def test_us_learning_fast_poll_keeps_holdings_ahead_of_warm_symbols(self) -> None:
        with (
            patch("app.web._active_operation_mode", return_value="learning"),
            patch("app.web._sticky_us_learning_symbols", return_value=("EDTK", "MSFT")),
        ):
            symbols = web_module._us_fast_poll_target_symbols(("F", "EDTK"))

        self.assertEqual(symbols, ("F", "EDTK", "MSFT"))

    def test_us_live_fast_poll_seeds_empty_watchlist(self) -> None:
        with web_module._live_lock:
            previous = dict(web_module._us_learning_watchlist_cache)
            web_module._us_learning_watchlist_cache.update(
                {"at": 0.0, "cash_usd": None, "symbols": ()}
            )
        try:
            with (
                patch("app.web._active_operation_mode", return_value="live_trading"),
                patch("app.web._sticky_us_learning_symbols", return_value=("AAPL", "MSFT")) as warm,
                patch.dict("os.environ", {"AUTO_RELIABILITY_US_WARM_SYMBOLS": "2"}),
            ):
                symbols = web_module._us_fast_poll_target_symbols(("F",))

            self.assertEqual(symbols, ("F", "AAPL", "MSFT"))
            warm.assert_called_once_with(2)
        finally:
            with web_module._live_lock:
                web_module._us_learning_watchlist_cache.clear()
                web_module._us_learning_watchlist_cache.update(previous)

    def test_last_live_account_basis_prefers_stabilized_snapshot(self) -> None:
        stable = {
            "equity": 200_000.0,
            "krw_cash": 100_000.0,
            "cash_equivalent_krw": 200_000.0,
            "positions": [],
            "source": "kis_live_account",
        }
        raw = {
            "account_checked": True,
            "actual_equity": 120_000.0,
            "krw_cash": 20_000.0,
            "cash_equivalent_krw": 120_000.0,
            "positions": [],
        }
        with web_module._live_lock:
            previous_stable = web_module._operation_mode_state.get("stable_account_basis")
            previous_connection = web_module._operation_mode_state.get("last_kis_connection")
            web_module._operation_mode_state["stable_account_basis"] = stable
            web_module._operation_mode_state["last_kis_connection"] = raw
        try:
            self.assertEqual(web_module._last_live_account_basis(), stable)
        finally:
            with web_module._live_lock:
                web_module._operation_mode_state["stable_account_basis"] = previous_stable
                web_module._operation_mode_state["last_kis_connection"] = previous_connection

    def test_kis_realtime_collector_symbols_are_capped_by_subscription_budget(self) -> None:
        account = AccountSnapshot(cash=100000.0, holdings=(), cash_by_currency={"KRW": 100000.0})
        many_symbols = tuple(f"{index:06d}" for index in range(1, 30))
        with (
            patch("app.web._live_account_snapshot_for_analysis", return_value=account),
            patch("app.web._load_realtime_collection_symbols", return_value=many_symbols),
            patch("app.web._live_affordable_buy_candidate_symbols", return_value=many_symbols),
            patch("app.web._cached_domestic_ranking_symbols", return_value=()),
            patch.dict(
                "os.environ",
                {"REALTIME_COLLECTOR_MAX_SYMBOLS": "40", "KIS_REALTIME_MAX_SUBSCRIPTIONS": "10"},
            ),
        ):
            symbols = web_module._kis_realtime_collector_symbols()

        self.assertLessEqual(len(symbols), 5)

    def test_kis_training_candidate_rejects_penny_wide_spread_symbol(self) -> None:
        store = SimpleNamespace(
            latest_tick=lambda _symbol: SimpleNamespace(price=119.0),
            latest_orderbook=lambda _symbol: SimpleNamespace(spread_bps=84.0),
        )

        self.assertFalse(
            web_module._kis_realtime_training_candidate_viable("252670", store)
        )

    def test_kis_realtime_collector_uses_live_client(self) -> None:
        with patch.dict("os.environ", {"KIS_PAPER_TRADING": "true"}, clear=False):
            client = web_module._kis_realtime_collector_client()

        self.assertFalse(client.paper)
        self.assertTrue(client.enabled)

    def test_realtime_runtime_status_uses_fast_llm_snapshot(self) -> None:
        client = TestClient(app)
        with (
            patch("app.web.event_llm_runtime_status", side_effect=TimeoutError("slow probe")),
            patch("app.web.RealtimeAccelerationPolicy") as acceleration_policy,
            patch("app.web.get_ontology_npu_classifier") as classifier,
            patch("app.web._safe_live_training_status_fast", return_value={"ok": True, "fast_status": True}),
        ):
            acceleration_policy.return_value.status.return_value = {"uses_npu": True}
            classifier.return_value.status.return_value = {"backend": "NPU"}
            response = client.get("/api/realtime/runtime")

        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertTrue(data["event_llm"]["probe_skipped"])
        self.assertEqual(data["live_training"], {"ok": True, "fast_status": True})

    def test_kiosk_orderable_cash_uses_cached_basis_only(self) -> None:
        basis = {
            "orderable_cash_by_currency": {"KRW": 120000.0, "USD": 12.5},
            "krw_cash": 120000.0,
            "foreign_cash_krw": 18000.0,
        }
        with (
            patch("app.web._refresh_live_account_basis_for_auto", side_effect=TimeoutError("broker refresh")),
            patch("app.web._last_live_account_basis", return_value=basis),
        ):
            cash = web_module._kiosk_orderable_cash()

        self.assertTrue(cash["available"])
        self.assertEqual(cash["krw"], 120000.0)
        self.assertEqual(cash["foreign_currency"], "USD")

    def test_account_dashboard_status_overlays_realtime_holding_price(self) -> None:
        now = datetime.now(timezone.utc)
        basis = {
            "source": "kis_live_account",
            "cash_equivalent_krw": 100000.0,
            "equity": 242000.0,
            "positions": [
                {
                    "ticker": "005930",
                    "market": "KR",
                    "currency": "KRW",
                    "quantity": 2,
                    "average_price": 70000.0,
                    "last_price": 71000.0,
                    "market_value_krw": 142000.0,
                    "purchase_amount_krw": 140000.0,
                }
            ],
        }
        tick = SimpleNamespace(price=73000.0, received_at=now, source="kis_realtime_websocket")
        with patch("app.web.RealtimeMarketDataStore") as store_cls:
            store_cls.return_value.latest_tick.return_value = tick
            store_cls.return_value.latest_orderbook.return_value = None
            updated = web_module._account_basis_with_realtime_holding_prices(basis)

        position = updated["positions"][0]
        self.assertEqual(position["last_price"], 73000.0)
        self.assertEqual(position["last_price_source"], "kis_realtime_websocket")
        self.assertEqual(position["market_value_krw"], 146000.0)
        self.assertEqual(position["unrealized_pnl_krw"], 6000.0)
        self.assertEqual(updated["equity"], 242000.0)

    def test_kis_realtime_collector_disconnect_is_recorded_as_reconnecting(self) -> None:
        with web_module._live_lock:
            previous_log = list(web_module._live_state.get("collection_log") or [])
            web_module._live_state["collection_log"] = []
        web_module._kis_realtime_collector_stop.clear()
        try:
            web_module._record_kis_realtime_collector_result(
                {"connection_closed": True, "ticks": 0, "orderbooks": 0}
            )
            with web_module._live_lock:
                latest = list(web_module._live_state.get("collection_log") or [])[-1]

            self.assertEqual(latest["status"], "reconnecting")
            self.assertIn("reconnecting", latest["message"])
        finally:
            web_module._kis_realtime_collector_stop.clear()
            with web_module._live_lock:
                web_module._live_state["collection_log"] = previous_log

    def test_overseas_appkey_conflict_is_waiting_not_generic_error(self) -> None:
        status, message = web_module._classify_kis_overseas_collector_cycle(
            {
                "subscriptions_accepted": 0,
                "subscriptions_rejected": 1,
                "appkey_already_in_use": 1,
                "subscription_errors_by_code": {"OPSP8996": 1},
            }
        )
        self.assertEqual(status, "waiting")
        self.assertIn("AppKey", message)

    def test_overseas_subscription_rejection_exposes_kis_code(self) -> None:
        status, message = web_module._classify_kis_overseas_collector_cycle(
            {
                "subscriptions_accepted": 0,
                "subscriptions_rejected": 2,
                "subscription_errors_by_code": {"OPSP9999": 2},
            }
        )
        self.assertEqual(status, "error")
        self.assertIn("OPSP9999=2", message)

    def test_kis_complete_subscription_symbol_feeds_fast_feature_sampler(self) -> None:
        with web_module._live_lock:
            previous = web_module._kis_realtime_complete_symbols
            web_module._kis_realtime_complete_symbols = ()
        try:
            web_module._record_kis_realtime_collector_result(
                {
                    "subscriptions_accepted": 2,
                    "accepted_subscription_pairs": [
                        {"symbol": "396500", "tr_id": "H0STCNT0"},
                        {"symbol": "396500", "tr_id": "H0STASP0"},
                    ],
                }
            )

            with patch("app.web._dashboard_krx_watch_symbols", return_value=()):
                symbols = web_module._krx_feature_frame_symbols()

            self.assertEqual(symbols, ("396500",))
        finally:
            with web_module._live_lock:
                web_module._kis_realtime_complete_symbols = previous

    def test_kis_unified_complete_subscription_symbol_feeds_fast_feature_sampler(self) -> None:
        with web_module._live_lock:
            previous = web_module._kis_realtime_complete_symbols
            web_module._kis_realtime_complete_symbols = ()
        try:
            web_module._record_kis_realtime_collector_result(
                {
                    "subscriptions_accepted": 2,
                    "accepted_subscription_pairs": [
                        {"symbol": "005930", "tr_id": "H0UNCNT0"},
                        {"symbol": "005930", "tr_id": "H0UNASP0"},
                    ],
                }
            )

            with patch("app.web._dashboard_krx_watch_symbols", return_value=()):
                symbols = web_module._krx_feature_frame_symbols()

            self.assertEqual(symbols, ("005930",))
        finally:
            with web_module._live_lock:
                web_module._kis_realtime_complete_symbols = previous

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
        self._isolate_live_account_refresh()
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
        self._isolate_live_account_refresh()
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
            patch("app.web._load_us_listed_exchange_map", return_value={}),
            patch("app.web._load_us_nasdaq_universe", return_value=("PENNY", "MICRO", "BIG")),
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
            patch("app.web._load_us_listed_exchange_map", return_value={}),
            patch("app.web._load_us_nasdaq_universe", return_value=("PENNY", "MICRO", "BIG")),
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

    def test_live_probe_reconciles_orderable_krw_with_kis_authority(self) -> None:
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
                return SimpleNamespace(
                    account=AccountSnapshot(
                        cash=0.0,
                        holdings=(),
                        cash_by_currency={"KRW": 0.0, "USD": 67.57},
                        orderable_cash_by_currency={"KRW": 0.0, "USD": 67.57},
                        cash_equivalent_krw=197_377.0,
                    )
                )

            def _get_domestic_orderable_cash(self) -> float:
                return 99_504.0

        with (
            patch("app.web.KisDevelopersApiClient", FakeKisClient),
            patch("app.web.audit.record") as record,
        ):
            connection = web_module._kis_connection_probe(paper=False, include_account=True)

        self.assertEqual(connection["orderable_cash_by_currency"]["KRW"], 99_504.0)
        self.assertTrue(connection["orderable_cash_reconciliation"]["mismatch"])
        self.assertEqual(
            connection["orderable_cash_reconciliation"]["authoritative_krw"],
            99_504.0,
        )
        record.assert_any_call(
            "kis_orderable_cash_mismatch",
            connection["orderable_cash_reconciliation"],
        )

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

    def test_reliability_does_not_require_krx_regular_ticks_during_opening_auction(self) -> None:
        """KRX 시가 단일가 중에는 KRX 정규장 틱을 요구하지 않는다.

        08:45 KST 에는 미국 시장도 함께 열려 있지 않다. 08:45 KST = 전일 19:45 ET 이고,
        KIS 공식 애프터마켓 주문 시간은 06:00~07:00 KST (Summer Time 05:00~07:00),
        즉 ET 16:00-17:00/18:00 이므로 19:45 ET 는 완전 마감이다.
        이전 구현이 애프터마켓을 20:00 ET 까지로 잘못 잡고 있어서 이 시각에 미국이
        "거래 가능"으로 보였고, 이 테스트도 그 값을 고정하고 있었다.
        근거: docs/kis_market_session_capability_matrix.md §5.1

        따라서 이 시각의 올바른 결과는 "요구되는 시장이 없음"이다. 미국이 실제로 열려
        있을 때 미국 틱이 요구되는지는
        ``test_reliability_requires_us_ticks_during_us_core_session`` 이 검증한다.
        """
        opening_auction = datetime(2026, 6, 30, 23, 45, tzinfo=timezone.utc)
        captured: list[tuple[str, ...]] = []

        def market_health(_now, groups):
            captured.append(tuple(groups))
            return {"ok": True, "healthy": {"US": ["AAPL", "MSFT"], "KRX": []}}

        with (
            patch("app.web._active_live_market_groups", return_value=("US", "KRX")),
            patch("app.web._cached_kis_connection_probe", return_value={
                "ok": True,
                "account_checked": True,
                "actual_equity": 200000.0,
            }),
            patch("app.web.evaluate_live_runtime_gates", return_value=SimpleNamespace(ok=True, failures=())),
            patch("app.web.load_short_horizon_strategy_config", return_value={
                "execution": {"live_trading_enabled": True},
            }),
            patch("app.web.TradingPolicySnapshot") as policy,
            patch("app.web._latest_model_reliability", return_value={"ok": True}),
            patch("app.web._auto_market_health", side_effect=market_health),
            patch.dict("os.environ", {"LIVE_TRADING_ENABLED": "true", "KIS_LIVE_ENABLED": "true"}),
        ):
            policy.from_environment.return_value.conflicts.return_value = ()
            result = web_module._evaluate_auto_reliability(opening_auction)

        self.assertEqual(captured, [()])
        self.assertNotIn(
            "KRX", result["components"]["market_data"]["required_markets"]
        )
        self.assertEqual(result["components"]["market_data"]["required_markets"], [])
        self.assertEqual(
            result["components"]["market_data"]["extended_order_markets"],
            ["US", "KRX"],
        )

    def test_reliability_requires_us_ticks_during_us_core_session(self) -> None:
        """미국 정규장 중에는 미국 틱이 실제로 요구된다 (KRX 는 마감이라 제외)."""
        us_core = datetime(2026, 7, 1, 15, 0, tzinfo=timezone.utc)  # 11:00 ET, 00:00 KST
        captured: list[tuple[str, ...]] = []

        def market_health(_now, groups):
            captured.append(tuple(groups))
            return {"ok": True, "healthy": {"US": ["AAPL", "MSFT"], "KRX": []}}

        with (
            patch("app.web._active_live_market_groups", return_value=("US", "KRX")),
            patch("app.web._cached_kis_connection_probe", return_value={
                "ok": True,
                "account_checked": True,
                "actual_equity": 200000.0,
            }),
            patch("app.web.evaluate_live_runtime_gates", return_value=SimpleNamespace(ok=True, failures=())),
            patch("app.web.load_short_horizon_strategy_config", return_value={
                "execution": {"live_trading_enabled": True},
            }),
            patch("app.web.TradingPolicySnapshot") as policy,
            patch("app.web._latest_model_reliability", return_value={"ok": True}),
            patch("app.web._auto_market_health", side_effect=market_health),
            patch.dict("os.environ", {"LIVE_TRADING_ENABLED": "true", "KIS_LIVE_ENABLED": "true"}),
        ):
            policy.from_environment.return_value.conflicts.return_value = ()
            result = web_module._evaluate_auto_reliability(us_core)

        self.assertEqual(captured, [("US",)])
        self.assertEqual(result["components"]["market_data"]["required_markets"], ["US"])

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
