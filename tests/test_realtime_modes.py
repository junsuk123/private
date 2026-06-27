from __future__ import annotations

import sys
import unittest
from datetime import datetime, timedelta, timezone
from pathlib import Path
from unittest.mock import patch

from fastapi.testclient import TestClient

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from app.realtime import OperationMode, OperationModeManager, ShortHorizonRiskPolicy
from app.schemas.domain import OrderAction, OrderIntent
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
            patch("app.web._kis_connection_probe", return_value={"ok": True, "mode": "paper"}) as kis_probe,
            patch("app.web._get_or_refresh_live") as refresh_live,
        ):
            data = client.post("/api/operation-mode/start", json={"mode": "paper_trading"}).json()

        self.assertTrue(data["ok"])
        self.assertEqual(data["paper_trading_status"], "background_collection_started")
        self.assertEqual(data["paper_trading_kind"], "kis_paper_api")
        self.assertEqual(data["demo_id"], "demo-test")
        self.assertEqual(data["demo_status"], "initialized")
        self.assertEqual(data["kis_connection"]["mode"], "paper")
        self.assertEqual(data["data_policy"]["analysis_input_stores"], ["data/store"])
        start_demo.assert_called_once()
        start_worker.assert_called_once_with("learning")
        kis_probe.assert_called_once_with(paper=True, include_account=True)
        refresh_live.assert_not_called()

    def test_live_api_test_checks_readiness_without_streaming_orders(self) -> None:
        client = TestClient(app)
        with (
            patch("app.web._start_live_worker") as start_worker,
            patch("app.web._start_streaming_demo") as start_demo,
            patch("app.web._kis_connection_probe", return_value={"ok": True, "mode": "live"}) as kis_probe,
            patch("app.web._get_or_refresh_live") as refresh_live,
        ):
            data = client.post("/api/operation-mode/start", json={"mode": "live_readiness"}).json()

        self.assertTrue(data["ok"])
        self.assertEqual(data["live_readiness_status"], "checked")
        self.assertEqual(data["live_readiness_kind"], "kis_live_readiness")
        self.assertEqual(data["kis_connection"]["mode"], "live")
        start_worker.assert_called_once_with("learning")
        start_demo.assert_not_called()
        kis_probe.assert_called_once_with(paper=False, include_account=False)
        refresh_live.assert_not_called()

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
