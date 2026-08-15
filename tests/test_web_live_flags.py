from __future__ import annotations

import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from fastapi.testclient import TestClient

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from app import web as web_module
from app.web import LIVE_FLAG_VALUES, app


class WebLiveFlagsTest(unittest.TestCase):
    def test_realtime_status_projects_the_live_trace_cycle_id(self) -> None:
        engine = SimpleNamespace(
            get_status=lambda: {
                "live_trace": {"cycle_id": "2026-08-13T15:00:00+00:00"},
                "last_summary": {"at": "2026-08-13T15:00:00+00:00"},
            }
        )
        worker = SimpleNamespace(is_alive=lambda: True)
        with (
            patch("app.web._realtime_trading_engine", engine),
            patch("app.web._realtime_trading_worker", worker),
        ):
            response = web_module.realtime_trading_status()

        payload = json.loads(response.body)
        self.assertEqual(
            payload["status"]["engine_cycle_id"],
            "2026-08-13T15:00:00+00:00",
        )
        self.assertEqual(
            payload["status"]["last_summary"]["engine_cycle_id"],
            "2026-08-13T15:00:00+00:00",
        )

    def test_live_flags_enable_validated_model_inference(self) -> None:
        self.assertEqual(
            LIVE_FLAG_VALUES["LIVE_SIGNAL_MODEL_INFERENCE_ENABLED"],
            "true",
        )

    def test_market_view_follows_authoritative_owned_symbol(self) -> None:
        engine = SimpleNamespace(
            get_status=lambda: {
                "strategy_session": {
                    "phase": "OWNED",
                    "selected_symbol": "NVDA",
                    "selected_strategy": "intraday_momentum",
                    "last_reason": "POSITION_OWNED_MONITORING",
                }
            }
        )
        with (
            patch("app.web._realtime_trading_engine", engine),
            patch(
                "app.web.build_strategy_market_view",
                return_value={
                    "symbol": "NVDA",
                    "selection": {
                        "strategy_id": "intraday_momentum",
                        "ontology_allowed": True,
                    },
                    "algorithm": None,
                },
            ) as build_view,
            patch("app.refactor_dashboard._algorithm", return_value={"strategy_id": "intraday_momentum"}),
        ):
            payload = web_module._strategy_market_view_with_live_session("005930", 30)

        build_view.assert_called_once()
        args, kwargs = build_view.call_args
        self.assertEqual(args, ("NVDA",))
        self.assertEqual(kwargs["limit"], 30)
        self.assertEqual(kwargs["selection_override"]["action"], "OWNED")
        self.assertEqual(
            kwargs["selection_override"]["strategy_id"],
            "intraday_momentum",
        )
        self.assertEqual(payload["symbol"], "NVDA")
        self.assertEqual(payload["selection"]["strategy_id"], "intraday_momentum")
        self.assertTrue(payload["selection"]["ontology_allowed"])

    def test_market_view_replaces_stale_symbol_with_current_scan_candidate(self) -> None:
        engine = SimpleNamespace(
            get_status=lambda: {
                "last_summary": {
                    "buy_candidate_sample": ["SOFI", "PFE"],
                },
                "strategy_session": {
                    "phase": "SCANNING",
                    "selected_symbol": None,
                    "last_reason": "GNN_NOT_LIVE_AUTHORIZED",
                },
            }
        )
        with (
            patch("app.web._realtime_trading_engine", engine),
            patch(
                "app.web.build_strategy_market_view",
                return_value={
                    "symbol": "SOFI",
                    "selection": {},
                    "algorithm": None,
                },
            ) as build_view,
        ):
            payload = web_module._strategy_market_view_with_live_session("005930", 30)

        build_view.assert_called_once()
        args, kwargs = build_view.call_args
        self.assertEqual(args, ("SOFI",))
        self.assertEqual(kwargs["limit"], 30)
        self.assertEqual(kwargs["selection_override"]["action"], "NO_TRADE")
        self.assertIn(
            "GNN_NOT_LIVE_AUTHORIZED",
            kwargs["selection_override"]["reason_codes"],
        )
        self.assertEqual(payload["symbol"], "SOFI")
        self.assertEqual(payload["live_trading"]["candidate_symbols"], ["SOFI", "PFE"])

    def test_apply_live_flags_requires_confirmation(self) -> None:
        client = TestClient(app)

        response = client.post("/api/live-flags/apply", json={})

        self.assertEqual(response.status_code, 400)

    def test_apply_live_flags_sets_process_env_without_orders(self) -> None:
        client = TestClient(app)
        readiness = {"ok": False, "gates": {"live_flags": True}, "failures": {"live_eligible_model": "missing"}}
        with (
            patch.dict(os.environ, {}, clear=True),
            patch("app.web._web_live_readiness_summary", return_value=readiness),
        ):
            response = client.post(
                "/api/live-flags/apply",
                json={"confirmation": "APPLY_LIVE_FLAGS"},
            )
            payload = response.json()

            self.assertEqual(response.status_code, 200)
            self.assertTrue(payload["ok"])
            self.assertFalse(payload["live_ready"])
            self.assertFalse(payload["orders_submitted"])
            for key, value in LIVE_FLAG_VALUES.items():
                self.assertEqual(os.environ[key], value)
                self.assertEqual(payload["flags"][key], value)

    def test_readiness_reports_model_artifact_reason_not_exception_class(self) -> None:
        with (
            patch("app.web.load_live_trading_safety_config"),
            patch("app.web.load_order_execution_config"),
            patch("app.web.validate_live_secret_file", return_value={"app_key": True, "app_secret": True, "account_no": True}),
            patch("app.web.evaluate_live_runtime_gates", return_value=type("Gate", (), {"ok": True, "failures": ()})()),
            patch("app.web.ModelArtifactRegistry") as registry_cls,
            patch(
                "app.web.live_training_status",
                return_value={
                    "training_rows": 0,
                    "feature_frame_lines": 0,
                    "realtime_store_exists": False,
                    "latest_ineligible_artifact": None,
                },
            ),
        ):
            registry_cls.return_value.load_latest_live_eligible.side_effect = RuntimeError(
                "NO_LIVE_ELIGIBLE_MODEL_ARTIFACT"
            )

            readiness = web_module._web_live_readiness_summary()

        self.assertFalse(readiness["ok"])
        self.assertIn("NO_LIVE_ELIGIBLE_MODEL_ARTIFACT", readiness["failures"]["live_eligible_model"])
        self.assertIn("training_rows=0", readiness["failures"]["live_eligible_model"])
        self.assertIn("realtime_store=missing", readiness["failures"]["live_eligible_model"])

    def test_homepage_inline_script_is_valid_javascript(self) -> None:
        if shutil.which("node") is None:
            self.skipTest("node is required for JavaScript syntax checking")
        client = TestClient(app)

        response = client.get("/?target_return_rate=20&period_minutes=720")
        self.assertIn('id="modeTestingButton"', response.text)
        self.assertIn('id="liveFlagsButton"', response.text)
        self.assertIn("function applyLiveFlags()", response.text)
        self.assertIn("function fetchWithOptionalTimeout", response.text)
        self.assertNotIn("AbortSignal.timeout", response.text)
        match = re.search(r"<script>(.*)</script>", response.text, re.S)
        self.assertIsNotNone(match)

        script_path = None
        try:
            with tempfile.NamedTemporaryFile(
                "w", encoding="utf-8", suffix=".js", delete=False
            ) as handle:
                script_path = Path(handle.name)
                handle.write(match.group(1))
            completed = subprocess.run(
                ["node", "--check", str(script_path)],
                check=False,
                capture_output=True,
                text=True,
            )
            self.assertEqual(completed.returncode, 0, completed.stderr)
        finally:
            if script_path is not None:
                script_path.unlink(missing_ok=True)


if __name__ == "__main__":
    unittest.main()
