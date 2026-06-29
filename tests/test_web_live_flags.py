from __future__ import annotations

import os
import re
import shutil
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from fastapi.testclient import TestClient

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from app.web import LIVE_FLAG_VALUES, app


class WebLiveFlagsTest(unittest.TestCase):
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
            self.assertFalse(payload["ok"])
            self.assertFalse(payload["orders_submitted"])
            for key, value in LIVE_FLAG_VALUES.items():
                self.assertEqual(os.environ[key], value)
                self.assertEqual(payload["flags"][key], value)

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
