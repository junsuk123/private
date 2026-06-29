from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from app.trading.live_runtime_guard import arming_failures, create_arming_file, evaluate_live_runtime_gates


class LiveArmingTest(unittest.TestCase):
    def test_arming_file_expires(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "armed.json"
            create_arming_file(path, ttl_seconds=-1)

            self.assertEqual(arming_failures(path), ["MANUAL_ARMING_EXPIRED"])

    def test_all_live_flags_and_arming_are_required(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "armed.json"
            create_arming_file(path, ttl_seconds=60)
            env = {
                "LIVE_TRADING_ENABLED": "true",
                "KIS_LIVE_ENABLED": "true",
                "KIS_PAPER_TRADING": "false",
                "LIVE_ORDER_SUBMIT_ENABLED": "true",
                "KILL_SWITCH_ENABLED": "false",
            }
            with patch("app.trading.live_runtime_guard.ARMING_FILE", path), patch.dict("os.environ", env, clear=True):
                result = evaluate_live_runtime_gates(require_manual_arming=True)

        self.assertTrue(result.ok, result.failures)


if __name__ == "__main__":
    unittest.main()
