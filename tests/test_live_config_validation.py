from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from app.config import LiveConfigError, load_live_trading_safety_config, load_order_execution_config


class LiveConfigValidationTest(unittest.TestCase):
    def test_example_live_configs_are_valid_and_limit_only(self) -> None:
        safety = load_live_trading_safety_config("config/live_trading_safety.json", allow_example=True)
        execution = load_order_execution_config("config/order_execution.json", allow_example=True)

        self.assertFalse(safety.market_orders_allowed)
        self.assertFalse(safety.allow_heuristic_fallback_in_live)
        self.assertEqual(execution.order_type, "LIMIT_ONLY")
        self.assertFalse(execution.allow_market_orders)

    def test_unsafe_market_order_config_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "live_trading_safety.json"
            payload = json.loads(Path("config/live_trading_safety.example.json").read_text(encoding="utf-8"))
            payload["market_orders_allowed"] = True
            path.write_text(json.dumps(payload), encoding="utf-8")

            with self.assertRaisesRegex(LiveConfigError, "limit-only"):
                load_live_trading_safety_config(path)


if __name__ == "__main__":
    unittest.main()
