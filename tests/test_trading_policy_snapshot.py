from __future__ import annotations

import sys
import unittest
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from app.trading.trading_policy import TradingPolicySnapshot


class TradingPolicySnapshotTest(unittest.TestCase):
    def test_version_is_deterministic_and_changes_with_policy(self) -> None:
        base = {
            "REALTIME_TAKE_PROFIT": "0.0025",
            "REALTIME_QUICK_TAKE_PROFIT_NET": "0.008",
            "REALTIME_STOP_LOSS_NET": "0.008",
            "REALTIME_HARD_STOP_LOSS": "0.03",
            "REALTIME_EMERGENCY_STOP_LOSS": "0.05",
            "REALTIME_ALLOW_LOSS_EXIT": "true",
        }
        with patch.dict("os.environ", base, clear=False):
            a = TradingPolicySnapshot.from_environment()
            b = TradingPolicySnapshot.from_environment()
        self.assertEqual(a.policy_version, b.policy_version)
        self.assertTrue(a.policy_version.startswith("tp_"))

        changed = {**base, "REALTIME_STOP_LOSS_NET": "0.012"}
        with patch.dict("os.environ", changed, clear=False):
            c = TradingPolicySnapshot.from_environment()
        self.assertNotEqual(a.policy_version, c.policy_version)

    def test_disabled_stop_loss_is_a_fail_conflict(self) -> None:
        env = {"REALTIME_ALLOW_LOSS_EXIT": "false", "REALTIME_STOP_LOSS_NET": "0.0"}
        with patch.dict("os.environ", env, clear=False):
            snap = TradingPolicySnapshot.from_environment()
        conflicts = snap.conflicts()
        codes = {c.code for c in conflicts}
        self.assertIn("STOP_LOSS_DISABLED", codes)
        self.assertEqual(
            next(c.severity for c in conflicts if c.code == "STOP_LOSS_DISABLED"), "FAIL"
        )

    def test_enabled_loss_exit_with_net_stop_has_no_stop_conflict(self) -> None:
        env = {
            "REALTIME_ALLOW_LOSS_EXIT": "true",
            "REALTIME_STOP_LOSS_NET": "0.008",
            "REALTIME_HARD_STOP_LOSS": "0.03",
            "REALTIME_EMERGENCY_STOP_LOSS": "0.05",
        }
        with patch.dict("os.environ", env, clear=False):
            snap = TradingPolicySnapshot.from_environment()
        codes = {c.code for c in snap.conflicts()}
        self.assertNotIn("STOP_LOSS_DISABLED", codes)
        self.assertNotIn("HARD_STOP_NOT_WIDER_THAN_STOP", codes)

    def test_position_weight_above_one_is_warning_not_fail(self) -> None:
        env = {"REALTIME_SMALL_ACCOUNT_MAX_POSITION_WEIGHT": "1.25"}
        with patch.dict("os.environ", env, clear=False):
            snap = TradingPolicySnapshot.from_environment()
        pw = [c for c in snap.conflicts() if c.code == "POSITION_WEIGHT_ABOVE_ONE"]
        self.assertTrue(pw)
        self.assertEqual(pw[0].severity, "WARNING")


class PolicyVersionParityTest(unittest.TestCase):
    def test_gate_and_exit_policy_share_version_and_decisions_are_stamped(self) -> None:
        from app.cost.profitability_gate import ProfitabilityGate, ProfitabilityInput
        from app.trading.dynamic_exit_policy import DynamicExitPolicy

        gate = ProfitabilityGate()
        exit_policy = DynamicExitPolicy()
        self.assertTrue(gate.policy_version)
        self.assertEqual(gate.policy_version, exit_policy.policy_version)

        decision = gate.evaluate(
            ProfitabilityInput(
                symbol="005930",
                action="BUY",
                market="KR",
                entry_price=10_000.0,
                quantity=1,
                expected_exit_price=10_100.0,
            )
        )
        self.assertEqual(decision.policy_version, gate.policy_version)
        self.assertEqual(decision.as_dict()["policy_version"], gate.policy_version)


if __name__ == "__main__":
    unittest.main()
