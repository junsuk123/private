from __future__ import annotations

import unittest
from unittest.mock import patch

from app.trading.dynamic_exit_policy import DynamicExitPolicy, LossExitEvidence


class DynamicExitPolicyTest(unittest.TestCase):
    def setUp(self) -> None:
        self.policy = DynamicExitPolicy()

    def test_take_profit_is_dynamic_with_volatility(self) -> None:
        calm = self.policy.resolve(all_in_cost_rate=0.003, realized_volatility=0.0)
        noisy = self.policy.resolve(all_in_cost_rate=0.003, realized_volatility=0.01)
        self.assertGreater(noisy.take_profit_rate, calm.take_profit_rate)
        # Take-profit always clears cost + the min net-profit buffer.
        self.assertGreaterEqual(calm.take_profit_rate, 0.003 + calm.min_net_profit_exit)

    def test_env_override_is_honored(self) -> None:
        with patch.dict("os.environ", {"REALTIME_QUICK_TAKE_PROFIT_NET": "0.02"}):
            levels = self.policy.resolve(all_in_cost_rate=0.003)
        self.assertEqual(levels.quick_take_profit_net, 0.02)

    def test_stop_loss_net_requires_separate_routine_loss_sell_optin(self) -> None:
        with patch.dict(
            "os.environ",
            {
                "REALTIME_STOP_LOSS_NET": "0.004",
                "REALTIME_ENABLE_ROUTINE_LOSS_SELL": "false",
            },
        ):
            levels = self.policy.resolve(all_in_cost_rate=0.003)
        self.assertEqual(levels.stop_loss_net, 0.0)

    def test_stop_loss_net_honored_when_routine_loss_sell_is_armed(self) -> None:
        with patch.dict(
            "os.environ",
            {
                "REALTIME_STOP_LOSS_NET": "0.004",
                "REALTIME_ENABLE_ROUTINE_LOSS_SELL": "true",
            },
        ):
            levels = self.policy.resolve(all_in_cost_rate=0.003)
        self.assertEqual(levels.stop_loss_net, 0.004)

    def test_hard_stop_always_permits_loss_exit(self) -> None:
        levels = self.policy.resolve(all_in_cost_rate=0.003)
        allowed, reason = self.policy.loss_exit_decision(
            levels, LossExitEvidence(pnl_rate=-0.10, net_pnl_rate=-0.10)
        )
        self.assertTrue(allowed)
        self.assertEqual(reason, "hard_stop_loss")

    def test_noise_band_loss_is_blocked(self) -> None:
        with patch.dict("os.environ", {"REALTIME_ALLOW_LOSS_EXIT": "true"}):
            levels = self.policy.resolve(all_in_cost_rate=0.003)
        allowed, reason = self.policy.loss_exit_decision(
            levels, LossExitEvidence(pnl_rate=-0.002, net_pnl_rate=-0.003)
        )
        self.assertFalse(allowed)
        self.assertEqual(reason, "LOSS_WITHIN_NOISE_BAND")

    def test_loss_exit_disabled_by_default(self) -> None:
        levels = self.policy.resolve(all_in_cost_rate=0.003)
        allowed, reason = self.policy.loss_exit_decision(
            levels, LossExitEvidence(pnl_rate=-0.02, net_pnl_rate=-0.02, ontology_score=0.0)
        )
        self.assertFalse(allowed)
        self.assertEqual(reason, "LOSS_EXIT_DISABLED")

    def test_ontology_dominance_permits_controlled_loss_exit(self) -> None:
        with patch.dict("os.environ", {"REALTIME_ALLOW_LOSS_EXIT": "true"}):
            levels = self.policy.resolve(all_in_cost_rate=0.003)
        allowed, reason = self.policy.loss_exit_decision(
            levels, LossExitEvidence(pnl_rate=-0.02, net_pnl_rate=-0.02, ontology_score=-0.8)
        )
        self.assertTrue(allowed)
        self.assertEqual(reason, "ontology_sell_dominance")

    def test_strong_negative_forecast_permits_loss_exit(self) -> None:
        with patch.dict("os.environ", {"REALTIME_ALLOW_LOSS_EXIT": "true"}):
            levels = self.policy.resolve(all_in_cost_rate=0.003)
        allowed, reason = self.policy.loss_exit_decision(
            levels,
            LossExitEvidence(pnl_rate=-0.02, net_pnl_rate=-0.02, predicted_net_return_bps=-25.0),
        )
        self.assertTrue(allowed)
        self.assertEqual(reason, "strong_negative_forecast")

    def test_resolved_levels_serialize(self) -> None:
        payload = self.policy.resolve(all_in_cost_rate=0.003).as_dict()
        for key in ("take_profit_rate", "hard_stop_rate", "allow_loss_exit", "block_sell_below_breakeven"):
            self.assertIn(key, payload)


if __name__ == "__main__":
    unittest.main()
