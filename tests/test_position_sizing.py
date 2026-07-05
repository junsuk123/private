from __future__ import annotations

import unittest

from app.risk.position_sizing import PositionSizer, SizingInputs


class PositionSizingTest(unittest.TestCase):
    def setUp(self) -> None:
        self.sizer = PositionSizer()

    def test_negative_expectancy_gets_zero_size(self) -> None:
        result = self.sizer.size(SizingInputs(net_expected_return=-0.01, target_net_return=0.008))
        self.assertEqual(result.position_weight, 0.0)

    def test_higher_edge_sizes_larger(self) -> None:
        low = self.sizer.size(SizingInputs(net_expected_return=0.002, target_net_return=0.008, confidence_score=0.8, liquidity_score=1.0))
        high = self.sizer.size(SizingInputs(net_expected_return=0.008, target_net_return=0.008, confidence_score=0.8, liquidity_score=1.0))
        self.assertGreater(high.position_weight, low.position_weight)

    def test_drawdown_reduces_size(self) -> None:
        base = self.sizer.size(SizingInputs(net_expected_return=0.008, target_net_return=0.008, confidence_score=0.8, account_drawdown_rate=0.0))
        drawn = self.sizer.size(SizingInputs(net_expected_return=0.008, target_net_return=0.008, confidence_score=0.8, account_drawdown_rate=-0.05))
        self.assertLess(drawn.position_weight, base.position_weight)
        self.assertLess(drawn.drawdown_multiplier, 1.0)

    def test_recent_loss_reduces_size(self) -> None:
        base = self.sizer.size(SizingInputs(net_expected_return=0.008, target_net_return=0.008, confidence_score=0.8))
        after_loss = self.sizer.size(SizingInputs(net_expected_return=0.008, target_net_return=0.008, confidence_score=0.8, recent_same_strategy_loss=True))
        self.assertLess(after_loss.position_weight, base.position_weight)

    def test_confidence_alone_does_not_exceed_base_weight(self) -> None:
        # With all multipliers at their max (edge=1, confidence=1, liquidity=1), the
        # weight cannot exceed the base weight — confidence never inflates size.
        result = self.sizer.size(SizingInputs(net_expected_return=0.008, target_net_return=0.008, confidence_score=1.0, liquidity_score=1.0))
        self.assertLessEqual(result.position_weight, self.sizer.config["base_position_weight"] + 1e-9)

    def test_weight_never_exceeds_max(self) -> None:
        result = self.sizer.size(SizingInputs(net_expected_return=10.0, target_net_return=0.008, confidence_score=1.0, liquidity_score=1.0, p_win=0.99, avg_win_net=0.05, avg_loss_net=0.001))
        self.assertLessEqual(result.position_weight, self.sizer.config["max_position_weight"])


if __name__ == "__main__":
    unittest.main()
