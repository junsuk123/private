from __future__ import annotations

import unittest
from datetime import datetime, timedelta, timezone

from app.evaluation import RealityCheckConfig, RealityCheckValidator, StrategyTradeObservation


class RealityCheckValidatorTest(unittest.TestCase):
    def test_validator_reports_gross_and_net_metrics_after_costs(self) -> None:
        trades = _profitable_trades()
        validator = RealityCheckValidator(
            RealityCheckConfig(train_size=8, test_size=4, bootstrap_iterations=50, target_net_return=0.0)
        )

        report = validator.validate(trades, strategy_name="technical_rule")

        self.assertTrue(report.validation_id.startswith("reality-"))
        self.assertGreater(report.gross_total_return, report.net_total_return)
        self.assertGreater(report.net_total_return, 0)
        self.assertGreater(report.out_of_sample_net_return, 0)
        self.assertGreater(report.out_of_sample_sharpe, 0)
        self.assertGreater(report.average_cost_per_trade, 0)
        self.assertGreater(report.average_net_profit_per_trade, 0)
        self.assertLess(report.break_even_failure_ratio, 0.50)
        self.assertLess(report.fee_converted_loss_ratio, 0.30)
        self.assertIsNotNone(report.reality_check_p_value)
        self.assertTrue(report.passed)
        self.assertIn("RealityCheckPassed", report.ontology_tags)

    def test_fee_converted_losses_block_validation(self) -> None:
        trades = _small_gross_return_trades()
        validator = RealityCheckValidator(
            RealityCheckConfig(train_size=6, test_size=3, bootstrap_iterations=30, target_net_return=0.0)
        )

        report = validator.validate(trades, strategy_name="short_term_reversal")

        self.assertGreater(report.gross_total_return, report.net_total_return)
        self.assertGreater(report.fee_converted_loss_ratio, 0.30)
        self.assertFalse(report.passed)
        self.assertIn("NoOutOfSampleValidation", report.ontology_tags)
        self.assertIn("DataSnoopingRisk", report.ontology_tags)

    def test_walk_forward_splits_do_not_overlap_future_data(self) -> None:
        report = RealityCheckValidator(
            RealityCheckConfig(train_size=5, test_size=2, step_size=2, bootstrap_iterations=20)
        ).validate(_profitable_trades(), strategy_name="technical_rule")

        self.assertTrue(report.walk_forward_splits)
        for split in report.walk_forward_splits:
            self.assertLess(max(split.train_indices), min(split.test_indices))

    def test_requires_trades(self) -> None:
        with self.assertRaises(ValueError):
            RealityCheckValidator().validate(())


def _profitable_trades() -> tuple[StrategyTradeObservation, ...]:
    start = datetime(2026, 1, 1, tzinfo=timezone.utc)
    returns = [0.018, 0.022, 0.016, 0.024, 0.019, 0.021, 0.017, 0.023, 0.020, 0.026, 0.018, 0.025]
    return tuple(
        StrategyTradeObservation(
            strategy_name="technical_rule",
            ticker="005930",
            entry_time=start + timedelta(days=index),
            exit_time=start + timedelta(days=index, minutes=30),
            entry_price=10_000,
            exit_price=10_000 * (1 + item),
            quantity=10,
        )
        for index, item in enumerate(returns)
    )


def _small_gross_return_trades() -> tuple[StrategyTradeObservation, ...]:
    start = datetime(2026, 1, 1, tzinfo=timezone.utc)
    returns = [0.0008, 0.0010, 0.0012, 0.0009, 0.0011, 0.0010, 0.0012, 0.0008, 0.0010]
    return tuple(
        StrategyTradeObservation(
            strategy_name="short_term_reversal",
            ticker="005930",
            entry_time=start + timedelta(days=index),
            exit_time=start + timedelta(days=index, minutes=15),
            entry_price=10_000,
            exit_price=10_000 * (1 + item),
            quantity=10,
        )
        for index, item in enumerate(returns)
    )


if __name__ == "__main__":
    unittest.main()
