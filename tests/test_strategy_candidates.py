from __future__ import annotations

import unittest
from datetime import datetime, timedelta

from app.schemas.domain import OrderAction, OrderIntent
from app.strategy.candidates import StrategyCandidate


class TestStrategyDataStructures(unittest.TestCase):
    def test_strategy_candidate_creation(self) -> None:
        """Test the creation of a StrategyCandidate instance."""
        now = datetime.now()
        candidate = StrategyCandidate(
            ticker="005930.KS",
            strategy_family="short_term_momentum",
            signal_name="MA_Cross_5_20",
            entry_price=75000.0,
            expected_exit_price=76500.0,
            expected_holding_minutes=60,
            gross_expected_return=0.02,
            confidence=0.85,
            features={"sma_5": 74000.0, "sma_20": 73500.0},
            ontology_tags=["momentum_positive", "golden_cross"],
            validation_id="valid-123",
            reason="5-day MA crossed above 20-day MA.",
            created_at=now,
        )
        self.assertEqual(candidate.ticker, "005930.KS")
        self.assertEqual(candidate.strategy_family, "short_term_momentum")
        self.assertEqual(candidate.expected_exit_price, 76500.0)
        self.assertEqual(candidate.confidence, 0.85)
        self.assertIn("golden_cross", candidate.ontology_tags)
        self.assertAlmostEqual(candidate.created_at.timestamp(), now.timestamp())

    def test_strategy_candidate_as_dict(self) -> None:
        """Test the as_dict method of StrategyCandidate."""
        candidate = StrategyCandidate(
            ticker="005930.KS",
            strategy_family="short_term_momentum",
            signal_name="MA_Cross_5_20",
            entry_price=75000.0,
            expected_exit_price=76500.0,
            expected_holding_minutes=60,
            gross_expected_return=0.02,
            confidence=0.85,
        )
        candidate_dict = candidate.as_dict()
        self.assertEqual(candidate_dict["ticker"], "005930.KS")
        self.assertEqual(candidate_dict["expected_holding_minutes"], 60)
        self.assertIn("created_at", candidate_dict)

    def test_order_intent_extension(self) -> None:
        """Test that OrderIntent can be created with new optional fields."""
        now = datetime.now()
        intent = OrderIntent(
            ticker="AAPL",
            market="US",
            action=OrderAction.BUY,
            suggested_weight=0.05,
            confidence=0.7,
            valid_until=now + timedelta(hours=1),
            reasoning_summary=("Test reason",),
            supporting_factors=("Factor1",),
            contradicting_factors=(),
            source_data_ids=("source1",),
            # New fields
            strategy_family="short_term_reversion",
            signal_name="RSI_Oversold",
            expected_exit_price=175.0,
            expected_holding_minutes=120,
            gross_expected_return=0.03,
            target_net_return=0.015,
            ontology_tags=("oversold", "mean_reversion"),
            strategy_metadata={"rsi_value": 25.0},
        )
        self.assertEqual(intent.ticker, "AAPL")
        self.assertEqual(intent.strategy_family, "short_term_reversion")
        self.assertEqual(intent.expected_exit_price, 175.0)
        self.assertEqual(intent.target_net_return, 0.015)
        self.assertIn("oversold", intent.ontology_tags)
        self.assertIn("rsi_value", intent.strategy_metadata)

    def test_candidate_converts_to_order_intent_without_executing(self) -> None:
        now = datetime.now()
        candidate = StrategyCandidate(
            ticker="005930.KS",
            strategy_family="short_term_momentum",
            signal_name="breakout",
            entry_price=75_000,
            expected_exit_price=76_500,
            expected_holding_minutes=45,
            gross_expected_return=0.02,
            confidence=0.91,
            features={"volume_ratio": 2.4},
            ontology_tags=["BreakoutWatch"],
            validation_id="validation-1",
            reason="Breakout candidate after volume expansion.",
        )

        intent = candidate.to_order_intent(
            market="KR",
            suggested_weight=0.02,
            valid_until=now + timedelta(minutes=45),
            source_data_ids=("quote:005930",),
            target_net_return=0.004,
        )

        self.assertEqual(intent.action, OrderAction.BUY)
        self.assertEqual(intent.expected_exit_price, candidate.expected_exit_price)
        self.assertEqual(intent.strategy_family, "short_term_momentum")
        self.assertEqual(intent.signal_name, "breakout")
        self.assertEqual(intent.validation_id, "validation-1")
        self.assertEqual(intent.ontology_tags, ("BreakoutWatch",))
        self.assertEqual(intent.strategy_metadata["features"]["volume_ratio"], 2.4)

    def test_candidate_requires_expected_exit_price(self) -> None:
        with self.assertRaises(ValueError):
            StrategyCandidate(
                ticker="005930.KS",
                strategy_family="short_term_momentum",
                signal_name="bad_signal",
                entry_price=75_000,
                expected_exit_price=0,
                expected_holding_minutes=45,
                gross_expected_return=0.0,
                confidence=0.5,
            )

    def test_order_intent_backward_compatibility(self) -> None:
        """Test that OrderIntent can be created without the new fields."""
        now = datetime.now()
        try:
            intent = OrderIntent(
                ticker="MSFT",
                market="US",
                action=OrderAction.BUY,
                suggested_weight=0.1,
                confidence=0.8,
                valid_until=now + timedelta(hours=1),
                reasoning_summary=("Legacy reason",),
                supporting_factors=(),
                contradicting_factors=(),
                source_data_ids=("source2",),
            )
            # Check that default values are set correctly
            self.assertEqual(intent.ticker, "MSFT")
            self.assertIsNone(intent.strategy_family)
            self.assertIsNone(intent.expected_exit_price)
            self.assertEqual(intent.ontology_tags, ())
        except TypeError as e:
            self.fail(f"OrderIntent instantiation failed with TypeError: {e}")


if __name__ == "__main__":
    unittest.main()
