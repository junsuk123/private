from __future__ import annotations

import unittest
from datetime import datetime, timezone, timedelta

from app.risk.manager import RiskManager
from app.schemas.domain import (
    AccountSnapshot,
    Holding,
    MarketSnapshot,
    OrderAction,
    OrderIntent,
    RiskRules,
    SourceMetadata,
)


def _create_mock_intent(
    expected_exit_price: float | None = None,
    target_net_return: float | None = 0.01,
    strategy_family: str | None = "test_family",
    validation_id: str | None = "valid-123",
    ontology_tags: tuple[str, ...] = (),
    strategy_metadata: dict | None = None,
) -> OrderIntent:
    return OrderIntent(
        ticker="005930.KS",
        market="KR",
        action=OrderAction.BUY,
        suggested_weight=0.05,
        confidence=0.8,
        valid_until=datetime.now(timezone.utc) + timedelta(hours=1),
        reasoning_summary=("Test",),
        supporting_factors=(),
        contradicting_factors=(),
        source_data_ids=("test-source",),
        expected_exit_price=expected_exit_price,
        target_net_return=target_net_return,
        strategy_family=strategy_family,
        validation_id=validation_id,
        ontology_tags=ontology_tags,
        strategy_metadata=strategy_metadata or {},
    )


def _create_mock_market_snapshot() -> MarketSnapshot:
    return MarketSnapshot(
        ticker="005930.KS",
        market="KR",
        company_name="Samsung Electronics",
        sector="Technology",
        last_price=75000.0,
        average_daily_trading_value=500_000_000_000,
        volatility_20d=0.02,
        source=SourceMetadata(
            source_name="test_source",
            retrieved_at=datetime.now(timezone.utc),
            quality_score=0.9,
            trust_level=4,
        ),
    )


def _create_mock_account_snapshot() -> AccountSnapshot:
    return AccountSnapshot(cash=10_000_000, holdings=())


class TestRiskManagerCostGate(unittest.TestCase):
    def setUp(self) -> None:
        self.market = _create_mock_market_snapshot()
        self.account = _create_mock_account_snapshot()

    def test_missing_expected_exit_price_rejected(self) -> None:
        """Test that an intent is rejected if expected_exit_price is missing for a BUY action."""
        intent = _create_mock_intent(expected_exit_price=None)
        risk_manager = RiskManager(rules=RiskRules(live_trading_enabled=False))
        result = risk_manager.validate(intent, self.account, self.market)
        self.assertFalse(result.approved)
        self.assertIn("MISSING_EXPECTED_EXIT_PRICE", result.rejection_reasons)
        self.assertIn("rejection_log", result.metadata)

    def test_missing_strategy_family_rejected(self) -> None:
        exit_price = self.market.last_price * 1.05
        intent = _create_mock_intent(expected_exit_price=exit_price, strategy_family=None)

        result = RiskManager(rules=RiskRules(live_trading_enabled=False)).validate(intent, self.account, self.market)

        self.assertFalse(result.approved)
        self.assertIn("MISSING_STRATEGY_FAMILY", result.rejection_reasons)

    def test_positive_gross_negative_net_rejected(self) -> None:
        """Test that an intent is rejected if net return is negative after costs."""
        # Set an exit price that is only slightly higher than the entry price,
        # ensuring gross return is positive but not enough to cover costs.
        exit_price = self.market.last_price * 1.001  # +0.1% gross return
        intent = _create_mock_intent(expected_exit_price=exit_price, target_net_return=0.01)
        risk_manager = RiskManager(rules=RiskRules(live_trading_enabled=False))
        result = risk_manager.validate(intent, self.account, self.market)
        self.assertFalse(result.approved)
        # The specific reason can vary based on cost engine config,
        # but it should be a cost-related rejection.
        self.assertTrue(
            any(
                reason in result.rejection_reasons
                for reason in [
                    "NET_RETURN_NOT_POSITIVE",
                    "BELOW_BREAK_EVEN_WITH_MARGIN",
                    "BELOW_TARGET_NET_RETURN_AFTER_COST",
                ]
            )
        )
        self.assertIn("rejection_log", result.metadata)

    def test_valid_candidate_passes_paper_gate(self) -> None:
        """Test that a valid candidate passes in paper trading mode."""
        # Set an exit price high enough to cover costs and meet the target net return.
        exit_price = self.market.last_price * 1.05  # +5% gross return
        intent = _create_mock_intent(expected_exit_price=exit_price, target_net_return=0.01)
        risk_manager = RiskManager(rules=RiskRules(live_trading_enabled=False))
        result = risk_manager.validate(intent, self.account, self.market)
        self.assertTrue(result.approved, msg=f"Validation failed with reasons: {result.rejection_reasons}")
        self.assertIsNotNone(result.final_order)
        self.assertGreater(result.final_order.quantity, 0)
        self.assertEqual(result.metadata["validation_required"], False)
        self.assertIn("cost_breakdown", result.metadata)

    def test_paper_candidate_without_validation_records_validation_required(self) -> None:
        exit_price = self.market.last_price * 1.05
        intent = _create_mock_intent(expected_exit_price=exit_price, target_net_return=0.01, validation_id=None)

        result = RiskManager(rules=RiskRules(live_trading_enabled=False)).validate(intent, self.account, self.market)

        self.assertTrue(result.approved, msg=f"Validation failed with reasons: {result.rejection_reasons}")
        self.assertEqual(result.metadata["validation_required"], True)

    def test_below_target_net_return_after_cost_rejected(self) -> None:
        exit_price = self.market.last_price * 1.03
        intent = _create_mock_intent(expected_exit_price=exit_price, target_net_return=0.05)

        result = RiskManager(rules=RiskRules(live_trading_enabled=False)).validate(intent, self.account, self.market)

        self.assertFalse(result.approved)
        self.assertIn("BELOW_TARGET_NET_RETURN_AFTER_COST", result.rejection_reasons)

    def test_cost_burden_high_rejected(self) -> None:
        exit_price = self.market.last_price * 1.05
        intent = _create_mock_intent(expected_exit_price=exit_price, target_net_return=0.01)
        risk_manager = RiskManager(rules=RiskRules(live_trading_enabled=False))
        risk_manager.cost_engine.config["gate"]["max_cost_to_alpha_ratio"] = 0.01

        result = risk_manager.validate(intent, self.account, self.market)

        self.assertFalse(result.approved)
        self.assertIn("COST_BURDEN_HIGH", result.rejection_reasons)

    def test_spread_too_wide_rejected(self) -> None:
        exit_price = self.market.last_price * 1.05
        intent = _create_mock_intent(
            expected_exit_price=exit_price,
            target_net_return=0.01,
            strategy_metadata={"orderbook_snapshot": {"best_bid": 74_000, "best_ask": 76_000}},
        )

        result = RiskManager(rules=RiskRules(live_trading_enabled=False)).validate(intent, self.account, self.market)

        self.assertFalse(result.approved)
        self.assertIn("SPREAD_TOO_WIDE", result.rejection_reasons)

    def test_slippage_risk_high_rejected(self) -> None:
        exit_price = self.market.last_price * 1.05
        intent = _create_mock_intent(expected_exit_price=exit_price, target_net_return=0.01)
        risk_manager = RiskManager(rules=RiskRules(live_trading_enabled=False))
        risk_manager.cost_engine.config["gate"]["max_slippage_rate"] = 0.0001

        result = risk_manager.validate(intent, self.account, self.market)

        self.assertFalse(result.approved)
        self.assertIn("SLIPPAGE_RISK_HIGH", result.rejection_reasons)

    def test_ontology_trade_forbidden_rejected(self) -> None:
        exit_price = self.market.last_price * 1.05
        intent = _create_mock_intent(expected_exit_price=exit_price, ontology_tags=("TradeForbidden",))

        result = RiskManager(rules=RiskRules(live_trading_enabled=False)).validate(intent, self.account, self.market)

        self.assertFalse(result.approved)
        self.assertIn("ONTOLOGY_TRADE_FORBIDDEN", result.rejection_reasons)

    def test_no_reality_check_live_rejected_by_live_safety_gate(self) -> None:
        """Live mode stays blocked regardless of strategy validation state."""
        exit_price = self.market.last_price * 1.05
        intent = _create_mock_intent(
            expected_exit_price=exit_price,
            target_net_return=0.01,
            validation_id=None, # No validation ID
        )
        # Enable live trading in rules
        risk_manager = RiskManager(rules=RiskRules(live_trading_enabled=True))
        result = risk_manager.validate(intent, self.account, self.market)
        self.assertFalse(result.approved)
        self.assertIn("live_trading_disabled", result.rejection_reasons)
        self.assertIn("MISSING_VALIDATION_ID", result.rejection_reasons)
        self.assertIn("REALITY_CHECK_NOT_PASSED", result.rejection_reasons)
    
    def test_valid_candidate_still_does_not_bypass_live_safety_gate(self) -> None:
        """Even a valid candidate must not bypass the disabled live-order safety gate."""
        exit_price = self.market.last_price * 1.05
        intent = _create_mock_intent(
            expected_exit_price=exit_price,
            target_net_return=0.01,
            validation_id="some-validation-id",
        )
        risk_manager = RiskManager(rules=RiskRules(live_trading_enabled=True))
        result = risk_manager.validate(intent, self.account, self.market)
        self.assertFalse(result.approved)
        self.assertIn("live_trading_disabled", result.rejection_reasons)
        self.assertIsNone(result.final_order)


if __name__ == "__main__":
    unittest.main()
