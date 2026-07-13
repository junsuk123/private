from __future__ import annotations

import unittest
from dataclasses import replace

from app.cost import ProfitabilityGate, ProfitabilityInput, load_policy


def _gate(**policy_overrides) -> ProfitabilityGate:
    policy = load_policy()
    if policy_overrides:
        policy = replace(policy, **policy_overrides)
    return ProfitabilityGate(policy=policy)


class ProfitabilityGateTest(unittest.TestCase):
    def test_profitable_kr_buy_is_allowed(self) -> None:
        gate = _gate()
        decision = gate.evaluate(
            ProfitabilityInput(
                symbol="005930",
                market="KR",
                venue="KRX",
                instrument_type="domestic_stock",
                entry_price=10_000,
                expected_exit_price=10_150,  # +1.5% gross clears KR cost + 0.8% floor
                quantity=1,
                spread_rate=0.0005,
                liquidity_score=0.9,
                realized_volatility=0.002,
                account_equity_krw=5_000_000,
            )
        )
        self.assertTrue(decision.allowed, decision.rejection_reasons)
        self.assertGreaterEqual(decision.net_expected_return, decision.required_min_net_return)
        self.assertGreater(decision.break_even_exit_price, decision.entry_price)

    def test_missing_expected_exit_price_is_rejected(self) -> None:
        decision = _gate().evaluate(
            ProfitabilityInput(symbol="005930", market="KR", entry_price=10_000, expected_exit_price=None, quantity=1)
        )
        self.assertFalse(decision.allowed)
        self.assertIn("MISSING_EXPECTED_EXIT_PRICE", decision.rejection_reasons)

    def test_net_return_below_required_is_rejected(self) -> None:
        # +0.3% gross does not clear a strict 0.8% KR floor after cost.
        decision = _gate(min_required_net_return={"default": 0.008, "KR": 0.008, "US": 0.012}).evaluate(
            ProfitabilityInput(
                symbol="005930",
                market="KR",
                entry_price=10_000,
                expected_exit_price=10_030,
                quantity=1,
                spread_rate=0.0005,
                liquidity_score=0.9,
            )
        )
        self.assertFalse(decision.allowed)
        self.assertIn("BELOW_TARGET_NET_RETURN_AFTER_COST", decision.rejection_reasons)

    def test_wide_spread_is_rejected(self) -> None:
        decision = _gate().evaluate(
            ProfitabilityInput(
                symbol="005930",
                market="KR",
                entry_price=10_000,
                expected_exit_price=10_300,
                quantity=1,
                spread_rate=0.02,  # 2% spread
                liquidity_score=0.9,
            )
        )
        self.assertFalse(decision.allowed)
        self.assertIn("SPREAD_TOO_WIDE", decision.rejection_reasons)

    def test_spread_consuming_alpha_is_rejected(self) -> None:
        # Spread within the absolute ceiling but large relative to a thin alpha.
        decision = _gate(max_spread_rate=0.01).evaluate(
            ProfitabilityInput(
                symbol="005930",
                market="KR",
                entry_price=10_000,
                expected_exit_price=10_090,  # ~0.9% gross
                quantity=1,
                spread_rate=0.006,  # 60% of alpha
                liquidity_score=0.9,
            )
        )
        self.assertFalse(decision.allowed)
        self.assertIn("SPREAD_CONSUMES_ALPHA", decision.rejection_reasons)

    def test_low_liquidity_is_rejected(self) -> None:
        decision = _gate().evaluate(
            ProfitabilityInput(
                symbol="005930",
                market="KR",
                entry_price=10_000,
                expected_exit_price=10_300,
                quantity=1,
                spread_rate=0.0005,
                liquidity_score=0.1,  # below the 0.3 floor
            )
        )
        self.assertFalse(decision.allowed)
        self.assertIn("LIQUIDITY_TOO_LOW", decision.rejection_reasons)

    def test_required_min_net_return_is_dynamic(self) -> None:
        # Higher realized volatility raises the required minimum net return.
        gate = _gate()
        calm = gate.evaluate(
            ProfitabilityInput(symbol="A", market="KR", entry_price=10_000, expected_exit_price=10_200, quantity=1, spread_rate=0.0005, liquidity_score=0.9, realized_volatility=0.0)
        )
        noisy = gate.evaluate(
            ProfitabilityInput(symbol="A", market="KR", entry_price=10_000, expected_exit_price=10_200, quantity=1, spread_rate=0.0005, liquidity_score=0.9, realized_volatility=0.02)
        )
        self.assertGreater(noisy.required_min_net_return, calm.required_min_net_return)

    def test_small_account_adds_extra_required_net(self) -> None:
        # Use a low market floor so the dynamic buffer path (not the floor) governs,
        # making the small-account buffer's effect observable. Set an explicit small-
        # account buffer so this exercises the MECHANISM regardless of the deployment's
        # tuned profitability_policy.yaml (which currently relaxes the buffer to 0.0).
        gate = _gate(
            min_required_net_return={"default": 0.0, "KR": 0.0, "US": 0.0},
            small_account_equity_krw=200_000.0,
            small_account_extra_net=0.002,
        )
        big = gate.evaluate(
            ProfitabilityInput(symbol="A", market="KR", entry_price=10_000, expected_exit_price=10_300, quantity=1, spread_rate=0.0005, liquidity_score=0.9, account_equity_krw=5_000_000)
        )
        small = gate.evaluate(
            ProfitabilityInput(symbol="A", market="KR", entry_price=10_000, expected_exit_price=10_300, quantity=1, spread_rate=0.0005, liquidity_score=0.9, account_equity_krw=150_000)
        )
        self.assertGreater(small.required_min_net_return, big.required_min_net_return)
        self.assertGreater(small.breakdown.account_buffer, 0.0)
        self.assertEqual(big.breakdown.account_buffer, 0.0)

    def test_explicit_target_only_tightens_never_loosens(self) -> None:
        gate = _gate()
        # A target below the market floor must not drop the requirement below the floor.
        decision = gate.evaluate(
            ProfitabilityInput(symbol="A", market="KR", entry_price=10_000, expected_exit_price=10_300, quantity=1, spread_rate=0.0005, liquidity_score=0.9, target_net_return=0.0)
        )
        self.assertGreaterEqual(decision.required_min_net_return, gate.policy.min_net_for_market("KR"))

    def test_sell_action_is_never_gated(self) -> None:
        decision = _gate().evaluate(
            ProfitabilityInput(symbol="A", action="SELL", market="KR", entry_price=10_000, expected_exit_price=9_000, quantity=5)
        )
        self.assertTrue(decision.allowed)
        self.assertIn("NON_BUY_ACTION_NOT_GATED", decision.warnings)

    def test_empty_orderbook_flagged_and_rejected(self) -> None:
        decision = _gate().evaluate(
            ProfitabilityInput(
                symbol="LCFYW",
                market="KR",
                entry_price=10_000,
                expected_exit_price=10_300,
                quantity=1,
                orderbook_snapshot={"best_bid": 0.0, "best_ask": 0.0},
                liquidity_score=0.9,
            )
        )
        self.assertFalse(decision.allowed)
        self.assertIn("SPREAD_TOO_WIDE", decision.rejection_reasons)
        self.assertIn("EMPTY_OR_INVALID_ORDERBOOK", decision.data_quality_flags)

    def test_decision_serializes(self) -> None:
        decision = _gate().evaluate(
            ProfitabilityInput(symbol="A", market="KR", entry_price=10_000, expected_exit_price=10_300, quantity=1, spread_rate=0.0005, liquidity_score=0.9)
        )
        payload = decision.as_dict()
        for key in (
            "allowed",
            "net_expected_return",
            "required_min_net_return",
            "break_even_exit_price",
            "all_in_cost_rate",
            "cost_to_alpha_ratio",
            "rejection_reasons",
        ):
            self.assertIn(key, payload)


if __name__ == "__main__":
    unittest.main()
