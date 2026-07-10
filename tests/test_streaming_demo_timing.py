from __future__ import annotations

import sys
import unittest
import os
from pathlib import Path
from unittest.mock import patch

from fastapi.testclient import TestClient

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from app.backtesting import StreamingAcceleratedDemo, TimeMode, TimeScalerConfig
from app.backtesting.streaming_demo import _currency_for_ticker
from app.web import _streaming_demos, app


TEST_TICKERS = ("AAPL", "MSFT", "NVDA", "005930.KS", "000660.KS")


class StreamingDemoTimingTest(unittest.TestCase):
    def test_twenty_minute_demo_has_twenty_visible_steps(self) -> None:
        demo = StreamingAcceleratedDemo(
            config=TimeScalerConfig(mode=TimeMode.REALTIME, acceleration_factor=1.0),
            period_minutes=20,
            initial_cash=10_000_000,
            seed=42,
            tickers=TEST_TICKERS,
        )

        results = demo.run_all_steps()

        self.assertEqual(len(results), 20)
        self.assertEqual(results[0].step_index, 15)
        self.assertEqual(results[-1].step_index, 34)
        self.assertEqual(demo.get_progress(), 100.0)
        self.assertTrue(demo.is_complete())
        final = demo.get_final_results()
        self.assertIsNotNone(final)
        self.assertEqual(final["final_positions"], {})
        self.assertEqual(results[-1].holdings, {})

    def test_compounding_demo_trends_upward_with_active_trading(self) -> None:
        demo = StreamingAcceleratedDemo(
            config=TimeScalerConfig(mode=TimeMode.REALTIME, acceleration_factor=1.0),
            period_minutes=20,
            initial_cash=10_000_000,
            seed=42,
            tickers=TEST_TICKERS,
        )

        results = demo.run_all_steps()
        values = [result.account_value for result in results]

        self.assertGreater(values[-1], values[0] * 1.03)
        self.assertGreaterEqual(results[-1].cumulative_trades, 30)
        self.assertTrue(
            any(
                "fast take-profit" in trade.reason
                for result in results
                for trade in result.trades_in_step
            )
        )

    def test_profit_gain_increases_for_shorter_more_aggressive_targets(self) -> None:
        calm = StreamingAcceleratedDemo(
            config=TimeScalerConfig(mode=TimeMode.REALTIME, acceleration_factor=1.0),
            period_minutes=390,
            initial_cash=10_000_000,
            target_return_rate=0.02,
            seed=42,
            tickers=TEST_TICKERS,
        )
        aggressive = StreamingAcceleratedDemo(
            config=TimeScalerConfig(mode=TimeMode.REALTIME, acceleration_factor=1.0),
            period_minutes=20,
            initial_cash=10_000_000,
            target_return_rate=0.08,
            seed=42,
            tickers=TEST_TICKERS,
        )

        calm_gain = calm._profit_gain_state()
        aggressive_gain = aggressive._profit_gain_state()

        self.assertGreater(aggressive_gain.gain, calm_gain.gain)
        self.assertGreater(aggressive_gain.max_single_stock_weight, calm_gain.max_single_stock_weight)
        self.assertLess(aggressive_gain.fast_take_profit, calm_gain.fast_take_profit)
        self.assertGreater(aggressive_gain.max_trades_per_day, calm_gain.max_trades_per_day)

    def test_profit_gain_multiplier_scales_dynamic_risk(self) -> None:
        normal = StreamingAcceleratedDemo(
            config=TimeScalerConfig(mode=TimeMode.REALTIME, acceleration_factor=1.0),
            period_minutes=120,
            initial_cash=10_000_000,
            target_return_rate=0.03,
            profit_gain_multiplier=1.0,
            seed=42,
            tickers=TEST_TICKERS,
        )
        boosted = StreamingAcceleratedDemo(
            config=TimeScalerConfig(mode=TimeMode.REALTIME, acceleration_factor=1.0),
            period_minutes=120,
            initial_cash=10_000_000,
            target_return_rate=0.03,
            profit_gain_multiplier=1.75,
            seed=42,
            tickers=TEST_TICKERS,
        )

        normal_gain = normal._profit_gain_state()
        boosted_gain = boosted._profit_gain_state()

        self.assertGreater(boosted_gain.gain, normal_gain.gain)
        self.assertGreater(boosted_gain.max_single_stock_weight, normal_gain.max_single_stock_weight)
        self.assertLess(boosted_gain.minimum_cash_reserve, normal_gain.minimum_cash_reserve)

    def test_final_step_liquidates_existing_holdings(self) -> None:
        demo = StreamingAcceleratedDemo(
            config=TimeScalerConfig(mode=TimeMode.REALTIME, acceleration_factor=1.0),
            period_minutes=1,
            initial_cash=10_000_000,
            seed=42,
            tickers=("AAPL",),
        )
        demo.initialize()
        demo._holdings["AAPL"] = 3

        result = demo.run_step()

        self.assertIsNotNone(result)
        self.assertEqual(result.holdings, {})
        self.assertTrue(
            any(
                trade.side == "SELL" and trade.reason == "mandatory final liquidation"
                for trade in result.trades_in_step
            )
        )

    def test_usd_buy_converts_krw_deposit_to_usd_deposit(self) -> None:
        previous_rate = os.environ.get("SIM_USD_KRW_RATE")
        os.environ["SIM_USD_KRW_RATE"] = "1350"
        try:
            demo = StreamingAcceleratedDemo(
                config=TimeScalerConfig(mode=TimeMode.REALTIME, acceleration_factor=1.0),
                period_minutes=1,
                initial_cash=1_350_000,
                seed=42,
                tickers=("AAPL",),
            )
            demo.initialize()

            self.assertEqual(_currency_for_ticker("AAPL"), "USD")
            approved = demo._ensure_cash_for_buy("USD", 100.0, 1350.0)

            self.assertTrue(approved)
            self.assertEqual(demo._cash_by_currency["USD"], 100.0)
            self.assertEqual(demo._cash_by_currency["KRW"], 1_215_000.0)
        finally:
            if previous_rate is None:
                os.environ.pop("SIM_USD_KRW_RATE", None)
            else:
                os.environ["SIM_USD_KRW_RATE"] = previous_rate

    def test_principal_protection_locks_initial_cash_after_first_doubling(self) -> None:
        demo = StreamingAcceleratedDemo(
            config=TimeScalerConfig(mode=TimeMode.REALTIME, acceleration_factor=1.0),
            period_minutes=1,
            initial_cash=1_000_000,
            seed=42,
            tickers=("AAPL",),
        )
        demo.initialize()

        demo._advance_capital_cycle(2_000_000)
        state = demo._principal_protection_state(2_000_000)

        self.assertTrue(state.principal_locked)
        self.assertEqual(state.protected_principal, 1_000_000)
        self.assertEqual(state.cycle_seed, 2_000_000)
        self.assertEqual(state.target_profit_amount, 2_000_000)
        self.assertEqual(state.target_equity, 4_000_000)

        demo._cash_by_currency["KRW"] = 1_050_000
        self.assertFalse(demo._ensure_cash_for_buy("KRW", 100_000, 1350.0))
        self.assertTrue(demo._ensure_cash_for_buy("KRW", 50_000, 1350.0))

    def test_principal_protection_sells_holdings_to_raise_cash_floor(self) -> None:
        demo = StreamingAcceleratedDemo(
            config=TimeScalerConfig(mode=TimeMode.REALTIME, acceleration_factor=1.0),
            period_minutes=1,
            initial_cash=1_000_000,
            seed=42,
            tickers=("005930.KS",),
        )
        demo.initialize()
        demo._advance_capital_cycle(2_000_000)
        demo._cash_by_currency["KRW"] = 100_000
        demo._holdings["005930.KS"] = 20
        demo._holding_currency_by_ticker["005930.KS"] = "KRW"

        trades = demo._raise_protected_cash_floor({"005930.KS": 100_000}, demo._timestamps[0])

        self.assertTrue(trades)
        self.assertGreaterEqual(demo._cash_by_currency["KRW"], 1_000_000)
        self.assertTrue(all(trade.reason == "principal protection reserve" for trade in trades))

    # --- Paper-trading removal (deprecation contract) --------------------------
    # Paper trading was removed; the app now trades live only. The old
    # start/step/terminate demo flow is gone. These verify the graceful deprecation.
    def test_paper_trading_start_endpoint_removed(self) -> None:
        client = TestClient(app)
        response = client.post(
            "/api/paper-trading/start",
            json={"target_return_rate": 0.02, "period_minutes": 20, "initial_cash": 10_000_000},
        )
        self.assertEqual(response.status_code, 410)
        payload = response.json()
        self.assertEqual(payload["status"], "removed")
        self.assertIn("removed", payload["message"].lower())

    def test_paper_trading_step_endpoint_removed(self) -> None:
        client = TestClient(app)
        response = client.post("/api/paper-trading/step", json={"demo_id": "does-not-exist"})
        self.assertEqual(response.status_code, 410)
        self.assertEqual(response.json()["status"], "removed")

    def test_paper_trading_terminate_endpoint_removed(self) -> None:
        client = TestClient(app)
        response = client.post("/api/paper-trading/terminate/nonexistent-demo-id")
        self.assertEqual(response.status_code, 410)
        payload = response.json()
        self.assertFalse(payload["ok"])
        self.assertEqual(payload["status"], "removed")



if __name__ == "__main__":
    unittest.main()
