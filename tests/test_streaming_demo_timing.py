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

    def test_paper_trading_step_waits_until_next_minute_is_due(self) -> None:
        previous_limit = os.environ.get("SIM_STREAMING_UNIVERSE_LIMIT")
        previous_target = os.environ.get("ONTOLOGY_FILTER1_TARGET_COUNT")
        os.environ["SIM_STREAMING_UNIVERSE_LIMIT"] = str(len(TEST_TICKERS))
        os.environ["ONTOLOGY_FILTER1_TARGET_COUNT"] = str(len(TEST_TICKERS))
        client = TestClient(app)
        try:
            start_response = client.post(
                "/api/paper-trading/start",
                json={
                    "target_return_rate": 0.02,
                    "period_minutes": 20,
                    "initial_cash": 10_000_000,
                    "acceleration_factor": 1,
                },
            )
            self.assertEqual(start_response.status_code, 200)
            demo_id = start_response.json()["demo_id"]

            early_step = client.post("/api/paper-trading/step", json={"demo_id": demo_id}).json()

            self.assertEqual(early_step["status"], "waiting")
            self.assertEqual(early_step["progress"], 0.0)
            self.assertGreater(early_step["retry_after_seconds"], 50)
            self.assertIn("account", early_step)
            self.assertEqual(early_step["account"]["cash"], 10_000_000)
            self.assertEqual(early_step["account"]["account_value"], 10_000_000)

            _streaming_demos[demo_id]._started_at_monotonic -= 60
            due_step = client.post("/api/paper-trading/step", json={"demo_id": demo_id}).json()

            self.assertEqual(due_step["status"], "running")
            self.assertEqual(due_step["step"], 1)
            self.assertEqual(due_step["raw_step"], 15)
            self.assertEqual(due_step["progress"], 5.0)
            self.assertIn("ontology_filter_1", due_step)
            self.assertLessEqual(due_step["ontology_filter_1"]["chart_fetch_count"], len(TEST_TICKERS))
            self.assertGreaterEqual(due_step["universe_count"], due_step["ontology_filter_1"]["chart_fetch_count"])
        finally:
            if previous_limit is None:
                os.environ.pop("SIM_STREAMING_UNIVERSE_LIMIT", None)
            else:
                os.environ["SIM_STREAMING_UNIVERSE_LIMIT"] = previous_limit
            if previous_target is None:
                os.environ.pop("ONTOLOGY_FILTER1_TARGET_COUNT", None)
            else:
                os.environ["ONTOLOGY_FILTER1_TARGET_COUNT"] = previous_target

    def test_paper_trading_mode_starts_buy_sell_loop(self) -> None:
        previous_limit = os.environ.get("SIM_STREAMING_UNIVERSE_LIMIT")
        previous_target = os.environ.get("ONTOLOGY_FILTER1_TARGET_COUNT")
        os.environ["SIM_STREAMING_UNIVERSE_LIMIT"] = str(len(TEST_TICKERS))
        os.environ["ONTOLOGY_FILTER1_TARGET_COUNT"] = str(len(TEST_TICKERS))
        client = TestClient(app)
        try:
            with (
                patch("app.web._start_live_worker"),
                patch("app.web._kis_connection_probe", return_value={"ok": True, "mode": "paper"}),
            ):
                start_response = client.post(
                    "/api/operation-mode/start",
                    json={
                        "mode": "paper_trading",
                        "target_return_rate": 0.02,
                        "period_minutes": 20,
                        "initial_cash": 10_000_000,
                        "acceleration_factor": 120,
                    },
                )
            self.assertEqual(start_response.status_code, 200)
            start_payload = start_response.json()
            demo_id = start_payload["demo_id"]
            self.assertEqual(start_payload["paper_trading_status"], "background_collection_started")
            self.assertIn(demo_id, _streaming_demos)

            _streaming_demos[demo_id]._started_at_monotonic -= 60
            step = client.post("/api/paper-trading/step", json={"demo_id": demo_id}).json()

            self.assertEqual(step["status"], "running")
            self.assertEqual(step["step"], 1)
            self.assertGreaterEqual(step["cumulative_trades"], 1)
            self.assertGreaterEqual(step["trades_in_step"], 1)
        finally:
            if previous_limit is None:
                os.environ.pop("SIM_STREAMING_UNIVERSE_LIMIT", None)
            else:
                os.environ["SIM_STREAMING_UNIVERSE_LIMIT"] = previous_limit
            if previous_target is None:
                os.environ.pop("ONTOLOGY_FILTER1_TARGET_COUNT", None)
            else:
                os.environ["ONTOLOGY_FILTER1_TARGET_COUNT"] = previous_target


if __name__ == "__main__":
    unittest.main()
