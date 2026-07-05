from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from app.execution.execution_quality import ExecutionQualityEngine, ExecutionQualityInput
from app.storage.execution_quality_store import ExecutionQualityStore


class ExecutionQualityTest(unittest.TestCase):
    def setUp(self) -> None:
        self.engine = ExecutionQualityEngine()

    def _req(self, **kw) -> ExecutionQualityInput:
        base = dict(
            symbol="005930",
            strategy_family="live_short_horizon",
            decision_reference_price=10_000.0,
            gross_expected_return=0.02,
            net_expected_return=0.015,
            required_min_net_return=0.008,
            best_bid=9_995.0,
            best_ask=10_005.0,
            bid_depth=100_000.0,
            ask_depth=100_000.0,
        )
        base.update(kw)
        return ExecutionQualityInput(**base)

    def test_clean_book_passes(self) -> None:
        a = self.engine.assess(self._req())
        self.assertTrue(a.allowed, a.reject_reason)
        self.assertGreater(a.fill_probability, 0.0)

    def test_wide_spread_consumes_alpha_rejected(self) -> None:
        a = self.engine.assess(self._req(best_bid=9_800.0, best_ask=10_200.0, gross_expected_return=0.01))
        self.assertFalse(a.allowed)
        self.assertEqual(a.reject_reason, "EXEC_SPREAD_CONSUMES_ALPHA")

    def test_execution_adjusted_net_below_min_rejected(self) -> None:
        # A thin net edge with meaningful spread slippage drops below the required min.
        a = self.engine.assess(
            self._req(best_bid=9_970.0, best_ask=10_030.0, gross_expected_return=0.02, net_expected_return=0.0085, required_min_net_return=0.008)
        )
        self.assertFalse(a.allowed)
        self.assertIn(a.reject_reason, {"EXEC_ADJUSTED_NET_BELOW_MIN", "EXEC_EXPECTED_SLIPPAGE_TOO_HIGH", "EXEC_SPREAD_CONSUMES_ALPHA"})

    def test_orderbook_pressure_sign(self) -> None:
        bid_heavy = self.engine.assess(self._req(bid_depth=200_000.0, ask_depth=50_000.0))
        self.assertGreater(bid_heavy.orderbook_pressure, 0.0)

    def test_realized_slippage_recorded_and_penalizes(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            store = ExecutionQualityStore(Path(tmp) / "eq.jsonl", window=10)
            engine = ExecutionQualityEngine(store=store)
            # Record several bad fills for the symbol.
            for _ in range(5):
                engine.record_fill(symbol="BADX", strategy_family="live_short_horizon", decision_reference_price=100.0, fill_price=101.0)
            avg = store.recent_average(symbol="BADX", strategy_family="live_short_horizon")
            self.assertIsNotNone(avg)
            self.assertGreater(avg, 0.006)
            a = engine.assess(
                ExecutionQualityInput(
                    symbol="BADX",
                    strategy_family="live_short_horizon",
                    decision_reference_price=100.0,
                    gross_expected_return=0.03,
                    net_expected_return=0.02,
                    required_min_net_return=0.008,
                    best_bid=99.95,
                    best_ask=100.05,
                    bid_depth=1_000.0,
                    ask_depth=1_000.0,
                )
            )
            self.assertFalse(a.allowed)
            self.assertEqual(a.reject_reason, "EXEC_SYMBOL_SLIPPAGE_HISTORY_BAD")

    def test_store_persists_and_reloads(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "eq.jsonl"
            store = ExecutionQualityStore(path)
            store.record(symbol="A", strategy_family="f", realized_slippage_rate=0.002)
            reloaded = ExecutionQualityStore(path)
            self.assertAlmostEqual(reloaded.recent_average(symbol="A", strategy_family="f"), 0.002)


if __name__ == "__main__":
    unittest.main()
