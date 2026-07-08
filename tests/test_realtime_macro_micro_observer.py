"""The realtime engine must invoke the advisory macro/micro observer each cycle
without letting it affect trading (failures swallowed, no order impact)."""

from __future__ import annotations

import unittest
from types import SimpleNamespace

from app.schemas.domain import AccountSnapshot
from app.trading.realtime_trading_engine import RealtimeTradingEngine


def _engine(observer):
    return RealtimeTradingEngine(
        decision_engine=SimpleNamespace(store=None),
        coordinator=SimpleNamespace(),
        account_provider=lambda: AccountSnapshot(cash=1000.0, holdings=(), cash_by_currency={"KRW": 1000.0}),
        candidate_symbols_provider=lambda: ("AAA", "BBB"),
        session_open_provider=lambda: True,
        macro_micro_observer=observer,
    )


class MacroMicroObserverTest(unittest.TestCase):
    def test_observer_called_with_context(self):
        calls = []
        engine = _engine(lambda account, held, candidates, dt: calls.append((held, candidates)))
        engine.run_once()
        self.assertEqual(len(calls), 1)
        held, candidates = calls[0]
        self.assertEqual(candidates, ("AAA", "BBB"))

    def test_observer_failure_does_not_break_cycle(self):
        def boom(*args):
            raise RuntimeError("observer down")

        engine = _engine(boom)
        summary = engine.run_once()  # must not raise
        self.assertIsInstance(summary, dict)

    def test_no_observer_is_fine(self):
        engine = RealtimeTradingEngine(
            decision_engine=SimpleNamespace(store=None),
            coordinator=SimpleNamespace(),
            account_provider=lambda: AccountSnapshot(cash=1000.0, holdings=(), cash_by_currency={"KRW": 1000.0}),
            candidate_symbols_provider=lambda: (),
            session_open_provider=lambda: True,
        )
        summary = engine.run_once()
        self.assertIsInstance(summary, dict)


if __name__ == "__main__":
    unittest.main()
