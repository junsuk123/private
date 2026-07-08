"""Phase 6: SharedLiveDecisionEngine.consume_bundle integration.

Proves the macro/micro layer is advisory and cannot bypass the authoritative
gates: macro BLOCK_BUY rejects before any gate/broker call; a net-negative BUY
is still rejected by the ProfitabilityGate; SELL/REDUCE is processed before BUY.
"""

from __future__ import annotations

import os
import unittest
from datetime import datetime, timezone
from types import SimpleNamespace

from app.graph.global_trade_arbiter import RankedTradeIntent
from app.graph.macro_micro_common import IntentType
from app.schemas.domain import AccountSnapshot, Holding
from app.trading.shared_decision_engine import SharedLiveDecisionEngine
from tests.test_shared_decision_engine_technical_prediction import (
    _BuyStore,
    _ThinEdgePredictor,
    _graph,
    _trend_up_frame,
)


def _now():
    return datetime(2026, 7, 9, tzinfo=timezone.utc)


def _intent(symbol, side, *, rank=0, net=15.0):
    return RankedTradeIntent(
        intent_type=IntentType.BUY if side == "BUY" else (IntentType.SELL if side == "SELL" else IntentType.REDUCE),
        symbol=symbol, side=side, rank=rank, score=100.0 - rank,
        expected_entry_price=100.0, expected_exit_price=101.0, expected_net_return_bps=net,
        downside_risk_bps=40.0, macro_regime="TREND_UP", micro_regime="MOMENTUM_CANDIDATE",
        selected_strategy="momentum", confidence=0.7, reason_codes=(), explanation_paths=(),
    )


def _bundle(intents, *, blocks_buy):
    macro = SimpleNamespace(blocks_buy=blocks_buy, as_dict=lambda: {"blocks_buy": blocks_buy, "market_regime": "TREND_UP"})
    return SimpleNamespace(macro_result=macro, ranked_trade_intents=tuple(intents))


def _engine(price=10_000.0, bps=30.0):
    engine = SharedLiveDecisionEngine(_BuyStore(price=price), predictor=_ThinEdgePredictor(bps))
    engine.feature_builder = SimpleNamespace(build=lambda symbol, decision_time=None: _trend_up_frame(price))
    return engine


class MacroMicroIntegrationTest(unittest.TestCase):
    def setUp(self):
        self._old = os.environ.get("REALTIME_MODEL_AUXILIARY_ONLY")
        os.environ["REALTIME_MODEL_AUXILIARY_ONLY"] = "true"

    def tearDown(self):
        if self._old is None:
            os.environ.pop("REALTIME_MODEL_AUXILIARY_ONLY", None)
        else:
            os.environ["REALTIME_MODEL_AUXILIARY_ONLY"] = self._old

    def _account(self, holdings=()):
        return AccountSnapshot(cash=1_000_000.0, holdings=tuple(holdings), cash_by_currency={"KRW": 1_000_000.0})

    def test_macro_block_buy_rejects_before_gate(self):
        engine = _engine()
        bundle = _bundle([_intent("000660", "BUY")], blocks_buy=True)
        results = engine.consume_bundle(bundle, self._account())
        self.assertEqual(len(results), 1)
        self.assertFalse(results[0].approved)
        self.assertIn("MACRO_BLOCK_BUY", results[0].reason_codes)
        self.assertIsNone(results[0].final_order)

    def test_net_negative_buy_rejected_by_profitability_gate(self):
        # Macro allows; but the thin model edge (30bps) is net-negative after KR
        # cost + floor. The ProfitabilityGate must reject regardless of the
        # advisory macro/micro intent.
        engine = _engine(bps=30.0)
        bundle = _bundle([_intent("000660", "BUY")], blocks_buy=False)
        results = engine.consume_bundle(bundle, self._account(), ontology_graph=_graph())
        self.assertFalse(results[0].approved)
        self.assertIsNone(results[0].final_order)  # no order regardless of advisory intent
        self.assertIn("PROFITABILITY_GATE_REJECTED", results[0].reason_codes)
        # Advisory context attached but non-authoritative.
        self.assertIn("macro_micro", results[0].diagnostics)

    def test_sell_reduce_processed_before_buy(self):
        engine = _engine()
        holding = Holding(ticker="005930", market="KR", company_name="S", sector="Tech",
                          quantity=10, average_price=9000.0, last_price=10_000.0)
        bundle = _bundle(
            [_intent("005930", "SELL", rank=0), _intent("000660", "BUY", rank=1)],
            blocks_buy=False,
        )
        results = engine.consume_bundle(bundle, self._account(holdings=(holding,)))
        # First result corresponds to the SELL intent (processed first).
        self.assertEqual(results[0].symbol, "005930")
        self.assertEqual(results[-1].symbol, "000660")

    def test_advisory_intent_has_no_broker_authority(self):
        intent = _intent("A", "BUY")
        for attr in ("final_order", "submit", "broker", "place_order"):
            self.assertFalse(hasattr(intent, attr))


if __name__ == "__main__":
    unittest.main()
