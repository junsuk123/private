"""Phase 7: technical prediction integration into SharedLiveDecisionEngine.

Verifies the technical layer feeds the gate advisorily but can never approve a
buy on its own — ProfitabilityGate + RiskManager stay authoritative.
"""

from __future__ import annotations

import os
import unittest
from datetime import datetime, timezone
from types import SimpleNamespace

from app.schemas.domain import AccountSnapshot
from app.technical.feature_builder import technical_feature_set_from_live_frame
from app.trading.shared_decision_engine import SharedLiveDecisionEngine


def _trend_up_frame(price: float):
    values = {
        "return_30s": 0.003,
        "return_1m": 0.006,
        "return_3m": 0.012,
        "distance_from_vwap": 0.0015,
        "spread_bps": 5.0,
        "orderbook_imbalance": 0.3,
        "bid_depth": 300000.0,
        "ask_depth": 100000.0,
        "depth_ratio": 3.0,
        "liquidity_score": 0.9,
        "realized_volatility_3m": 0.004,
        "max_drop_3m": 0.0,
        "cost_to_volatility_ratio": 0.15,
        "principal_cushion_ratio": 1.0,
        "news_sentiment": 0.0,
        "rsi_14": 62.0,
        "macd_histogram": 0.4,
        "bollinger_percent_b": 0.7,
        "ema_gap_bps": 20.0,
        "donchian_breakout": 0.0,
        "volume_spike_ratio": 2.0,
    }
    return SimpleNamespace(
        symbol="000660",
        mark_price=price,
        feature_schema_hash="unit-schema",
        provenance=SimpleNamespace(source_record_ids=("unit-frame",)),
        as_feature_dict=lambda: dict(values),
    )


class _ThinEdgePredictor:
    def __init__(self, bps: float) -> None:
        self._bps = bps

    def predict(self, frame):
        return SimpleNamespace(
            probability_success=0.7,
            expected_net_return_bps=self._bps,
            uncertainty_score=0.2,
            approved=True,
            reason_codes=(),
            model_artifact_id="unit",
            feature_schema_hash=frame.feature_schema_hash,
            provider="trained_model",
            is_fallback=False,
        )


class _BuyStore:
    def __init__(self, price: float) -> None:
        now = datetime.now(timezone.utc)
        self._tick = SimpleNamespace(price=price, received_at=now, exchange_timestamp=now, sequence_key="t")

    def latest_tick(self, symbol):
        return self._tick

    def latest_orderbook(self, symbol):
        return None


def _graph(support=("InformedOrderFlowImbalance", "ForeignInstitutionJointBuying")):
    class _G:
        def matching(self, subject=None, predicate=None, obj=None):
            if predicate == "supportsSignal":
                return [SimpleNamespace(object=o) for o in support]
            return []

    return _G()


def _engine(price: float, bps: float) -> SharedLiveDecisionEngine:
    engine = SharedLiveDecisionEngine(_BuyStore(price=price), predictor=_ThinEdgePredictor(bps))
    engine.feature_builder = SimpleNamespace(build=lambda symbol, decision_time=None: _trend_up_frame(price))
    return engine


def _run(engine):
    account = AccountSnapshot(cash=1_000_000.0, holdings=(), cash_by_currency={"KRW": 1_000_000.0})
    old = os.environ.get("REALTIME_MODEL_AUXILIARY_ONLY")
    os.environ["REALTIME_MODEL_AUXILIARY_ONLY"] = "true"
    try:
        return engine.evaluate_buy("000660", account, suggested_weight=0.01, ontology_graph=_graph())
    finally:
        if old is None:
            os.environ.pop("REALTIME_MODEL_AUXILIARY_ONLY", None)
        else:
            os.environ["REALTIME_MODEL_AUXILIARY_ONLY"] = old


class FrameMappingTest(unittest.TestCase):
    def test_frame_maps_to_feature_set(self):
        fs = technical_feature_set_from_live_frame(_trend_up_frame(10_000.0), "000660")
        self.assertEqual(fs.symbol, "000660")
        self.assertEqual(fs.price, 10_000.0)
        self.assertEqual(fs.rsi, 62.0)
        self.assertEqual(fs.macd_histogram, 0.4)
        self.assertEqual(fs.return_30s, 0.003)
        self.assertAlmostEqual(fs.vwap_distance_bps, 15.0, places=3)
        self.assertIsNotNone(fs.ema_fast)
        self.assertGreater(fs.ema_fast, fs.ema_slow)  # positive ema_gap_bps

    def test_completed_minute_context_overrides_sparse_tick_neutral_defaults(self):
        frame = _trend_up_frame(10_000.0)
        fast = frame.as_feature_dict()
        fast.update({"ema_gap_bps": 0.0, "rsi_14": 50.0, "volume_spike_ratio": 1.0})
        slow = {
            "slow_technical:ema_fast": 10_020.0,
            "slow_technical:ema_slow": 10_000.0,
            "slow_technical:rsi": 64.0,
            "slow_technical:volume_spike_ratio": 2.4,
            "slow_technical:realized_volatility": 0.003,
            "slow_technical:bar_count": 80.0,
        }
        frame.as_feature_dict = lambda: dict(fast)
        frame.as_context_dict = lambda: {**fast, **slow}

        fs = technical_feature_set_from_live_frame(frame, "INTC")

        self.assertEqual(fs.ema_fast, 10_020.0)
        self.assertEqual(fs.rsi, 64.0)
        self.assertEqual(fs.volume_spike_ratio, 2.4)
        self.assertEqual(fs.realized_volatility, 0.003)


class TechnicalIntegrationTest(unittest.TestCase):
    def test_technical_prediction_in_diagnostics(self):
        engine = _engine(price=10_000.0, bps=200.0)
        self.assertTrue(engine._technical_enabled)
        result = _run(engine)
        self.assertIn("technical_prediction", result.diagnostics)
        self.assertIsNotNone(result.diagnostics["technical_prediction"])

    def test_technical_cannot_approve_net_negative_buy(self):
        # Thin model edge (net-negative after KR cost + floor). A healthy technical
        # BUY must NOT rescue it — the ProfitabilityGate still rejects.
        engine = _engine(price=10_000.0, bps=30.0)
        result = _run(engine)
        self.assertFalse(result.approved, result.reason_codes)
        self.assertIn("PROFITABILITY_GATE_REJECTED", result.reason_codes)

    def test_technical_does_not_inflate_expected_exit(self):
        # With a thin model edge, the gate's expected net must not exceed the
        # model's honest estimate because of the technical layer (min() rule).
        engine = _engine(price=10_000.0, bps=30.0)
        result = _run(engine)
        pd = result.diagnostics.get("profitability_decision", {})
        # gross expected return should reflect <= 30 bps, i.e. <= 0.003.
        self.assertLessEqual(pd.get("gross_expected_return", 1.0), 0.0031)

    def test_disabled_technical_yields_none(self):
        engine = _engine(price=10_000.0, bps=200.0)
        engine._technical_enabled = False
        result = _run(engine)
        self.assertIsNone(result.diagnostics.get("technical_prediction"))


if __name__ == "__main__":
    unittest.main()
