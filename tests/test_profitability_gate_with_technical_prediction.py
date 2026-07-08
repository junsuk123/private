from __future__ import annotations

from app.cost import ProfitabilityGate, ProfitabilityInput
from app.technical.prediction import PredictionConfig, TechnicalPredictionEngine
from tests.test_technical_signals import trend_up_features


def _prediction(**cfg):
    engine = TechnicalPredictionEngine(config=PredictionConfig(min_confidence=0.3, **cfg))
    return engine.predict(trend_up_features(symbol="000660"))


class TestGateConsumesTechnicalExit:
    def test_gate_uses_technical_expected_exit_price(self):
        pred = _prediction()
        assert pred.tradable and pred.expected_exit_price is not None
        gate = ProfitabilityGate()
        decision = gate.evaluate(
            ProfitabilityInput(
                symbol="000660",
                action="BUY",
                market="KR",
                venue="KRX",
                instrument_type="domestic_stock",
                entry_price=pred.entry_price,
                expected_exit_price=pred.expected_exit_price,
                quantity=1,
                spread_rate=0.0005,
                liquidity_score=0.9,
                realized_volatility=0.004,
                account_equity_krw=1_000_000.0,
            )
        )
        # The gate's gross return must reflect the technical exit price we fed it.
        expected_gross = pred.expected_exit_price / pred.entry_price - 1.0
        assert abs(decision.gross_expected_return - expected_gross) < 1e-6

    def test_gate_remains_authoritative_rejecting_thin_edge(self):
        # A tiny technical edge does not clear the KR cost floor -> gate rejects,
        # regardless of the technical layer's optimism.
        pred = _prediction()
        gate = ProfitabilityGate()
        decision = gate.evaluate(
            ProfitabilityInput(
                symbol="000660",
                action="BUY",
                market="KR",
                venue="KRX",
                instrument_type="domestic_stock",
                entry_price=pred.entry_price,
                expected_exit_price=pred.expected_exit_price,
                quantity=1,
                spread_rate=0.0005,
                liquidity_score=0.9,
                realized_volatility=0.004,
                account_equity_krw=1_000_000.0,
            )
        )
        assert not decision.allowed
        assert decision.net_expected_return < decision.required_min_net_return


class TestHorizonEdgeBuffer:
    def test_large_horizon_buffer_forces_no_trade(self):
        # The 60s momentum/breakout horizon gets a huge extra required buffer ->
        # the otherwise-tradable prediction becomes NO_TRADE.
        pred = _prediction(horizon_edge_buffer_bps={5: 0.0, 15: 0.0, 30: 0.0, 60: 100_000.0, 300: 0.0})
        assert not pred.tradable

    def test_zero_buffer_keeps_tradable(self):
        pred = _prediction(horizon_edge_buffer_bps={60: 0.0})
        assert pred.tradable
