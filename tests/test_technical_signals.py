from __future__ import annotations

from app.technical import reason_codes as rc
from app.technical.regime import MarketRegime
from app.technical.signals import (
    BreakoutSignalProvider,
    CompositeTechnicalSignalEngine,
    MeanReversionSignalProvider,
    MomentumTrendSignalProvider,
    SignalDirection,
    TechnicalFeatureSet,
    VolatilityBandSignalProvider,
    VwapVolumeSignalProvider,
)


def trend_up_features(**over) -> TechnicalFeatureSet:
    base = dict(
        symbol="A",
        price=100.0,
        ema_fast=100.2,
        ema_slow=100.0,
        macd=0.5,
        macd_signal=0.1,
        macd_histogram=0.4,
        short_return=0.006,
        momentum_persistence=0.8,
        rsi=62,
        bb_percent_b=0.7,
        bb_bandwidth=0.03,
        vwap=99.8,
        vwap_distance_bps=15,
        vwap_slope=3.0,
        relative_volume=1.6,
        volume_spike_ratio=2.0,
        donchian_high=100.05,
        donchian_low=99.0,
        breakout_strength=0.0,
        false_breakout_risk=0.1,
        atr_pct=0.008,
        realized_volatility=0.004,
        volatility_expansion=0.9,
        liquidity_score=0.9,
        spread_bps=5.0,
        orderbook_imbalance=0.3,
        expected_slippage_bps=2.0,
    )
    base.update(over)
    return TechnicalFeatureSet(**base)


class TestProviderSchema:
    def test_all_providers_output_valid_schema(self):
        f = trend_up_features()
        for provider in (
            MomentumTrendSignalProvider(),
            BreakoutSignalProvider(),
            MeanReversionSignalProvider(),
            VwapVolumeSignalProvider(),
            VolatilityBandSignalProvider(),
        ):
            s = provider.evaluate(f)
            assert -1.0 <= s.score <= 1.0
            assert 0.0 <= s.confidence <= 1.0
            assert s.expected_edge_bps >= 0.0
            assert s.expected_horizon_seconds > 0
            assert isinstance(s.reason_codes, tuple)

    def test_unavailable_when_data_missing(self):
        empty = TechnicalFeatureSet(symbol="A")
        s = MomentumTrendSignalProvider().evaluate(empty)
        assert not s.available
        assert s.reason_codes == (rc.TECHNICAL_SIGNAL_UNAVAILABLE,)
        assert s.score == 0.0

    def test_edge_is_zero_without_volatility_proxy(self):
        f = trend_up_features(realized_volatility=None, atr_pct=None)
        s = MomentumTrendSignalProvider().evaluate(f)
        # No fabricated floor: no vol proxy -> no expected edge.
        assert s.expected_edge_bps == 0.0


class TestProviderDirection:
    def test_momentum_buy_in_uptrend(self):
        s = MomentumTrendSignalProvider().evaluate(trend_up_features())
        assert s.direction == SignalDirection.BUY
        assert s.score > 0
        assert rc.MOMENTUM_CONFIRMED in s.reason_codes

    def test_momentum_sell_in_downtrend(self):
        f = trend_up_features(ema_fast=99.8, macd=-0.5, macd_histogram=-0.4, short_return=-0.006, momentum_persistence=0.2)
        s = MomentumTrendSignalProvider().evaluate(f)
        assert s.direction == SignalDirection.SELL
        assert s.score < 0

    def test_breakout_confirmed(self):
        s = BreakoutSignalProvider().evaluate(trend_up_features(breakout_strength=0.001))
        assert s.direction == SignalDirection.BUY
        assert rc.BREAKOUT_CONFIRMED in s.reason_codes

    def test_breakout_missing_volume_flag(self):
        s = BreakoutSignalProvider().evaluate(trend_up_features(volume_spike_ratio=1.0, breakout_strength=0.001))
        assert rc.VOLUME_CONFIRMATION_MISSING in s.reason_codes

    def test_reversion_buy_when_oversold(self):
        f = trend_up_features(rsi=22, bb_percent_b=0.03)
        s = MeanReversionSignalProvider().evaluate(f)
        assert s.direction == SignalDirection.BUY
        assert rc.MEAN_REVERSION_CANDIDATE in s.reason_codes

    def test_vwap_breakdown_flag(self):
        s = VwapVolumeSignalProvider().evaluate(trend_up_features(vwap_distance_bps=-10))
        assert rc.VWAP_BREAKDOWN in s.reason_codes
        assert s.score < 0

    def test_volatility_band_reduce_on_expansion(self):
        f = trend_up_features(volatility_expansion=2.0, atr_pct=0.05, bb_bandwidth=0.12)
        s = VolatilityBandSignalProvider().evaluate(f)
        assert s.score < 0
        assert s.direction == SignalDirection.REDUCE


class TestCompositeEngine:
    def setup_method(self):
        self.engine = CompositeTechnicalSignalEngine()

    def test_buy_in_clean_uptrend(self):
        c = self.engine.evaluate(trend_up_features())
        assert c.direction == SignalDirection.BUY
        assert not c.blocks_buy
        assert c.expected_edge_bps >= 0.0
        assert c.selected_methodology
        assert rc.VWAP_CONFIRMATION_OK in c.reason_codes

    def test_blocked_in_low_liquidity(self):
        c = self.engine.evaluate(trend_up_features(liquidity_score=0.05))
        assert c.blocks_buy
        assert c.direction == SignalDirection.HOLD
        assert rc.LOW_LIQUIDITY_TECHNICAL_BLOCK in c.reason_codes

    def test_blocked_in_high_volatility(self):
        c = self.engine.evaluate(trend_up_features(realized_volatility=0.05))
        assert c.blocks_buy
        assert rc.HIGH_VOLATILITY_TECHNICAL_BLOCK in c.reason_codes

    def test_buy_requires_vwap_confirmation(self):
        # Strong momentum but price below VWAP -> no BUY (confirmation layer).
        c = self.engine.evaluate(trend_up_features(vwap_distance_bps=-12, vwap_slope=-3.0))
        assert c.direction == SignalDirection.HOLD
        assert not c.blocks_buy  # not a risk block, just unconfirmed
        assert rc.VWAP_BREAKDOWN in c.reason_codes

    def test_no_single_indicator_buys_alone(self):
        # Only RSI present (deep oversold), nothing else -> cannot form BUY.
        f = TechnicalFeatureSet(symbol="A", price=100.0, rsi=10, liquidity_score=0.9)
        c = self.engine.evaluate(f)
        assert c.direction != SignalDirection.BUY

    def test_reversion_blocked_in_downtrend(self):
        # Oversold but strong downtrend -> reversion contributor blocked.
        f = trend_up_features(
            ema_fast=99.8,
            ema_slow=100.0,
            macd=-0.5,
            macd_histogram=-0.5,
            short_return=-0.01,
            vwap_distance_bps=-40,
            rsi=22,
            bb_percent_b=0.03,
            volume_spike_ratio=1.0,
            breakout_strength=-0.02,
        )
        c = self.engine.evaluate(f)
        assert c.direction != SignalDirection.BUY
        assert rc.MEAN_REVERSION_BLOCKED_BY_DOWNTREND in c.reason_codes

    def test_composite_determinism(self):
        f = trend_up_features()
        a = self.engine.evaluate(f)
        b = self.engine.evaluate(f)
        assert a.direction == b.direction
        assert a.score == b.score
        assert a.confidence == b.confidence

    def test_exit_deterioration_codes(self):
        f = trend_up_features(vwap_distance_bps=-10, macd_histogram=-0.3, volatility_expansion=2.0)
        codes = self.engine.evaluate_exit_deterioration(f)
        assert rc.TECHNICAL_EXIT_DETERIORATION in codes
        assert rc.VWAP_BREAKDOWN in codes

    def test_no_exit_deterioration_when_healthy(self):
        codes = self.engine.evaluate_exit_deterioration(trend_up_features())
        assert codes == ()
