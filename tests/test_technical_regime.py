from __future__ import annotations

from app.technical.regime import (
    MarketRegime,
    RegimeConfig,
    RegimeInput,
    TechnicalRegimeClassifier,
)


def _clf():
    return TechnicalRegimeClassifier(RegimeConfig())


class TestRiskRegimes:
    def test_low_liquidity_blocks_buy(self):
        r = _clf().classify(RegimeInput(symbol="A", liquidity_score=0.1, price=100))
        assert r.regime == MarketRegime.LOW_LIQUIDITY_RISK
        assert r.blocks_buy is True

    def test_wide_spread_is_low_liquidity(self):
        r = _clf().classify(RegimeInput(symbol="A", spread_bps=120, price=100, liquidity_score=0.9))
        assert r.regime == MarketRegime.LOW_LIQUIDITY_RISK
        assert r.blocks_buy is True

    def test_high_volatility_blocks_buy(self):
        r = _clf().classify(
            RegimeInput(symbol="A", realized_volatility=0.05, liquidity_score=0.9, price=100)
        )
        assert r.regime == MarketRegime.HIGH_VOLATILITY_RISK
        assert r.blocks_buy is True

    def test_high_atr_pct_is_volatility_risk(self):
        r = _clf().classify(RegimeInput(symbol="A", atr_pct=0.05, liquidity_score=0.9, price=100))
        assert r.regime == MarketRegime.HIGH_VOLATILITY_RISK

    def test_risk_gate_precedes_opportunity(self):
        # Even a clean uptrend is overridden by a liquidity risk.
        r = _clf().classify(
            RegimeInput(
                symbol="A",
                price=100,
                ema_fast=101,
                ema_slow=100,
                macd_histogram=0.5,
                vwap_distance_bps=20,
                short_return=0.01,
                rsi=60,
                bb_percent_b=0.7,
                liquidity_score=0.05,
            )
        )
        assert r.regime == MarketRegime.LOW_LIQUIDITY_RISK


class TestDataSufficiency:
    def test_no_trade_when_insufficient(self):
        r = _clf().classify(RegimeInput(symbol="A", price=100, liquidity_score=0.9))
        assert r.regime == MarketRegime.NO_TRADE
        assert r.blocks_buy is True
        assert r.confidence == 0.0
        assert "price" not in r.missing_features  # price was provided
        assert "rsi" in r.missing_features


class TestDirectionalRegimes:
    def test_trend_up(self):
        r = _clf().classify(
            RegimeInput(
                symbol="A",
                price=100,
                ema_fast=100.2,  # +20 bps gap
                ema_slow=100.0,
                macd_histogram=0.4,
                vwap_distance_bps=15,
                short_return=0.006,
                rsi=62,
                bb_percent_b=0.7,
                liquidity_score=0.9,
            )
        )
        assert r.regime == MarketRegime.TREND_UP
        assert 0.0 < r.confidence <= 1.0
        assert not r.blocks_buy

    def test_trend_down(self):
        r = _clf().classify(
            RegimeInput(
                symbol="A",
                price=100,
                ema_fast=99.8,
                ema_slow=100.0,
                macd_histogram=-0.4,
                vwap_distance_bps=-15,
                short_return=-0.006,
                rsi=40,
                bb_percent_b=0.3,
                liquidity_score=0.9,
            )
        )
        assert r.regime == MarketRegime.TREND_DOWN

    def test_range_bound(self):
        r = _clf().classify(
            RegimeInput(
                symbol="A",
                price=100,
                ema_fast=100.01,  # ~1 bps, within range band
                ema_slow=100.0,
                macd_histogram=0.0,
                vwap_distance_bps=1,
                short_return=0.0,
                rsi=50,
                bb_percent_b=0.5,
                liquidity_score=0.9,
            )
        )
        assert r.regime == MarketRegime.RANGE_BOUND

    def test_breakout_candidate_with_volume(self):
        r = _clf().classify(
            RegimeInput(
                symbol="A",
                price=100,
                ema_fast=100.1,
                ema_slow=100.0,
                macd_histogram=0.2,
                vwap_distance_bps=12,
                short_return=0.004,
                rsi=68,
                bb_percent_b=0.97,
                breakout_strength=0.001,  # just above range high
                volume_spike_ratio=2.5,
                liquidity_score=0.9,
            )
        )
        assert r.regime == MarketRegime.BREAKOUT_CANDIDATE
        assert any("volume" in reason for reason in r.reasons)

    def test_breakout_without_volume_is_weaker(self):
        r = _clf().classify(
            RegimeInput(
                symbol="A",
                price=100,
                ema_fast=100.05,
                ema_slow=100.0,
                macd_histogram=0.1,
                vwap_distance_bps=5,
                short_return=0.001,
                rsi=55,
                bb_percent_b=0.6,
                breakout_strength=0.001,
                volume_spike_ratio=1.0,  # no confirmation
                liquidity_score=0.9,
            )
        )
        # Unconfirmed breakout should not dominate as a high-confidence breakout.
        assert r.regime != MarketRegime.BREAKOUT_CANDIDATE or r.confidence < 0.7

    def test_mean_reversion_candidate(self):
        # Oversold dip in a NON-trending context (flat ema/macd, mild pullback).
        # Reversion should win here; it must NOT win in a strong downtrend, which
        # is covered by test_reversion_does_not_beat_strong_downtrend below.
        r = _clf().classify(
            RegimeInput(
                symbol="A",
                price=100,
                ema_fast=100.0,
                ema_slow=100.0,
                macd_histogram=0.0,
                vwap_distance_bps=-6,
                short_return=-0.0004,
                rsi=25,  # oversold
                bb_percent_b=0.02,
                liquidity_score=0.9,
            )
        )
        assert r.regime == MarketRegime.MEAN_REVERSION_CANDIDATE

    def test_reversion_does_not_beat_strong_downtrend(self):
        # Oversold but clearly falling -> TREND_DOWN, so the reversion methodology
        # is correctly blocked from buying against the trend.
        r = _clf().classify(
            RegimeInput(
                symbol="A",
                price=100,
                ema_fast=99.8,
                ema_slow=100.0,
                macd_histogram=-0.5,
                vwap_distance_bps=-40,
                short_return=-0.01,
                rsi=25,
                bb_percent_b=0.02,
                liquidity_score=0.9,
            )
        )
        assert r.regime == MarketRegime.TREND_DOWN


class TestDiagnostics:
    def test_contributions_and_scores_present(self):
        r = _clf().classify(
            RegimeInput(
                symbol="A",
                price=100,
                ema_fast=100.2,
                ema_slow=100.0,
                macd_histogram=0.4,
                vwap_distance_bps=15,
                short_return=0.006,
                rsi=62,
                bb_percent_b=0.7,
                liquidity_score=0.9,
            )
        )
        assert "ema_gap_bps" in r.feature_contributions
        assert r.scores  # non-empty
        d = r.as_dict()
        assert d["regime"] == "TREND_UP"
        assert isinstance(d["reasons"], list)

    def test_determinism(self):
        inp = RegimeInput(
            symbol="A",
            price=100,
            ema_fast=100.2,
            ema_slow=100.0,
            macd_histogram=0.4,
            vwap_distance_bps=15,
            short_return=0.006,
            rsi=62,
            bb_percent_b=0.7,
            liquidity_score=0.9,
        )
        a = _clf().classify(inp)
        b = _clf().classify(inp)
        assert a.regime == b.regime and a.confidence == b.confidence
