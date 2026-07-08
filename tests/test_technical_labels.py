from __future__ import annotations

from datetime import datetime, timedelta, timezone

from app.features.schemas import OHLCVBar
from app.technical.feature_builder import build_technical_feature_set
from app.technical.labels import LabelBuilder, LabelConfig, is_synthetic_source


def _bars(prices, vols=None):
    start = datetime(2026, 1, 1, 9, 0, tzinfo=timezone.utc)
    return [
        OHLCVBar("A", start + timedelta(minutes=i), p, p * 1.001, p * 0.999, p, (vols[i] if vols else 1000.0))
        for i, p in enumerate(prices)
    ]


class TestFeatureBuilder:
    def test_builds_populated_feature_set(self):
        prices = [100 + i * 0.1 for i in range(40)]
        fs = build_technical_feature_set(_bars(prices), symbol="A", liquidity_score=0.8)
        assert fs.symbol == "A"
        assert fs.price is not None
        assert fs.ema_fast is not None and fs.ema_slow is not None
        assert fs.rsi is not None
        assert fs.vwap is not None
        assert fs.macd_histogram is not None
        assert fs.liquidity_score == 0.8

    def test_short_data_yields_none_not_crash(self):
        fs = build_technical_feature_set(_bars([100, 101]), symbol="A")
        assert fs.price == 101
        assert fs.rsi is None  # insufficient
        assert fs.macd_histogram is None

    def test_orderbook_fields_extracted(self):
        class OB:
            best_bid = 99.9
            best_ask = 100.1
            imbalance = 0.2
            spread_bps = 20.0

        fs = build_technical_feature_set(_bars([100 + i for i in range(30)]), symbol="A", orderbook=OB())
        assert fs.spread_bps == 20.0
        assert fs.orderbook_imbalance == 0.2


class TestSyntheticGuard:
    def test_synthetic_sources_detected(self):
        assert is_synthetic_source("synthetic-feed")
        assert is_synthetic_source("sample-indicator:005930")
        assert is_synthetic_source("reference:AAPL")
        assert not is_synthetic_source("kis_realtime_websocket")

    def test_builder_rejects_synthetic_source(self):
        labels = LabelBuilder().build(
            symbol="A", entry_price=100.0, future_path=[(5, 101)], source="synthetic"
        )
        assert labels is None


class TestLabels:
    def _path(self):
        # seconds-after-decision, price. Rises then dips.
        return [(5, 100.5), (15, 101.0), (30, 100.2), (60, 101.5), (120, 99.0), (300, 102.0)]

    def test_future_returns(self):
        labels = LabelBuilder().build(symbol="A", entry_price=100.0, future_path=self._path())
        assert labels is not None
        assert abs(labels.future_return_5s - 0.005) < 1e-9
        assert abs(labels.future_return_60s - 0.015) < 1e-9
        assert abs(labels.future_return_5m - 0.02) < 1e-9

    def test_missing_horizon_is_none(self):
        labels = LabelBuilder().build(symbol="A", entry_price=100.0, future_path=[(5, 100.5)])
        assert labels.future_return_5s is not None
        assert labels.future_return_60s is None
        assert labels.future_return_5m is None

    def test_mfe_mae(self):
        labels = LabelBuilder().build(symbol="A", entry_price=100.0, future_path=self._path())
        # within max horizon 300s: max 102.0 (+200bps), min 99.0 (-100bps)
        assert abs(labels.max_favorable_excursion_bps - 200.0) < 1e-6
        assert abs(labels.max_adverse_excursion_bps + 100.0) < 1e-6

    def test_tp_before_stop(self):
        cfg = LabelConfig(take_profit_bps=40.0, stop_loss_bps=40.0)
        # rises to +50bps at 5s before any -40bps -> hit_tp = 1
        labels = LabelBuilder(config=cfg).build(
            symbol="A", entry_price=100.0, future_path=[(5, 100.5), (10, 99.0)]
        )
        assert labels.hit_take_profit_before_stop_label == 1

    def test_stop_before_tp(self):
        cfg = LabelConfig(take_profit_bps=40.0, stop_loss_bps=40.0)
        labels = LabelBuilder(config=cfg).build(
            symbol="A", entry_price=100.0, future_path=[(5, 99.5), (10, 101.0)]
        )
        assert labels.hit_take_profit_before_stop_label == 0

    def test_net_label_without_cost_engine(self):
        # +15bps at 60s, cost 5bps spread -> net +10bps -> label 1
        labels = LabelBuilder().build(
            symbol="A", entry_price=100.0, future_path=[(60, 100.15)], spread_bps=5.0
        )
        assert labels.net_profitable_after_cost_label == 1

    def test_net_label_negative_after_cost(self):
        # +3bps at 60s, cost 10bps -> net negative -> label 0
        labels = LabelBuilder().build(
            symbol="A", entry_price=100.0, future_path=[(60, 100.03)], spread_bps=10.0
        )
        assert labels.net_profitable_after_cost_label == 0

    def test_net_label_uses_cost_engine_when_provided(self):
        from app.cost.trading_cost_engine import TradingCostEngine

        engine = TradingCostEngine()
        labels = LabelBuilder(cost_engine=engine).build(
            symbol="005930", entry_price=10000.0, future_path=[(60, 10050.0)], quantity=1
        )
        assert labels is not None
        assert labels.net_profitable_after_cost_label in (0, 1)
        assert labels.metadata["cost_rate_assumed"] is not None

    def test_no_lookahead_only_uses_provided_path(self):
        # If the path stops at 30s, horizons beyond 30s must be None (no invention).
        labels = LabelBuilder().build(
            symbol="A", entry_price=100.0, future_path=[(5, 100.1), (15, 100.2), (30, 100.3)]
        )
        assert labels.future_return_30s is not None
        assert labels.future_return_60s is None
        assert labels.future_return_5m is None
