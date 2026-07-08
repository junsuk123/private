from __future__ import annotations

from datetime import datetime, timedelta, timezone
from types import SimpleNamespace

from app.features.macro_feature_frame import (
    build_macro_feature_frame,
    macro_feature_frame_from_store,
)


def _now():
    return datetime(2026, 7, 9, 3, 0, tzinfo=timezone.utc)


class TestBuilder:
    def test_rising_universe_positive_trend_full_breadth(self):
        closes = {"A": [100 + i for i in range(30)], "B": [50 + i * 0.5 for i in range(30)]}
        f = build_macro_feature_frame(closes, timestamp=_now())
        assert f.index_trend is not None and f.index_trend > 0
        assert f.market_breadth == 1.0
        assert f.symbol_count == 2

    def test_mixed_breadth(self):
        closes = {"UP": [1 + i for i in range(30)], "DOWN": [30 - i for i in range(30)]}
        f = build_macro_feature_frame(closes, timestamp=_now())
        assert f.market_breadth == 0.5

    def test_sector_aggregation(self):
        closes = {"A": [100 + i for i in range(30)], "B": [100 + i for i in range(30)], "C": [30 - i for i in range(29)] + [1]}
        f = build_macro_feature_frame(closes, timestamp=_now(), sector_of={"A": "tech", "B": "tech", "C": "energy"})
        assert "tech" in f.sector_snapshots and f.sector_snapshots["tech"]["strength"] > 0
        assert "energy" in f.sector_snapshots and f.sector_snapshots["energy"]["strength"] < 0

    def test_unknown_sector_ignored(self):
        f = build_macro_feature_frame({"A": [1 + i for i in range(30)]}, timestamp=_now(), sector_of={"A": "Unknown"})
        assert f.sector_snapshots == {}

    def test_insufficient_data_yields_none(self):
        f = build_macro_feature_frame({"A": [100.0, 101.0]}, timestamp=_now())
        assert f.index_trend is None
        assert f.market_breadth is None
        assert f.symbol_count == 0

    def test_as_macro_kwargs_shape(self):
        f = build_macro_feature_frame({"A": [100 + i for i in range(30)]}, timestamp=_now())
        kw = f.as_macro_kwargs()
        assert "index_snapshots" in kw and kw["index_snapshots"]["COMPOSITE"]["trend"] is not None
        assert "market_breadth" in kw and "market_volatility" in kw


class TestStoreAdapter:
    def _store(self, price_paths):
        start = _now() - timedelta(minutes=30)

        class _Store:
            def recent_ticks(self, symbol, since):
                prices = price_paths.get(symbol, [])
                return [
                    SimpleNamespace(price=p, volume=100, exchange_timestamp=start + timedelta(seconds=i))
                    for i, p in enumerate(prices)
                ]

        return _Store()

    def test_from_store_rising(self):
        store = self._store({"A": [100 + i for i in range(30)], "B": [200 + i for i in range(30)]})
        f = macro_feature_frame_from_store(store, ["A", "B"], now=_now())
        assert f.index_trend is not None and f.index_trend > 0
        assert f.total_trading_value is not None and f.total_trading_value > 0

    def test_from_store_no_ticks_all_none(self):
        store = self._store({})
        f = macro_feature_frame_from_store(store, ["A", "B"], now=_now())
        assert f.index_trend is None
        assert f.symbol_count == 0

    def test_from_store_resilient_to_symbol_error(self):
        class _Store:
            def recent_ticks(self, symbol, since):
                if symbol == "BAD":
                    raise RuntimeError("db error")
                return [SimpleNamespace(price=100 + i, volume=10, exchange_timestamp=_now()) for i in range(30)]

        f = macro_feature_frame_from_store(_Store(), ["GOOD", "BAD"], now=_now())
        assert "GOOD" in f.per_symbol_return and "BAD" not in f.per_symbol_return
