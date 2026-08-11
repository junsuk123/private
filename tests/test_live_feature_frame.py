from __future__ import annotations

import sys
import tempfile
import unittest
from datetime import datetime, timedelta, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from app.data.realtime_store import RealtimeMarketDataStore
from app.data.realtime_types import KIS_REALTIME_SOURCE, OrderbookLevel, RealtimeOrderbookSnapshot, RealtimeTradeTick
from app.features.feature_schema import LIVE_SHORT_HORIZON_SCHEMA
from app.features.live_feature_frame import LiveFeatureFrameBuilder


class FeatureSchemaIdentityTest(unittest.TestCase):
    """The model vector must carry market STATE, never instrument IDENTITY.

    A per-symbol-constant column pins the top of a pooled cross-sectional ranking
    to one instrument: on 2026-08-05 raw ``ask_depth``/``bid_depth`` put 44 of the
    top 53 predictions on a single liquidity-provider-quoted ETF, giving a top-k
    net return of -51bps and precision@k 0.148 against a 0.395 base rate, while
    deciles 2-10 ranked monotonically. Removing them moved precision@k to 0.444 and
    top-k net to +18bps.
    """

    # Raw sizes and raw price levels. Each had a between-symbol variance ratio
    # above 0.75 (1.0 = pure identity) when measured over the KR training rows.
    IDENTITY_FEATURES = (
        "bid_depth",
        "ask_depth",
        "liquidity_score",
        "realized_volatility_3m",
        "box_high",
        "box_low",
        "box_mid",
        "box_previous_close",
    )

    # The scale-free columns that carry the same information. If one of these is
    # ever dropped, the identity column it replaced must not come back instead.
    SCALE_FREE_COUNTERPARTS = (
        "depth_ratio",
        "orderbook_imbalance",
        "box_position",
        "box_width_pct",
        "breakout_distance_bps",
    )

    def test_model_vector_excludes_instrument_identity_features(self) -> None:
        present = sorted(
            set(self.IDENTITY_FEATURES) & set(LIVE_SHORT_HORIZON_SCHEMA.feature_names)
        )
        self.assertEqual(
            present,
            [],
            "identity features are back in the model vector; see this class's docstring "
            "-- put raw levels in the strategy layer, which compares a symbol to itself",
        )

    def test_scale_free_counterparts_are_retained(self) -> None:
        for name in self.SCALE_FREE_COUNTERPARTS:
            self.assertIn(name, LIVE_SHORT_HORIZON_SCHEMA.feature_names, name)

    def test_schema_is_internally_consistent(self) -> None:
        names = LIVE_SHORT_HORIZON_SCHEMA.feature_names
        self.assertEqual(len(names), len(set(names)), "duplicate feature name")
        self.assertEqual(len(names), len(LIVE_SHORT_HORIZON_SCHEMA.dtypes))

    def test_dropping_identity_features_changed_the_schema_hash(self) -> None:
        # A v4 artifact scored with a v5 vector would read the wrong column for
        # every weight, so the hash MUST differ from the pre-removal schema.
        self.assertNotEqual(LIVE_SHORT_HORIZON_SCHEMA.schema_hash, "3bbec7413bcb24f89d55d995")


class LiveFeatureFrameTest(unittest.TestCase):
    def test_feature_frame_has_schema_hash_and_provenance(self) -> None:
        now = datetime(2026, 6, 29, 9, 30, tzinfo=timezone.utc)
        with tempfile.TemporaryDirectory() as tmp:
            store = RealtimeMarketDataStore(Path(tmp) / "rt.sqlite3")
            _seed(store, now)
            frame = LiveFeatureFrameBuilder(store, journal_path=Path(tmp) / "features.jsonl").build("005930", decision_time=now)

        self.assertEqual(frame.feature_schema_hash, LIVE_SHORT_HORIZON_SCHEMA.schema_hash)
        self.assertEqual(len(frame.values), len(LIVE_SHORT_HORIZON_SCHEMA.feature_names))
        self.assertGreater(len(frame.provenance.source_record_ids), 0)
        values = frame.as_feature_dict()
        self.assertIn("return_1s", values)
        self.assertIn("aggressor_imbalance_5s", values)
        # Two distinct recent prints plus a contemporaneous book now constitute
        # the minimum causal overseas window; expected-move volatility can come
        # from completed minute history when a third print is absent.
        self.assertEqual(values["second_data_ready"], 1.0)

    def test_feature_frame_computes_true_second_level_microstructure(self) -> None:
        now = datetime(2026, 6, 29, 9, 30, tzinfo=timezone.utc)
        with tempfile.TemporaryDirectory() as tmp:
            store = RealtimeMarketDataStore(Path(tmp) / "rt.sqlite3")
            ticks = tuple(
                RealtimeTradeTick(
                    symbol="005930",
                    exchange_timestamp=now - timedelta(seconds=4 - index),
                    received_at=now - timedelta(seconds=4 - index),
                    source=KIS_REALTIME_SOURCE,
                    price=70000 + index * 10,
                    volume=100 + index,
                    trade_direction="BUY" if index >= 2 else "SELL",
                    sequence_key=f"tick:{index}",
                )
                for index in range(5)
            )
            books = tuple(
                RealtimeOrderbookSnapshot(
                    symbol="005930",
                    exchange_timestamp=now - timedelta(seconds=4 - index),
                    received_at=now - timedelta(seconds=4 - index),
                    source=KIS_REALTIME_SOURCE,
                    levels=(OrderbookLevel(70000 + index * 10, 1000 + index * 20, 70020 + index * 10, 900),),
                    sequence_key=f"book:{index}",
                )
                for index in range(5)
            )
            store.save_ticks(ticks)
            store.save_orderbooks(books)
            frame = LiveFeatureFrameBuilder(
                store,
                journal_path=Path(tmp) / "features.jsonl",
            ).build("005930", decision_time=now)

        values = frame.as_context_dict()
        self.assertEqual(values["second_data_ready"], 1.0)
        self.assertEqual(values["tick_count_5s"], 5.0)
        self.assertGreater(values["return_5s"], 0.0)
        self.assertGreater(values["aggressor_imbalance_5s"], 0.0)

    def test_second_window_uses_causal_receive_time_when_exchange_feed_is_delayed(self) -> None:
        now = datetime(2026, 6, 29, 9, 30, tzinfo=timezone.utc)
        with tempfile.TemporaryDirectory() as tmp:
            store = RealtimeMarketDataStore(Path(tmp) / "rt.sqlite3")
            ticks = tuple(
                RealtimeTradeTick(
                    symbol="INTC",
                    exchange_timestamp=now - timedelta(seconds=30 - index),
                    received_at=now - timedelta(seconds=3 - index * 2),
                    source=KIS_REALTIME_SOURCE,
                    price=100.0 + index * 0.05,
                    volume=100,
                    trade_direction="BUY",
                    sequence_key=f"delayed:{index}",
                    latency_ms=27_000.0,
                )
                for index in range(2)
            )
            books = tuple(
                RealtimeOrderbookSnapshot(
                    symbol="INTC",
                    exchange_timestamp=now - timedelta(seconds=30 - index),
                    received_at=now - timedelta(seconds=3 - index * 2),
                    source=KIS_REALTIME_SOURCE,
                    levels=(OrderbookLevel(99.9, 1000, 100.1, 900),),
                    sequence_key=f"delayed-book:{index}",
                    latency_ms=27_000.0,
                )
                for index in range(2)
            )
            store.save_ticks(ticks)
            store.save_orderbooks(books)
            frame = LiveFeatureFrameBuilder(
                store, journal_path=Path(tmp) / "features.jsonl"
            ).build("INTC", decision_time=now)

        values = frame.as_context_dict()
        self.assertEqual(values["second_data_ready"], 1.0)
        self.assertGreater(values["return_5s"], 0.0)
        self.assertEqual(values["source_latency_ms_10s"], 27_000.0)

    def test_overseas_inside_spread_prints_get_causal_quote_direction(self) -> None:
        now = datetime(2026, 6, 29, 9, 30, tzinfo=timezone.utc)
        with tempfile.TemporaryDirectory() as tmp:
            store = RealtimeMarketDataStore(Path(tmp) / "rt.sqlite3")
            ticks = tuple(
                RealtimeTradeTick(
                    symbol="INTC",
                    exchange_timestamp=now - timedelta(seconds=3 - index * 2),
                    received_at=now - timedelta(seconds=3 - index * 2),
                    source=KIS_REALTIME_SOURCE,
                    price=100.06 + index * 0.02,
                    volume=100,
                    trade_direction=None,
                    sequence_key=f"us-tick:{index}",
                )
                for index in range(2)
            )
            books = tuple(
                RealtimeOrderbookSnapshot(
                    symbol="INTC",
                    exchange_timestamp=now - timedelta(seconds=4 - index * 2),
                    received_at=now - timedelta(seconds=4 - index * 2),
                    source=KIS_REALTIME_SOURCE,
                    levels=(OrderbookLevel(100.00, 1000, 100.10, 900),),
                    sequence_key=f"us-book:{index}",
                )
                for index in range(2)
            )
            store.save_ticks(ticks)
            store.save_orderbooks(books)
            frame = LiveFeatureFrameBuilder(
                store, journal_path=Path(tmp) / "features.jsonl"
            ).build("INTC", decision_time=now)

        values = frame.as_feature_dict()
        self.assertEqual(values["second_data_ready"], 1.0)
        self.assertEqual(values["tick_count_5s"], 2.0)
        self.assertGreater(values["aggressor_imbalance_5s"], 0.0)

    def test_feature_frame_can_use_kis_orderbook_when_trade_ticks_are_sparse(self) -> None:
        now = datetime(2026, 6, 29, 9, 30, tzinfo=timezone.utc)
        with tempfile.TemporaryDirectory() as tmp:
            store = RealtimeMarketDataStore(Path(tmp) / "rt.sqlite3")
            _seed_orderbooks_only(store, now)
            frame = LiveFeatureFrameBuilder(store, journal_path=Path(tmp) / "features.jsonl").build("005930", decision_time=now)

        self.assertEqual(frame.feature_schema_hash, LIVE_SHORT_HORIZON_SCHEMA.schema_hash)
        self.assertGreater(len(frame.provenance.source_record_ids), 0)


def _seed(store: RealtimeMarketDataStore, now: datetime) -> None:
    ticks = tuple(
        RealtimeTradeTick(
            symbol="005930",
            exchange_timestamp=now - timedelta(seconds=120 - i * 10),
            received_at=now - timedelta(seconds=120 - i * 10),
            source=KIS_REALTIME_SOURCE,
            price=70000 + i * 10,
            volume=100 + i,
            sequence_key=f"tick:{i}",
        )
        for i in range(13)
    )
    store.save_ticks(ticks)
    store.save_orderbooks(
        (
            RealtimeOrderbookSnapshot(
                symbol="005930",
                exchange_timestamp=now,
                received_at=now,
                source=KIS_REALTIME_SOURCE,
                levels=(OrderbookLevel(70100, 1000, 70150, 800),),
                sequence_key="book:1",
            ),
        )
    )


def _seed_orderbooks_only(store: RealtimeMarketDataStore, now: datetime) -> None:
    books = tuple(
        RealtimeOrderbookSnapshot(
            symbol="005930",
            exchange_timestamp=now - timedelta(seconds=120 - i * 10),
            received_at=now - timedelta(seconds=120 - i * 10),
            source=KIS_REALTIME_SOURCE,
            levels=(OrderbookLevel(70000 + i * 10, 1000 + i, 70100 + i * 10, 900 + i),),
            sequence_key=f"book:{i}",
        )
        for i in range(13)
    )
    store.save_orderbooks(books)


if __name__ == "__main__":
    unittest.main()
