"""Causality, parity and family-scoring behaviour.

The property that matters: a value computed at ``as_of`` must never change when
later data arrives. Everything else here follows from that.
"""

from __future__ import annotations

import sys
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from app.features.schemas import OHLCVBar
from app.technical.causal_bars import (
    CausalBarSet,
    completed_bars,
    floor_to_minute,
    forming_bar,
    load_causal_bars,
)
from app.technical.indicator_families import (
    COMPACT_MODEL_FEATURES,
    FAMILY_NAMES,
    MEAN_REVERSION,
    TREND,
    build_families,
    compact_model_features,
    confirmation,
)

START = datetime(2026, 8, 6, 9, 0, tzinfo=timezone.utc)


@dataclass
class Row:
    """Minimal stand-in for RealtimeMinuteBar."""

    symbol: str
    minute_start: datetime
    open: float
    high: float
    low: float
    close: float
    volume: float
    source_record_ids: tuple = ()


def _rows(count: int, *, start=START, step=0.5, base=100.0):
    out = []
    for index in range(count):
        close = base + step * index
        out.append(
            Row(
                symbol="T",
                minute_start=start + timedelta(minutes=index),
                open=close,
                high=close + 0.5,
                low=close - 0.5,
                close=close,
                volume=1000.0,
            )
        )
    return out


class TestCompletedBarsAreCausal:
    def test_forming_minute_is_always_dropped(self):
        rows = _rows(10)
        # as_of sits inside the last row's minute, so that row is still forming.
        as_of = rows[-1].minute_start + timedelta(seconds=30)
        result = completed_bars(rows, symbol="T", as_of=as_of)
        assert result.bar_count == 9
        assert result.dropped_forming_bar
        assert result.last_bar_start == rows[-2].minute_start

    def test_appending_future_rows_does_not_change_past_bars(self):
        rows = _rows(30)
        as_of = rows[20].minute_start
        before = completed_bars(rows[:21], symbol="T", as_of=as_of)
        after = completed_bars(rows, symbol="T", as_of=as_of)
        assert before.bars == after.bars

    def test_forming_bar_is_available_separately_for_timing(self):
        rows = _rows(10)
        as_of = rows[-1].minute_start + timedelta(seconds=15)
        bar = forming_bar(rows, symbol="T", as_of=as_of)
        assert bar is not None and bar.as_of == rows[-1].minute_start
        # ...and it is NOT in the indicator input.
        assert bar not in completed_bars(rows, symbol="T", as_of=as_of).bars

    def test_duplicate_minutes_are_collapsed(self):
        rows = _rows(5)
        as_of = rows[-1].minute_start + timedelta(minutes=1)
        doubled = rows + rows
        assert completed_bars(doubled, symbol="T", as_of=as_of).bar_count == 5

    def test_malformed_bar_is_dropped_not_repaired(self):
        rows = _rows(5)
        rows[2].high = 1.0  # high < low
        as_of = rows[-1].minute_start + timedelta(minutes=1)
        assert completed_bars(rows, symbol="T", as_of=as_of).bar_count == 4

    def test_out_of_order_input_is_sorted(self):
        rows = _rows(6)
        as_of = rows[-1].minute_start + timedelta(minutes=1)
        shuffled = [rows[3], rows[0], rows[5], rows[1], rows[4], rows[2]]
        bars = completed_bars(shuffled, symbol="T", as_of=as_of).bars
        assert list(bars) == sorted(bars, key=lambda item: item.as_of)


class TestTimeframeAggregation:
    def test_five_minute_bars_aggregate_completed_minutes(self):
        rows = _rows(20)
        as_of = rows[-1].minute_start + timedelta(minutes=1)
        result = completed_bars(rows, symbol="T", as_of=as_of, timeframe_minutes=5)
        assert result.timeframe_minutes == 5
        assert result.bar_count == 4  # 20 completed minutes / 5
        first = result.bars[0]
        assert first.open == rows[0].open
        assert first.close == rows[4].close
        assert first.high == max(row.high for row in rows[:5])
        assert first.low == min(row.low for row in rows[:5])
        assert first.volume == sum(row.volume for row in rows[:5])

    def test_partial_trailing_window_is_not_emitted(self):
        # 12 completed minutes: two full 5m windows, and 2 minutes that are not a
        # 5m bar. Emitting a short aggregate would misstate the timeframe.
        rows = _rows(12)
        as_of = rows[-1].minute_start + timedelta(minutes=1)
        result = completed_bars(rows, symbol="T", as_of=as_of, timeframe_minutes=5)
        assert result.bar_count == 2

    def test_aggregation_is_causal(self):
        rows = _rows(40)
        as_of = rows[20].minute_start
        before = completed_bars(rows[:21], symbol="T", as_of=as_of, timeframe_minutes=5)
        after = completed_bars(rows, symbol="T", as_of=as_of, timeframe_minutes=5)
        assert before.bars == after.bars

    def test_unsupported_timeframe_is_rejected(self):
        try:
            completed_bars(_rows(5), symbol="T", as_of=START, timeframe_minutes=7)
        except ValueError:
            return
        raise AssertionError("unsupported timeframe must raise")

    def test_floor_to_minute_aligns_windows(self):
        moment = datetime(2026, 8, 6, 9, 13, 47, tzinfo=timezone.utc)
        assert floor_to_minute(moment, 5).minute == 10
        assert floor_to_minute(moment, 1).second == 0


class TestProvenanceAndWarmup:
    def test_warmup_flag_reflects_available_history(self):
        rows = _rows(10)
        as_of = rows[-1].minute_start + timedelta(minutes=1)
        short = completed_bars(rows, symbol="T", as_of=as_of, warmup_required=64)
        assert not short.warmup_complete
        long = completed_bars(
            _rows(100), symbol="T", as_of=START + timedelta(minutes=200),
            warmup_required=64,
        )
        assert long.warmup_complete

    def test_provenance_is_serialisable_and_complete(self):
        import json

        rows = _rows(10)
        payload = completed_bars(
            rows, symbol="T", as_of=rows[-1].minute_start, warmup_required=5
        ).as_provenance()
        json.dumps(payload)
        for key in (
            "timeframe_minutes", "as_of", "bar_count", "warmup_complete",
            "dropped_forming_bar", "source_record_ids", "last_bar_start",
        ):
            assert key in payload

    def test_store_failure_degrades_to_empty_not_exception(self):
        class Broken:
            def recent_minute_bars(self, *a, **k):
                raise RuntimeError("db down")

        result = load_causal_bars(Broken(), "T", as_of=START)
        assert isinstance(result, CausalBarSet)
        assert result.bar_count == 0
        assert not result.warmup_complete


# --------------------------------------------------------------------------- #
# Families                                                                     #
# --------------------------------------------------------------------------- #
def _bars(count, step=0.5, base=100.0, volume=1000.0):
    return tuple(
        OHLCVBar(
            ticker="T",
            as_of=START + timedelta(minutes=index),
            open=base + step * index,
            high=base + step * index + 0.5,
            low=base + step * index - 0.5,
            close=base + step * index,
            volume=volume,
        )
        for index in range(count)
    )


class TestAvailabilityMask:
    def test_missing_data_reports_unavailable_not_neutral(self):
        bundle = build_families(())
        for name in FAMILY_NAMES:
            assert not bundle.available(name)
        mask = bundle.availability_mask()
        assert set(mask.values()) == {0.0}

    def test_available_families_report_the_flag(self):
        bundle = build_families(_bars(200))
        assert bundle.available(TREND)
        assert bundle.availability_mask()["trend_available"] == 1.0

    def test_compact_features_are_always_finite_and_complete(self):
        import math

        for count in (0, 3, 60, 200):
            bars = _bars(count)
            payload = compact_model_features(bars, build_families(bars))
            assert set(payload) == set(COMPACT_MODEL_FEATURES)
            assert all(math.isfinite(value) for value in payload.values())

    def test_neutral_value_is_paired_with_a_zero_flag(self):
        # The pairing is the whole point: a neutral number with no flag is
        # indistinguishable from a real neutral reading.
        payload = compact_model_features((), build_families(()))
        assert payload["trend_family_score"] == 0.0
        assert payload["trend_available"] == 0.0
        assert payload["envelope_position"] == 0.5  # documented neutral


class TestFamilyIndependence:
    def test_two_independent_families_are_required(self):
        bundle = build_families(_bars(200))
        result = confirmation(bundle, regime="TREND_UP", adx=30.0)
        assert result.confirmed
        assert len(result.agreeing_families) >= 2

    def test_no_families_available_means_no_direction(self):
        result = confirmation(build_families(()), regime="TREND_UP")
        assert result.direction == 0
        assert "FAMILY_EVIDENCE_UNAVAILABLE" in result.reason_codes

    def test_conflicting_families_reduce_conviction_without_blocking(self):
        bundle = build_families(_bars(200))
        result = confirmation(bundle, regime="TREND_UP", adx=30.0)
        # An uptrend genuinely conflicts with the reversion family; that must cost
        # conviction rather than veto the setup.
        assert result.opposing_families
        assert "FAMILY_CONFLICT" in result.reason_codes
        assert 0.0 < result.conviction < 1.0

    def test_high_adx_suppresses_mean_reversion(self):
        bundle = build_families(_bars(200))
        trending = confirmation(bundle, regime="RANGE_BOUND", adx=40.0)
        calm = confirmation(bundle, regime="RANGE_BOUND", adx=5.0)
        assert "MEAN_REVERSION_SUPPRESSED_BY_ADX" in trending.reason_codes
        assert "MEAN_REVERSION_SUPPRESSED_BY_ADX" not in calm.reason_codes
        # Suppressing the reversion family moves the vote toward the trend side.
        assert trending.weighted_score > calm.weighted_score

    def test_regime_changes_the_weighting(self):
        bundle = build_families(_bars(200))
        trend = confirmation(bundle, regime="TREND_UP", adx=10.0)
        range_bound = confirmation(bundle, regime="RANGE_BOUND", adx=10.0)
        assert trend.weighted_score != range_bound.weighted_score

    def test_direction_flips_with_the_tape(self):
        up = confirmation(build_families(_bars(200)), regime="TREND_UP", adx=30.0)
        down = confirmation(
            build_families(_bars(200, step=-0.5, base=200.0)),
            regime="TREND_DOWN",
            adx=30.0,
        )
        assert up.direction == 1
        assert down.direction == -1


class TestParity:
    def test_live_and_replay_share_one_builder(self):
        """Same as-of, same rows -> identical bars regardless of extra history.

        This is what makes a replay number comparable to a live number: both call
        completed_bars, and the only input that matters is as_of.
        """
        rows = _rows(120)
        as_of = rows[80].minute_start
        live_view = completed_bars(rows[:81], symbol="T", as_of=as_of)
        replay_view = completed_bars(rows, symbol="T", as_of=as_of)
        assert live_view.bars == replay_view.bars

        live_features = compact_model_features(
            live_view.bars, build_families(live_view.bars)
        )
        replay_features = compact_model_features(
            replay_view.bars, build_families(replay_view.bars)
        )
        assert live_features == replay_features

    def test_cadence_difference_does_not_change_bar_meaning(self):
        # US REST cadence can miss minutes; KR websocket does not. A gap must
        # produce FEWER bars, never bars of a different duration.
        dense = _rows(30)
        sparse = [row for index, row in enumerate(dense) if index % 3 == 0]
        as_of = dense[-1].minute_start + timedelta(minutes=1)
        dense_set = completed_bars(dense, symbol="T", as_of=as_of)
        sparse_set = completed_bars(sparse, symbol="T", as_of=as_of)
        assert sparse_set.bar_count < dense_set.bar_count
        assert sparse_set.timeframe_minutes == dense_set.timeframe_minutes


class TestSchemaOrdering:
    def test_schema_contains_every_compact_feature_in_order(self):
        from app.features.feature_schema import LIVE_FEATURE_NAMES

        positions = [LIVE_FEATURE_NAMES.index(name) for name in COMPACT_MODEL_FEATURES]
        assert positions == sorted(positions), "compact block must stay contiguous"

    def test_schema_hash_changed_from_v5(self):
        from app.features.feature_schema import LIVE_SHORT_HORIZON_SCHEMA

        assert LIVE_SHORT_HORIZON_SCHEMA.schema_hash != "5d01d34609767a3a62ddb8fb"
