from __future__ import annotations

import math
from datetime import datetime, timedelta, timezone

from app.features.schemas import OHLCVBar
from app.technical.labels import LabelBuilder
from app.technical.replay import ReplayConfig, TechnicalReplayEvaluator


def _series(n=80, seed_trend=0.001):
    start = datetime(2026, 7, 1, 9, 0, tzinfo=timezone.utc)
    bars = []
    price = 100.0
    for i in range(n):
        price *= (1.0 + seed_trend * math.sin(i / 5.0) + seed_trend)
        bars.append(
            OHLCVBar("A", start + timedelta(minutes=i), price, price * 1.002, price * 0.998, price, 1000.0 + i)
        )
    return bars


def _evaluator():
    return TechnicalReplayEvaluator(ReplayConfig(warmup_bars=30, walk_forward_splits=3))


class TestNoLookahead:
    def test_prediction_uses_past_only(self):
        # Predictions at overlapping indices must be identical whether or not
        # future bars exist beyond them (feature isolation => no look-ahead).
        bars = _series(80)
        ev = _evaluator()
        short_rows = {r["index"]: r for r in ev.evaluate_rows("A", bars[:55])}
        long_rows = {r["index"]: r for r in ev.evaluate_rows("A", bars[:80])}
        overlap = set(short_rows) & set(long_rows)
        assert overlap
        for i in overlap:
            s, l = short_rows[i], long_rows[i]
            assert s["tradable"] == l["tradable"]
            assert s["methodology"] == l["methodology"]
            assert s["regime"] == l["regime"]
            assert s["predicted_net_bps"] == l["predicted_net_bps"]

    def test_future_spike_changes_label_not_prediction(self):
        bars = _series(80)
        ev = _evaluator()
        base = {r["index"]: r for r in ev.evaluate_rows("A", bars)}
        # Inject a huge upward spike at bars 40-42 (just after the checked region).
        spiked = list(bars)
        for j in (40, 41, 42):
            b = spiked[j]
            spiked[j] = OHLCVBar(b.ticker, b.as_of, b.open, b.high * 2, b.low, b.close * 2, b.volume)
        after = {r["index"]: r for r in ev.evaluate_rows("A", spiked)}
        # Predictions at indices whose PAST excludes the spike are unchanged.
        for i in range(30, 40):
            assert base[i]["predicted_net_bps"] == after[i]["predicted_net_bps"]
            assert base[i]["tradable"] == after[i]["tradable"]
        # Labels within ~5 bars BEFORE the spike see it in their FUTURE window.
        changed = any(base[i]["mfe_bps"] != after[i]["mfe_bps"] for i in (37, 38, 39))
        assert changed

    def test_labels_only_use_future_path(self):
        # A label at t must not depend on data at/before t beyond the entry price.
        lb = LabelBuilder()
        labels = lb.build(symbol="A", entry_price=100.0, future_path=[(5, 100.5), (60, 101.0)])
        assert labels.future_return_5s == 100.5 / 100.0 - 1.0
        # No path point beyond 60s -> 5m return is None (never invented).
        assert labels.future_return_5m is None


class TestReplayReport:
    def test_report_structure_and_metrics(self):
        report = _evaluator().evaluate("A", _series(120))
        assert report["no_lookahead"] is True
        assert report["overall"]["n"] > 0
        assert "by_methodology" in report and "by_regime" in report
        wf = report["walk_forward"]
        assert len(wf) >= 2  # walk-forward segmented
        # Each split reports its own metrics.
        assert all("n" in seg for seg in wf)

    def test_deterministic(self):
        bars = _series(90)
        a = _evaluator().evaluate("A", bars)
        b = _evaluator().evaluate("A", bars)
        assert a == b
