from __future__ import annotations

import math
from datetime import datetime, timedelta, timezone

from app.features.schemas import OHLCVBar
from app.graph.macro_micro_replay import MacroMicroReplayConfig, MacroMicroReplayEvaluator


def _series(symbol, n=70, phase=0.0, drift=0.0009):
    start = datetime(2026, 7, 1, 9, 0, tzinfo=timezone.utc)
    bars, price = [], 100.0
    for i in range(n):
        price *= (1.0 + drift * math.sin(i / 5.0 + phase) + drift)
        bars.append(OHLCVBar(symbol, start + timedelta(minutes=i), price, price * 1.002, price * 0.998, price, 1000.0 + i))
    return bars


def _symbol_bars(n=70):
    return {"AAA": _series("AAA", n, phase=0.0), "BBB": _series("BBB", n, phase=1.0)}


def _evaluator():
    return MacroMicroReplayEvaluator(MacroMicroReplayConfig(warmup_bars=30))


class TestNoLookahead:
    def test_predictions_use_past_only(self):
        ev = _evaluator()
        bars = _symbol_bars(70)
        short_rows = {(r["step"], r["symbol"]): r for r in ev.evaluate_rows({k: v[:50] for k, v in bars.items()})}
        long_rows = {(r["step"], r["symbol"]): r for r in ev.evaluate_rows(bars)}
        overlap = set(short_rows) & set(long_rows)
        assert overlap
        for key in overlap:
            s, l = short_rows[key], long_rows[key]
            # Macro + micro decisions depend only on past bars -> identical.
            assert s["entry_signal"] == l["entry_signal"]
            assert s["predicted_net_bps"] == l["predicted_net_bps"]
            assert s["macro_regime"] == l["macro_regime"]

    def test_future_spike_changes_realized_not_prediction(self):
        ev = _evaluator()
        bars = _symbol_bars(70)
        base = {(r["step"], r["symbol"]): r for r in ev.evaluate_rows(bars)}
        spiked = {k: list(v) for k, v in bars.items()}
        for j in (40, 41, 42):
            b = spiked["AAA"][j]
            spiked["AAA"][j] = OHLCVBar(b.ticker, b.as_of, b.open, b.high * 2, b.low, b.close * 2, b.volume)
        after = {(r["step"], r["symbol"]): r for r in ev.evaluate_rows(spiked)}
        # Prediction for AAA at steps whose PAST excludes the spike is unchanged.
        for step in range(30, 40):
            key = (step, "AAA")
            if key in base and key in after:
                assert base[key]["predicted_net_bps"] == after[key]["predicted_net_bps"]
        # A realized value near the spike changed.
        changed = any(
            base.get((s, "AAA"), {}).get("realized_net_bps") != after.get((s, "AAA"), {}).get("realized_net_bps")
            for s in range(37, 40)
        )
        assert changed


class TestReport:
    def test_report_structure(self):
        report = _evaluator().evaluate(_symbol_bars(90))
        assert report["no_lookahead"] is True
        assert report["symbols"] == ["AAA", "BBB"]
        assert report["rows"] > 0
        assert "regime_distribution" in report

    def test_expected_vs_realized_present_when_buys(self):
        report = _evaluator().evaluate(_symbol_bars(90))
        # Whether or not any BUYs formed, the report exposes the comparison keys.
        assert "avg_predicted_net_bps" in report
        assert "avg_edge_error_bps" in report

    def test_deterministic(self):
        bars = _symbol_bars(80)
        assert _evaluator().evaluate(bars) == _evaluator().evaluate(bars)
