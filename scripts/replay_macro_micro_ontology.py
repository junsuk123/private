"""CLI: replay the macro–micro ontology reasoning over stored/offline bars.

Runs the no-look-ahead MacroMicroReplayEvaluator across a multi-symbol bar
series and writes a reproducible JSON report under
``data/models/macro_micro_replay_reports/``.

Usage:
    python scripts/replay_macro_micro_ontology.py --from-bars bars.json
    python scripts/replay_macro_micro_ontology.py --symbol SYMBOL_A --symbol SYMBOL_B --since 2026-07-01

``--from-bars`` reads a JSON list of {ticker,as_of,open,high,low,close,volume}
rows (multiple tickers), so the harness runs offline / in CI.
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from app.features.schemas import OHLCVBar  # noqa: E402
from app.graph.macro_micro_replay import MacroMicroReplayConfig, MacroMicroReplayEvaluator  # noqa: E402

REPORT_DIR = Path("data/models/macro_micro_replay_reports")


def _bars_from_json(path: Path) -> dict[str, list[OHLCVBar]]:
    rows = json.loads(path.read_text(encoding="utf-8"))
    by_symbol: dict[str, list[OHLCVBar]] = {}
    for r in rows:
        bar = OHLCVBar(str(r["ticker"]), datetime.fromisoformat(r["as_of"]),
                       float(r["open"]), float(r["high"]), float(r["low"]), float(r["close"]), float(r["volume"]))
        by_symbol.setdefault(bar.ticker, []).append(bar)
    for s in by_symbol:
        by_symbol[s].sort(key=lambda b: b.as_of)
    return by_symbol


def _bars_from_store(symbols: list[str], since: datetime) -> dict[str, list[OHLCVBar]]:
    from app.data.realtime_store import RealtimeMarketDataStore

    store = RealtimeMarketDataStore()
    out: dict[str, list[OHLCVBar]] = {}
    for symbol in symbols:
        buckets: dict[datetime, list[float]] = {}
        vols: dict[datetime, float] = {}
        for tick in store.recent_ticks(symbol, since):
            minute = tick.exchange_timestamp.replace(second=0, microsecond=0)
            buckets.setdefault(minute, []).append(float(tick.price))
            vols[minute] = vols.get(minute, 0.0) + float(max(0, tick.volume))
        out[symbol] = [
            OHLCVBar(symbol, m, b[0], max(b), min(b), b[-1], vols[m]) for m, b in sorted(buckets.items())
        ]
    return out


def main() -> int:
    parser = argparse.ArgumentParser(description="Macro–micro ontology replay / walk-forward validation.")
    parser.add_argument("--symbol", action="append", default=[])
    parser.add_argument("--since", default=None)
    parser.add_argument("--from-bars", default=None)
    parser.add_argument("--warmup", type=int, default=30)
    parser.add_argument("--stamp", default="latest")
    args = parser.parse_args()

    try:
        from app.cost.trading_cost_engine import TradingCostEngine

        cost_engine = TradingCostEngine()
    except Exception:
        cost_engine = None
    evaluator = MacroMicroReplayEvaluator(MacroMicroReplayConfig(warmup_bars=args.warmup), cost_engine=cost_engine)

    if args.from_bars:
        series = _bars_from_json(Path(args.from_bars))
    else:
        if not args.symbol or not args.since:
            parser.error("provide --from-bars, or both --symbol and --since")
        series = _bars_from_store(args.symbol, datetime.fromisoformat(args.since))

    report = evaluator.evaluate(series)
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    out_path = REPORT_DIR / f"macro_micro_replay.{args.stamp}.json"
    out_path.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"wrote {out_path}")
    print(f"  symbols={report['symbols']} steps={report['steps']} buys={report['buy_candidates']} "
          f"edge_error_bps={report['avg_edge_error_bps']} regimes={report['regime_distribution']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
