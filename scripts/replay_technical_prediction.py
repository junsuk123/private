"""CLI: replay the technical prediction layer over stored realtime data.

Reads minute bars for one or more symbols from the realtime store, runs the
no-look-ahead :class:`TechnicalReplayEvaluator`, and writes a reproducible JSON
report under ``data/models/technical_replay_reports/``.

Usage:
    python scripts/replay_technical_prediction.py --symbol 005930 --since 2026-07-01
    python scripts/replay_technical_prediction.py --from-bars path/to/bars.json

The ``--from-bars`` mode reads a JSON list of
``{ticker,as_of,open,high,low,close,volume}`` rows (no broker/store needed) so
the harness is runnable offline and in CI.
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from app.features.schemas import OHLCVBar  # noqa: E402
from app.technical.replay import ReplayConfig, TechnicalReplayEvaluator  # noqa: E402

REPORT_DIR = Path("data/models/technical_replay_reports")


def _bars_from_json(path: Path) -> dict[str, list[OHLCVBar]]:
    rows = json.loads(path.read_text(encoding="utf-8"))
    by_symbol: dict[str, list[OHLCVBar]] = {}
    for r in rows:
        bar = OHLCVBar(
            ticker=str(r["ticker"]),
            as_of=datetime.fromisoformat(r["as_of"]),
            open=float(r["open"]),
            high=float(r["high"]),
            low=float(r["low"]),
            close=float(r["close"]),
            volume=float(r["volume"]),
        )
        by_symbol.setdefault(bar.ticker, []).append(bar)
    for symbol in by_symbol:
        by_symbol[symbol].sort(key=lambda b: b.as_of)
    return by_symbol


def _bars_from_store(symbol: str, since: datetime) -> list[OHLCVBar]:
    from app.data.realtime_store import RealtimeMarketDataStore

    store = RealtimeMarketDataStore()
    ticks = store.recent_ticks(symbol, since)
    # Aggregate ticks into 1-minute OHLCV bars.
    buckets: dict[datetime, list[float]] = {}
    vols: dict[datetime, float] = {}
    for tick in ticks:
        minute = tick.exchange_timestamp.replace(second=0, microsecond=0)
        buckets.setdefault(minute, []).append(float(tick.price))
        vols[minute] = vols.get(minute, 0.0) + float(max(0, tick.volume))
    bars: list[OHLCVBar] = []
    for minute in sorted(buckets):
        prices = buckets[minute]
        bars.append(
            OHLCVBar(symbol, minute, prices[0], max(prices), min(prices), prices[-1], vols[minute])
        )
    return bars


def main() -> int:
    parser = argparse.ArgumentParser(description="Technical prediction replay / walk-forward validation.")
    parser.add_argument("--symbol", action="append", default=[], help="symbol(s) to replay from the store")
    parser.add_argument("--since", default=None, help="ISO date/time lower bound for store ticks")
    parser.add_argument("--from-bars", default=None, help="JSON file of OHLCV rows (offline mode)")
    parser.add_argument("--warmup", type=int, default=30)
    parser.add_argument("--splits", type=int, default=3)
    parser.add_argument("--stamp", default=None, help="report filename stamp (default: 'latest')")
    args = parser.parse_args()

    config = ReplayConfig(warmup_bars=args.warmup, walk_forward_splits=args.splits)
    try:
        from app.cost.trading_cost_engine import TradingCostEngine

        cost_engine = TradingCostEngine()
    except Exception:
        cost_engine = None
    evaluator = TechnicalReplayEvaluator(config, cost_engine=cost_engine)

    series: dict[str, list[OHLCVBar]] = {}
    if args.from_bars:
        series = _bars_from_json(Path(args.from_bars))
    else:
        if not args.symbol or not args.since:
            parser.error("provide --from-bars, or both --symbol and --since")
        since = datetime.fromisoformat(args.since)
        for symbol in args.symbol:
            series[symbol] = _bars_from_store(symbol, since)

    reports = {symbol: evaluator.evaluate(symbol, bars) for symbol, bars in series.items()}
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    stamp = args.stamp or "latest"
    out_path = REPORT_DIR / f"technical_replay.{stamp}.json"
    out_path.write_text(json.dumps(reports, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"wrote {out_path}")
    for symbol, report in reports.items():
        o = report["overall"]
        print(f"  {symbol}: n={o.get('n')} tradable={o.get('tradable_count')} "
              f"hit_rate={o.get('hit_rate')} avg_realized_net_bps={o.get('avg_realized_net_bps')} "
              f"edge_error_bps={o.get('avg_edge_error_bps')}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
