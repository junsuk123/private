from __future__ import annotations

import argparse
import asyncio
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from app.data.kis_realtime import QueueMessageSource, KisRealtimeSubscriptionManager
from app.data.market_data_health import evaluate_market_data_health
from app.data.realtime_store import RealtimeMarketDataStore


def main() -> int:
    parser = argparse.ArgumentParser(description="Check realtime market-data freshness without orders.")
    parser.add_argument("--symbols", nargs="+", default=["005930"])
    parser.add_argument("--duration-seconds", type=int, default=0)
    parser.add_argument("--fixture", type=Path, default=None)
    parser.add_argument("--db-path", type=Path, default=Path("data/store/realtime_market_data.sqlite3"))
    args = parser.parse_args()

    symbols = _symbols(args.symbols)
    store = RealtimeMarketDataStore(args.db_path)
    counts = {"messages": 0, "ticks": 0, "orderbooks": 0}
    if args.fixture is not None:
        messages = tuple(line.strip() for line in args.fixture.read_text(encoding="utf-8").splitlines() if line.strip())
        manager = KisRealtimeSubscriptionManager(store, QueueMessageSource(messages))
        manager.subscribe(symbols)
        counts = asyncio.run(manager.run_forever())
    elif args.duration_seconds <= 0:
        print("No fixture supplied and duration is zero; running store freshness check only.")
    else:
        print("Live KIS WebSocket transport is not configured in this local script yet; no orders submitted.")

    health = [
        evaluate_market_data_health(store, symbol, now=datetime.now(timezone.utc))
        for symbol in symbols
    ]
    report = {
        "checked_at": datetime.now(timezone.utc).isoformat(),
        "symbols": symbols,
        "counts": counts,
        "health": [
            {
                "symbol": item.symbol,
                "ok_for_live_buy": item.ok_for_live_buy,
                "reason_codes": item.reason_codes,
                "quote_count": item.quote_count,
                "orderbook_count": item.orderbook_count,
            }
            for item in health
        ],
        "no_orders": True,
    }
    print(json.dumps(report, indent=2, default=str))
    return 0 if all(item.ok_for_live_buy for item in health) else 1


def _symbols(raw: list[str]) -> list[str]:
    symbols: list[str] = []
    for arg in raw:
        for item in arg.split(","):
            text = item.strip()
            if text:
                symbols.append(text.zfill(6) if text.isdigit() else text)
    return symbols


if __name__ == "__main__":
    raise SystemExit(main())
