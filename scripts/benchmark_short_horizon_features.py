from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from app.features.schemas import OHLCVBar  # noqa: E402
from app.features.short_horizon_features import ShortHorizonFeatureBuilder  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description="Benchmark short-horizon feature build latency.")
    parser.add_argument("--symbols", type=int, default=512)
    parser.add_argument("--bars", type=int, default=120)
    parser.add_argument("--iterations", type=int, default=10)
    args = parser.parse_args()

    histories = {
        f"SIM{i:05d}": _bars(f"SIM{i:05d}", args.bars, i)
        for i in range(args.symbols)
    }
    builder = ShortHorizonFeatureBuilder()
    values: list[float] = []
    missing_counts: list[int] = []
    for _ in range(max(1, args.iterations)):
        started = time.perf_counter()
        for bars in histories.values():
            features = builder.build(bars)
            missing_counts.append(len(features.missing_fields))
        values.append((time.perf_counter() - started) * 1000.0)

    print(
        json.dumps(
            {
                "symbols": args.symbols,
                "bars_per_symbol": args.bars,
                "iterations": args.iterations,
                "indicator_build_ms": _summary(values),
                "missing_fields_median": statistics.median(missing_counts) if missing_counts else 0,
            },
            indent=2,
            sort_keys=True,
        )
    )


def _bars(ticker: str, count: int, offset: int) -> tuple[OHLCVBar, ...]:
    start = datetime.now(timezone.utc) - timedelta(minutes=count)
    rows = []
    price = 100.0 + offset * 0.01
    for index in range(count):
        price *= 1.0 + ((index % 9) - 4) * 0.0002
        rows.append(
            OHLCVBar(
                ticker=ticker,
                as_of=start + timedelta(minutes=index),
                open=price * 0.999,
                high=price * 1.002,
                low=price * 0.998,
                close=price,
                volume=100_000 + (index % 17) * 1_000 + offset,
            )
        )
    return tuple(rows)


def _summary(values: list[float]) -> dict[str, float]:
    ordered = sorted(values)
    p95_index = min(len(ordered) - 1, int(round((len(ordered) - 1) * 0.95)))
    return {
        "min": round(min(values), 3),
        "median": round(statistics.median(values), 3),
        "p95": round(ordered[p95_index], 3),
        "max": round(max(values), 3),
    }


if __name__ == "__main__":
    main()
