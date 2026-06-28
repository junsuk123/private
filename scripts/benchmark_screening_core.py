from __future__ import annotations

import argparse
import json
import os
import statistics
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from app.trading_pipeline import (  # noqa: E402
    _ontology_filter_1_python_loop,
    build_lightweight_market_snapshots,
    ontology_filter_1,
    universe_from_tickers,
)


def main() -> None:
    parser = argparse.ArgumentParser(description="Benchmark ontology_filter_1 screening hot paths.")
    parser.add_argument("--symbols", type=int, default=4096)
    parser.add_argument("--top-k", type=int, default=50)
    parser.add_argument("--iterations", type=int, default=10)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    snapshots = build_lightweight_market_snapshots(
        universe_from_tickers(tuple(f"SIM{i:05d}" for i in range(args.symbols))),
        seed=args.seed,
    )
    os.environ["ONTOLOGY_NPU_ENABLED"] = "false"
    legacy_ms = _measure(lambda: _ontology_filter_1_python_loop(snapshots, target_count=args.top_k), args.iterations)
    vector_ms = _measure(lambda: ontology_filter_1(snapshots, target_count=args.top_k), args.iterations)
    result = ontology_filter_1(snapshots, target_count=args.top_k)

    payload = {
        "symbols": args.symbols,
        "top_k": args.top_k,
        "iterations": args.iterations,
        "candidate_screen_ms": _summary(vector_ms),
        "legacy_candidate_screen_ms": _summary(legacy_ms),
        "speedup": round((statistics.median(legacy_ms) / max(1e-9, statistics.median(vector_ms))), 3),
        "latest_metrics": result.metrics,
    }
    print(json.dumps(payload, indent=2, sort_keys=True))


def _measure(fn, iterations: int) -> list[float]:
    values: list[float] = []
    for _ in range(max(1, iterations)):
        started = time.perf_counter()
        fn()
        values.append((time.perf_counter() - started) * 1000.0)
    return values


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
