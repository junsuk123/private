from __future__ import annotations

import argparse
import json
import os
import time
import tracemalloc
from pathlib import Path

from app.trading_pipeline import build_lightweight_market_snapshots, ontology_filter_1, universe_from_tickers


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="CPU")
    parser.add_argument("--output", default="docs/npu_realtime_benchmark_results.md")
    args = parser.parse_args()
    os.environ["OPENVINO_DEVICE"] = args.device

    tracemalloc.start()
    rows = []
    for name, count in (
        ("small_universe", 128),
        ("medium_universe", 1024),
        ("large_universe", 4096),
        ("extra_large_universe", 10000),
    ):
        snapshots = build_lightweight_market_snapshots(universe_from_tickers(tuple(f"SIM{i:05d}" for i in range(count))))
        started = time.perf_counter()
        result = ontology_filter_1(snapshots, target_count=50, cache_key=None)
        total_ms = (time.perf_counter() - started) * 1000
        current, peak = tracemalloc.get_traced_memory()
        row = dict(result.metrics)
        row.update(
            {
                "scenario": name,
                "total_pipeline_ms": round(total_ms, 3),
                "process_memory_mb": round(current / 1024 / 1024, 3),
                "peak_memory_mb": round(peak / 1024 / 1024, 3),
            }
        )
        rows.append(row)
    _write_markdown(Path(args.output), args.device, rows)
    print(json.dumps(rows, indent=2))


def _write_markdown(path: Path, device: str, rows: list[dict[str, float | int | str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "# Realtime Pipeline Benchmark Results",
        "",
        f"Requested device: `{device}`",
        "",
        "| scenario | input | hard_filter | topk | device | scoring_ms | total_pipeline_ms | peak_memory_mb |",
        "|---|---:|---:|---:|---|---:|---:|---:|",
    ]
    for row in rows:
        lines.append(
            "| {scenario} | {candidate_count_input} | {candidate_count_after_hard_filter} | "
            "{candidate_count_after_npu_topk} | {device} | {total_ms} | {total_pipeline_ms} | "
            "{peak_memory_mb} |".format(**row)
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
