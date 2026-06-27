from __future__ import annotations

import argparse
import json
import os
import time
import tracemalloc
from pathlib import Path

import numpy as np

from app.graph.npu_classifier import OntologyNpuLinearScorer


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="CPU")
    parser.add_argument("--output", default="docs/npu_benchmark_results.md")
    args = parser.parse_args()
    os.environ["OPENVINO_DEVICE"] = args.device

    scenarios = (128, 1024, 4096, 10000)
    rows = []
    tracemalloc.start()
    for count in scenarios:
        rng = np.random.default_rng(42 + count)
        features = rng.normal(0.1, 0.5, size=(count, 8)).astype(np.float32)
        scorer = OntologyNpuLinearScorer(batch_size="auto")
        started = time.perf_counter()
        result = scorer.score_candidates(tuple(f"SIM{i:05d}" for i in range(count)), features, top_k=50)
        elapsed = (time.perf_counter() - started) * 1000
        profile = dict(result.profile)
        profile["scenario"] = count
        profile["wall_ms"] = round(elapsed, 3)
        profile["process_memory_mb"] = _memory_mb()
        profile["peak_memory_mb"] = _peak_memory_mb()
        rows.append(profile)

    _write_markdown(Path(args.output), args.device, rows)
    print(json.dumps(rows, indent=2))


def _memory_mb() -> float:
    current, _peak = tracemalloc.get_traced_memory()
    return round(current / 1024 / 1024, 3)


def _peak_memory_mb() -> float:
    _current, peak = tracemalloc.get_traced_memory()
    return round(peak / 1024 / 1024, 3)


def _write_markdown(path: Path, device: str, rows: list[dict[str, float | int | str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "# NPU Benchmark Results",
        "",
        f"Requested device: `{device}`",
        "",
        "| scenario | device | batch | top_k | preprocess_ms | inference_ms | postprocess_ms | total_ms | memory_mb |",
        "|---:|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        lines.append(
            "| {scenario} | {device} | {batch_bucket} | {top_k} | {preprocess_ms} | "
            "{inference_ms} | {postprocess_ms} | {total_ms} | {process_memory_mb} |".format(**row)
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
