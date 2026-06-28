from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from fastapi.testclient import TestClient  # noqa: E402

from app.web import app  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description="Profile realtime dashboard API responsiveness.")
    parser.add_argument("--iterations", type=int, default=8)
    parser.add_argument("--force-refresh", action="store_true")
    args = parser.parse_args()

    client = TestClient(app)
    endpoints = {
        "status": lambda: client.get("/api/status"),
        "diagnostics": lambda: client.get("/api/research/diagnostics"),
        "live_snapshot": lambda: client.post(
            "/api/live-snapshot",
            json={"force_refresh": bool(args.force_refresh), "include_graph": False},
        ),
        "operation_mode_status": lambda: client.get("/api/operation-mode/status"),
    }
    api_timings: dict[str, list[float]] = {name: [] for name in endpoints}
    latest_payloads = {}
    for _ in range(max(1, args.iterations)):
        for name, call in endpoints.items():
            started = time.perf_counter()
            response = call()
            api_timings[name].append((time.perf_counter() - started) * 1000.0)
            response.raise_for_status()
            latest_payloads[name] = response.json()

    live_snapshot = latest_payloads.get("live_snapshot", {})
    diagnostics = live_snapshot.get("diagnostics") or latest_payloads.get("diagnostics", {})
    progress = client.get("/api/live-progress").json().get("progress", {})
    payload = {
        "api_response_p50_ms": {
            name: round(statistics.median(values), 3)
            for name, values in api_timings.items()
        },
        "api_response_p95_ms": {
            name: _p95(values)
            for name, values in api_timings.items()
        },
        "indicator_build_ms": _metric(progress, "indicator_build_ms"),
        "graph_build_ms": _metric(progress, "graph_build_ms"),
        "reasoning_ms": _metric(progress, "reasoning_ms"),
        "storage_write_ms": _metric(progress, "storage_write_ms"),
        "risk_validation_ms": _metric(progress, "risk_validation_ms"),
        "diagnostics": {
            "is_refreshing": diagnostics.get("is_refreshing"),
            "graph_triples_count": diagnostics.get("graph_triples_count"),
            "store_summary": diagnostics.get("store_summary", {}),
        },
    }
    print(json.dumps(payload, indent=2, sort_keys=True))


def _p95(values: list[float]) -> float:
    ordered = sorted(values)
    index = min(len(ordered) - 1, int(round((len(ordered) - 1) * 0.95)))
    return round(ordered[index], 3)


def _metric(payload: dict, key: str) -> float | None:
    value = payload.get(key)
    return round(float(value), 3) if isinstance(value, (int, float)) else None


if __name__ == "__main__":
    main()
