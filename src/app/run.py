from __future__ import annotations

import argparse
import json
import socket
import sys
from dataclasses import asdict, is_dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import uvicorn

from app.audit import AuditLogger
from app.cli import run_demo
from app.research import ResearchService
from app.storage import LocalResearchStore


def main() -> None:
    _configure_stdout()
    parser = argparse.ArgumentParser(description="Run the complete local investment system")
    parser.add_argument("--host", default="127.0.0.1", help="Web server host")
    parser.add_argument("--port", default=8000, type=int, help="Web server port")
    parser.add_argument(
        "--strict-port",
        action="store_true",
        help="Fail if the requested port is already in use instead of selecting the next free port",
    )
    parser.add_argument(
        "--research-config",
        default="config/research_sources.live.json",
        help="Research source config JSON",
    )
    parser.add_argument(
        "--skip-startup-checks",
        action="store_true",
        help="Start the web server without running research and demo checks first",
    )
    args = parser.parse_args()

    if not args.skip_startup_checks:
        startup = run_startup_checks(Path(args.research_config))
        print(json.dumps(_to_jsonable(startup), indent=2, ensure_ascii=False, sort_keys=True))

    port = args.port if args.strict_port else _find_available_port(args.host, args.port)
    if port != args.port:
        print(f"Port {args.port} is already in use. Using {port} instead.")

    print(f"Web UI: http://{args.host}:{port}")
    uvicorn.run("app.web:app", host=args.host, port=port, app_dir="src", reload=False)


def run_startup_checks(research_config: Path) -> dict[str, Any]:
    audit = AuditLogger(Path("logs/startup.jsonl"))
    store = LocalResearchStore()
    research_result = ResearchService(archive=None).run_from_config(research_config)
    stored_counts = store.save_research_result(research_result)
    demo_result = run_demo(research_result)
    graph_counts = store.save_graph_and_reasoning(
        demo_result["graph_triples"],
        demo_result["reasoning_paths"],
    )

    summary = {
        "startup": "ok",
        "research_config": str(research_config),
        "research_events": len(research_result.events),
        "research_raw_records": len(research_result.raw_records),
        "research_market_snapshots": len(research_result.market_snapshots),
        "research_macro_metrics": len(research_result.macro_metrics),
        "research_skipped_sources": research_result.skipped_sources,
        "research_diagnostics": research_result.diagnostics,
        "graph_triples": len(demo_result["graph_triples"]),
        "classified_events": len(demo_result["classified_events"]),
        "reasoning_paths": len(demo_result["reasoning_paths"]),
        "strategy_signals": len(demo_result["strategy_signals"]),
        "order_intents": len(demo_result["order_intents"]),
        "risk_results": len(demo_result["risk_results"]),
        "stored_new_records": {**stored_counts, **graph_counts},
        "store_summary": store.summary(),
        "store_path": str(store.root),
        "live_trading_enabled": False,
    }
    audit.record("startup_checks", summary)
    return summary


def _to_jsonable(value: Any) -> Any:
    if is_dataclass(value):
        return _to_jsonable(asdict(value))
    if isinstance(value, dict):
        return {str(key): _to_jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_to_jsonable(item) for item in value]
    if isinstance(value, datetime):
        return value.isoformat()
    return value


def _configure_stdout() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")


def _find_available_port(host: str, preferred_port: int, attempts: int = 50) -> int:
    for port in range(preferred_port, preferred_port + attempts):
        if _is_port_available(host, port):
            return port
    raise RuntimeError(
        f"No available port found from {preferred_port} to {preferred_port + attempts - 1}."
    )


def _is_port_available(host: str, port: int) -> bool:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.settimeout(0.2)
        return sock.connect_ex((host, port)) != 0


if __name__ == "__main__":
    main()
