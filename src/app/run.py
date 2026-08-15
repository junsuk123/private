from __future__ import annotations

import argparse
import asyncio
import json
import os
import signal
import socket
import subprocess
import sys
import threading
import time
import traceback
from dataclasses import asdict, is_dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import uvicorn

from app.audit import AuditLogger
from app.cli import run_demo
from app.data.llm_classifier import configure_default_event_llm_env
from app.research import ResearchService
from app.storage import LocalResearchStore
from app.trading.live_runtime_guard import env_bool as runtime_env_bool
from app.web_access_guard import (
    ExternalAccessDenied,
    binds_externally,
    require_token_for_external_bind,
)
from app.trading_pipeline import load_short_horizon_strategy_config
from app.config.refactor_profile import load_refactor_profile


def main() -> None:
    _configure_stdout()
    _configure_windows_event_loop_policy()
    configure_default_event_llm_env()
    parser = argparse.ArgumentParser(description="Run the complete local investment system")
    parser.add_argument(
        "--host",
        default=os.getenv("APP_HOST", "127.0.0.1"),
        help=(
            "Web server host. Stays 127.0.0.1 unless APP_HOST or this flag says "
            "otherwise; binding elsewhere requires APP_ACCESS_TOKEN because this "
            "app has no authentication and accepts live-trading requests."
        ),
    )
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

    # Checked BEFORE anything else starts: the failure mode this prevents is a
    # server that came up reachable on the LAN with shutdown and live-trading
    # endpoints open, which is not something to discover after the fact.
    try:
        require_token_for_external_bind(args.host)
    except ExternalAccessDenied as exc:
        print(f"REFUSING TO START: {exc}", file=sys.stderr)
        raise SystemExit(2)

    _stop_existing_app_servers(args.host, args.port)

    if not args.skip_startup_checks:
        _run_startup_checks_in_background(Path(args.research_config))

    port = args.port if args.strict_port else _find_available_port(args.host, args.port)
    if port != args.port:
        print(f"Port {args.port} is already in use. Using {port} instead.")

    print(f"Web UI: http://{args.host}:{port}")
    if binds_externally(args.host):
        # ``http://0.0.0.0:8000`` is not an address anything can open. Print the
        # LAN address the phone actually needs, and say plainly what is now
        # reachable, since this is the one line an operator will read before
        # typing the URL into a device.
        lan = _lan_address()
        if lan:
            print(f"Reachable on this network: http://{lan}:{port}/account?token=...")
        print(
            "This server accepts live-trading and shutdown requests. "
            "Non-loopback clients must present APP_ACCESS_TOKEN."
        )
    try:
        uvicorn.run(
            "app.web:app",
            host=args.host,
            port=port,
            app_dir="src",
            reload=False,
            access_log=False,
        )
    except KeyboardInterrupt:
        print("Server stopped.")


def _configure_windows_event_loop_policy() -> None:
    """Avoid a Windows Proactor accept-loop failure that can silently drop HTTP.

    The market-data worker threads can keep running after the listening socket is
    lost, which makes the process look healthy while the GUI/API is unreachable.
    Uvicorn's selector loop is sufficient for this application and does not use
    the IOCP accept path that raised WinError 64 in production.
    """
    if sys.platform != "win32":
        return
    policy_factory = getattr(asyncio, "WindowsSelectorEventLoopPolicy", None)
    if policy_factory is not None:
        asyncio.set_event_loop_policy(policy_factory())


def _run_startup_checks_in_background(research_config: Path) -> None:
    def worker() -> None:
        audit = AuditLogger(Path("logs/startup.jsonl"))
        try:
            startup = run_startup_checks(research_config)
            print(json.dumps(_to_jsonable(startup), indent=2, ensure_ascii=False, sort_keys=True))
        except Exception as exc:  # noqa: BLE001 - startup checks must not block the web server.
            audit.record(
                "startup_checks_failed",
                {"error": str(exc), "traceback": traceback.format_exc()},
            )
            print(f"Startup checks failed in background: {str(exc) or exc.__class__.__name__}")

    threading.Thread(target=worker, name="startup-checks", daemon=True).start()


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

    strategy_config = load_short_horizon_strategy_config()
    refactor_profile = load_refactor_profile()
    execution = dict(strategy_config.get("execution", {}) or {})
    live_enabled_by_config = bool(execution.get("live_trading_enabled", False))
    live_enabled_by_env = runtime_env_bool("LIVE_TRADING_ENABLED", False) and runtime_env_bool("KIS_LIVE_ENABLED", False)

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
        "live_trading_enabled": live_enabled_by_config and live_enabled_by_env,
        "live_trading_enabled_by_config": live_enabled_by_config,
        "live_trading_enabled_by_env": live_enabled_by_env,
        "refactor_runtime": {
            "mode": refactor_profile.mode.value,
            "broker_submission_enabled": refactor_profile.broker_submission_enabled,
            "maximum_order_notional": refactor_profile.maximum_order_notional,
            "allowed_symbol_count": len(refactor_profile.allowed_symbols),
            "flags": asdict(refactor_profile.flags),
        },
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


def _lan_address() -> str | None:
    """This machine's address on the local network, or ``None``.

    Uses a UDP socket to a routable address purely to ask the OS which local
    interface it would route through; no packet is sent and nothing is contacted.
    """
    probe = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    try:
        probe.connect(("8.8.8.8", 80))
        return str(probe.getsockname()[0])
    except OSError:
        return None
    finally:
        probe.close()


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


def _stop_existing_app_servers(host: str, preferred_port: int) -> None:
    current_pid = os.getpid()
    workspace = Path(__file__).resolve().parents[2]
    processes = _list_processes()
    protected_pids = {current_pid, *_ancestor_pids(current_pid, processes)}
    listening_pids = _listening_pids(preferred_port, preferred_port + 50)
    matched: set[int] = set()
    for process in processes:
        pid = int(process.get("pid", 0) or 0)
        if pid <= 0 or pid in protected_pids:
            continue
        command = str(process.get("command", "") or "")
        if _is_existing_app_server_command(command, workspace) or (
            pid in listening_pids and _looks_like_app_server_command(command)
        ):
            matched.add(pid)

    if not matched:
        return

    descendants_by_parent = _descendants_by_parent(processes)
    for pid in sorted(matched):
        _terminate_process_tree(pid, descendants_by_parent, current_pid)

    deadline = time.time() + 5.0
    while time.time() < deadline:
        if _is_port_available(host, preferred_port):
            return
        time.sleep(0.25)


def _is_existing_app_server_command(command: str, workspace: Path) -> bool:
    lowered = command.lower()
    if not _looks_like_app_server_command(lowered):
        return False
    workspace_text = str(workspace).lower()
    return workspace_text in lowered


def _looks_like_app_server_command(command: str) -> bool:
    lowered = command.lower()
    return ("python" in lowered or "uvicorn" in lowered) and (
        "run.py" in lowered or "app.web:app" in lowered
    )


def _list_processes() -> list[dict[str, object]]:
    if os.name == "nt":
        script = (
            "[Console]::OutputEncoding = [System.Text.Encoding]::UTF8; "
            "Get-CimInstance Win32_Process | "
            "Select-Object ProcessId,ParentProcessId,CommandLine | ConvertTo-Json -Compress"
        )
        try:
            result = subprocess.run(
                ["powershell", "-NoProfile", "-Command", script],
                check=False,
                capture_output=True,
                text=True,
                encoding="utf-8",
                errors="replace",
                timeout=10,
            )
        except (OSError, subprocess.SubprocessError):
            return []
        if result.returncode != 0 or not result.stdout.strip():
            return []
        try:
            payload = json.loads(result.stdout)
        except json.JSONDecodeError:
            return []
        rows = payload if isinstance(payload, list) else [payload]
        return [
            {
                "pid": row.get("ProcessId"),
                "ppid": row.get("ParentProcessId"),
                "command": row.get("CommandLine") or "",
            }
            for row in rows
            if isinstance(row, dict)
        ]

    try:
            result = subprocess.run(
                ["ps", "-eo", "pid=,ppid=,args="],
                check=False,
                capture_output=True,
                text=True,
                errors="replace",
                timeout=10,
            )
    except (OSError, subprocess.SubprocessError):
        return []
    rows: list[dict[str, object]] = []
    for line in result.stdout.splitlines():
        parts = line.strip().split(None, 2)
        if len(parts) < 3:
            continue
        rows.append({"pid": parts[0], "ppid": parts[1], "command": parts[2]})
    return rows


def _listening_pids(start_port: int, end_port: int) -> set[int]:
    if os.name == "nt":
        try:
            result = subprocess.run(
                ["netstat", "-ano", "-p", "tcp"],
                check=False,
                capture_output=True,
                text=True,
                errors="replace",
                timeout=10,
            )
        except (OSError, subprocess.SubprocessError):
            return set()
        pids: set[int] = set()
        for line in result.stdout.splitlines():
            parts = line.split()
            if len(parts) < 5 or parts[0].lower() != "tcp" or parts[3].upper() != "LISTENING":
                continue
            port = _port_from_endpoint(parts[1])
            if start_port <= port <= end_port:
                pids.add(_safe_int(parts[-1]))
        return {pid for pid in pids if pid > 0}

    try:
        result = subprocess.run(
            ["sh", "-c", f"lsof -nP -iTCP:{start_port}-{end_port} -sTCP:LISTEN -Fp 2>/dev/null"],
            check=False,
            capture_output=True,
            text=True,
            errors="replace",
            timeout=10,
        )
    except (OSError, subprocess.SubprocessError):
        return set()
    return {_safe_int(line[1:]) for line in result.stdout.splitlines() if line.startswith("p")} - {0}


def _port_from_endpoint(endpoint: str) -> int:
    if endpoint.startswith("["):
        _, _, suffix = endpoint.rpartition("]:")
        return _safe_int(suffix)
    _, _, suffix = endpoint.rpartition(":")
    return _safe_int(suffix)


def _descendants_by_parent(processes: list[dict[str, object]]) -> dict[int, list[int]]:
    children: dict[int, list[int]] = {}
    for process in processes:
        pid = _safe_int(process.get("pid"))
        ppid = _safe_int(process.get("ppid"))
        if pid > 0 and ppid > 0:
            children.setdefault(ppid, []).append(pid)
    return children


def _ancestor_pids(pid: int, processes: list[dict[str, object]]) -> set[int]:
    parent_by_pid = {
        _safe_int(process.get("pid")): _safe_int(process.get("ppid"))
        for process in processes
    }
    ancestors: set[int] = set()
    parent = parent_by_pid.get(pid, 0)
    while parent > 0 and parent not in ancestors:
        ancestors.add(parent)
        parent = parent_by_pid.get(parent, 0)
    return ancestors


def _terminate_process_tree(pid: int, descendants_by_parent: dict[int, list[int]], current_pid: int) -> None:
    for child_pid in descendants_by_parent.get(pid, ()):
        if child_pid != current_pid:
            _terminate_process_tree(child_pid, descendants_by_parent, current_pid)
    if pid == current_pid:
        return
    if os.name == "nt":
        subprocess.run(["taskkill", "/PID", str(pid), "/T", "/F"], check=False, capture_output=True, text=True)
        return
    try:
        os.kill(pid, signal.SIGTERM)
    except OSError:
        return
    time.sleep(0.1)
    try:
        os.kill(pid, signal.SIGKILL)
    except OSError:
        pass


def _safe_int(value: object) -> int:
    try:
        return int(str(value).strip())
    except (TypeError, ValueError):
        return 0


if __name__ == "__main__":
    main()
