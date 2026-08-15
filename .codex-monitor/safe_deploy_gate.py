from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
SUCCESS_STATUSES = {"NO_CHANGE_NEEDED", "DIAGNOSTICS_ADDED", "PATCHED_AND_TESTED"}


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _server_pids(port: int) -> list[int]:
    if os.name != "nt":
        return []
    command = (
        f"@(Get-NetTCPConnection -LocalPort {int(port)} -State Listen "
        "-ErrorAction SilentlyContinue | Select-Object -ExpandProperty OwningProcess "
        "-Unique) -join ','"
    )
    completed = subprocess.run(
        ["powershell.exe", "-NoProfile", "-NonInteractive", "-Command", command],
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        timeout=15,
        check=False,
    )
    if completed.returncode != 0:
        return []
    return sorted(
        {int(value) for value in completed.stdout.strip().split(",") if value.strip().isdigit()}
    )


def _git_snapshot() -> dict[str, Any]:
    def run(*args: str) -> str:
        completed = subprocess.run(
            ["git", *args], cwd=ROOT, capture_output=True, text=True,
            encoding="utf-8", errors="replace", check=False,
        )
        return completed.stdout.strip() if completed.returncode == 0 else ""

    status = run("status", "--porcelain", "--untracked-files=all")
    return {
        "commit": run("rev-parse", "HEAD") or None,
        "branch": run("branch", "--show-current") or None,
        "dirty": bool(status),
        "status_sha256": hashlib.sha256(status.encode("utf-8")).hexdigest(),
    }


def build_dry_run_plan(result: dict[str, Any], *, port: int = 8010) -> dict[str, Any]:
    commands = [str(command) for command in result.get("test_commands") or []]
    tests = result.get("test_results") or []
    results_by_command = {
        str(item.get("command")): int(item.get("exit_code", -1))
        for item in tests
        if isinstance(item, dict)
    }
    tests_passed = bool(commands) and all(
        command in results_by_command and results_by_command[command] == 0
        for command in commands
    )
    safety = result.get("safety_gate_results") or {}
    changed_files = [str(path) for path in result.get("changed_files") or []]
    allowed_scope = bool(safety.get("sensitive_paths_unchanged", False)) and bool(
        safety.get("secret_scan_passed", False)
    )
    pids = _server_pids(port)
    unique_server = len(pids) == 1
    pre_restart = _git_snapshot()
    eligible_after_approval = (
        result.get("status") in SUCCESS_STATUSES
        and tests_passed
        and allowed_scope
        and unique_server
    )
    return {
        "evaluated_at": _now(),
        "mode": "disabled_and_dry_run",
        "run_id": result.get("run_id"),
        "restart_executed": False,
        "deployment_executed": False,
        "operator_approval_required": True,
        "eligible_after_approval": eligible_after_approval,
        "approved_restart": {
            "command": ".\\run.ps1",
            "headless": False,
            "managed_gui_required": True,
            "direct_run_py_forbidden": True,
        },
        "checks": {
            "result_status_successful": result.get("status") in SUCCESS_STATUSES,
            "required_tests_recorded_and_passed": tests_passed,
            "changed_files_recorded": bool(changed_files) or result.get("status") == "NO_CHANGE_NEEDED",
            "allowed_change_scope": allowed_scope,
            "secret_scan_passed": bool(safety.get("secret_scan_passed", False)),
            "server_pid_unambiguous": unique_server,
        },
        "server": {"port": port, "pids": pids, "selected_pid": pids[0] if unique_server else None},
        "pre_restart_snapshot": pre_restart,
        "post_restart_verification": [
            {"method": "GET", "url": f"http://127.0.0.1:{port}/api/system/restart-safety"},
            {"method": "GET", "url": f"http://127.0.0.1:{port}/api/refactor/market-view"},
            {"method": "GET", "url": f"http://127.0.0.1:{port}/account"},
        ],
        "rollback": {
            "automatic": False,
            "procedure": [
                "Keep the pre-restart commit and dirty-status digest.",
                "If a post-restart health check fails, leave live order submission fail-closed.",
                "Restore the reviewed prior artifact only after explicit operator approval.",
            ],
        },
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Evaluate deployment readiness without restarting")
    parser.add_argument("result", type=Path)
    parser.add_argument("--port", type=int, default=8010)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args(argv)
    value = json.loads(args.result.read_text(encoding="utf-8"))
    plan = build_dry_run_plan(value, port=args.port)
    rendered = json.dumps(plan, ensure_ascii=False, indent=2) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        temporary = args.output.with_name(f".{args.output.name}.writing")
        temporary.write_text(rendered, encoding="utf-8")
        os.replace(temporary, args.output)
    print(rendered, end="")
    return 0 if plan["eligible_after_approval"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
