#!/usr/bin/env python3
"""Start/stop the local news LLM only when the Raspberry Pi has spare capacity."""

from __future__ import annotations

import argparse
import os
import subprocess
import time
import urllib.request
from datetime import datetime, time as dtime
from zoneinfo import ZoneInfo


SERVICE = os.getenv("LLM_OPPORTUNISTIC_SERVICE", "llama-server.service")
APP_HEALTH_URL = os.getenv("LLM_OPPORTUNISTIC_APP_HEALTH_URL", "http://127.0.0.1:8010/api/trade-explanations")
LLM_HEALTH_URL = os.getenv("LLM_OPPORTUNISTIC_LLM_HEALTH_URL", "http://127.0.0.1:8080/v1/models")
START_LOAD = float(os.getenv("LLM_OPPORTUNISTIC_START_LOAD_PER_CORE", "0.35"))
STOP_LOAD = float(os.getenv("LLM_OPPORTUNISTIC_STOP_LOAD_PER_CORE", "0.85"))
APP_MAX_SECONDS = float(os.getenv("LLM_OPPORTUNISTIC_APP_MAX_SECONDS", "1.5"))


def _run(*args: str, check: bool = False) -> subprocess.CompletedProcess[str]:
    return subprocess.run(args, text=True, capture_output=True, check=check)


def _active(service: str) -> bool:
    return _run("systemctl", "is-active", "--quiet", service).returncode == 0


def _market_open(now_utc: datetime | None = None) -> bool:
    now_utc = now_utc or datetime.now(ZoneInfo("UTC"))
    kr_now = now_utc.astimezone(ZoneInfo("Asia/Seoul"))
    if kr_now.weekday() < 5 and dtime(9, 0) <= kr_now.time() <= dtime(15, 30):
        return True
    us_now = now_utc.astimezone(ZoneInfo("America/New_York"))
    if us_now.weekday() < 5 and dtime(9, 30) <= us_now.time() <= dtime(16, 0):
        return True
    return False


def _load_per_core() -> float:
    cores = os.cpu_count() or 1
    return os.getloadavg()[0] / cores


def _http_ok(url: str, timeout: float) -> tuple[bool, float]:
    start = time.monotonic()
    try:
        with urllib.request.urlopen(url, timeout=timeout) as response:
            elapsed = time.monotonic() - start
            return response.status < 500, elapsed
    except Exception:
        return False, time.monotonic() - start


def decide() -> tuple[str, str]:
    active = _active(SERVICE)
    market_open = _market_open()
    load = _load_per_core()
    app_ok, app_seconds = _http_ok(APP_HEALTH_URL, APP_MAX_SECONDS)

    if market_open:
        return ("stop" if active else "keep-stopped", f"market_open load={load:.2f} app={app_ok}/{app_seconds:.2f}s")
    if active and (not app_ok or load > STOP_LOAD):
        return ("stop", f"protect_pi load={load:.2f}>{STOP_LOAD:.2f} app={app_ok}/{app_seconds:.2f}s")
    if not active and app_ok and load <= START_LOAD:
        return ("start", f"idle_enough load={load:.2f}<={START_LOAD:.2f} app={app_seconds:.2f}s")
    if active:
        llm_ok, llm_seconds = _http_ok(LLM_HEALTH_URL, 2.0)
        return ("keep-running", f"running load={load:.2f} app={app_seconds:.2f}s llm={llm_ok}/{llm_seconds:.2f}s")
    return ("keep-stopped", f"not_idle load={load:.2f}>{START_LOAD:.2f} app={app_ok}/{app_seconds:.2f}s")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    action, reason = decide()
    print(f"llm-opportunistic action={action} reason={reason}")
    if args.dry_run:
        return 0
    if action == "start":
        _run("systemctl", "start", SERVICE, check=False)
    elif action == "stop":
        _run("systemctl", "stop", SERVICE, check=False)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
