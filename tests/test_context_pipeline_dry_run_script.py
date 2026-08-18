"""The production dry run is part of the suite, not a thing someone remembers to run.

It is the only test that exercises the real components in the real order, so a regression
that only shows up in the assembled system fails here rather than in production.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "context_pipeline_dry_run.py"


@pytest.fixture(scope="module")
def dry_run(tmp_path_factory) -> dict:
    store = tmp_path_factory.mktemp("dryrun") / "state.sqlite3"
    environment = {
        **os.environ,
        "TRADING_STATE_DB_PATH": str(store),
        # The dry run must never be able to reach a broker, whatever the operator's
        # environment happens to say.
        "LIVE_TRADING_ENABLED": "false",
        "KIS_LIVE_ENABLED": "false",
    }
    environment.pop("PYTHONPATH", None)
    result = subprocess.run(
        [sys.executable, str(SCRIPT), "--store", str(store), "--json"],
        cwd=ROOT,
        env=environment,
        capture_output=True,
        text=True,
        timeout=600,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    return json.loads(result.stdout)


def test_the_dry_run_passes_every_stage(dry_run: dict) -> None:
    failed = [stage["stage"] for stage in dry_run["stages"] if not stage["ok"]]
    assert failed == [], f"failing stages: {failed}"
    assert dry_run["ok"] is True


@pytest.mark.parametrize(
    "stage",
    [
        "calendar_session",
        "live_cycle",
        "health_consistency",
        "idempotency",
        "duplicate_prevention",
        "partial_fill",
        "fill_reconciliation",
        "restart_recovery",
        "unknown_state_on_timeout",
        "position_reconciliation",
        "account_reconciliation",
        "no_orders_submitted",
    ],
)
def test_every_declared_stage_runs(dry_run: dict, stage: str) -> None:
    names = {item["stage"] for item in dry_run["stages"]}
    assert stage in names


def test_no_order_reached_the_broker(dry_run: dict) -> None:
    stage = next(
        item for item in dry_run["stages"] if item["stage"] == "no_orders_submitted"
    )
    assert stage["ok"]


def test_the_run_writes_a_report(dry_run: dict) -> None:
    assert Path(dry_run["report_path"]).exists()


def test_readiness_never_contradicts_module_health(dry_run: dict) -> None:
    stage = next(
        item for item in dry_run["stages"] if item["stage"] == "health_consistency"
    )
    assert stage["ok"]
    if stage["gnn"] == "OFFLINE" or stage["data"] == "STALE":
        assert stage["new_entry_permitted"] is False
        assert stage["final_gate"] == "BLOCK"
