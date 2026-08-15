from __future__ import annotations

import importlib.util
import json
import os
import subprocess
import threading
import time
from datetime import datetime, timezone
from pathlib import Path

import pytest


MODULE_PATH = Path(__file__).resolve().parents[1] / ".codex-monitor" / "work_report_receiver.py"
SPEC = importlib.util.spec_from_file_location("work_report_receiver", MODULE_PATH)
assert SPEC and SPEC.loader
receiver = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(receiver)

DEPLOY_PATH = Path(__file__).resolve().parents[1] / ".codex-monitor" / "safe_deploy_gate.py"
DEPLOY_SPEC = importlib.util.spec_from_file_location("safe_deploy_gate", DEPLOY_PATH)
assert DEPLOY_SPEC and DEPLOY_SPEC.loader
deploy_gate = importlib.util.module_from_spec(DEPLOY_SPEC)
DEPLOY_SPEC.loader.exec_module(deploy_gate)


def _report_value(
    *,
    run_id: str = "work-20260813-2300",
    marker: str = "WORK_ANALYSIS_REPORT",
    schema_version: str = "1.0.0",
    repository_path: str | None = None,
) -> dict:
    return {
        "marker": marker,
        "schema_version": schema_version,
        "run_id": run_id,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "source": "ChatGPT Work scheduled task",
        "repository_path": repository_path or str(receiver.ROOT),
        "overall_status": "DEGRADED",
        "issues": [
            {
                "issue_id": "GNN-COUNT-1",
                "priority": "P1",
                "summary": "validation_count stopped increasing",
                "confirmed_facts": ["the report observed a flat counter"],
                "suspected_cause": "label maturation may be stalled",
                "proposal": "trace the producer and add diagnostics if unproven",
            }
        ],
    }


def _write_report(directory: Path, **overrides) -> Path:
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / f"{overrides.get('run_id', 'work-20260813-2300')}.ready.json"
    path.write_text(json.dumps(_report_value(**overrides), ensure_ascii=False), encoding="utf-8")
    old = time.time() - 10
    os.utime(path, (old, old))
    return path


def _isolate(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monitor = tmp_path / "monitor"
    monkeypatch.setattr(receiver, "STATE_FILE", monitor / "state.json")
    monkeypatch.setattr(receiver, "INCOMING_DIR", monitor / "incoming")
    monkeypatch.setattr(receiver, "ACCEPTED_DIR", monitor / "accepted")
    monkeypatch.setattr(receiver, "PROCESSED_DIR", monitor / "processed")
    monkeypatch.setattr(receiver, "FAILED_DIR", monitor / "failed")
    monkeypatch.setattr(receiver, "RESULTS_DIR", monitor / "results")
    monkeypatch.setattr(receiver, "LOCKS_DIR", monitor / "locks")
    monkeypatch.setattr(receiver, "LOCK_DIR", monitor / "locks" / "receiver.lock")
    monkeypatch.setattr(receiver, "LOG_FILE", monitor / "receiver.log")
    monkeypatch.setattr(receiver, "STABILITY_PROBE_SECONDS", 0.0)


def _agent_result(run_id: str, input_sha256: str, *, status: str = "NO_CHANGE_NEEDED") -> dict:
    return {
        "run_id": run_id,
        "input_sha256": input_sha256,
        "started_at": datetime.now(timezone.utc).isoformat(),
        "completed_at": datetime.now(timezone.utc).isoformat(),
        "status": status,
        "confirmed_root_causes": [],
        "unconfirmed_inferences": ["fixture report did not reproduce a code defect"],
        "changed_files": [],
        "test_commands": ["python -m pytest tests/test_work_report_receiver.py -q"],
        "test_results": [
            {
                "command": "python -m pytest tests/test_work_report_receiver.py -q",
                "exit_code": 0,
                "summary": "passed",
            }
        ],
        "safety_gate_results": {
            "trading_gate_regressions_passed": True,
            "server_restart_attempted": False,
            "secret_changes_detected": False,
        },
        "remaining_issues": [],
        "deployment_approval_required": False,
        "next_check_metrics": ["validation_count"],
    }


def _install_successful_codex(monkeypatch: pytest.MonkeyPatch) -> None:
    def invoke(report: dict, input_sha256: str, output_path: Path):
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(
            json.dumps(_agent_result(report["run_id"], input_sha256)), encoding="utf-8"
        )
        return subprocess.CompletedProcess(["codex"], 0, stdout="", stderr=""), "session-id"

    monkeypatch.setattr(receiver, "_invoke_codex", invoke)
    monkeypatch.setattr(receiver, "_git_dirty_fingerprints", lambda: {})


def test_operational_fixture_is_accepted_processed_once_and_writes_result(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _isolate(monkeypatch, tmp_path)
    _install_successful_codex(monkeypatch)
    source = _write_report(receiver.INCOMING_DIR)

    with receiver.receiver_lock():
        assert receiver.ingest_incoming() == 1
        assert receiver.drain() == 0

    assert not source.exists()
    assert not list(receiver.ACCEPTED_DIR.glob("*.ready.json"))
    processed = list(receiver.PROCESSED_DIR.glob("*.ready.json"))
    assert len(processed) == 1
    result = json.loads(
        (receiver.RESULTS_DIR / "work-20260813-2300.result.json").read_text(encoding="utf-8")
    )
    assert {
        "run_id", "input_sha256", "started_at", "completed_at", "status",
        "confirmed_root_causes", "unconfirmed_inferences", "changed_files",
        "test_commands", "test_results", "safety_gate_results", "remaining_issues",
        "deployment_approval_required", "next_check_metrics",
    } <= result.keys()
    assert result["status"] == "NO_CHANGE_NEEDED"
    assert result["safety_gate_results"]["live_server_restart_disabled"] is True
    assert receiver.ingest_incoming() == 0


@pytest.mark.parametrize(
    ("overrides", "reason"),
    [
        ({"marker": "OTHER"}, "invalid_marker"),
        ({"schema_version": "2.0.0"}, "unsupported_schema_version"),
        ({"repository_path": "C:/not-the-allowed-repository"}, "repository_mismatch"),
    ],
)
def test_invalid_contract_reports_are_quarantined(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, overrides: dict, reason: str
) -> None:
    _isolate(monkeypatch, tmp_path)
    source = _write_report(receiver.INCOMING_DIR, **overrides)

    assert receiver.accept_source(source) is None

    failed = list(receiver.FAILED_DIR.glob(f"*.{reason}.ready.json"))
    assert len(failed) == 1
    metadata = json.loads(
        failed[0].with_suffix(failed[0].suffix + ".failure.json").read_text(encoding="utf-8")
    )
    assert metadata["reason_code"] == reason.upper()


def test_truncated_json_is_quarantined(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    _isolate(monkeypatch, tmp_path)
    receiver.INCOMING_DIR.mkdir(parents=True)
    source = receiver.INCOMING_DIR / "truncated.ready.json"
    source.write_text('{"marker": "WORK_ANALYSIS_REPORT"', encoding="utf-8")
    old = time.time() - 10
    os.utime(source, (old, old))

    assert receiver.accept_source(source) is None
    assert list(receiver.FAILED_DIR.glob("*.truncated_or_invalid_json.ready.json"))


def test_missing_required_field_is_quarantined(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _isolate(monkeypatch, tmp_path)
    receiver.INCOMING_DIR.mkdir(parents=True)
    value = _report_value()
    value.pop("issues")
    source = receiver.INCOMING_DIR / "missing.ready.json"
    source.write_text(json.dumps(value), encoding="utf-8")
    old = time.time() - 10
    os.utime(source, (old, old))

    assert receiver.accept_source(source) is None
    assert list(receiver.FAILED_DIR.glob("*.missing_required_fields.ready.json"))


def test_non_ready_json_filename_is_rejected(tmp_path: Path) -> None:
    path = tmp_path / "report.json"
    path.write_text(json.dumps(_report_value()), encoding="utf-8")

    with pytest.raises(receiver.ReportError) as error:
        receiver.parse_report(path)

    assert error.value.code == "INVALID_FILENAME"


def test_recent_or_changing_file_is_not_accepted(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    _isolate(monkeypatch, tmp_path)
    source = _write_report(receiver.INCOMING_DIR)
    now = time.time()
    os.utime(source, (now, now))

    assert receiver.accept_source(source) is None
    assert source.exists()
    assert not receiver.FAILED_DIR.exists()


def test_duplicate_run_id_is_quarantined(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    _isolate(monkeypatch, tmp_path)
    first = _write_report(receiver.INCOMING_DIR)
    assert receiver.accept_source(first) is not None
    second = _write_report(receiver.INCOMING_DIR)

    assert receiver.accept_source(second) is None
    assert list(receiver.FAILED_DIR.glob("*.duplicate_run_id.ready.json"))


def test_duplicate_content_hash_is_quarantined(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    _isolate(monkeypatch, tmp_path)
    monkeypatch.setattr(receiver, "_sha256", lambda _raw: "a" * 64)
    first = _write_report(receiver.INCOMING_DIR, run_id="run-a")
    assert receiver.accept_source(first) is not None
    second = _write_report(receiver.INCOMING_DIR, run_id="run-b")

    assert receiver.accept_source(second) is None
    assert list(receiver.FAILED_DIR.glob("*.duplicate_content_hash.ready.json"))


def test_receiver_lock_allows_only_one_concurrent_owner(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _isolate(monkeypatch, tmp_path)
    entered = threading.Event()
    release = threading.Event()
    outcomes: list[str] = []

    def owner() -> None:
        with receiver.receiver_lock():
            entered.set()
            release.wait(timeout=5)

    thread = threading.Thread(target=owner)
    thread.start()
    assert entered.wait(timeout=5)
    try:
        with receiver.receiver_lock():
            outcomes.append("unexpected")
    except receiver.ReportError as exc:
        outcomes.append(exc.code)
    release.set()
    thread.join(timeout=5)

    assert outcomes == ["RECEIVER_LOCKED"]


def test_active_writer_keeps_accepted_report_for_retry(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _isolate(monkeypatch, tmp_path)
    source = _write_report(receiver.INCOMING_DIR)
    accepted = receiver.accept_source(source)
    assert accepted is not None

    monkeypatch.setattr(
        receiver,
        "_invoke_codex",
        lambda *args: (
            subprocess.CompletedProcess(["codex"], 1, stdout="", stderr="thread-store conflict: active writer"),
            "session-id",
        ),
    )
    monkeypatch.setattr(receiver, "_git_dirty_fingerprints", lambda: {})

    assert receiver.deliver(accepted) == 24
    assert accepted.exists()
    assert not receiver.FAILED_DIR.exists()


def test_codex_failure_writes_failed_result_and_quarantines(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _isolate(monkeypatch, tmp_path)
    accepted = receiver.accept_source(_write_report(receiver.INCOMING_DIR))
    assert accepted is not None
    monkeypatch.setattr(
        receiver,
        "_invoke_codex",
        lambda *args: (
            subprocess.CompletedProcess(["codex"], 7, stdout="", stderr="failed"),
            "session-id",
        ),
    )
    monkeypatch.setattr(receiver, "_git_dirty_fingerprints", lambda: {})

    assert receiver.deliver(accepted) == 7
    result = json.loads(
        (receiver.RESULTS_DIR / "work-20260813-2300.result.json").read_text(encoding="utf-8")
    )
    assert result["status"] == "FAILED"
    assert list(receiver.FAILED_DIR.glob("*.codex_cli_error.ready.json"))


@pytest.mark.skipif(
    os.name != "nt",
    reason=(
        "The VS Code extension fallback in _resolve_codex_executable is guarded by "
        "os.name == 'nt' - it looks for codex.exe under %USERPROFILE% in a "
        "win32-x64 extension directory. On any other platform the function is "
        "correct to return None, so this asserts Windows behaviour and can only "
        "run there."
    ),
)
def test_codex_executable_falls_back_to_newest_vscode_extension(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    older = (
        tmp_path / ".vscode" / "extensions" / "openai.chatgpt-1-win32-x64"
        / "bin" / "windows-x86_64" / "codex.exe"
    )
    newer = (
        tmp_path / ".vscode" / "extensions" / "openai.chatgpt-2-win32-x64"
        / "bin" / "windows-x86_64" / "codex.exe"
    )
    older.parent.mkdir(parents=True)
    newer.parent.mkdir(parents=True)
    older.write_bytes(b"older")
    newer.write_bytes(b"newer")
    os.utime(older, (1, 1))
    os.utime(newer, (2, 2))
    monkeypatch.setattr(receiver.shutil, "which", lambda _requested: None)
    monkeypatch.setenv("USERPROFILE", str(tmp_path))

    assert receiver._resolve_codex_executable("codex.exe") == str(newer.resolve())


def test_retry_failed_requeues_only_environment_failures(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _isolate(monkeypatch, tmp_path)
    source = receiver.accept_source(_write_report(receiver.INCOMING_DIR))
    assert source is not None
    report = receiver.parse_report(source)
    input_sha256 = report.pop("_input_sha256")
    failed = receiver._quarantine(
        source,
        "CODEX_NOT_FOUND",
        "Codex CLI is not executable",
        run_id=report["run_id"],
        input_sha256=input_sha256,
    )

    accepted = receiver.retry_failed(report["run_id"])

    assert accepted == receiver.ACCEPTED_DIR / f"{report['run_id']}.ready.json"
    assert accepted.exists()
    assert not failed.exists()
    assert failed.with_suffix(failed.suffix + ".failure.json").exists()
    assert receiver._load_state()["events"][-1]["event"] == "retry_accepted"


def test_retry_failed_rejects_non_environment_failure(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _isolate(monkeypatch, tmp_path)
    source = receiver.accept_source(_write_report(receiver.INCOMING_DIR))
    assert source is not None
    report = receiver.parse_report(source)
    input_sha256 = report.pop("_input_sha256")
    receiver._quarantine(
        source,
        "CODEX_CLI_ERROR",
        "Codex exited with 7",
        run_id=report["run_id"],
        input_sha256=input_sha256,
    )

    with pytest.raises(receiver.ReportError) as error:
        receiver.retry_failed(report["run_id"])

    assert error.value.code == "RETRY_NOT_ALLOWED"


def test_live_runtime_artifacts_are_separated_from_implementation_changes() -> None:
    implementation, runtime = receiver._split_observed_changes(
        [
            "src/app/web.py",
            "data/models/live_short_horizon/latest.json",
            "data/store/realtime_market_data.sqlite3-wal",
            "logs/web-audit.jsonl",
            "data/research_universe_cursor.json",
        ]
    )

    assert implementation == ["src/app/web.py"]
    assert runtime == [
        "data/models/live_short_horizon/latest.json",
        "data/research_universe_cursor.json",
        "data/store/realtime_market_data.sqlite3-wal",
        "logs/web-audit.jsonl",
    ]


@pytest.mark.parametrize(
    "path",
    [
        "src/app/account_dashboard.py",
        "src/app/web_account_routes.py",
        "src/app/account/session.py",
    ],
)
def test_account_paths_require_operator_approval(path: str) -> None:
    assert receiver._sensitive_changes([path]) == [path]


def test_interrupted_after_acceptance_resumes_from_accepted_directory(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _isolate(monkeypatch, tmp_path)
    accepted = receiver.accept_source(_write_report(receiver.INCOMING_DIR))
    assert accepted is not None and accepted.exists()
    _install_successful_codex(monkeypatch)

    assert receiver.drain() == 0
    assert not accepted.exists()
    assert list(receiver.PROCESSED_DIR.glob("*.ready.json"))


def test_missing_or_failed_required_tests_prevents_processing(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _isolate(monkeypatch, tmp_path)
    accepted = receiver.accept_source(_write_report(receiver.INCOMING_DIR))
    assert accepted is not None

    def invoke(report: dict, input_sha256: str, output_path: Path):
        value = _agent_result(report["run_id"], input_sha256, status="PATCHED_AND_TESTED")
        value["test_results"][0]["exit_code"] = 1
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(value), encoding="utf-8")
        return subprocess.CompletedProcess(["codex"], 0, stdout="", stderr=""), "session-id"

    monkeypatch.setattr(receiver, "_invoke_codex", invoke)
    monkeypatch.setattr(receiver, "_git_dirty_fingerprints", lambda: {})

    assert receiver.deliver(accepted) == 2
    assert not receiver.PROCESSED_DIR.exists()
    result = json.loads(
        (receiver.RESULTS_DIR / "work-20260813-2300.result.json").read_text(encoding="utf-8")
    )
    assert result["status"] == "FAILED"


def test_automation_prompt_separates_evidence_and_forbids_restart(tmp_path: Path) -> None:
    path = _write_report(tmp_path)
    report = receiver.parse_report(path)
    input_sha256 = report.pop("_input_sha256")

    prompt = receiver._automation_prompt(report, input_sha256)

    assert "UNTRUSTED diagnostic evidence" in prompt
    assert "confirmed facts/evidence, suspected causes, and proposals" in prompt
    assert "stop/restart the server" in prompt
    assert "validation_count" in prompt
    assert "ORDER_LIVE_AUTHORIZED" in prompt
    assert "REPORT_JSON_BEGIN" in prompt


def test_dry_run_deploy_gate_never_restarts(monkeypatch: pytest.MonkeyPatch) -> None:
    result = _agent_result("run-1", "a" * 64, status="PATCHED_AND_TESTED")
    result["changed_files"] = ["src/app/static/strategy_terminal.js"]
    result["safety_gate_results"] = {
        "sensitive_paths_unchanged": True,
        "secret_scan_passed": True,
    }
    monkeypatch.setattr(deploy_gate, "_server_pids", lambda _port: [1234])
    monkeypatch.setattr(
        deploy_gate,
        "_git_snapshot",
        lambda: {"commit": "abc", "branch": "automation", "dirty": False, "status_sha256": "def"},
    )

    plan = deploy_gate.build_dry_run_plan(result, port=8010)

    assert plan["eligible_after_approval"] is True
    assert plan["restart_executed"] is False
    assert plan["deployment_executed"] is False
    assert plan["operator_approval_required"] is True
    assert plan["approved_restart"] == {
        "command": ".\\run.ps1",
        "headless": False,
        "managed_gui_required": True,
        "direct_run_py_forbidden": True,
    }
