"""Automatic retraining and promotion of the context GNN checkpoint.

Without a checkpoint the runtime is OFFLINE and every new entry is blocked, so the
recovery has to happen on its own — at startup, and then on a cadence — rather than
waiting for someone to notice and run a script.
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import pytest

import app.web as web
from app.models.gnn_runtime import GnnHealthState
from app.models.temporal_hetero_gnn import (
    SUPERVISED_HEADS,
    TemporalHeteroGnn,
    TemporalHeteroGnnConfig,
)
from app.models.temporal_hetero_gnn_training import evaluate_promotion

CONFIG = TemporalHeteroGnnConfig(max_nodes=24, feature_dim=8, time_steps=3)


class _FakeChild:
    """Enough of Popen for the reader path: it is communicate() that is exercised."""

    def __init__(self, stdout: str, stderr: str = "", returncode: int = 0) -> None:
        self._stdout = stdout
        self._stderr = stderr
        self.returncode = returncode
        self.pid = -1

    def communicate(self, timeout: float | None = None) -> tuple[str, str]:
        return self._stdout, self._stderr

    def poll(self) -> int:
        return self.returncode


# --------------------------------------------------------------------------- #
# Promotion
# --------------------------------------------------------------------------- #
def test_a_challenger_is_promoted_when_no_checkpoint_is_deployed(tmp_path) -> None:
    verdict = evaluate_promotion(
        TemporalHeteroGnn(CONFIG), [], incumbent_path=tmp_path / "absent.npz"
    )
    assert verdict["promote"] is True
    assert verdict["reason"] == "NO_INCUMBENT"


def test_an_unreadable_incumbent_cannot_defend_its_slot(tmp_path) -> None:
    corrupt = tmp_path / "corrupt.npz"
    corrupt.write_bytes(b"not an npz")
    verdict = evaluate_promotion(
        TemporalHeteroGnn(CONFIG), [], incumbent_path=corrupt
    )
    assert verdict["promote"] is True
    assert verdict["reason"].startswith("INCUMBENT_UNREADABLE")


def test_an_equal_challenger_does_not_displace_the_champion(tmp_path) -> None:
    """Ties go to the incumbent: churn costs reproducibility and buys nothing."""
    from tests.test_temporal_hetero_gnn_training import CONFIG as FULL_CONFIG
    from tests.test_temporal_hetero_gnn_training import _examples

    deployed = tmp_path / "latest.npz"
    champion = TemporalHeteroGnn(FULL_CONFIG)
    champion.save_checkpoint(deployed)

    verdict = evaluate_promotion(
        TemporalHeteroGnn(FULL_CONFIG), _examples(4), incumbent_path=deployed
    )
    assert verdict["promote"] is False
    assert verdict["reason"] == "INCUMBENT_RETAINED"
    assert verdict["candidate_loss"] == verdict["incumbent_loss"]


# --------------------------------------------------------------------------- #
# The worker
# --------------------------------------------------------------------------- #
def test_training_runs_once_immediately_rather_than_after_the_first_interval(
    monkeypatch,
) -> None:
    """A restart is exactly when the checkpoint is most likely to be missing."""
    calls: list[int] = []

    def _fake_run() -> dict[str, object]:
        calls.append(1)
        web._temporal_gnn_stop.set()  # one pass only
        return {"promoted": False, "promotion": {"reason": "INCUMBENT_RETAINED"}}

    monkeypatch.setattr(web, "_run_temporal_gnn_training_once", _fake_run)
    web._temporal_gnn_stop.clear()
    web._temporal_gnn_training_loop()

    assert calls == [1], "the loop must train before it waits, not after"


def test_a_promoted_checkpoint_is_reloaded_without_a_server_restart(
    monkeypatch,
) -> None:
    """GnnRuntime.reload is the only exit from a latched OFFLINE, and nothing else
    calls it after construction — so a promotion that skipped it would leave the
    running server ignoring the checkpoint it just published."""
    reloaded: list[str] = []

    monkeypatch.setattr(
        web,
        "_run_temporal_gnn_training_once",
        lambda: (web._temporal_gnn_stop.set(), {"promoted": True, "checkpoint": "x.npz"})[1],
    )
    monkeypatch.setattr(
        web, "_reload_temporal_gnn_runtime", lambda: reloaded.append("DEGRADED") or "DEGRADED"
    )
    web._temporal_gnn_stop.clear()
    web._temporal_gnn_training_loop()

    assert reloaded == ["DEGRADED"]
    assert web._temporal_gnn_heartbeat["promoted"] is True
    assert web._temporal_gnn_heartbeat["health_state"] == "DEGRADED"


def test_a_failing_trainer_is_reported_not_raised(monkeypatch) -> None:
    """학습 실패가 서버를 죽여서는 안 된다."""
    monkeypatch.setattr(
        web,
        "_run_temporal_gnn_training_once",
        lambda: (web._temporal_gnn_stop.set(), {"error": "boom"})[1],
    )
    web._temporal_gnn_stop.clear()
    web._temporal_gnn_training_loop()

    assert web._temporal_gnn_heartbeat["ok"] is False
    assert web._temporal_gnn_heartbeat["error"] == "boom"


def test_a_child_that_writes_no_report_becomes_an_error(monkeypatch) -> None:
    """A crashed trainer must surface its last line, not a silent empty report."""
    import subprocess

    monkeypatch.setattr(
        subprocess, "Popen", lambda *a, **k: _FakeChild("", "Traceback: nope", 1)
    )
    report = web._run_temporal_gnn_training_once()
    assert "Traceback: nope" in str(report["error"])


def test_a_child_report_is_parsed_and_returned(monkeypatch) -> None:
    import subprocess

    monkeypatch.setattr(
        subprocess,
        "Popen",
        lambda *a, **k: _FakeChild(json.dumps({"promoted": True, "checkpoint": "a.npz"})),
    )
    report = web._run_temporal_gnn_training_once()
    assert report["promoted"] is True
    assert report["checkpoint"] == "a.npz"


def test_the_trainer_child_is_capped_and_niced() -> None:
    """The fit saturates every core it is given; the trading loop is latency-sensitive."""
    command = web._temporal_gnn_training_command()
    if os.name != "nt":
        assert "nice" in command[0]
    assert "--limit" in command and "--epochs" in command


# --------------------------------------------------------------------------- #
# The fit must not outlive the server that started it
# --------------------------------------------------------------------------- #
def test_shutdown_kills_the_trainer_child() -> None:
    """A fit runs for hours in its own session, so nothing else will stop it.

    ``run.ps1`` force-kills the app server and the next one trains again on boot, so a
    child left behind accumulates: a few restarts and several fits compete for the CPU
    the trading loop needs.
    """
    import subprocess

    isolation = (
        {"creationflags": subprocess.CREATE_NEW_PROCESS_GROUP}
        if os.name == "nt"
        else {"start_new_session": True}
    )
    child = subprocess.Popen(
        [sys.executable, "-c", "import time; time.sleep(120)"],
        stdout=subprocess.DEVNULL,
        **isolation,
    )
    try:
        web._temporal_gnn_child = child
        web._terminate_temporal_gnn_child()
        assert child.poll() is not None, "the child survived shutdown"
    finally:
        web._temporal_gnn_child = None
        if child.poll() is None:  # pragma: no cover - only on a failed assertion
            child.kill()


def test_terminating_when_nothing_runs_is_harmless() -> None:
    web._temporal_gnn_child = None
    web._terminate_temporal_gnn_child()


def test_stopping_the_worker_ends_the_fit_rather_than_waiting_it_out(monkeypatch) -> None:
    """The worker thread parks inside ``communicate`` and cannot see the stop flag."""
    killed: list[bool] = []
    monkeypatch.setattr(web, "_terminate_temporal_gnn_child", lambda: killed.append(True))
    web._stop_temporal_gnn_training_worker()
    assert killed == [True]


def test_a_second_trainer_refuses_to_run_while_one_holds_the_lock(tmp_path) -> None:
    """flock, because the kernel drops it even when the holder is SIGKILLed."""
    import importlib.util

    spec = importlib.util.spec_from_file_location(
        "trainer_script", Path("scripts/train_temporal_hetero_gnn.py")
    )
    trainer = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(trainer)

    target = tmp_path / "latest.npz"
    with trainer._single_writer(target) as first:
        assert first is True
        with trainer._single_writer(target) as second:
            assert second is False, "two fits must not publish over each other"
    with trainer._single_writer(target) as third:
        assert third is True, "the lock must be released when the holder exits"


def test_a_refusal_that_exits_non_zero_is_not_reported_as_a_healthy_cycle(
    monkeypatch, tmp_path
) -> None:
    """The trainer refuses on thin data and exits 1; the runtime then stays OFFLINE.

    That state blocks every new entry, and the heartbeat used to render it exactly like
    the healthy "challenger lost to the incumbent" case: ok=True, error=None,
    promoted=False, checkpoint=None. An operator reading the dashboard could not tell a
    blocked system from a working one.
    """
    monkeypatch.setattr(
        web,
        "_run_temporal_gnn_training_once",
        lambda: (
            web._temporal_gnn_stop.set(),
            {
                "refused": "only 0 resolved decisions; 40 required. The runtime stays OFFLINE",
                "checkpoint": None,
                "training_examples": 0,
                "exit_code": 1,
            },
        )[1],
    )
    monkeypatch.setattr(web, "DEFAULT_GNN_CHECKPOINT_PATH", tmp_path / "missing.npz")
    web._temporal_gnn_stop.clear()
    web._temporal_gnn_training_loop()

    assert web._temporal_gnn_heartbeat["ok"] is False
    assert web._temporal_gnn_heartbeat["blocks_new_entries"] is True
    assert "OFFLINE" in web._temporal_gnn_heartbeat["refused"]
    assert web._temporal_gnn_heartbeat["training_examples"] == 0


def test_thin_challenger_does_not_block_when_an_incumbent_exists(
    monkeypatch, tmp_path
) -> None:
    checkpoint = tmp_path / "latest.npz"
    checkpoint.write_bytes(b"incumbent")
    monkeypatch.setattr(web, "DEFAULT_GNN_CHECKPOINT_PATH", checkpoint)
    monkeypatch.setattr(
        web,
        "_run_temporal_gnn_training_once",
        lambda: (
            web._temporal_gnn_stop.set(),
            {"refused": "thin challenger", "exit_code": 1, "training_examples": 120},
        )[1],
    )

    web._temporal_gnn_stop.clear()
    web._temporal_gnn_training_loop()

    assert web._temporal_gnn_heartbeat["ok"] is True
    assert web._temporal_gnn_heartbeat["blocks_new_entries"] is False


def test_a_lock_skip_exits_zero_and_stays_a_healthy_cycle(monkeypatch) -> None:
    """`refused` also covers a benign skip: another run held the checkpoint lock.

    That one exits 0 and leaves a working incumbent in place, so it must NOT be
    escalated. The exit code is the only thing telling the two refusals apart.
    """
    monkeypatch.setattr(
        web,
        "_run_temporal_gnn_training_once",
        lambda: (
            web._temporal_gnn_stop.set(),
            {
                "refused": "another training run already holds the checkpoint lock",
                "checkpoint": None,
                "exit_code": 0,
            },
        )[1],
    )
    web._temporal_gnn_stop.clear()
    web._temporal_gnn_training_loop()

    assert web._temporal_gnn_heartbeat["ok"] is True
    assert web._temporal_gnn_heartbeat["blocks_new_entries"] is False


def test_the_child_exit_code_reaches_the_caller(monkeypatch) -> None:
    """A valid report on a non-zero exit must not lose the exit code."""
    import subprocess

    monkeypatch.setattr(
        subprocess,
        "Popen",
        lambda *a, **k: _FakeChild(json.dumps({"refused": "thin data"}), "", 1),
    )
    report = web._run_temporal_gnn_training_once()
    assert report["exit_code"] == 1
    assert report["refused"] == "thin data"
