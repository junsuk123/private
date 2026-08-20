"""Cross-process worker liveness.

The property under test is the one that was wrong in production: an observer
process that runs no workers must report the SYSTEM's state, not its own.
"""
from __future__ import annotations

from datetime import datetime, timedelta, timezone

from app.monitoring import worker_heartbeat


def test_absent_mark_is_not_alive(tmp_path):
    assert worker_heartbeat.is_alive("trading_engine", directory=tmp_path) is False


def test_a_fresh_mark_is_alive(tmp_path):
    worker_heartbeat.record("trading_engine", directory=tmp_path)
    assert worker_heartbeat.is_alive("trading_engine", directory=tmp_path) is True


def test_a_stale_mark_expires(tmp_path):
    """A SIGKILLed process leaves its last mark behind; it must stop counting."""
    worker_heartbeat.record("trading_engine", directory=tmp_path)
    later = datetime.now(timezone.utc) + timedelta(seconds=10_000)
    assert (
        worker_heartbeat.is_alive("trading_engine", directory=tmp_path, now=later) is False
    )


def test_observer_without_the_worker_reports_the_system(tmp_path):
    """The exact production defect: read-only instance said 'engine stopped'."""
    worker_heartbeat.record("trading_engine", directory=tmp_path)
    assert (
        worker_heartbeat.running("trading_engine", False, directory=tmp_path) is True
    )


def test_local_thread_wins_without_any_mark(tmp_path):
    """Direct knowledge needs no freshness argument."""
    assert worker_heartbeat.running("trading_engine", True, directory=tmp_path) is True


def test_no_mark_and_no_thread_is_stopped(tmp_path):
    assert worker_heartbeat.running("trading_engine", False, directory=tmp_path) is False


def test_detail_round_trips(tmp_path):
    worker_heartbeat.record(
        "trading_engine", detail={"cycles": 445, "errors": 0}, directory=tmp_path
    )
    mark = worker_heartbeat.read("trading_engine", directory=tmp_path)
    assert mark is not None
    assert mark["detail"]["cycles"] == 445
    assert mark["pid"] > 0


def test_workers_do_not_collide(tmp_path):
    worker_heartbeat.record("trading_engine", directory=tmp_path)
    assert worker_heartbeat.is_alive("model_training", directory=tmp_path) is False


def test_unreadable_mark_is_not_alive(tmp_path):
    path = tmp_path / "trading_engine.json"
    path.write_text("{not json", encoding="utf-8")
    assert worker_heartbeat.is_alive("trading_engine", directory=tmp_path) is False


def test_record_never_raises_on_a_bad_directory(tmp_path):
    blocker = tmp_path / "blocked"
    blocker.write_text("i am a file", encoding="utf-8")
    worker_heartbeat.record("trading_engine", directory=blocker)
    assert worker_heartbeat.is_alive("trading_engine", directory=blocker) is False
