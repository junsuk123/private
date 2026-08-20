"""Journal paths must be redirectable, so a test run cannot write to the real ones.

The audit trail a funded account is reconciled against had 6,162 ``LAB`` order
records in ``logs/live-orders.jsonl`` and 665 demo-issuer entries in
``logs/audit.jsonl``. None came from trading. They came from tests constructing a
``LiveOrderJournal``, a ``DecisionLogger`` or a ``RiskManager`` without naming a
path, because the paths were default ARGUMENTS pointing at ``logs/`` -- so "did not
pass a path" silently meant "write to production".

These tests pin the two properties that fix it: the default resolves through
``OBAITS_LOG_DIR``, and it resolves on CALL rather than being bound at import.
"""

from __future__ import annotations

import os
from pathlib import Path

import pytest

from app.audit import AuditLogger, log_path
from app.audit.logger import DEFAULT_LOG_DIR


def test_the_suite_is_not_pointed_at_the_operator_log_directory() -> None:
    """conftest must have redirected the whole set before any module was imported."""
    active = os.environ.get("OBAITS_LOG_DIR")

    assert active, "OBAITS_LOG_DIR is unset, so the defaults resolve into logs/"
    assert Path(active).resolve() != Path(DEFAULT_LOG_DIR).resolve()


def test_default_resolves_under_the_configured_directory(monkeypatch, tmp_path) -> None:
    monkeypatch.setenv("OBAITS_LOG_DIR", str(tmp_path))

    assert log_path("live-orders.jsonl") == tmp_path / "live-orders.jsonl"


def test_unset_directory_keeps_the_production_default(monkeypatch) -> None:
    """Production behaviour must not depend on the variable being present."""
    monkeypatch.delenv("OBAITS_LOG_DIR", raising=False)

    assert log_path("audit.jsonl") == Path(DEFAULT_LOG_DIR) / "audit.jsonl"


@pytest.mark.parametrize("value", ["", "   "])
def test_a_blank_directory_is_not_treated_as_the_filesystem_root(
    monkeypatch, value
) -> None:
    """An empty value must fall back, not resolve to ``/live-orders.jsonl``."""
    monkeypatch.setenv("OBAITS_LOG_DIR", value)

    assert log_path("live-orders.jsonl") == Path(DEFAULT_LOG_DIR) / "live-orders.jsonl"


def test_resolution_happens_on_call_not_at_import(monkeypatch, tmp_path) -> None:
    """The original bug in one assertion.

    A default argument is evaluated once, when the module is imported, so no amount of
    later environment setting could move it. Re-resolving per call is what makes the
    redirect work regardless of import order.
    """
    monkeypatch.setenv("OBAITS_LOG_DIR", str(tmp_path / "first"))
    first = log_path("decision-log.jsonl")
    monkeypatch.setenv("OBAITS_LOG_DIR", str(tmp_path / "second"))
    second = log_path("decision-log.jsonl")

    assert first != second


# -- the sinks that actually leaked ------------------------------------------- #

def _writes_under(directory: Path) -> bool:
    return any(directory.rglob("*.jsonl"))


def test_live_order_journal_default_goes_to_the_redirected_directory(
    monkeypatch, tmp_path
) -> None:
    monkeypatch.setenv("OBAITS_LOG_DIR", str(tmp_path))
    from app.execution.live_order_journal import LiveOrderJournal

    LiveOrderJournal().record("live_order_submitted", {"ticker": "LAB"})

    assert (tmp_path / "live-orders.jsonl").exists()
    assert _writes_under(tmp_path)


def test_decision_logger_default_goes_to_the_redirected_directory(
    monkeypatch, tmp_path
) -> None:
    monkeypatch.setenv("OBAITS_LOG_DIR", str(tmp_path))
    from app.trading.decision_logger import DecisionLogger

    assert DecisionLogger().path == tmp_path / "decision-log.jsonl"


def test_risk_manager_without_a_logger_does_not_write_to_production(
    monkeypatch, tmp_path
) -> None:
    """The fallback logger was the least visible of the three leaks."""
    monkeypatch.setenv("OBAITS_LOG_DIR", str(tmp_path))
    from app.risk.manager import RiskManager

    assert RiskManager().audit_logger.path == tmp_path / "principal-protection.jsonl"


def test_an_explicit_path_still_wins(tmp_path) -> None:
    """Redirection must not override a caller that named its own file."""
    from app.execution.live_order_journal import LiveOrderJournal

    explicit = tmp_path / "named.jsonl"
    LiveOrderJournal(explicit).record("live_order_submitted", {"ticker": "005930"})

    assert explicit.exists()


def test_audit_logger_still_redacts_after_the_change(tmp_path) -> None:
    """A path refactor must not disturb the redaction the journal exists for."""
    path = tmp_path / "audit.jsonl"
    AuditLogger(path).record("kis_call", {"app_key": "AAAA", "ticker": "005930"})

    body = path.read_text(encoding="utf-8")
    assert "AAAA" not in body
    assert "005930" in body


def test_every_default_path_entry_point_works_with_no_argument(monkeypatch, tmp_path):
    """Calling each sink with no path must resolve, not crash.

    Guards the regression this refactor actually caused: the signatures were changed
    from ``path = "logs/..."`` to ``path = None`` while one body kept doing
    ``Path(path)``, which raised on ``None``. It surfaced seven files away, in the US
    watchlist tests, because the broken helper was called transitively -- so the cheap
    check is to invoke every one of them here with the argument omitted.
    """
    monkeypatch.setenv("OBAITS_LOG_DIR", str(tmp_path))

    from app import web
    from app.execution.live_order_journal import LiveOrderJournal
    from app.risk.manager import RiskManager
    from app.storage.execution_quality_store import ExecutionQualityStore
    from app.trading.decision_logger import DecisionLogger

    assert LiveOrderJournal().audit.path == tmp_path / "live-orders.jsonl"
    assert DecisionLogger().path == tmp_path / "decision-log.jsonl"
    assert ExecutionQualityStore().path == tmp_path / "execution-quality.jsonl"
    assert RiskManager().audit_logger.path == tmp_path / "principal-protection.jsonl"
    # These two only read, but a reader pointed somewhere other than the writer is a
    # worse failure than either being wrong alone.
    assert web._recent_live_buy_tickers() == set()
    assert web._live_order_journal_snapshot()["path"] == str(
        tmp_path / "live-orders.jsonl"
    )
