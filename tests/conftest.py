"""Test isolation for the process-wide, on-disk stores.

Two stores are keyed off a default path under ``data/store`` and are reached
through module-level singletons: the strategy performance store (realized
per-strategy outcomes) and the change-point detector state. Without redirection a
test run would

* write realized outcomes into the operator's real ``data/store``, and
* leak those outcomes between tests — the exit-decision tests read
  ``recent_performance`` from the same store, so a session test that closes a
  winning position silently changes their expected behaviour.

Both are session-scoped and autouse: no test should have to remember this.
"""

from __future__ import annotations

import os

import pytest


@pytest.fixture(scope="session", autouse=True)
def isolated_process_wide_stores(tmp_path_factory):
    root = tmp_path_factory.mktemp("process-stores")
    previous = {
        name: os.environ.get(name)
        for name in ("STRATEGY_PERFORMANCE_STORE_PATH", "CHANGE_POINT_STATE_PATH")
    }
    os.environ["STRATEGY_PERFORMANCE_STORE_PATH"] = str(root / "strategy-performance.sqlite3")
    os.environ["CHANGE_POINT_STATE_PATH"] = str(root / "change-point-state.json")

    # The singletons may already exist if an earlier import touched them.
    from app.graph import change_point
    from app.trading import strategy_performance_store

    strategy_performance_store.reset_default_store()
    change_point.reset_default_detector()
    try:
        yield root
    finally:
        strategy_performance_store.reset_default_store()
        change_point.reset_default_detector()
        for name, value in previous.items():
            if value is None:
                os.environ.pop(name, None)
            else:
                os.environ[name] = value


@pytest.fixture(autouse=True)
def restore_process_environment():
    """Undo process-environment writes made by product code during a test.

    Not every env write comes from a test. ``KisDevelopersApiClient.__init__``
    calls ``load_kis_env_file()``, which copies ``config/secrets/kis.env`` into
    ``os.environ``. That file carries trading-policy keys as well as credentials,
    so merely CONSTRUCTING a KIS client exported ``REALTIME_ALLOW_LOSS_EXIT=true``
    and friends for the rest of the session.

    The result was five order-dependent failures: the exit-policy tests assert
    the *disabled-by-default* behaviour, passed when run alone, and failed in the
    full suite only because an alphabetically earlier dashboard test had built a
    client first. The load itself is correct -- it is setdefault-shaped, so an
    operator's ``run.ps1`` values still win in production -- but a leak across
    test boundaries makes the suite order-dependent, which hides real breakage.

    ``monkeypatch`` already covers tests that set env themselves; this covers the
    ones where the code under test does it.
    """
    snapshot = dict(os.environ)
    try:
        yield
    finally:
        if os.environ != snapshot:
            os.environ.clear()
            os.environ.update(snapshot)


@pytest.fixture(autouse=True)
def clean_strategy_performance_history():
    """Every test starts with an empty realized-outcome history.

    Realized performance now feeds sizing, tuning and election, so a leftover row
    from another test is a behaviour change, not harmless residue.
    """
    from app.trading.strategy_performance_store import default_store

    store = default_store()
    yield
    try:
        import sqlite3
        from contextlib import closing

        with closing(sqlite3.connect(store.path, timeout=15)) as conn:
            conn.execute("delete from strategy_outcomes")
            conn.commit()
        store._cache.clear()  # noqa: SLF001 - test-only cache reset
    except Exception:  # noqa: BLE001 - a store that was never created needs no reset.
        pass
