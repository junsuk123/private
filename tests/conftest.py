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
import tempfile

import pytest

# --------------------------------------------------------------------------- #
# The operator's journals, redirected before ANY product module is imported.
# --------------------------------------------------------------------------- #
# This runs at conftest import, not inside a fixture, and that is the whole point.
# pytest imports the test modules -- and therefore ``app.web`` and friends -- before
# any fixture body executes, and ``app.web`` resolves its audit path at module level.
# A fixture would be too late for exactly the sink that writes most.
#
# What it prevents, measured on 2026-08-19 before the redirect existed: the journals
# a funded account is reconciled against had accumulated 6,162 ``LAB`` order records
# in ``logs/live-orders.jsonl`` and 665 demo-issuer (900001/900002) entries in
# ``logs/audit.jsonl``, written by tests that constructed a ``LiveOrderJournal``,
# ``DecisionLogger`` or ``RiskManager`` without naming a path. Those paths were
# default ARGUMENTS pointing at ``logs/``, so "did not pass a path" meant "wrote to
# production". A synthetic order sitting in the real order journal is not untidiness;
# it is an audit trail that can no longer be trusted to mean what it says.
#
# ``setdefault``, so a deliberate OBAITS_LOG_DIR (a CI artifact directory, say) wins.
os.environ.setdefault(
    "OBAITS_LOG_DIR", tempfile.mkdtemp(prefix="obaits-test-logs-")
)


@pytest.fixture(scope="session", autouse=True)
def isolated_process_wide_stores(tmp_path_factory):
    root = tmp_path_factory.mktemp("process-stores")
    previous = {
        name: os.environ.get(name)
        for name in (
            "STRATEGY_PERFORMANCE_STORE_PATH",
            "CHANGE_POINT_STATE_PATH",
            "TRADING_STATE_DB_PATH",
            "ADAPTIVE_THRESHOLDS_STORE_PATH",
            "DIRECTIONAL_SHADOW_STORE_PATH",
        )
    }
    os.environ["STRATEGY_PERFORMANCE_STORE_PATH"] = str(root / "strategy-performance.sqlite3")
    os.environ["CHANGE_POINT_STATE_PATH"] = str(root / "change-point-state.json")
    # The context/decision/order store. Without this a test run would write real order
    # intents and gate verdicts into the operator's data/store, and the seasonality
    # baselines would accumulate synthetic observations across runs.
    os.environ["TRADING_STATE_DB_PATH"] = str(root / "trading-state.sqlite3")
    # Learned entry policy: adapted thresholds and the edge calibrator. Reached
    # from every algorithm trigger, and it WRITES -- so without redirection a test
    # run edits the strictness a funded account trades on. It also reads, which is
    # not merely untidy: the calibrator had learned enough from the live tape to
    # refuse a trade a signal-engine test asserted, so the suite's result depended
    # on how the real system happened to be doing that morning.
    os.environ["ADAPTIVE_THRESHOLDS_STORE_PATH"] = str(root / "adaptive-thresholds.sqlite3")
    # The directional shadow tape. It is now READ by the cost authority every entry
    # path consults (``app.cost.round_trip``), so leaving it pointed at the operator's
    # store made the round-trip cost -- and therefore every trigger floor and every
    # coverage verdict in the suite -- a function of that morning's live fills.
    os.environ["DIRECTIONAL_SHADOW_STORE_PATH"] = str(root / "directional-shadow.sqlite3")

    from app.cost import round_trip as _round_trip

    _round_trip.reset_caches()

    # The singletons may already exist if an earlier import touched them.
    from app.graph import change_point
    from app.storage import trading_state_store
    from app.technical import adaptive_thresholds
    from app.trading import strategy_performance_store

    strategy_performance_store.reset_default_store()
    change_point.reset_default_detector()
    trading_state_store.reset_default_trading_state_store()
    adaptive_thresholds.reset_default_adaptive_thresholds()
    try:
        yield root
    finally:
        strategy_performance_store.reset_default_store()
        change_point.reset_default_detector()
        trading_state_store.reset_default_trading_state_store()
        adaptive_thresholds.reset_default_adaptive_thresholds()
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


@pytest.fixture(autouse=True)
def relax_promotion_sample_floors(monkeypatch):
    """Promotion SAMPLE-SIZE floors are a deployment policy, not trainer behaviour.

    The trainer fixtures deliberately use tiny synthetic datasets to exercise the
    fitting and registry paths. Production floors (hundreds of validation rows
    across several symbols) would reject those artifacts and make every trainer
    test fail for a reason that has nothing to do with what it is testing.

    Only the SIZE floors are relaxed. The qualitative guards stay live in every
    test: holdout must exist, top-k net must be positive and clear the runtime
    minimum, and its lower bound must be positive. Tests that assert the size
    floors set these variables explicitly.
    """
    monkeypatch.setenv("LIVE_MODEL_PROMOTION_MIN_VALIDATION_ROWS", "0")
    monkeypatch.setenv("LIVE_MODEL_PROMOTION_MIN_VALIDATION_SYMBOLS", "0")
