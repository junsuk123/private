"""The scheduled investor-flow refresh: one shared routine, failures visible.

``residual_relative_strength`` treats informed flow as mandatory, and KIS only
reports it per business day. So if this refresh stalls, the stored 30-day window
ages out and the strategy silently returns to being unevaluable — the same class of
failure that hid six strategies for the life of the training set. These tests pin
the behaviours that keep that visible.
"""

from __future__ import annotations

import sqlite3

import pytest

from app.data.investor_flow_collector import (
    krx_symbols_with_bars,
    refresh_investor_flow,
)
from app.data.investor_flow_store import InvestorFlowStore


def _bar_database(tmp_path, rows: dict[str, int]):
    path = tmp_path / "bars.sqlite3"
    conn = sqlite3.connect(path)
    conn.execute("create table realtime_minute_bars (symbol text, minute_start text)")
    for symbol, count in rows.items():
        conn.executemany(
            "insert into realtime_minute_bars (symbol, minute_start) values (?, ?)",
            [(symbol, f"2026-07-31T{index:04d}") for index in range(count)],
        )
    conn.commit()
    conn.close()
    return path


class _FakeClient:
    """Stands in for the KIS client; records calls, can fail per symbol."""

    def __init__(self, *, fail: set[str] | None = None) -> None:
        self.calls: list[str] = []
        self.fail = fail or set()

    def get_domestic_investor_flow(self, symbol: str):
        self.calls.append(symbol)
        if symbol in self.fail:
            raise RuntimeError("broker said no")
        return (
            {
                "symbol": symbol,
                "business_date": "20260730",
                "close_price": 100.0,
                "retail_net_buy_value": -10.0,
                "foreign_net_buy_value": 6.0,
                "institution_net_buy_value": 4.0,
            },
            {
                "symbol": symbol,
                "business_date": "20260731",
                "close_price": 101.0,
                "retail_net_buy_value": -20.0,
                "foreign_net_buy_value": 12.0,
                "institution_net_buy_value": 8.0,
            },
        )


def test_worklist_comes_from_the_bar_store_not_a_hardcoded_universe(tmp_path) -> None:
    """Collecting flow for a symbol the labeller discards is wasted API budget."""
    database = _bar_database(tmp_path, {"005930": 150, "000660": 120, "073240": 10})
    symbols = krx_symbols_with_bars(database, minimum_bars=100)
    assert set(symbols) == {"005930", "000660"}
    assert "073240" not in symbols


def test_worklist_excludes_non_krx_symbols(tmp_path) -> None:
    database = _bar_database(tmp_path, {"005930": 150, "AAPL": 150, "0015G0": 150})
    # Six digits only: US tickers have no KIS investor-flow endpoint, and values
    # like 0015G0 are non-cash instruments.
    assert set(krx_symbols_with_bars(database, minimum_bars=100)) == {"005930"}


def test_refresh_writes_every_eligible_symbol(tmp_path) -> None:
    database = _bar_database(tmp_path, {"005930": 150, "000660": 120})
    store = InvestorFlowStore(tmp_path / "flow.sqlite3")
    client = _FakeClient()

    result = refresh_investor_flow(
        client=client, store=store, database=database, delay_seconds=0.0
    )
    assert set(client.calls) == {"005930", "000660"}
    assert result.symbols_attempted == 2
    assert result.rows_written == 4
    assert result.failures == []
    assert result.coverage["symbols"] == 2


def test_one_bad_symbol_does_not_abort_the_sweep(tmp_path) -> None:
    """A single unreadable symbol must not cost us every other symbol's data."""
    database = _bar_database(tmp_path, {"005930": 150, "000660": 120, "252670": 110})
    store = InvestorFlowStore(tmp_path / "flow.sqlite3")
    client = _FakeClient(fail={"000660"})

    result = refresh_investor_flow(
        client=client, store=store, database=database, delay_seconds=0.0
    )
    assert len(client.calls) == 3, "the sweep must continue past the failure"
    assert result.rows_written == 4  # the two healthy symbols
    assert len(result.failures) == 1
    assert "000660" in result.failures[0]


def test_partial_sweep_reports_an_exact_failure_count(tmp_path) -> None:
    """The truncated list must never make a partial run look complete."""
    database = _bar_database(tmp_path, {f"00{index:04d}": 150 for index in range(30)})
    store = InvestorFlowStore(tmp_path / "flow.sqlite3")
    failing = {f"00{index:04d}" for index in range(25)}
    client = _FakeClient(fail=failing)

    result = refresh_investor_flow(
        client=client, store=store, database=database, delay_seconds=0.0
    )
    payload = result.as_dict()
    assert payload["failed_symbols"] == 25
    assert len(payload["failures"]) == 20, "list is truncated"
    assert payload["failed_symbols"] > len(payload["failures"])


def test_empty_bar_store_is_reported_not_silently_successful(tmp_path) -> None:
    database = _bar_database(tmp_path, {"073240": 5})
    store = InvestorFlowStore(tmp_path / "flow.sqlite3")
    client = _FakeClient()

    result = refresh_investor_flow(
        client=client, store=store, database=database, delay_seconds=0.0
    )
    assert client.calls == []
    assert result.skipped_reason == "NO_KRX_SYMBOLS_WITH_ENOUGH_BARS"
    assert result.rows_written == 0


def test_refresh_is_idempotent_and_corrects_the_current_day(tmp_path) -> None:
    """Today's figures move while the session runs, so a re-read must overwrite."""
    database = _bar_database(tmp_path, {"005930": 150})
    store = InvestorFlowStore(tmp_path / "flow.sqlite3")
    client = _FakeClient()

    refresh_investor_flow(client=client, store=store, database=database, delay_seconds=0.0)
    refresh_investor_flow(client=client, store=store, database=database, delay_seconds=0.0)

    history = store.history("005930")
    assert len(history) == 2, "re-running must not duplicate business days"
    assert store.coverage()["rows"] == 2


def test_missing_bar_database_does_not_raise(tmp_path) -> None:
    assert krx_symbols_with_bars(tmp_path / "nope.sqlite3", minimum_bars=100) == ()


def test_informed_flow_is_foreign_plus_institution(tmp_path) -> None:
    database = _bar_database(tmp_path, {"005930": 150})
    store = InvestorFlowStore(tmp_path / "flow.sqlite3")
    refresh_investor_flow(
        client=_FakeClient(), store=store, database=database, delay_seconds=0.0
    )
    latest = store.history("005930")[-1]
    assert latest.informed_net_buy_value == pytest.approx(20.0)  # 12 + 8
