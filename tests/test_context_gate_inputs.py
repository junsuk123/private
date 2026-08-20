"""The hard trade gates are fail-closed, so the caller has to supply what it knows.

Every hard gate reads ``None`` as "not established" and refuses. That is correct, but it
means a caller that passes nothing produces reason codes — WS_DISCONNECTED,
ACCOUNT_RECONCILIATION_FAIL, TRADING_HALT — which read as findings of fact about a socket,
an account and a venue that were never actually consulted.
"""

from __future__ import annotations

import pytest

import app.web as web
from app.execution.kis_real import _kis_trading_halted


# --------------------------------------------------------------------------- #
# Broker-reported halt
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize(
    ("payload", "expected"),
    [
        ({"trht_yn": "Y"}, True),
        ({"trht_yn": "N"}, False),
        ({"temp_stop_yn": "Y"}, True),
        ({"iscd_stat_cls_code": "58"}, True),   # 거래정지
        ({"iscd_stat_cls_code": "00"}, False),  # 그 외
        ({"iscd_stat_cls_code": "51"}, False),  # 관리종목: restricted, not suspended
        ({"stck_prpr": "72000"}, None),         # field absent entirely
        ({}, None),
    ],
)
def test_halt_is_read_from_the_broker_or_left_unknown(payload, expected) -> None:
    assert _kis_trading_halted(payload) is expected


def test_an_absent_halt_field_is_not_an_all_clear() -> None:
    """``False`` here would turn "the feed did not say" into a pass on a hard gate."""
    assert _kis_trading_halted({"acml_vol": "1000"}) is None


# --------------------------------------------------------------------------- #
# What the refresher supplies
# --------------------------------------------------------------------------- #
def test_the_venue_flag_never_claims_a_halt_from_a_closed_market(monkeypatch) -> None:
    """Closed and halted are different findings, and this source cannot tell them apart."""
    monkeypatch.setattr(web, "_is_live_market_extended_open", lambda group: False)
    assert web._context_venue_halted() is None

    monkeypatch.setattr(web, "_is_live_market_extended_open", lambda group: True)
    assert web._context_venue_halted() is False


def test_the_venue_flag_stays_unknown_when_the_calendar_raises(monkeypatch) -> None:
    def _boom(group):
        raise RuntimeError("calendar unavailable")

    monkeypatch.setattr(web, "_is_live_market_extended_open", _boom)
    assert web._context_venue_halted() is None


def test_reconciled_comes_from_the_broker_cross_check_not_from_having_a_snapshot(
    monkeypatch,
) -> None:
    """Numbers we never compared against the broker are the state the gate refuses."""
    basis = {"krw_cash": 1000.0, "positions": []}
    monkeypatch.setattr(web, "_refresh_live_account_basis_for_auto", lambda: basis)
    monkeypatch.setattr(web, "_last_live_account_basis", lambda: basis)
    assert web._context_account_state().reconciled is None

    basis["orderable_cash_reconciliation"] = {"mismatch": False, "error": None}
    assert web._context_account_state().reconciled is True

    basis["orderable_cash_reconciliation"] = {"mismatch": True, "error": None}
    assert web._context_account_state().reconciled is False

    # A failed check is not a passed one.
    basis["orderable_cash_reconciliation"] = {"mismatch": False, "error": "timeout"}
    assert web._context_account_state().reconciled is None


def test_no_account_basis_yields_no_claim(monkeypatch) -> None:
    monkeypatch.setattr(web, "_refresh_live_account_basis_for_auto", lambda: None)
    monkeypatch.setattr(web, "_last_live_account_basis", lambda: None)
    assert web._context_account_state() is None


# --------------------------------------------------------------------------- #
# Websocket transport state
# --------------------------------------------------------------------------- #
def test_websocket_state_is_unknown_until_the_collector_runs() -> None:
    web._kis_realtime_ws_state.update({"connected": None, "changed_at": None})
    assert web._kis_realtime_websocket_connected() is None


def test_websocket_state_tracks_the_connect_and_return_boundary() -> None:
    web._mark_kis_realtime_ws(True)
    assert web._kis_realtime_websocket_connected() is True
    web._mark_kis_realtime_ws(False)
    assert web._kis_realtime_websocket_connected() is False


def test_the_refresher_supplies_all_three_inputs(monkeypatch) -> None:
    """The bug this file exists for: refresh() used to be called with no arguments."""
    captured: dict[str, object] = {}

    class _Service:
        def refresh(self, **kwargs):
            captured.update(kwargs)
            web._context_refresh_stop.set()

    monkeypatch.setattr(web, "get_context_runtime", lambda: _Service())
    monkeypatch.setattr(web, "_context_account_state", lambda: "ACCOUNT")
    monkeypatch.setattr(web, "_kis_realtime_websocket_connected", lambda: True)
    monkeypatch.setattr(web, "_context_venue_halted", lambda: False)

    web._context_refresh_stop.clear()
    web._context_refresh_loop()

    assert captured == {
        "account": "ACCOUNT",
        "websocket_connected": True,
        "trading_halted": False,
    }
