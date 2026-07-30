from __future__ import annotations

import sys
from datetime import date
from pathlib import Path
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from app.execution.kis_real import KisApiError, KisDevelopersApiClient


def _client(get_response: dict) -> KisDevelopersApiClient:
    """Build a KIS client without network/env for parsing tests."""
    client = KisDevelopersApiClient.__new__(KisDevelopersApiClient)
    client.paper = False
    client.credentials = SimpleNamespace(account_no="12345678", account_product_code="01")
    client._get = lambda path, tr_id, params: get_response  # type: ignore[attr-defined]
    return client


def test_realized_pnl_reads_summary_total() -> None:
    resp = {
        "rt_cd": "0",
        "output1": [{"pdno": "005930", "rlzt_pfls": "10000"}],
        "output2": {"tot_rlzt_pfls": "123456", "tot_fee": "300"},
    }
    client = _client(resp)
    assert client.get_domestic_realized_pnl(date(2026, 7, 2), date(2026, 7, 2)) == 123456.0


def test_realized_pnl_handles_comma_and_alt_field() -> None:
    resp = {"rt_cd": "0", "output2": [{"rlzt_pfls": "1,250,000"}]}
    client = _client(resp)
    assert client.get_domestic_realized_pnl("20260702", "20260702") == 1_250_000.0


def test_realized_pnl_falls_back_to_row_sum_when_summary_missing() -> None:
    resp = {
        "rt_cd": "0",
        "output1": [
            {"pdno": "005930", "rlzt_pfls": "5000"},
            {"pdno": "000660", "rlzt_pfls": "-1500"},
        ],
        "output2": {},
    }
    client = _client(resp)
    assert client.get_domestic_realized_pnl(date(2026, 7, 2), date(2026, 7, 2)) == 3500.0


def test_realized_pnl_raises_on_error_response() -> None:
    resp = {"rt_cd": "1", "msg1": "lookup failed"}
    client = _client(resp)
    try:
        client.get_domestic_realized_pnl(date(2026, 7, 2), date(2026, 7, 2))
        raise AssertionError("expected KisApiError")
    except KisApiError:
        pass


def test_overseas_realized_pnl_reads_summary_total() -> None:
    resp = {
        "rt_cd": "0",
        "output1": [{"ovrs_pdno": "AAPL", "ovrs_rlzt_pfls_amt": "5000"}],
        "output2": {"ovrs_rlzt_pfls_tot_amt": "77777", "smtl_fee1": "10"},
    }
    client = _client(resp)
    assert client.get_overseas_realized_pnl(date(2026, 7, 2), date(2026, 7, 2)) == 77777.0


def test_overseas_realized_pnl_falls_back_to_row_sum() -> None:
    resp = {
        "rt_cd": "0",
        "output1": [
            {"ovrs_pdno": "AAPL", "ovrs_rlzt_pfls_amt": "12,000"},
            {"ovrs_pdno": "MSFT", "ovrs_rlzt_pfls_amt": "-2000"},
        ],
        "output2": {},
    }
    client = _client(resp)
    assert client.get_overseas_realized_pnl("20260702", "20260702") == 10_000.0


def test_overseas_realized_pnl_raises_on_error_response() -> None:
    resp = {"rt_cd": "1", "msg1": "overseas lookup failed"}
    client = _client(resp)
    try:
        client.get_overseas_realized_pnl(date(2026, 7, 2), date(2026, 7, 2))
        raise AssertionError("expected KisApiError")
    except KisApiError:
        pass


def test_overseas_period_profit_requests_krw_and_all_exchanges() -> None:
    captured: dict = {}
    client = _client({"rt_cd": "0", "output2": {"ovrs_rlzt_pfls_tot_amt": "0"}})
    client._get = lambda path, tr_id, params: captured.update(  # type: ignore[attr-defined]
        {"path": path, "tr_id": tr_id, "params": params}
    ) or {"rt_cd": "0", "output2": {"ovrs_rlzt_pfls_tot_amt": "0"}}
    client.get_overseas_realized_pnl("20260702", "20260702")
    assert captured["tr_id"] == "TTTS3039R"
    assert captured["path"].endswith("/overseas-stock/v1/trading/inquire-period-profit")
    # KRW mode so amounts need no FX conversion; blank exchange/currency = all.
    # Live KIS returns foreign-currency values for 01 and KRW-converted
    # settlement values for 02.
    assert captured["params"]["WCRC_FRCR_DVSN_CD"] == "02"
    assert captured["params"]["OVRS_EXCG_CD"] == ""
    assert captured["params"]["CRCY_CD"] == ""


def test_account_basis_passes_realized_pnl_through() -> None:
    import app.web as web

    connection = {
        "account_checked": True,
        "krw_cash": 1_000_000.0,
        "equity": 1_000_000.0,
        "cash_by_currency": {"KRW": 1_000_000.0},
        "positions": [],
        "realized_pnl_today_krw": 42_000.0,
    }
    basis = web._account_basis_from_kis_connection(connection)
    assert basis is not None
    assert basis["realized_pnl_today_krw"] == 42_000.0


def test_dashboard_surfaces_realized_pnl_from_status() -> None:
    from app.account_dashboard import AccountDashboardService

    def status_provider() -> dict:
        return {
            "account_checked": True,
            "basis_source": "kis_live_account",
            "krw_cash": 1_000_000.0,
            "equity": 1_000_000.0,
            "cash_by_currency": {"KRW": 1_000_000.0},
            "positions": [],
            "realized_pnl_today_krw": 88_800.0,
        }

    service = AccountDashboardService(status_provider=status_provider, logs_provider=lambda: {})
    dashboard = service.build_dashboard(persist=False)
    assert dashboard["snapshot"]["realized_pnl_today_krw"] == 88_800.0
