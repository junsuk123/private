from __future__ import annotations

import pytest

from app.portfolio import build_portfolio_report, valuation_complete
from app.schemas.domain import AccountSnapshot, Holding


def _holding(ticker="AAPL", market="NASDAQ", quantity=1, price=100, sector="Technology"):
    return Holding(ticker, market, ticker, sector, quantity, price, price)


def test_mixed_holdings_and_cash_share_one_measured_currency():
    account = AccountSnapshot(cash=100_000, holdings=(_holding(), _holding("005930", "KRX", price=60_000)), cash_by_currency={"KRW": 100_000, "USD": 100}, fx_rate_by_currency={"USD": 1400}, total_equity_krw=440_000)
    report = build_portfolio_report(account)
    assert valuation_complete(account)
    assert report.equity == 440_000
    assert report.position_weights["AAPL"] == pytest.approx(140_000 / 440_000)
    assert report.position_weights["005930"] == pytest.approx(60_000 / 440_000)
    assert report.sector_weights["Technology"] == pytest.approx(200_000 / 440_000)
    assert report.cash_weight == pytest.approx(240_000 / 440_000)


def test_missing_fx_is_explicitly_incomplete_and_cannot_undercount_positions():
    account = AccountSnapshot(cash=100_000, holdings=(_holding(),), total_equity_krw=240_000)
    report = build_portfolio_report(account)
    assert not valuation_complete(account)
    assert report.position_weights["AAPL"] >= 1
    assert report.sector_weights["Technology"] >= 1
    assert report.cash_weight == 0


def test_single_native_currency_needs_no_fx():
    usd = AccountSnapshot(cash=100, holdings=(_holding(),), base_currency="USD", realized_pnl_today=10)
    report = build_portfolio_report(usd)
    assert valuation_complete(usd)
    assert report.equity == 200
    assert report.position_weights["AAPL"] == report.cash_weight == .5
    assert report.daily_pnl_ratio == .05
    krw = AccountSnapshot(cash=100_000, holdings=(_holding("005930", "KRX", price=100_000),))
    report = build_portfolio_report(krw)
    assert valuation_complete(krw)
    assert report.equity == 200_000
    assert report.cash_weight == .5


def test_explicit_zero_cash_is_not_replaced_and_zero_foreign_balance_needs_no_fx():
    account = AccountSnapshot(cash=100_000, holdings=(_holding("005930", "KRX", price=100_000),), cash_by_currency={"KRW": 0, "USD": 0})
    report = build_portfolio_report(account)
    assert valuation_complete(account)
    assert report.equity == 100_000
    assert report.cash_weight == 0


def test_base_usd_pnl_converts_when_report_total_is_krw():
    account = AccountSnapshot(cash=100, holdings=(_holding(),), base_currency="USD", total_equity_krw=280_000, realized_pnl_today=10, fx_rate_by_currency={"USD": 1400})
    report = build_portfolio_report(account)
    assert valuation_complete(account)
    assert report.daily_pnl_ratio == .05


@pytest.mark.parametrize("bad", [float("nan"), float("inf"), -1])
def test_invalid_fx_stays_unknown(bad):
    account = AccountSnapshot(cash=100_000, holdings=(_holding(),), total_equity_krw=240_000, fx_rate_by_currency={"USD": bad})
    assert not valuation_complete(account)
    assert build_portfolio_report(account).position_weights["AAPL"] == 1


def test_zero_total_does_not_resurrect_old_holdings():
    account = AccountSnapshot(cash=100_000, holdings=(_holding("005930", "KRX", price=100_000),), total_equity_krw=0)
    assert build_portfolio_report(account).equity == 0
