from __future__ import annotations

import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from app.risk.manager import RiskManager, _is_non_common_equity_ticker
from app.schemas.domain import (
    AccountSnapshot,
    Holding,
    MarketSnapshot,
    OrderAction,
    OrderIntent,
    RiskRules,
    SourceMetadata,
)


def _market(ticker: str) -> MarketSnapshot:
    return MarketSnapshot(
        ticker=ticker,
        market="NASDAQ",
        company_name=ticker,
        sector="Tech",
        last_price=10.0,
        average_daily_trading_value=100_000_000_000,
        volatility_20d=0.02,
        source=SourceMetadata("KIS", datetime.now(timezone.utc), source_type="broker_api", trust_level=5, quality_score=1.0, is_realtime=True),
    )


def _intent(ticker: str, action: OrderAction = OrderAction.BUY) -> OrderIntent:
    return OrderIntent(
        ticker=ticker, market="NASDAQ", action=action, suggested_weight=0.02, confidence=0.8,
        valid_until=datetime.now(timezone.utc) + timedelta(hours=1),
        reasoning_summary=("t",), supporting_factors=(), contradicting_factors=(),
        source_data_ids=("s",), strategy_family="unit", signal_name="s",
        expected_exit_price=10.6, target_net_return=0.0,
    )


# --- ticker heuristic --------------------------------------------------------
def test_detects_warrant_unit_right_tickers() -> None:
    assert _is_non_common_equity_ticker("LCFYW")   # warrant (5th letter W)
    assert _is_non_common_equity_ticker("LAFAU")   # unit (5th letter U)
    assert _is_non_common_equity_ticker("ABCDR")   # right (5th letter R)
    assert _is_non_common_equity_ticker("LCFY.WS")
    assert _is_non_common_equity_ticker("ABCD-UN")
    assert _is_non_common_equity_ticker("ABCD.RT")


def test_does_not_flag_common_stocks() -> None:
    assert not _is_non_common_equity_ticker("AAPL")     # common
    assert not _is_non_common_equity_ticker("LCFY")     # the common share of the warrant
    assert not _is_non_common_equity_ticker("MSFT")
    assert not _is_non_common_equity_ticker("005930")   # KRX numeric
    assert not _is_non_common_equity_ticker("005930.KS")
    assert not _is_non_common_equity_ticker("GOOGL")    # 5 letters but ends in L


# --- RiskManager integration -------------------------------------------------
def test_riskmanager_blocks_warrant_buy_by_default() -> None:
    result = RiskManager().validate(_intent("LCFYW"), AccountSnapshot(cash=10_000_000, holdings=()), _market("LCFYW"))
    assert not result.approved
    assert "NON_COMMON_INSTRUMENT_BUY_BLOCKED" in result.rejection_reasons


def test_riskmanager_allows_warrant_sell_of_held_position() -> None:
    # Must still be able to EXIT a warrant we are already stuck holding.
    holding = Holding(ticker="LCFYW", market="NASDAQ", company_name="Locafy Warrant", sector="Tech",
                      quantity=1, average_price=10.0, last_price=10.5)
    account = AccountSnapshot(cash=1_000_000, holdings=(holding,))
    result = RiskManager().validate(_intent("LCFYW", action=OrderAction.SELL), account, _market("LCFYW"))
    assert result.checks["tradable_instrument_type"] is True
    assert "NON_COMMON_INSTRUMENT_BUY_BLOCKED" not in result.rejection_reasons


def test_riskmanager_allows_warrant_buy_when_explicitly_enabled() -> None:
    rules = RiskRules(warrant_unit_buys_allowed=True)
    result = RiskManager(rules).validate(_intent("LCFYW"), AccountSnapshot(cash=10_000_000, holdings=()), _market("LCFYW"))
    assert result.checks["tradable_instrument_type"] is True
    assert "NON_COMMON_INSTRUMENT_BUY_BLOCKED" not in result.rejection_reasons


def test_riskmanager_allows_common_stock_buy() -> None:
    result = RiskManager().validate(_intent("AAPL"), AccountSnapshot(cash=10_000_000, holdings=()), _market("AAPL"))
    assert result.checks["tradable_instrument_type"] is True
