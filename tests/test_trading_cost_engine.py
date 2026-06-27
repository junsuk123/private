from __future__ import annotations

from datetime import datetime, timedelta, timezone

from app.backtesting import StreamingAcceleratedDemo, TimeMode, TimeScalerConfig
from app.cost import TradingCostEngine
from app.risk import RiskManager
from app.schemas.domain import (
    AccountSnapshot,
    MarketSnapshot,
    OrderAction,
    OrderIntent,
    SourceMetadata,
)


def test_break_even_domestic_stock_krx_reflects_fees_tax_and_slippage() -> None:
    cost = TradingCostEngine().estimate(
        symbol="005930",
        market="KR",
        venue="KRX",
        instrument_type="domestic_stock",
        entry_price=10_000,
        expected_exit_price=10_100,
        quantity=10,
        target_net_return=0.0,
    )

    assert cost.break_even_return > 0
    assert cost.break_even_return > 0.002
    assert cost.sell_tax > cost.sell_fee
    assert cost.net_expected_return < cost.gross_expected_return


def test_reject_positive_gross_negative_net_after_costs() -> None:
    cost = TradingCostEngine().estimate(
        symbol="005930",
        market="KR",
        venue="KRX",
        instrument_type="domestic_stock",
        entry_price=10_000,
        expected_exit_price=10_010,
        quantity=10,
        target_net_return=0.0,
    )

    assert cost.gross_expected_return > 0
    assert cost.net_expected_return <= 0
    assert not cost.tradable
    assert cost.reject_reason == "NET_RETURN_NOT_POSITIVE"


def test_required_exit_price_for_target_net_return_exceeds_break_even() -> None:
    cost = TradingCostEngine().estimate(
        symbol="005930",
        market="KR",
        venue="NXT",
        instrument_type="domestic_stock",
        entry_price=10_000,
        expected_exit_price=10_500,
        quantity=10,
        target_net_return=0.01,
    )

    assert cost.required_exit_price > cost.break_even_exit_price
    assert cost.net_expected_return > 0.01


def test_risk_manager_records_cost_breakdown_and_blocks_unprofitable_buy() -> None:
    now = datetime.now(timezone.utc)
    source = SourceMetadata(
        source_name="KIS broker quote",
        retrieved_at=now,
        observed_at=now,
        source_type="broker_api",
        trust_level=5,
        quality_score=1.0,
        is_realtime=True,
    )
    market = MarketSnapshot(
        ticker="005930",
        market="KR",
        company_name="Samsung Electronics",
        sector="Semiconductor",
        last_price=10_000,
        average_daily_trading_value=100_000_000_000,
        volatility_20d=0.02,
        source=source,
    )
    intent = OrderIntent(
        ticker="005930",
        market="KR",
        action=OrderAction.BUY,
        suggested_weight=0.01,
        confidence=0.01,
        valid_until=now + timedelta(hours=1),
        reasoning_summary=("low edge",),
        supporting_factors=("test",),
        contradicting_factors=(),
        source_data_ids=("quote",),
        strategy_family="unit_test",
        signal_name="low_edge",
        expected_exit_price=10_010,
        expected_holding_minutes=60,
        gross_expected_return=0.001,
    )

    result = RiskManager().validate(intent, AccountSnapshot(cash=10_000_000, holdings=()), market)

    assert not result.approved
    assert "cost_breakdown" in result.metadata
    assert "NET_RETURN_NOT_POSITIVE" in result.rejection_reasons


def test_streaming_demo_records_domestic_trade_costs() -> None:
    demo = StreamingAcceleratedDemo(
        config=TimeScalerConfig(mode=TimeMode.REALTIME, acceleration_factor=1.0),
        period_minutes=1,
        initial_cash=10_000_000,
        seed=42,
        tickers=("005930.KS",),
    )
    demo.initialize()
    demo._holdings["005930.KS"] = 3
    demo._average_cost_by_ticker["005930.KS"] = 60_000
    demo._holding_currency_by_ticker["005930.KS"] = "KRW"

    result = demo.run_step()

    assert result is not None
    domestic_sells = [trade for trade in result.trades_in_step if trade.ticker == "005930.KS" and trade.side == "SELL"]
    assert domestic_sells
    assert domestic_sells[0].trading_cost > 0
    assert domestic_sells[0].net_value < domestic_sells[0].value
