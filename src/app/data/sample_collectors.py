from __future__ import annotations

from datetime import datetime, timezone

from app.schemas.domain import AccountSnapshot, Holding, InvestorFlowSnapshot, MarketSnapshot, SourceMetadata


def collect_sample_account() -> AccountSnapshot:
    return AccountSnapshot(
        cash=1_000_000,
        holdings=(),
        realized_pnl_today=0,
        unrealized_pnl_today=0,
    )


def collect_sample_market() -> tuple[MarketSnapshot, ...]:
    now = datetime.now(timezone.utc)
    return (
        MarketSnapshot(
            ticker="005930",
            market="KOSPI",
            company_name="Samsung Electronics",
            sector="Semiconductor",
            last_price=74_800,
            average_daily_trading_value=650_000_000_000,
            volatility_20d=0.026,
            source=SourceMetadata("sample_market", now, source_id="market-005930"),
            investor_flow=InvestorFlowSnapshot(
                ticker="005930",
                market="KOSPI",
                foreign_net_buy=18_000_000_000,
                institution_net_buy=9_500_000_000,
                retail_net_buy=-21_000_000_000,
                program_net_buy=4_000_000_000,
                volume_change_rate=0.42,
                price_change_rate=0.018,
                trading_value=650_000_000_000,
                observed_at=now,
                source=SourceMetadata("sample_market_flow", now, source_id="flow-005930"),
            ),
        ),
        MarketSnapshot(
            ticker="000660",
            market="KOSPI",
            company_name="SK hynix",
            sector="Semiconductor",
            last_price=198_000,
            average_daily_trading_value=420_000_000_000,
            volatility_20d=0.041,
            source=SourceMetadata("sample_market", now, source_id="market-000660"),
            investor_flow=InvestorFlowSnapshot(
                ticker="000660",
                market="KOSPI",
                foreign_net_buy=7_000_000_000,
                institution_net_buy=-14_000_000_000,
                retail_net_buy=8_000_000_000,
                program_net_buy=-3_500_000_000,
                volume_change_rate=0.28,
                price_change_rate=-0.006,
                trading_value=420_000_000_000,
                observed_at=now,
                source=SourceMetadata("sample_market_flow", now, source_id="flow-000660"),
            ),
        ),
    )
