from __future__ import annotations

from datetime import datetime, timezone

from app.schemas.domain import AccountSnapshot, Holding, MarketSnapshot, SourceMetadata


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
        ),
    )
