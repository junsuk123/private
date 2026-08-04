from __future__ import annotations

import os
from datetime import datetime, timezone

from app.schemas.domain import AccountSnapshot, Holding, InvestorFlowSnapshot, MarketSnapshot, SourceMetadata


def collect_sample_account() -> AccountSnapshot:
    return AccountSnapshot(
        cash=1_000_000,
        holdings=(),
        realized_pnl_today=0,
        unrealized_pnl_today=0,
    )


def collect_sample_market(symbols: tuple[str, ...] | None = None) -> tuple[MarketSnapshot, ...]:
    """Build issuer-neutral synthetic markets for explicit offline/demo runs."""
    now = datetime.now(timezone.utc)
    configured = tuple(
        token.strip().upper()
        for token in os.getenv("SAMPLE_MARKET_SYMBOLS", "").split(",")
        if token.strip()
    )
    selected = tuple(symbols or configured or ("900001", "900002"))
    snapshots: list[MarketSnapshot] = []
    for index, ticker in enumerate(dict.fromkeys(selected)):
        trading_value = 650_000_000_000 / (index + 1)
        direction = 1.0 if index % 2 == 0 else -1.0
        snapshots.append(
            MarketSnapshot(
                ticker=ticker,
                market="KOSPI",
                company_name=f"Demo issuer {index + 1}",
                sector="Synthetic",
                last_price=50_000.0 + index * 25_000.0,
                average_daily_trading_value=trading_value,
                volatility_20d=0.026 + index * 0.007,
                source=SourceMetadata("sample_market", now, source_id=f"market-{ticker}"),
                investor_flow=InvestorFlowSnapshot(
                    ticker=ticker,
                    market="KOSPI",
                    foreign_net_buy=direction * 18_000_000_000,
                    institution_net_buy=direction * 9_500_000_000,
                    retail_net_buy=-direction * 21_000_000_000,
                    program_net_buy=direction * 4_000_000_000,
                    volume_change_rate=0.42 - min(index, 4) * 0.04,
                    price_change_rate=direction * 0.018,
                    trading_value=trading_value,
                    observed_at=now,
                    source=SourceMetadata("sample_market_flow", now, source_id=f"flow-{ticker}"),
                ),
            )
        )
    return tuple(snapshots)
