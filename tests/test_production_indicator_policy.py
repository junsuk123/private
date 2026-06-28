from __future__ import annotations

import sys
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from app.indicators import (  # noqa: E402
    build_sample_indicators,
    build_trusted_indicators_from_markets,
    is_sample_or_hash_indicator,
)
from app.pipeline import build_analysis_context  # noqa: E402
from app.research import ResearchRunResult  # noqa: E402
from app.schemas.domain import MarketSnapshot, SourceMetadata  # noqa: E402


def test_sample_indicators_are_marked_untrusted_for_live_paths() -> None:
    market = _market("AAPL")
    indicators = build_sample_indicators((market,))

    assert is_sample_or_hash_indicator(indicators["AAPL"])


def test_trusted_indicator_builder_uses_measured_market_source_only() -> None:
    market = _market("AAPL")
    indicators = build_trusted_indicators_from_markets((market,))

    assert indicators["AAPL"].source_ids == ("broker:AAPL",)
    assert not is_sample_or_hash_indicator(indicators["AAPL"])
    assert indicators["AAPL"].revenue_growth is None


def test_production_context_does_not_promote_sample_indicators_to_trusted_evidence() -> None:
    research = ResearchRunResult(
        events=(),
        raw_records=(),
        market_snapshots=(_market("AAPL"),),
        macro_metrics=(),
        skipped_sources=(),
        archived_paths=(),
        diagnostics={},
    )

    context = build_analysis_context(research_result=research)

    assert context.indicators
    assert all(not is_sample_or_hash_indicator(indicator) for indicator in context.indicators.values())
    assert all(source_id.startswith("broker:") for indicator in context.indicators.values() for source_id in indicator.source_ids)


def _market(ticker: str) -> MarketSnapshot:
    return MarketSnapshot(
        ticker=ticker,
        market="US",
        company_name=ticker,
        sector="Technology",
        last_price=100.0,
        average_daily_trading_value=5_000_000_000,
        volatility_20d=0.03,
        source=SourceMetadata(
            source_name="broker_api",
            retrieved_at=datetime.now(timezone.utc),
            source_type="broker_api",
            trust_level=5,
            quality_score=0.95,
            is_realtime=True,
            source_id=f"broker:{ticker}",
        ),
    )
