"""Point-in-time indicator provenance and market applicability relations.

This graph is a data contract, independent of learned GNN weights. An exchange
close or a daily macro release can be slow context; neither is a live quote.
Unknown providers, future timestamps and stale readings cannot become evidence.
"""
from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from typing import Any, Iterable


@dataclass(frozen=True)
class IndicatorContract:
    origin_market: str
    targets: tuple[str, ...]
    providers: tuple[str, ...]
    max_age_seconds: float
    role: str


# These are adapter identifiers implemented in public_collectors.py, not
# promises that the corresponding source is enabled, licensed or realtime.
_DAILY_PRICE_PROVIDERS = ("stooq", "yahoo_chart")
_US_EQUITY = IndicatorContract("US", ("US", "KR"), ("kis", "kis_realtime", "nasdaq", "nyse", "fred", "fred_public_csv", "alpha_vantage_daily", *_DAILY_PRICE_PROVIDERS), 86400, "equity_reference")
_US_RATES = IndicatorContract("US", ("US", "KR"), ("fred", "fred_public_csv", "us_treasury"), 259200, "rates_reference")
_COMMODITY = IndicatorContract("GLOBAL", ("US", "KR"), ("fred", "fred_public_csv", *_DAILY_PRICE_PROVIDERS), 86400, "commodity_reference")
_FUTURES = IndicatorContract("US", ("US", "KR"), ("cme", "kis", "kis_realtime"), 3600, "futures_reference")
_CONTRACTS = {
    **{name: _US_EQUITY for name in ("SP500", "NASDAQ", "DOW", "RUSSELL2000", "SOX", "NVDA", "AMD", "AVGO", "MU", "TSM", "INTC")},
    "KOSPI": IndicatorContract("KR", ("KR",), ("kis", "kis_realtime", "krx"), 300, "local_index"),
    "KOSDAQ": IndicatorContract("KR", ("KR",), ("kis", "kis_realtime", "krx"), 300, "local_index"),
    "USDKRW": IndicatorContract("GLOBAL", ("KR",), ("ecos", "bok", "kis", "fred", "fred_public_csv"), 259200, "fx_reference"),
    "KR_BASE_RATE": IndicatorContract("KR", ("KR",), ("ecos", "bok"), 6048000, "rates_reference"),
    "VIX": IndicatorContract("US", ("US", "KR"), ("cboe", "fred", "fred_public_csv", *_DAILY_PRICE_PROVIDERS), 259200, "volatility_reference"),
    "DXY": IndicatorContract("GLOBAL", ("US", "KR"), ("fred", "fred_public_csv", "ice"), 259200, "fx_reference"),
    "US2Y": _US_RATES, "US10Y": _US_RATES,
    **{name: _COMMODITY for name in ("WTI", "GOLD", "COPPER")},
    "NIKKEI": IndicatorContract("JP", ("US", "KR"), ("fred", "fred_public_csv", *_DAILY_PRICE_PROVIDERS), 86400, "asia_equity_reference"),
    "HANGSENG": IndicatorContract("HK", ("US", "KR"), _DAILY_PRICE_PROVIDERS, 86400, "asia_equity_reference"),
    "CSI300": IndicatorContract("CN", ("US", "KR"), _DAILY_PRICE_PROVIDERS, 86400, "asia_equity_reference"),
    **{name: _FUTURES for name in ("ES", "NQ", "YM", "US_INDEX_FUTURES")},
}


@dataclass(frozen=True)
class IndicatorRelation:
    indicator: str
    source: str
    target_market: str
    origin_market: str | None
    role: str | None
    observed_at: str
    age_seconds: float
    usable: bool
    relation: str
    reason_codes: tuple[str, ...]

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


def evaluate_market_indicators(
    observations: Iterable[Any], market: str, *, captured_at: datetime
) -> tuple[IndicatorRelation, ...]:
    from app.data.market_capabilities import normalize_market_group

    group = normalize_market_group(market)
    target = group.value if group is not None else str(market)
    now = captured_at if captured_at.tzinfo else captured_at.replace(tzinfo=timezone.utc)
    result = []
    for observation in observations:
        name = str(observation.name).strip().upper()
        source = str(observation.source or "").strip().lower()
        observed = observation.observed_at
        observed = observed if observed.tzinfo else observed.replace(tzinfo=timezone.utc)
        age = (now - observed).total_seconds()
        contract = _CONTRACTS.get(name)
        reasons = []
        if contract is None:
            reasons.append("INDICATOR_CONTRACT_UNKNOWN")
        else:
            if target not in contract.targets:
                reasons.append("INDICATOR_MARKET_NOT_APPLICABLE")
            # Adapter name plus optional series identifier, not substring matching.
            if source.split(":", 1)[0] not in contract.providers:
                reasons.append("INDICATOR_SOURCE_UNVERIFIED")
            if age > contract.max_age_seconds:
                reasons.append("INDICATOR_STALE")
        if age < 0:
            reasons.append("INDICATOR_FROM_FUTURE")
        try:
            valid = not isinstance(observation.value, bool) and math.isfinite(float(observation.value))
        except (TypeError, ValueError):
            valid = False
        if not valid:
            reasons.append("INDICATOR_VALUE_INVALID")
        result.append(IndicatorRelation(
            indicator=name, source=source, target_market=target,
            origin_market=contract.origin_market if contract else None,
            role=contract.role if contract else None, observed_at=observed.isoformat(),
            age_seconds=round(age, 3), usable=not reasons,
            relation="CONFIRMS" if contract and contract.origin_market == target else "INFLUENCES",
            reason_codes=tuple(reasons),
        ))
    return tuple(result)
