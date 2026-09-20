"""Independent KR/US long-only operating plans, without order authority.

The plan provides strategy research preferences by market and regime. Its CASH
mode recommends waiting; it is not an execution veto. Empty preferences leave
existing ontology eligibility and compatibility intact. Hard abstention remains
the responsibility of symbol/regime ontology, risk and broker gates. Extended
sessions come from the broker capability service, never wall-clock guesses.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from functools import lru_cache
from pathlib import Path
from typing import Any, Mapping

from app.data.market_capabilities import MarketSessionService, default_service
from app.strategy.catalog import STRATEGY_IDS, is_short_strategy

_CONFIG = Path(__file__).resolve().parents[3] / "config" / "multi_market_policy.yaml"
_ALIASES = {
    "BULL": "TREND_UP", "BULLISH": "TREND_UP", "UPTREND": "TREND_UP",
    "BEAR": "TREND_DOWN", "BEARISH": "TREND_DOWN", "DOWNTREND": "TREND_DOWN",
    "SIDEWAYS": "RANGE", "RANGING": "RANGE", "MEAN_REVERTING": "RANGE",
    "LOW_VOL_MEAN_REVERTING": "RANGE",
    "LOW_VOL_TRENDING": "TREND", "TREND_LOW_VOL": "TREND",
    "TREND_HIGH_VOL": "HIGH_VOL", "RANGE_LOW_VOL": "RANGE", "RANGE_HIGH_VOL": "HIGH_VOL",
    "HIGH_VOL_TRENDING": "HIGH_VOL", "HIGH_VOL_MEAN_REVERTING": "HIGH_VOL",
    "VOLATILITY_SHOCK": "STRESS", "CRISIS": "STRESS", "RISK_OFF": "STRESS",
    "SHOCK": "STRESS", "DISLOCATED": "STRESS", "NO_TRADE": "STRESS",
}


@dataclass(frozen=True)
class MarketOperatingPlan:
    market: str
    currency: str
    regime: str
    mode: str
    preferred_strategy_ids: tuple[str, ...]
    active_sessions: tuple[str, ...]
    entry_sessions: tuple[str, ...]
    live_entry_sessions: tuple[str, ...]
    data_available: bool
    exit_allowed: bool
    maximum_position_weight: float
    reason_codes: tuple[str, ...]

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


@lru_cache(maxsize=1)
def load_market_profiles() -> Mapping[str, Any]:
    import yaml

    document = yaml.safe_load(_CONFIG.read_text(encoding="utf-8"))
    profiles = document["markets"]
    for market in ("KR", "US"):
        profile = profiles[market]
        if not 0 < float(profile["maximum_position_weight"]) <= 1:
            raise ValueError(f"invalid cash-account position cap: {market}")
        for strategies in profile["regimes"].values():
            if not isinstance(strategies, list):
                raise ValueError(f"strategy profile must contain lists: {market}")
            for strategy in strategies:
                if strategy not in STRATEGY_IDS or is_short_strategy(strategy):
                    raise ValueError(f"not a catalogued LONG strategy: {market}/{strategy}")
    return profiles


def build_market_operating_plans(
    regimes: Mapping[str, str] | None = None,
    *,
    now_utc: datetime | None = None,
    service: MarketSessionService | None = None,
) -> dict[str, MarketOperatingPlan]:
    current = now_utc or datetime.now(timezone.utc)
    if current.tzinfo is None:
        current = current.replace(tzinfo=timezone.utc)
    sessions = service or default_service()
    profiles = load_market_profiles()
    result: dict[str, MarketOperatingPlan] = {}
    for market in ("KR", "US"):
        profile = profiles[market]
        regime = str((regimes or {}).get(market, "UNKNOWN") or "UNKNOWN").upper().strip()
        regime = _ALIASES.get(regime, regime)
        preferred = tuple(profile["regimes"].get(regime, ()))
        active = sessions.active_capabilities(market, current)
        entry = tuple(item for item in active if item.new_entry_allowed)
        live = tuple(item for item in entry if item.live_order_authorized)
        reasons: list[str] = []
        if not preferred:
            reasons.append("MARKET_POLICY_CASH_REGIME")
        if not entry:
            reasons.extend(sessions.new_entry_block_reasons(market, current))
        if entry and not live:
            reasons.append("SESSION_LIVE_AUTHORIZATION_REQUIRED")
        data_available = any(item.data_available for item in active)
        mode = "LONG" if preferred and entry else "OBSERVE" if preferred and data_available else "CASH"
        # This plan's cap describes its recommended posture; CASH's zero is not
        # an execution veto. Orders independently enforce the market profile cap,
        # their exact route's potentially tighter cap and account risk budget.
        cap = min(
            float(profile["maximum_position_weight"]),
            max((item.policy.maximum_position_weight for item in entry), default=0.0),
        ) if mode == "LONG" else 0.0
        result[market] = MarketOperatingPlan(
            market=market, currency="KRW" if market == "KR" else "USD",
            regime=regime, mode=mode, preferred_strategy_ids=preferred,
            active_sessions=tuple(dict.fromkeys(item.session.value for item in active)),
            entry_sessions=tuple(dict.fromkeys(item.session.value for item in entry)),
            live_entry_sessions=tuple(dict.fromkeys(item.session.value for item in live)),
            data_available=data_available, exit_allowed=any(item.exit_allowed for item in active),
            maximum_position_weight=cap, reason_codes=tuple(dict.fromkeys(reasons)),
        )
    return result


def strategy_profile_preferred(market: str, regime: str | None, strategy_id: str) -> bool:
    """A soft research prior, never eligibility or evidence of positive return."""
    return strategy_id in market_profile_strategy_ids(market, regime)


def market_profile_strategy_ids(market: str, regime: str | None) -> tuple[str, ...]:
    from app.data.market_capabilities import normalize_market_group

    group = normalize_market_group(market)
    if group is None:
        return ()
    state = str(regime or "UNKNOWN").strip().upper()
    state = _ALIASES.get(state, state)
    return tuple(load_market_profiles()[group.value]["regimes"].get(state, ()))
