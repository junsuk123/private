from datetime import datetime, timedelta, timezone
from zoneinfo import ZoneInfo

import pytest

from app.context.global_context import GlobalContextBuilder, IndicatorObservation, load_global_indicator_config
from app.data.market_capabilities import MarketSessionConfig, MarketSessionService, SessionId
from app.ontology.indicator_graph import evaluate_market_indicators


def kst(hour, minute=0, second=0):
    return datetime(2026, 8, 5, hour, minute, second, tzinfo=ZoneInfo("Asia/Seoul"))


@pytest.mark.parametrize("hour,minute,second,session,active", [
    (9, 0, 0, SessionId.NXT_REGULAR, False),
    (9, 0, 29, SessionId.NXT_REGULAR, False),
    (9, 0, 30, SessionId.NXT_REGULAR, True),
    (9, 0, 31, SessionId.NXT_REGULAR, True),
    (15, 20, 0, SessionId.NXT_REGULAR, False),
    (15, 30, 0, SessionId.NXT_POST, False),
    (15, 39, 59, SessionId.NXT_POST, False),
    (15, 40, 0, SessionId.NXT_POST, True),
    (19, 59, 59, SessionId.NXT_POST, True),
    (20, 0, 0, SessionId.NXT_POST, False),
])
def test_nxt_continuous_execution_boundaries(hour, minute, second, session, active):
    service = MarketSessionService()
    capabilities = service.active_capabilities("KR", kst(hour, minute, second))
    assert (session in {item.session for item in capabilities}) is active
    if session is SessionId.NXT_REGULAR and active:
        capability = next(item for item in capabilities if item.session is session)
        assert capability.session_start == kst(9, 0, 30)


def test_nxt_auction_gap_does_not_allow_continuous_entries_even_when_policy_enabled(monkeypatch):
    monkeypatch.setenv("KIS_ENV", "real")
    monkeypatch.setenv("KIS_PAPER_TRADING", "false")
    monkeypatch.setenv("TRADING_ALLOW_ENTRY_NXT_POST", "true")
    monkeypatch.setenv("TRADING_ALLOW_ENTRY_US_DAYTIME", "true")
    service = MarketSessionService()
    assert not service.new_entry_allowed("KR", kst(15, 39, 59))
    assert service.new_entry_allowed("US", kst(15, 39, 59))
    assert service.new_entry_allowed("KR", kst(15, 40))
    assert service.new_entry_allowed("US", kst(15, 40))
    nxt = next(item for item in service.active_capabilities("KR", kst(15, 40)) if item.session is SessionId.NXT_POST)
    assert nxt.live_order_authorized is False


def test_nxt_fallback_schedule_matches_configured_schedule(tmp_path):
    configured = MarketSessionConfig.load()
    fallback = MarketSessionConfig.load(tmp_path / "absent.yaml")
    for session in (SessionId.NXT_REGULAR, SessionId.NXT_POST):
        assert fallback.windows[session] == configured.windows[session]


def test_every_configured_global_indicator_has_an_applicability_contract():
    moment = kst(12)
    config = load_global_indicator_config()
    observations = tuple(
        IndicatorObservation(name, 100, moment, source="unverified")
        for group in config.groups.values() for name in group.members
    )
    for market in ("KR", "US"):
        relations = evaluate_market_indicators(observations, market, captured_at=moment)
        assert not any("INDICATOR_CONTRACT_UNKNOWN" in item.reason_codes for item in relations)
        assert all("INDICATOR_SOURCE_UNVERIFIED" in item.reason_codes for item in relations)


@pytest.mark.parametrize("name,source", [
    ("WTI", "fred"), ("GOLD", "yahoo_chart"), ("COPPER", "fred_public_csv"),
    ("NIKKEI", "fred"), ("HANGSENG", "stooq"), ("CSI300", "yahoo_chart"),
    ("YM", "kis_realtime"), ("US_INDEX_FUTURES", "kis"),
    ("SP500", "stooq"), ("NVDA", "alpha_vantage_daily"), ("VIX", "yahoo_chart"),
])
def test_existing_collector_identifiers_preserve_configured_context(name, source):
    moment = kst(12)
    observation = IndicatorObservation(name, 100, moment, source=source, change_ratio=0.01)
    context = GlobalContextBuilder().build((observation,), captured_at=moment, market="US")
    assert context.indicator_relations[0]["usable"] is True
    assert any(name in group.observed_members for group in context.groups.values())
    assert "GLOBAL_UNKNOWN_INDICATOR" not in context.reason_codes


def test_daily_reference_sources_never_bypass_freshness():
    moment = kst(12)
    observation = IndicatorObservation("GOLD", 100, moment - timedelta(days=2), source="yahoo_chart")
    relation = evaluate_market_indicators((observation,), "KR", captured_at=moment)[0]
    assert relation.usable is False
    assert relation.reason_codes == ("INDICATOR_STALE",)
