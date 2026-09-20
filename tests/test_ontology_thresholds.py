from dataclasses import asdict, replace
from datetime import datetime, timedelta, timezone
import math

import pytest

from app.ontology.policy_evidence import PolicyObservation, project_market_evidence
from app.risk.ontology_thresholds import PolicyLimits, resolve_ontology_policy


NOW = datetime(2026, 9, 21, 1, 30, tzinfo=timezone.utc)
REQUIRED = ("realized_volatility", "spread_rate", "liquidity_score", "quote_age_seconds", "regime_confidence")


def _projection(*, market="KR", regime="TREND_UP", changes=None, observation_changes=None, as_of=NOW):
    values = dict(realized_volatility=.002, spread_rate=.00015, liquidity_score=.9,
                  quote_age_seconds=.1, regime_confidence=.9, data_quality_score=.95,
                  model_uncertainty=.1, market_breadth=.4, trend_strength=.6,
                  change_point_probability=.05, drawdown_rate=0.0)
    values.update(changes or {})
    observations = []
    for metric, value in values.items():
        fields = dict(metric=metric, value=value, market=market, observed_at=NOW,
                      source="kis_realtime", max_age_seconds=30.0,
                      unit="seconds" if metric == "quote_age_seconds" else "ratio",
                      horizon_seconds=60.0 if metric == "realized_volatility" else None)
        fields.update((observation_changes or {}).get(metric, {}))
        observations.append(PolicyObservation(**fields))
    return project_market_evidence(market, as_of=as_of, observations=observations,
                                   required_metrics=REQUIRED, regime=regime, context_id="market-cycle-42")


def _policy(projection=None, **kwargs):
    fields = dict(symbol="005930", all_in_cost_rate=.001, forecast_gross_bps=500,
                  requested_horizon_seconds=900, limits=PolicyLimits())
    fields.update(kwargs)
    return resolve_ontology_policy(projection or _projection(), **fields)


def test_complete_typed_market_evidence_generates_finite_bounded_operating_policy():
    policy = _policy()
    assert policy.valid_for_entry, policy.reason_codes
    assert policy.market == "KR"
    assert 0 < policy.position_cap <= .15
    assert 0 < policy.soft_stop_rate <= policy.hard_stop_rate <= .02
    assert policy.hard_stop_rate <= policy.emergency_stop_rate <= .035
    assert policy.target_return_rate > policy.all_in_cost_rate + policy.net_profit_floor_rate
    assert policy.is_current(NOW)
    for value in asdict(policy).values():
        if isinstance(value, float):
            assert math.isfinite(value)


@pytest.mark.parametrize(("field", "changes", "reason"), [
    ("spread_rate", {"market": "US"}, "ONTO_POLICY_MARKET_MISMATCH"),
    ("liquidity_score", {"observed_at": NOW - timedelta(seconds=31)}, "ONTO_POLICY_STALE"),
    ("regime_confidence", {"observed_at": NOW + timedelta(microseconds=1)}, "ONTO_POLICY_FROM_FUTURE"),
    ("realized_volatility", {"horizon_seconds": None}, "ONTO_POLICY_HORIZON_MISSING"),
    ("realized_volatility", {"horizon_seconds": float("inf")}, "ONTO_POLICY_HORIZON_INVALID"),
    ("spread_rate", {"value": float("nan")}, "ONTO_POLICY_VALUE_INVALID"),
    ("liquidity_score", {"value": float("inf")}, "ONTO_POLICY_VALUE_INVALID"),
])
def test_invalid_required_observation_disables_new_entries(field, changes, reason):
    projection = _projection(observation_changes={field: changes})
    assert not projection.usable
    policy = _policy(projection)
    assert policy.valid_for_entry is False
    assert policy.position_cap == 0
    assert reason in policy.reason_codes
    assert 0 < policy.hard_stop_rate <= .02, "Missing evidence must retain a bounded exit fallback"


def test_optional_rejected_observation_does_not_invalidate_complete_required_evidence():
    # Add a bad optional field; required typed values remain usable.
    projection = _projection(changes={"global_extra": .8}, observation_changes={"global_extra": {"source": "untrusted_blog"}})
    assert projection.usable
    assert "ONTO_POLICY_SOURCE_UNVERIFIED" in projection.reason_codes
    assert _policy(projection).valid_for_entry


def test_more_cost_requires_larger_net_profit_floor_and_target():
    cheap = _policy(all_in_cost_rate=.001)
    expensive = _policy(all_in_cost_rate=.006)
    assert expensive.net_profit_floor_rate > cheap.net_profit_floor_rate
    assert expensive.target_return_rate > cheap.target_return_rate
    assert expensive.max_trades_per_day <= cheap.max_trades_per_day


def test_higher_market_stress_reduces_exposure_loss_budget_and_exit_patience():
    calm = _policy()
    stressed = _policy(_projection(regime="TREND_DOWN", changes=dict(
        liquidity_score=.4, model_uncertainty=.7, trend_strength=-.6,
        market_breadth=-.6, change_point_probability=.7, drawdown_rate=.02,
    )))
    assert calm.valid_for_entry and stressed.valid_for_entry
    assert stressed.stress > calm.stress
    assert stressed.position_cap < calm.position_cap
    assert stressed.daily_loss_budget_rate < calm.daily_loss_budget_rate
    assert stressed.trade_loss_budget_rate < calm.trade_loss_budget_rate
    assert stressed.minimum_cash_reserve > calm.minimum_cash_reserve
    assert stressed.maximum_holding_seconds < calm.maximum_holding_seconds
    assert stressed.trailing_giveback < calm.trailing_giveback


def test_new_volatility_cannot_widen_an_owned_positions_loss_barriers():
    entry = _policy()
    later = _policy(_projection(changes={"realized_volatility": .015}), forecast_gross_bps=1000)
    tightened = later.tighten_for_position(entry)
    for field in ("soft_stop_rate", "hard_stop_rate", "emergency_stop_rate", "trailing_stop_rate", "maximum_holding_seconds"):
        assert getattr(tightened, field) <= getattr(entry, field)
        assert getattr(tightened, field) <= getattr(later, field)
    assert later.tighten_for_position(replace(entry, symbol="AAPL")) is later


@pytest.mark.parametrize("volatility", [.0001, .002, .008])
@pytest.mark.parametrize("uncertainty,downside", [(0., 0.), (.5, 100.), (1., 1000.)])
def test_forward_shadow_graph_can_only_tighten_market_policy_risk(volatility, uncertainty, downside):
    # Shadow payoffs do not certify adaptive live exits. Even a highly optimistic
    # checkpoint may not relax market-only permission, capital or exit limits.
    base = _projection(changes={"realized_volatility": volatility})
    observations = tuple(item for item in base.observations if item.metric != "model_uncertainty")
    base = replace(base, observations=observations)
    graph = replace(base, observations=(*observations,
        PolicyObservation("model_uncertainty", uncertainty, "KR", NOW, "validated_temporal_rgcn", 5),
        PolicyObservation("expected_downside_net_bps", downside, "KR", NOW, "validated_temporal_rgcn", 5, unit="bps"),
    ))
    baseline, advised = _policy(base), _policy(graph)
    for name in ("position_cap", "sector_cap", "trade_loss_budget_rate", "daily_loss_budget_rate",
                 "soft_stop_rate", "hard_stop_rate", "emergency_stop_rate", "trailing_stop_rate",
                 "maximum_holding_seconds", "early_exit_confirmations", "max_quote_age_seconds"):
        assert getattr(advised, name) <= getattr(baseline, name), name
    for name in ("minimum_cash_reserve", "net_profit_floor_rate", "minimum_reward_risk"):
        assert getattr(advised, name) >= getattr(baseline, name), name
    assert not advised.valid_for_entry or baseline.valid_for_entry
    assert advised.policy_id != baseline.policy_id
    if advised.valid_for_entry:
        assert advised.noise_band_rate < advised.hard_stop_rate


def test_positive_forecast_below_cash_cost_and_risk_reward_is_not_enough():
    policy = _policy(forecast_gross_bps=5)
    assert policy.valid_for_entry is False
    assert "POLICY_NET_REWARD_INSUFFICIENT" in policy.reason_codes
    assert policy.position_cap == 0


@pytest.mark.parametrize("cost", [None, float("nan"), float("inf"), -.001])
def test_invalid_cash_cost_is_not_assumed_free(cost):
    policy = _policy(all_in_cost_rate=cost)
    assert policy.valid_for_entry is False
    assert "POLICY_COST_UNKNOWN" in policy.reason_codes


def test_policy_has_no_validity_before_its_evidence_time_or_after_expiry():
    policy = _policy()
    assert not policy.is_current(NOW - timedelta(microseconds=1))
    assert not policy.is_current(policy.expires_at + timedelta(microseconds=1))
    future = _policy(_projection(as_of=NOW + timedelta(seconds=10)))
    assert not future.is_current(NOW)


def test_earliest_required_observation_expiry_bounds_policy_lifetime():
    projection = _projection(observation_changes={
        "realized_volatility": {"observed_at": NOW - timedelta(seconds=29)},
    })
    policy = _policy(projection)
    assert policy.valid_for_entry
    assert policy.expires_at <= NOW + timedelta(seconds=1)


def test_short_lived_risk_evidence_cannot_outlive_its_source():
    projection = _projection(observation_changes={"model_uncertainty": {"max_age_seconds": 2}})
    policy = _policy(projection)
    assert policy.expires_at <= NOW + timedelta(seconds=2)


def test_emergency_override_tightens_all_loss_barriers(monkeypatch):
    monkeypatch.setenv("REALTIME_EMERGENCY_STOP_LOSS", "0.004")
    policy = _policy()
    assert policy.soft_stop_rate <= policy.hard_stop_rate <= policy.emergency_stop_rate <= .004


def test_legacy_environment_can_tighten_but_never_enlarge_safety_ceilings(monkeypatch):
    monkeypatch.setenv("REALTIME_HARD_STOP_LOSS", "0.90")
    monkeypatch.setenv("REALTIME_EMERGENCY_STOP_LOSS", "0.99")
    monkeypatch.setenv("REALTIME_SMALL_ACCOUNT_MAX_POSITION_WEIGHT", "3")
    monkeypatch.setenv("REALTIME_DAILY_REALIZED_LOSS_BUY_STOP_RATE", "0.80")
    policy = _policy(_projection(changes={"realized_volatility": .03}), forecast_gross_bps=1000)
    assert policy.hard_stop_rate <= .02
    assert policy.emergency_stop_rate <= .035
    assert policy.position_cap <= .15
    assert policy.daily_loss_budget_rate <= .01
    monkeypatch.setenv("REALTIME_HARD_STOP_LOSS", "0.005")
    assert _policy().hard_stop_rate <= .005


def test_market_position_limits_remain_distinct_without_currency_absolute_values():
    kr = _policy()
    us = _policy(_projection(market="US"), symbol="AAPL")
    assert kr.market == "KR" and us.market == "US"
    assert kr.position_cap <= .15
    assert us.position_cap <= .12
    assert kr.all_in_cost_rate == us.all_in_cost_rate


def test_unknown_or_halted_market_regime_never_grants_entry():
    for regime in ("UNKNOWN", "HALTED", "DISLOCATED"):
        policy = _policy(_projection(regime=regime))
        assert policy.valid_for_entry is False
        assert policy.position_cap == 0


@pytest.mark.parametrize("reason", ["POLICY_NET_REWARD_INSUFFICIENT", "POLICY_SPREAD_TOO_WIDE",
                                   "POLICY_NOISE_EXCEEDS_LOSS_BUDGET", "POLICY_REGIME_NO_ENTRY"])
def test_fresh_adverse_entry_context_remains_usable_for_position_risk(reason):
    policy = replace(_policy(), valid_for_entry=False, reason_codes=(reason,))
    assert policy.has_current_market_evidence(NOW)
    assert not policy.has_current_market_evidence(policy.expires_at + timedelta(seconds=1))
    assert not replace(policy, reason_codes=(reason, "POLICY_QUOTE_STALE")).has_current_market_evidence(NOW)
    assert not replace(policy, reason_codes=()).has_current_market_evidence(NOW)
