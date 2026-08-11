"""Coverage matrix, validation metrics, and the lifecycle transition whitelist."""

from __future__ import annotations

import random
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone

import pytest

from app.context import MarketContextBuilder, SymbolContextInputs
from app.monitoring.strategy_drift import StrategyDriftConfig, StrategyDriftMonitor
from app.strategy.coverage import (
    COVERAGE_GAP_REASON,
    CONTEXT_DIMENSIONS,
    StrategyCoverageAnalyzer,
    bucket_for_context,
)
from app.strategy.spec import StrategyLifecycleState
from app.strategy_validation import (
    PromotionGates,
    StrategyValidationRecord,
    StrategyValidationRegistry,
    TradeObservation,
    compute_metrics,
    cost_stress,
    effective_sample_count,
    parameter_stability,
    purged_kfold_splits,
    regime_breakdown,
    walk_forward,
)
from app.technical.signals import TechnicalFeatureSet

AT = datetime(2026, 8, 11, 4, 0, 0, tzinfo=timezone.utc)


def _context(**overrides):
    values = {
        "symbol": "005930",
        "price": 70_000.0,
        "ema_fast": 70_400.0,
        "ema_slow": 69_900.0,
        "spread_bps": 8.0,
        "orderbook_imbalance": 0.3,
        "liquidity_score": 0.7,
        "aggressor_imbalance_5s": 0.4,
        "realized_volatility": 0.003,
        "realized_volatility_10s": 0.002,
        "second_data_ready": 1.0,
    }
    values.update(overrides)
    return MarketContextBuilder().build(
        SymbolContextInputs(
            symbol="005930",
            features=TechnicalFeatureSet(**values),
            context_id="ctx-cov",
            tick_freshness_sec=0.5,
            orderbook_freshness_sec=0.8,
            history_bar_count=40,
        ),
        captured_at=AT,
    )


# --------------------------------------------------------------------------- #
# Coverage                                                                     #
# --------------------------------------------------------------------------- #
def test_bucketing_is_deterministic_and_within_declared_labels() -> None:
    context = _context()
    first = bucket_for_context(context)
    assert first == bucket_for_context(context)
    for axis, value in first.as_dict().items():
        assert value in CONTEXT_DIMENSIONS[axis], f"{axis}={value} is not a declared label"


def test_trend_and_flow_buckets_track_the_context() -> None:
    up = bucket_for_context(_context(ema_fast=71_000.0, ema_slow=69_000.0))
    down = bucket_for_context(_context(ema_fast=69_000.0, ema_slow=71_000.0))
    assert up.trend == "strong_up"
    assert down.trend == "strong_down"

    selling = bucket_for_context(_context(aggressor_imbalance_5s=-0.5))
    assert selling.microstructure == "sell_pressure"


@dataclass(frozen=True)
class _Candidate:
    strategy_id: str
    eligible: bool
    entry_ready: bool
    selectable: bool
    expected_net_return_bps: float = 10.0


@dataclass(frozen=True)
class _Selection:
    ranked_candidates: tuple
    is_no_trade: bool


def test_coverage_gap_is_recorded_when_nothing_is_selectable() -> None:
    analyzer = StrategyCoverageAnalyzer(state_path=None)
    context = _context()
    for _ in range(6):
        analyzer.record_selection(
            context,
            _Selection(
                ranked_candidates=(
                    _Candidate("intraday_momentum", True, True, False),
                    _Candidate("vwap_mean_reversion", True, False, False),
                ),
                is_no_trade=True,
            ),
        )
    gaps = analyzer.gaps(minimum_observations=5)
    assert gaps, "a repeatedly-empty bucket must be reported as a gap"
    assert gaps[0].is_coverage_gap
    assert gaps[0].no_trade_rate == pytest.approx(1.0)

    candidates = analyzer.research_candidates(minimum_observations=5)
    assert candidates
    assert candidates[0]["reason"] == COVERAGE_GAP_REASON
    # The nearest firing strategy is REPORTED, not promoted into the gap.
    assert "intraday_momentum" in candidates[0]["nearest_firing_strategies"]


def test_bucket_with_a_selectable_strategy_is_not_a_gap() -> None:
    analyzer = StrategyCoverageAnalyzer(state_path=None)
    context = _context()
    analyzer.record_selection(
        context,
        _Selection(
            ranked_candidates=(_Candidate("intraday_momentum", True, True, True),),
            is_no_trade=False,
        ),
    )
    assert analyzer.gaps(minimum_observations=1) == ()


def test_best_net_is_none_when_never_measurable() -> None:
    analyzer = StrategyCoverageAnalyzer(state_path=None)
    observation = analyzer.record(
        _context(),
        eligible_count=3,
        entry_ready_count=0,
        validated_positive_count=0,
        no_trade=True,
    )
    assert observation.best_strategy_expected_net_bps is None


# --------------------------------------------------------------------------- #
# Metrics                                                                      #
# --------------------------------------------------------------------------- #
def _trades(count: int, *, mean: float, spacing_minutes: int = 30, horizon_minutes: int = 10):
    generator = random.Random(11)
    return [
        TradeObservation(
            strategy_id="intraday_momentum",
            symbol="005930",
            market="KR",
            regime="TREND_UP",
            opened_at=AT + timedelta(minutes=spacing_minutes * index),
            closed_at=AT + timedelta(minutes=spacing_minutes * index + horizon_minutes),
            gross_return_bps=(value := generator.gauss(mean, 40.0)) + 28.0,
            net_return_bps=value,
            cost_bps=28.0,
            evidence_source="LIVE",
            session_phase="regular",
            predicted_net_bps=mean,
            predicted_probability=0.55,
            max_adverse_excursion_bps=30.0,
            max_favorable_excursion_bps=60.0,
        )
        for index in range(count)
    ]


def test_empty_metrics_are_none_not_zero() -> None:
    metrics = compute_metrics("intraday_momentum", [])
    assert metrics.trigger_count == 0
    assert metrics.net_ev_bps is None
    assert metrics.lower_confidence_bound_bps is None
    assert "METRICS_NO_TRADES" in metrics.reason_codes


def test_overlapping_trades_are_discounted() -> None:
    non_overlapping = _trades(20, mean=20.0, spacing_minutes=30, horizon_minutes=10)
    overlapping = _trades(20, mean=20.0, spacing_minutes=1, horizon_minutes=60)
    assert effective_sample_count(non_overlapping) == pytest.approx(20.0, abs=0.01)
    assert effective_sample_count(overlapping) < 5.0

    metrics = compute_metrics("intraday_momentum", overlapping, minimum_samples=10)
    assert "METRICS_OVERLAPPING_TRADES_DISCOUNTED" in metrics.reason_codes
    # The discount widens the interval, which is the whole point.
    assert metrics.lower_confidence_bound_bps < compute_metrics(
        "intraday_momentum", non_overlapping, minimum_samples=10
    ).lower_confidence_bound_bps


def test_single_trade_bucket_reports_none_rather_than_that_trade() -> None:
    metrics = compute_metrics("intraday_momentum", _trades(1, mean=50.0))
    assert metrics.market_breakdown.get("KR") is None


def test_no_live_evidence_is_flagged() -> None:
    shadow = [
        TradeObservation(
            strategy_id="s",
            symbol="X",
            market="KR",
            regime="TREND_UP",
            opened_at=AT + timedelta(minutes=index),
            closed_at=AT + timedelta(minutes=index + 1),
            gross_return_bps=50.0,
            net_return_bps=22.0,
            cost_bps=28.0,
            evidence_source="SHADOW",
        )
        for index in range(30)
    ]
    assert "METRICS_NO_LIVE_EVIDENCE" in compute_metrics("s", shadow).reason_codes


# --------------------------------------------------------------------------- #
# Cost stress / purged CV / walk-forward                                       #
# --------------------------------------------------------------------------- #
def test_break_even_cost_multiple_and_fragility() -> None:
    fragile = cost_stress("s", _trades(40, mean=2.0))
    robust = cost_stress("s", _trades(40, mean=80.0))
    assert fragile.break_even_cost_multiple < robust.break_even_cost_multiple
    assert robust.survives_all
    assert not fragile.survives_all


def test_purged_folds_remove_overlapping_training_trades() -> None:
    trades = _trades(40, mean=20.0, spacing_minutes=5, horizon_minutes=60)
    splits = purged_kfold_splits(trades, folds=4)
    assert splits
    for split in splits:
        assert split.test_indices
        assert split.purged_indices or split.embargoed_indices, (
            "overlapping horizons must cost the training fold something"
        )
        assert not (set(split.train_indices) & set(split.test_indices))


def test_walk_forward_reports_out_of_sample_stability() -> None:
    result = walk_forward("s", _trades(60, mean=30.0), windows=4)
    assert result.windows
    assert result.out_of_sample_stability is not None
    assert 0.0 <= result.out_of_sample_stability <= 1.0


def test_regime_breakdown_says_it_cannot_tell_when_it_cannot() -> None:
    breakdown = regime_breakdown("s", _trades(6, mean=20.0))
    assert breakdown.discriminates is None


def test_parameter_stability_flags_a_spike() -> None:
    def evaluate(values):
        # A cliff at exactly 10.0: the configured value works and neighbours do not.
        return 100.0 if abs(values["threshold"] - 10.0) < 1e-9 else 1.0

    result = parameter_stability(
        "s", parameters={"threshold": 10.0}, evaluate=evaluate
    )
    assert result.stable is False
    assert "threshold" in result.fragile_parameters


def test_parameter_stability_accepts_a_plateau() -> None:
    result = parameter_stability(
        "s", parameters={"threshold": 10.0}, evaluate=lambda _values: 50.0
    )
    assert result.stable is True


# --------------------------------------------------------------------------- #
# Lifecycle registry                                                           #
# --------------------------------------------------------------------------- #
def _record(**overrides) -> StrategyValidationRecord:
    values = {
        "strategy_id": "intraday_momentum",
        "validated_at": AT,
        "validation_version": "audit-1",
        "algorithm_version": "spec-v1",
        "sample_count": 50,
        "effective_sample_count": 45.0,
        "net_ev_bps": 30.0,
        "lower_confidence_bound_bps": 8.0,
        "break_even_cost_multiple": 1.8,
        "out_of_sample_stability": 0.8,
        "parameter_stability": True,
        "evidence_mix": {"LIVE": 50},
    }
    values.update(overrides)
    return StrategyValidationRecord(**values)


def test_two_rung_promotion_is_refused() -> None:
    registry = StrategyValidationRegistry(state_path=None)
    registry.seed_from({"intraday_momentum": StrategyLifecycleState.SHADOW})
    applied, reasons = registry.transition(
        "intraday_momentum",
        to_state=StrategyLifecycleState.LIVE,
        actor="test",
        record=_record(),
    )
    assert not applied
    assert any("TRANSITION_NOT_ALLOWED" in reason for reason in reasons)


def test_promotion_without_a_record_is_refused() -> None:
    registry = StrategyValidationRegistry(state_path=None)
    registry.seed_from({"intraday_momentum": StrategyLifecycleState.SHADOW})
    applied, reasons = registry.transition(
        "intraday_momentum", to_state=StrategyLifecycleState.LIVE_PROBE, actor="test"
    )
    assert not applied
    assert "LIFECYCLE_NO_VALIDATION_RECORD" in reasons


def test_shadow_only_evidence_cannot_reach_live() -> None:
    registry = StrategyValidationRegistry(state_path=None)
    registry.seed_from({"intraday_momentum": StrategyLifecycleState.LIVE_PROBE})
    applied, reasons = registry.transition(
        "intraday_momentum",
        to_state=StrategyLifecycleState.LIVE,
        actor="test",
        record=_record(evidence_mix={"SHADOW": 200}),
    )
    assert not applied
    assert "LIFECYCLE_NO_LIVE_EVIDENCE" in reasons


def test_negative_lower_bound_cannot_reach_live_probe() -> None:
    registry = StrategyValidationRegistry(state_path=None)
    registry.seed_from({"intraday_momentum": StrategyLifecycleState.SHADOW})
    applied, reasons = registry.transition(
        "intraday_momentum",
        to_state=StrategyLifecycleState.LIVE_PROBE,
        actor="test",
        record=_record(lower_confidence_bound_bps=-2.0),
    )
    assert not applied
    assert "LIFECYCLE_LOWER_BOUND_NOT_POSITIVE" in reasons


def test_valid_one_rung_promotion_is_applied() -> None:
    registry = StrategyValidationRegistry(state_path=None)
    registry.seed_from({"intraday_momentum": StrategyLifecycleState.SHADOW})
    applied, _ = registry.transition(
        "intraday_momentum",
        to_state=StrategyLifecycleState.LIVE_PROBE,
        actor="test",
        record=_record(),
    )
    assert applied
    assert registry.state("intraday_momentum") is StrategyLifecycleState.LIVE_PROBE


def test_demotion_needs_no_evidence() -> None:
    registry = StrategyValidationRegistry(state_path=None)
    registry.seed_from({"intraday_momentum": StrategyLifecycleState.LIVE})
    applied, _ = registry.transition(
        "intraday_momentum", to_state=StrategyLifecycleState.DEGRADED, actor="drift"
    )
    assert applied
    assert registry.state("intraday_momentum") is StrategyLifecycleState.DEGRADED


def test_storing_evidence_never_changes_state() -> None:
    registry = StrategyValidationRegistry(state_path=None)
    registry.seed_from({"intraday_momentum": StrategyLifecycleState.SHADOW})
    registry.upsert_record(_record())
    assert registry.state("intraday_momentum") is StrategyLifecycleState.SHADOW


# --------------------------------------------------------------------------- #
# Drift demotion                                                               #
# --------------------------------------------------------------------------- #
def test_one_loss_does_not_demote() -> None:
    monitor = StrategyDriftMonitor(config=StrategyDriftConfig(minimum_samples=20))
    monitor.record(strategy_id="s", net_return_bps=-300.0, at=AT, is_live=True)
    proposals = monitor.demotion_proposals({"s": StrategyLifecycleState.LIVE})
    assert proposals == ()


def test_sustained_negative_ev_demotes_one_rung() -> None:
    monitor = StrategyDriftMonitor(config=StrategyDriftConfig(minimum_samples=20))
    for index in range(30):
        monitor.record(
            strategy_id="s",
            net_return_bps=-40.0 + (index % 3),
            at=AT + timedelta(minutes=index),
            cost_bps=28.0,
            is_live=True,
        )
    proposals = monitor.demotion_proposals({"s": StrategyLifecycleState.LIVE})
    assert len(proposals) == 1
    assert proposals[0].to_state is StrategyLifecycleState.DEGRADED
    assert "DRIFT_ROLLING_NET_EV_NEGATIVE" in proposals[0].reason_codes


def test_shadow_only_losses_recommend_research_not_retire() -> None:
    """RETIRE is terminal, so a simulated sample must not reach it."""
    from app.strategy_validation import AuditClassification, StrategyAuditRunner

    shadow_losers = [
        TradeObservation(
            strategy_id="liquidity_shock_reversal",
            symbol="AAPL",
            market="US",
            regime="TREND_DOWN",
            opened_at=AT + timedelta(minutes=10 * index),
            closed_at=AT + timedelta(minutes=10 * index + 5),
            gross_return_bps=-90.0,
            net_return_bps=-120.0,
            cost_bps=30.0,
            evidence_source="SHADOW",
        )
        for index in range(80)
    ]
    report = StrategyAuditRunner().run(
        {"liquidity_shock_reversal": shadow_losers},
        strategy_ids=["liquidity_shock_reversal"],
    )
    audit = report.audits[0]
    assert audit.classification is AuditClassification.RESEARCH
    assert "AUDIT_NO_LIVE_EVIDENCE" in audit.reason_codes
    assert audit.recommended_lifecycle is not StrategyLifecycleState.RETIRED

    live_losers = [
        TradeObservation(**{**vars(trade), "evidence_source": "LIVE"})
        for trade in shadow_losers
    ]
    live_report = StrategyAuditRunner().run(
        {"liquidity_shock_reversal": live_losers},
        strategy_ids=["liquidity_shock_reversal"],
    )
    assert live_report.audits[0].classification is AuditClassification.RETIRE


def test_profitable_strategy_is_not_demoted() -> None:
    monitor = StrategyDriftMonitor(config=StrategyDriftConfig(minimum_samples=20))
    for index in range(30):
        monitor.record(
            strategy_id="s",
            net_return_bps=40.0,
            at=AT + timedelta(minutes=index),
            cost_bps=28.0,
            is_live=True,
        )
    assert monitor.demotion_proposals({"s": StrategyLifecycleState.LIVE}) == ()
