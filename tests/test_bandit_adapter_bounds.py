"""The bandit correction is bounded, shrunk, and reports when it was clamped."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta, timezone

import pytest

from app.routing.bandit_adapter import (
    BANDIT_CORRECTION_CLAMPED,
    BANDIT_NO_HISTORY,
    BANDIT_POSTERIOR_UNAVAILABLE,
    BANDIT_SHRUNK_TO_PARENT,
    BanditAdapterConfig,
    BanditContextKey,
    StrategyBanditAdapter,
)

AT = datetime(2026, 8, 11, 4, 0, 0, tzinfo=timezone.utc)


@dataclass(frozen=True)
class _Posterior:
    sample_count: int
    posterior_mean_net_bps: float
    effective_sample_count: float
    conservative_edge_bps: float = 0.0
    loss_streak: int = 0


@dataclass(frozen=True)
class _Outcome:
    recorded_at: datetime


class _Store:
    def __init__(self, posterior: _Posterior | None, *, outcomes=()):
        self._posterior = posterior
        self._outcomes = tuple(outcomes)

    def posterior(self, _strategy_id, **_kwargs):
        if self._posterior is None:
            raise RuntimeError("no posterior")
        return self._posterior

    def recent_outcomes(self, _strategy_id, **_kwargs):
        return self._outcomes


def _adapter(store, **config_overrides) -> StrategyBanditAdapter:
    config = BanditAdapterConfig(
        max_correction_bps=20.0,
        minimum_samples=8,
        half_life_seconds=None,
        max_age_seconds=None,
        **config_overrides,
    )
    return StrategyBanditAdapter(store=store, config=config)


def _correct(adapter, *, predicted: float | None):
    return adapter.correct(
        strategy_id="intraday_momentum",
        market="KR",
        regime="TREND_UP",
        volatility_percentile=0.5,
        predicted_net_bps=predicted,
        now=AT,
    )


def test_correction_is_clamped_and_says_so() -> None:
    adapter = _adapter(
        _Store(_Posterior(sample_count=40, posterior_mean_net_bps=-300.0, effective_sample_count=40.0))
    )
    correction = _correct(adapter, predicted=50.0)
    assert correction.correction_bps == pytest.approx(-20.0)
    assert correction.raw_correction_bps < -20.0
    assert correction.clamped
    assert BANDIT_CORRECTION_CLAMPED in correction.reason_codes


def test_positive_correction_is_clamped_symmetrically() -> None:
    adapter = _adapter(
        _Store(_Posterior(sample_count=40, posterior_mean_net_bps=400.0, effective_sample_count=40.0))
    )
    correction = _correct(adapter, predicted=10.0)
    assert correction.correction_bps == pytest.approx(20.0)


def test_no_history_means_no_correction_not_a_penalty() -> None:
    adapter = _adapter(_Store(_Posterior(sample_count=0, posterior_mean_net_bps=0.0, effective_sample_count=0.0)))
    correction = _correct(adapter, predicted=50.0)
    assert correction.correction_bps == 0.0
    assert BANDIT_NO_HISTORY in correction.reason_codes


def test_unavailable_store_means_no_correction() -> None:
    adapter = _adapter(_Store(None))
    correction = _correct(adapter, predicted=50.0)
    assert correction.correction_bps == 0.0
    assert BANDIT_POSTERIOR_UNAVAILABLE in correction.reason_codes


def test_thin_sample_is_shrunk_toward_zero() -> None:
    # Kept inside the +/-20bps bound on purpose: if both raw corrections clamped, the test
    # would be measuring the clamp rather than the shrinkage.
    thin = _adapter(
        _Store(_Posterior(sample_count=2, posterior_mean_net_bps=-16.0, effective_sample_count=2.0))
    )
    thick = _adapter(
        _Store(_Posterior(sample_count=40, posterior_mean_net_bps=-16.0, effective_sample_count=40.0))
    )
    thin_correction = _correct(thin, predicted=0.0)
    thick_correction = _correct(thick, predicted=0.0)
    assert not thin_correction.clamped and not thick_correction.clamped
    assert thin_correction.correction_bps == pytest.approx(-4.0)
    assert thick_correction.correction_bps == pytest.approx(-16.0)
    assert BANDIT_SHRUNK_TO_PARENT in thin_correction.reason_codes


def test_recency_decay_reduces_an_old_posterior() -> None:
    old = _Store(
        _Posterior(sample_count=40, posterior_mean_net_bps=-40.0, effective_sample_count=40.0),
        outcomes=(_Outcome(AT - timedelta(days=60)),) * 5,
    )
    fresh = _Store(
        _Posterior(sample_count=40, posterior_mean_net_bps=-40.0, effective_sample_count=40.0),
        outcomes=(_Outcome(AT - timedelta(hours=1)),) * 5,
    )
    decayed = StrategyBanditAdapter(
        store=old,
        config=BanditAdapterConfig(half_life_seconds=7 * 24 * 3600.0, max_age_seconds=None),
    )
    current = StrategyBanditAdapter(
        store=fresh,
        config=BanditAdapterConfig(half_life_seconds=7 * 24 * 3600.0, max_age_seconds=None),
    )
    assert abs(_correct(decayed, predicted=0.0).correction_bps) < abs(
        _correct(current, predicted=0.0).correction_bps
    )


def test_context_key_excludes_symbol_and_never_widens_market_or_direction() -> None:
    key = BanditContextKey(
        strategy_id="intraday_momentum",
        market="KR",
        regime_cluster="TREND_UP",
        volatility_bucket="HIGH",
        direction="LONG",
    )
    assert "symbol" not in key.as_dict()
    parent = key.parent
    assert parent.market == key.market
    assert parent.direction == key.direction
    assert parent.regime_cluster == "ALL"
    assert parent.volatility_bucket == "ALL"


def test_volatility_bucket_is_config_driven() -> None:
    config = BanditAdapterConfig(volatility_buckets=(0.33, 0.66))
    assert config.volatility_bucket(None) == "UNKNOWN"
    assert config.volatility_bucket(0.1) == "LOW"
    assert config.volatility_bucket(0.5) == "MID"
    assert config.volatility_bucket(0.9) == "HIGH"


def test_bandit_cannot_bypass_a_gate_because_it_only_returns_a_number() -> None:
    adapter = _adapter(
        _Store(_Posterior(sample_count=40, posterior_mean_net_bps=9_999.0, effective_sample_count=40.0))
    )
    correction = _correct(adapter, predicted=0.0)
    assert isinstance(correction.correction_bps, float)
    assert abs(correction.correction_bps) <= adapter.config.max_correction_bps
    # And no method on the adapter can authorise anything.
    assert not hasattr(adapter, "select")
    assert not hasattr(adapter, "submit")
