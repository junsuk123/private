"""Rolling seasonality: shrinkage, decay, persistence and leakage."""

from __future__ import annotations

import math
from datetime import datetime, timedelta, timezone

import pytest

from app.config.temporal_config import SeasonalityConfig
from app.features.seasonality import (
    GLOBAL_BUCKET,
    SeasonalityEngine,
    key_for,
    keys_from_context,
)
from app.storage.trading_state_store import TradingStateStore

BASE = datetime(2026, 8, 3, 0, 30, tzinfo=timezone.utc)


def _engine(**overrides) -> SeasonalityEngine:
    config = SeasonalityConfig(**{"baseline_window": 500, "shrinkage_k": 30.0, **overrides})
    return SeasonalityEngine(store=None, config=config, persist=False)


def _key(day: str = "MON", phase: str = "OPENING", regime: str = "TREND_UP"):
    return key_for(
        "open_return_bps",
        market_group="KR",
        day_of_week=day,
        session_phase=phase,
        regime=regime,
    )


def _fill(engine: SeasonalityEngine, key, values, *, start: datetime = BASE) -> None:
    for index, value in enumerate(values):
        engine.observe(key, value, observed_at=start + timedelta(minutes=index))


# --------------------------------------------------------------------------- #
# Normalisation
# --------------------------------------------------------------------------- #
def test_z_score_matches_the_declared_formula() -> None:
    engine = _engine(shrinkage_k=0.0)
    key = _key()
    _fill(engine, key, [10.0, 12.0, 8.0, 11.0, 9.0])
    baseline = engine.baseline(key)
    score = engine.score(key, 20.0)
    expected = (20.0 - baseline.mean) / (baseline.stdev + engine.config.epsilon)
    assert score.z_score == pytest.approx(expected)


def test_repeated_value_gives_a_zero_z_score() -> None:
    engine = _engine(shrinkage_k=0.0)
    key = _key()
    _fill(engine, key, [5.0] * 40)
    assert engine.score(key, 5.0).z_score == pytest.approx(0.0, abs=1e-6)


def test_cold_start_is_reported_and_does_not_explode() -> None:
    engine = _engine()
    score = engine.score(_key(), 42.0)
    assert score.cold_start
    assert score.sample_count == 0
    assert score.z_score == 0.0
    assert score.confidence == 0.0


# --------------------------------------------------------------------------- #
# Shrinkage
# --------------------------------------------------------------------------- #
def test_thin_bucket_is_shrunk_toward_the_global_baseline() -> None:
    engine = _engine(shrinkage_k=30.0)
    monday = _key("MON")
    tuesday = _key("TUE")
    # A large, well-populated global baseline centred on 0.
    _fill(engine, tuesday, [0.0, 1.0, -1.0] * 60)
    # One thin Monday bucket far from it.
    engine.observe(monday, 100.0, observed_at=BASE + timedelta(days=1))

    score = engine.score(monday, 100.0)
    assert score.shrinkage_weight == pytest.approx(1.0 / 31.0)
    # The estimate must sit close to the global mean, not to the single Monday reading.
    assert abs(score.mean) < 10.0


def test_shrinkage_weight_follows_n_over_n_plus_k() -> None:
    engine = _engine(shrinkage_k=30.0)
    key = _key()
    _fill(engine, key, [1.0] * 30)
    score = engine.score(key, 1.0)
    assert score.effective_sample_size == pytest.approx(30.0, rel=0.05)
    assert score.shrinkage_weight == pytest.approx(
        score.effective_sample_size / (score.effective_sample_size + 30.0)
    )
    assert score.confidence == pytest.approx(score.shrinkage_weight)


def test_confidence_rises_with_sample_count() -> None:
    engine = _engine()
    key = _key()
    confidences = []
    for count in (1, 10, 100):
        fresh = _engine()
        _fill(fresh, key, [1.0, 2.0, 3.0] * count)
        confidences.append(fresh.score(key, 2.0).confidence)
    assert confidences == sorted(confidences)
    # 300 observations against a 500-wide exponential window give an effective n of
    # 500*(1 - (1 - 1/500)^300) ~= 226, hence a confidence of 226/(226+30).
    assert confidences[-1] > 0.85
    del engine


# --------------------------------------------------------------------------- #
# Conditioning
# --------------------------------------------------------------------------- #
def test_buckets_are_conditioned_on_day_phase_and_regime() -> None:
    engine = _engine(shrinkage_k=0.0)
    trend = _key(regime="TREND_UP")
    range_regime = _key(regime="RANGE_LOW_VOL")
    _fill(engine, trend, [50.0] * 60)
    _fill(engine, range_regime, [-50.0] * 60, start=BASE + timedelta(days=7))
    # The same raw value is unremarkable in one regime and extreme in the other.
    assert engine.score(trend, 50.0).z_score == pytest.approx(0.0, abs=1e-3)
    assert engine.score(range_regime, 50.0).z_score > 5.0


def test_global_bucket_is_shared_across_conditioned_buckets() -> None:
    engine = _engine()
    _fill(engine, _key("MON"), [3.0] * 20)
    _fill(engine, _key("TUE"), [3.0] * 20, start=BASE + timedelta(days=1))
    global_baseline = engine.baseline(_key("MON").global_key)
    assert global_baseline.key.day_of_week == GLOBAL_BUCKET
    assert global_baseline.sample_count == 40


# --------------------------------------------------------------------------- #
# Leakage
# --------------------------------------------------------------------------- #
def test_observation_is_scored_before_it_joins_the_baseline() -> None:
    engine = _engine(shrinkage_k=0.0)
    key = _key()
    _fill(engine, key, [10.0] * 50)
    before = engine.baseline(key)
    score = engine.observe(key, 10.0, observed_at=BASE + timedelta(hours=5))
    after = engine.baseline(key)
    assert score.mean == pytest.approx(before.mean)
    assert score.sample_count == before.sample_count
    assert after.sample_count == before.sample_count + 1


def test_out_of_order_observation_is_rejected_not_merged() -> None:
    engine = _engine()
    key = _key()
    engine.observe(key, 1.0, observed_at=BASE + timedelta(days=2))
    stale = engine.observe(key, 999.0, observed_at=BASE)
    baseline = engine.baseline(key)
    assert baseline.sample_count == 1
    assert baseline.out_of_order_rejected == 1
    assert baseline.mean == pytest.approx(1.0)
    # The score is still returned so the caller can act on it; only the baseline is
    # protected.
    assert stale.z_score != 0.0
    assert engine.report()["out_of_order_rejected"] >= 1


def test_non_finite_observation_is_refused() -> None:
    engine = _engine()
    with pytest.raises(ValueError):
        engine.observe(_key(), math.inf, observed_at=BASE)


# --------------------------------------------------------------------------- #
# Rolling window
# --------------------------------------------------------------------------- #
def test_effective_sample_size_saturates_at_the_window() -> None:
    engine = _engine(baseline_window=50)
    key = _key()
    _fill(engine, key, [1.0] * 500)
    baseline = engine.baseline(key)
    assert baseline.sample_count == 500
    assert baseline.weight == pytest.approx(50.0, rel=0.05)


def test_old_regime_decays_out_of_the_baseline() -> None:
    engine = _engine(baseline_window=20, shrinkage_k=0.0)
    key = _key()
    _fill(engine, key, [100.0] * 100)
    _fill(engine, key, [0.0] * 100, start=BASE + timedelta(days=1))
    assert engine.baseline(key).mean == pytest.approx(0.0, abs=1.0)


def test_staleness_is_reported() -> None:
    engine = _engine(staleness_days=10)
    key = _key()
    engine.observe(key, 1.0, observed_at=BASE)
    fresh = engine.score(key, 1.0, now=BASE + timedelta(days=1))
    aged = engine.score(key, 1.0, now=BASE + timedelta(days=60))
    assert not fresh.stale
    assert aged.stale


# --------------------------------------------------------------------------- #
# Persistence
# --------------------------------------------------------------------------- #
def test_baselines_survive_a_restart(tmp_path) -> None:
    store = TradingStateStore(tmp_path / "state.sqlite3")
    key = _key()
    first = SeasonalityEngine(store=store, config=SeasonalityConfig())
    _fill(first, key, [7.0, 9.0, 8.0, 10.0, 6.0])

    second = SeasonalityEngine(store=store, config=SeasonalityConfig())
    reloaded = second.baseline(key)
    assert reloaded.sample_count == 5
    assert reloaded.mean == pytest.approx(first.baseline(key).mean)
    assert reloaded.baseline_updated_at is not None

    row = store.fetch_one(
        "select * from seasonality_baseline where metric = ? and day_of_week = ?",
        (key.metric, key.day_of_week),
    )
    assert row is not None
    assert row["baseline_window"] == SeasonalityConfig().baseline_window
    assert row["confidence"] > 0.0


def test_report_counts_thin_and_populated_buckets(tmp_path) -> None:
    engine = SeasonalityEngine(
        store=TradingStateStore(tmp_path / "state.sqlite3"),
        config=SeasonalityConfig(minimum_samples=5),
    )
    _fill(engine, _key("MON"), [1.0] * 10)
    engine.observe(_key("TUE"), 1.0, observed_at=BASE + timedelta(days=1))
    report = engine.report()
    assert report["bucket_count"] == 2
    assert report["thin_bucket_count"] == 1
    assert report["total_observations"] == 11
    assert report["metrics"] == ["open_return_bps"]


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #
def test_keys_from_context_normalises_and_drops_absent_values() -> None:
    pairs = keys_from_context(
        {"Open_Return_BPS": 1.0, "volume_z": None, "spread": float("nan")},
        market_group="kr",
        day_of_week="mon",
        session_phase="opening",
        regime="trend_up",
    )
    assert [key.metric for key, _ in pairs] == ["open_return_bps"]
    key = pairs[0][0]
    assert (key.market_group, key.day_of_week, key.session_phase, key.regime) == (
        "KR",
        "MON",
        "OPENING",
        "TREND_UP",
    )
