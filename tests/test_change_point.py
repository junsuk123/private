from __future__ import annotations

from datetime import datetime, timedelta, timezone

from app.graph.change_point import (
    CHANGE_POINT_DETECTED,
    CHANGE_POINT_INSUFFICIENT_HISTORY,
    CHANGE_POINT_REGIME_YOUNG,
    BayesianOnlineChangePointDetector,
    ChangePointConfig,
    unavailable_result,
)


NOW = datetime(2026, 7, 28, 1, 0, tzinfo=timezone.utc)


def _detector(**overrides) -> BayesianOnlineChangePointDetector:
    config = ChangePointConfig(
        min_channel_observations=overrides.pop("min_channel_observations", 5),
        corroborating_channels=overrides.pop("corroborating_channels", 2),
        regime_trust_age_seconds=overrides.pop("regime_trust_age_seconds", 0.0),
        **overrides,
    )
    return BayesianOnlineChangePointDetector(config, state_path=None)


def _feed(detector, values_by_channel, count, *, start=NOW, step=60):
    result = None
    for index in range(count):
        result = detector.update(
            {name: value(index) for name, value in values_by_channel.items()},
            timestamp=start + timedelta(seconds=step * index),
            persist=False,
        )
    return result


def test_no_history_reports_unknown_not_stable():
    detector = _detector()
    result = detector.update({"index_return": 0.001}, timestamp=NOW, persist=False)
    assert result.change_point_probability == 0.0
    assert CHANGE_POINT_INSUFFICIENT_HISTORY in result.reason_codes
    # "I cannot tell" must never be reported as trustworthy history.
    assert not result.history_trustworthy


def test_stationary_series_does_not_declare_a_change_point():
    detector = _detector()
    values = {
        "index_return": lambda i: 0.0005 * (1 if i % 2 else -1),
        "market_volatility": lambda i: 0.004 + 0.0001 * (i % 3),
    }
    result = _feed(detector, values, 40)
    assert result is not None
    assert CHANGE_POINT_DETECTED not in result.reason_codes
    assert result.change_point_probability < 0.5
    assert result.regime_stability > 0.0


def test_regime_shift_in_two_channels_is_detected():
    detector = _detector()
    calm = {
        "index_return": lambda i: 0.0005 * (1 if i % 2 else -1),
        "market_volatility": lambda i: 0.004 + 0.0001 * (i % 3),
    }
    _feed(detector, calm, 40)
    # A KOSPI-style repricing: both channels move to a new level at once.
    shocked = {
        "index_return": lambda i: -0.05 - 0.001 * (i % 3),
        "market_volatility": lambda i: 0.06 + 0.001 * (i % 3),
    }
    results = [
        detector.update(
            {name: value(index) for name, value in shocked.items()},
            timestamp=NOW + timedelta(hours=1, seconds=60 * index),
            persist=False,
        )
        for index in range(6)
    ]
    detected = [item for item in results if CHANGE_POINT_DETECTED in item.reason_codes]
    assert detected, [item.reason_codes for item in results]
    assert detected[0].change_point_probability >= 0.5
    assert not detected[0].history_trustworthy
    # The break is detected promptly, not several minutes late.
    assert results.index(detected[0]) <= 2
    # ...and once the new level has persisted, the detector stops crying wolf:
    # a permanent stand-down would be as useless as never detecting anything.
    assert CHANGE_POINT_DETECTED not in results[-1].reason_codes


def test_single_noisy_channel_cannot_declare_a_change_point_alone():
    """Corroboration: one channel going haywire must not freeze all trading."""
    detector = _detector(corroborating_channels=2)
    calm = {
        "index_return": lambda i: 0.0005 * (1 if i % 2 else -1),
        "market_volatility": lambda i: 0.004 + 0.0001 * (i % 3),
        "market_breadth": lambda i: 0.5 + 0.01 * (i % 3),
    }
    _feed(detector, calm, 40)
    noisy = {
        "index_return": lambda i: 0.0005 * (1 if i % 2 else -1),
        "market_volatility": lambda i: 0.004 + 0.0001 * (i % 3),
        "market_breadth": lambda i: 12.0,  # one obviously broken channel
    }
    result = _feed(detector, noisy, 4, start=NOW + timedelta(hours=1))
    assert result is not None
    assert result.channel_probabilities["market_breadth"] > 0.5
    assert CHANGE_POINT_DETECTED not in result.reason_codes


def test_young_regime_is_not_yet_trustworthy():
    detector = _detector(regime_trust_age_seconds=3_600.0)
    values = {
        "index_return": lambda i: 0.0005 * (1 if i % 2 else -1),
        "market_volatility": lambda i: 0.004 + 0.0001 * (i % 3),
    }
    result = _feed(detector, values, 10, step=10)
    assert result is not None
    assert CHANGE_POINT_REGIME_YOUNG in result.reason_codes
    assert not result.history_trustworthy
    assert result.regime_stability < 1.0


def test_state_round_trips_through_disk(tmp_path):
    path = tmp_path / "cp.json"
    first = BayesianOnlineChangePointDetector(
        ChangePointConfig(min_channel_observations=5, regime_trust_age_seconds=0.0),
        state_path=path,
    )
    for index in range(20):
        first.update(
            {"index_return": 0.0005 * (1 if index % 2 else -1)},
            timestamp=NOW + timedelta(seconds=60 * index),
        )
    assert path.exists()
    second = BayesianOnlineChangePointDetector(
        ChangePointConfig(min_channel_observations=5, regime_trust_age_seconds=0.0),
        state_path=path,
    )
    assert second.snapshot()["observation_count"] == 20
    assert second.snapshot()["channels"]["index_return"] == 20


def test_non_finite_input_is_skipped_not_imputed():
    detector = _detector()
    result = detector.update(
        {"index_return": float("nan"), "market_volatility": None}, timestamp=NOW, persist=False
    )
    assert result.channel_probabilities == {}
    assert result.observation_count == 0


def test_unavailable_result_is_honest_about_not_knowing():
    result = unavailable_result(NOW)
    assert result.change_point_probability == 0.0
    assert result.regime_stability == 0.0
    assert not result.history_trustworthy
