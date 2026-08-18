"""Data freshness: three timestamps, three states, and what each permits."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pytest

from app.data.freshness import (
    DataFreshnessRegistry,
    FreshnessPolicy,
    FreshnessState,
    load_freshness_policies,
)

NOW = datetime(2026, 8, 19, 1, 30, tzinfo=timezone.utc)


@pytest.fixture()
def registry() -> DataFreshnessRegistry:
    return DataFreshnessRegistry()


def test_shipped_policy_marks_the_order_path_critical() -> None:
    policies, default = load_freshness_policies()
    assert policies[("kis_realtime", "trade")].critical
    assert policies[("kis_realtime", "orderbook")].critical
    assert policies[("kis_rest", "account")].critical
    # A macro series is not on the order path and must not be able to stop trading.
    assert not policies[("fred", "macro_series")].critical
    assert default.critical is False


def test_fresh_observation_is_healthy(registry: DataFreshnessRegistry) -> None:
    reading = registry.record_event(
        "kis_realtime", "trade", NOW, received_time=NOW, processed_time=NOW, now=NOW
    )
    assert reading.state is FreshnessState.HEALTHY
    assert not reading.blocks_new_entry
    assert reading.age_seconds == pytest.approx(0.0, abs=1.0)


def test_ageing_walks_healthy_to_degraded_to_stale(registry: DataFreshnessRegistry) -> None:
    policy = registry.policy_for("kis_realtime", "trade")
    registry.record_event("kis_realtime", "trade", NOW)
    healthy = registry.reading("kis_realtime", "trade", now=NOW)
    degraded = registry.reading(
        "kis_realtime",
        "trade",
        now=NOW + timedelta(seconds=policy.healthy_max_age_seconds + 1),
    )
    stale = registry.reading(
        "kis_realtime",
        "trade",
        now=NOW + timedelta(seconds=policy.degraded_max_age_seconds + 1),
    )
    assert healthy.state is FreshnessState.HEALTHY
    assert degraded.state is FreshnessState.DEGRADED
    assert stale.state is FreshnessState.STALE
    assert stale.blocks_new_entry


def test_never_observed_stream_is_stale_not_absent(registry: DataFreshnessRegistry) -> None:
    registry.expect("kis_realtime", "orderbook")
    reading = registry.reading("kis_realtime", "orderbook", now=NOW)
    assert reading.state is FreshnessState.STALE
    assert "FRESHNESS_NO_OBSERVATION" in reading.reason_codes
    assert reading.blocks_new_entry


def test_receive_lag_is_distinguished_from_age(registry: DataFreshnessRegistry) -> None:
    reading = registry.record_event(
        "kis_realtime",
        "trade",
        NOW,
        received_time=NOW + timedelta(seconds=8),
        processed_time=NOW + timedelta(seconds=8),
        now=NOW + timedelta(seconds=8),
    )
    assert reading.state is FreshnessState.DEGRADED
    assert "FRESHNESS_RECEIVE_LAG" in reading.reason_codes
    assert "FRESHNESS_AGE_STALE" not in reading.reason_codes


def test_process_lag_is_distinguished_from_receive_lag(
    registry: DataFreshnessRegistry,
) -> None:
    reading = registry.record_event(
        "kis_realtime",
        "trade",
        NOW,
        received_time=NOW,
        processed_time=NOW + timedelta(seconds=9),
        now=NOW,
    )
    assert "FRESHNESS_PROCESS_LAG" in reading.reason_codes
    assert "FRESHNESS_RECEIVE_LAG" not in reading.reason_codes


def test_future_event_time_is_treated_as_a_clock_problem(
    registry: DataFreshnessRegistry,
) -> None:
    reading = registry.record_event(
        "kis_realtime", "trade", NOW + timedelta(minutes=5), received_time=NOW, now=NOW
    )
    assert reading.state is FreshnessState.STALE
    assert "FRESHNESS_CLOCK_SKEW" in reading.reason_codes


def test_blocking_reasons_name_the_offending_stream(
    registry: DataFreshnessRegistry,
) -> None:
    registry.record_event("kis_realtime", "trade", NOW - timedelta(hours=1), scope_key="005930")
    reasons = registry.blocking_reasons(now=NOW)
    assert reasons == ("STALE_DATA:kis_realtime/trade:005930",)


def test_non_critical_staleness_does_not_block(registry: DataFreshnessRegistry) -> None:
    registry.record_event("fred", "macro_series", NOW - timedelta(days=30))
    assert registry.blocking_reasons(now=NOW) == ()
    assert registry.worst_state(now=NOW) is FreshnessState.STALE


def test_report_summarises_states_and_blocks(registry: DataFreshnessRegistry) -> None:
    registry.record_event("kis_realtime", "trade", NOW)
    registry.record_event("kis_rest", "account", NOW - timedelta(hours=2))
    report = registry.report(now=NOW)
    assert report["counts"]["HEALTHY"] == 1
    assert report["counts"]["STALE"] == 1
    assert report["blocking_reasons"] == ["STALE_DATA:kis_rest/account"]
    assert report["worst_state"] == "STALE"


def test_policy_rejects_an_inverted_band() -> None:
    with pytest.raises(ValueError):
        FreshnessPolicy(
            source="x",
            data_type="y",
            healthy_max_age_seconds=100.0,
            degraded_max_age_seconds=10.0,
        )
