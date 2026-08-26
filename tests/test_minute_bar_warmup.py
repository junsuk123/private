from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
import threading
import time
from types import SimpleNamespace

from app.data.minute_bar_warmup import (
    AdaptiveConcurrencyController,
    HistoricalDataCoordinator,
    HistoricalDependency,
    RequirementResolver,
    SymbolReadiness,
    detect_missing_ranges,
)
from app.data.kis_minute_history import _parse_rows
from app.data.market_capabilities import MarketGroup, Venue
from zoneinfo import ZoneInfo


NOW = datetime.now(timezone.utc).replace(second=0, microsecond=0)


@dataclass
class _Spec:
    strategy_id: str
    minimum_history_bars: int

    def permits_market(self, market: str) -> bool:
        return True


class _Registry:
    def __init__(self):
        self.specs = {"fast": _Spec("fast", 10), "slow": _Spec("slow", 30)}

    def require(self, name):
        return self.specs[name]


class _FeatureProvider:
    @staticmethod
    def historical_dependencies(*, market: str):
        return (HistoricalDependency("feature:test", 1, 40, 50),)


class _Repo:
    def __init__(self, bars=()):
        self.bars = list(bars)
        self.lock = threading.Lock()

    def bars_for_requirement(self, requirement, *, as_of):
        with self.lock:
            return tuple(self.bars[-requirement.preferred_observations :])

    def merge_bars(self, bars):
        with self.lock:
            existing = {bar.minute_start: bar for bar in self.bars}
            existing.update({bar.minute_start: bar for bar in bars})
            self.bars = [existing[key] for key in sorted(existing)]


def _bars(count: int):
    anchor = datetime.now(timezone.utc).replace(second=0, microsecond=0)
    return tuple(
        SimpleNamespace(minute_start=anchor - timedelta(minutes=count - index))
        for index in range(count)
    )


def test_requirement_is_maximum_of_actual_component_metadata() -> None:
    resolver = RequirementResolver(_Registry(), providers=(_FeatureProvider,))
    result = resolver.resolve("ABC", "US", applicable_strategy_ids=("fast", "slow"))
    assert result.minimum_observations == 40
    assert result.preferred_observations == 50
    assert set(result.components) == {"strategy:fast", "strategy:slow", "feature:test"}


def test_gap_detection_skips_non_session_minutes() -> None:
    resolver = RequirementResolver(_Registry())
    requirement = resolver.resolve("ABC", "US", applicable_strategy_ids=("fast",))
    bars = _bars(10)
    missing = detect_missing_ranges(
        requirement,
        bars[:-1],
        as_of=NOW,
        is_expected_bar=lambda stamp, market: stamp != bars[-1].minute_start,
    )
    assert all(item.start != bars[-1].minute_start for item in missing)


def test_complete_cache_never_calls_provider() -> None:
    calls = []
    coordinator = HistoricalDataCoordinator(
        repository=_Repo(_bars(50)),
        fetch_bars=lambda requirement, missing: calls.append((requirement, missing)) or (),
        resolver=RequirementResolver(_Registry(), providers=(_FeatureProvider,)),
        expected_bar=lambda *_: True,
    )
    result = coordinator.request("ABC", "US", applicable_strategy_ids=("slow",)).result(timeout=2)
    assert result["ready"] is True
    assert calls == []
    assert coordinator.status()["metrics"]["historical_bars_reused_from_cache"] == 50


def test_synchronous_cache_observation_is_visible_and_idempotent() -> None:
    bars = _bars(50)
    coordinator = HistoricalDataCoordinator(
        repository=_Repo(bars),
        fetch_bars=lambda *_: (),
        resolver=RequirementResolver(_Registry(), providers=(_FeatureProvider,)),
        expected_bar=lambda *_: True,
    )
    requirement = coordinator.resolver.resolve(
        "ABC", "US", applicable_strategy_ids=("slow",)
    )
    coordinator.observe_ready_cache(requirement, bars)
    coordinator.observe_ready_cache(requirement, bars)
    status = coordinator.status()
    assert status["readiness"]["symbols"]["US:ABC"]["state"] == "DATA_READY"
    assert status["metrics"]["historical_bars_reused_from_cache"] == 50
    assert status["metrics"]["warmup_requests_total"] == 0


def test_stale_cache_uses_one_tail_request_not_each_empty_minute() -> None:
    old_anchor = datetime.now(timezone.utc).replace(second=0, microsecond=0) - timedelta(hours=2)
    stale = tuple(
        SimpleNamespace(minute_start=old_anchor - timedelta(minutes=50 - index))
        for index in range(50)
    )
    calls = []
    coordinator = HistoricalDataCoordinator(
        repository=_Repo(stale),
        fetch_bars=lambda requirement, missing: calls.append(missing) or (),
        resolver=RequirementResolver(_Registry(), providers=(_FeatureProvider,)),
        expected_bar=lambda *_: True,
    )
    result = coordinator.request("ABC", "US", applicable_strategy_ids=("slow",)).result(timeout=2)
    assert result["ready"] is False
    assert len(calls) == 1
    assert len(calls[0]) == 1
    assert "HISTORY_STALE" in coordinator.status()["readiness"]["symbols"]["US:ABC"]["reasons"]


def test_stale_cache_without_progress_uses_retry_cooldown() -> None:
    old_anchor = datetime.now(timezone.utc).replace(second=0, microsecond=0) - timedelta(hours=2)
    stale = tuple(
        SimpleNamespace(minute_start=old_anchor - timedelta(minutes=50 - index))
        for index in range(50)
    )
    calls = 0

    def fetch(*_):
        nonlocal calls
        calls += 1
        return stale[-5:]

    coordinator = HistoricalDataCoordinator(
        repository=_Repo(stale),
        fetch_bars=fetch,
        resolver=RequirementResolver(_Registry(), providers=(_FeatureProvider,)),
        expected_bar=lambda *_: True,
    )
    first = coordinator.request("ABC", "US", applicable_strategy_ids=("slow",))
    assert first.result(timeout=2)["ready"] is False
    second = coordinator.request("ABC", "US", applicable_strategy_ids=("slow",))
    assert second.result(timeout=1)["suppressed"] is True
    assert calls == 1
    metrics = coordinator.status()["metrics"]
    assert metrics["warmup_requests_total"] == 1
    assert metrics["warmup_requests_suppressed"] == 1
    assert "US:ABC:1m" in coordinator.status()["backfill_retry_cooldowns"]


def test_simultaneous_identical_requests_are_coalesced() -> None:
    gate = threading.Event()
    calls = 0

    def fetch(requirement, missing):
        nonlocal calls
        calls += 1
        gate.wait(1)
        return _bars(50)

    coordinator = HistoricalDataCoordinator(
        repository=_Repo(),
        fetch_bars=fetch,
        resolver=RequirementResolver(_Registry(), providers=(_FeatureProvider,)),
        expected_bar=lambda *_: True,
    )
    first = coordinator.request("ABC", "US", applicable_strategy_ids=("slow",))
    second = coordinator.request("ABC", "US", applicable_strategy_ids=("slow",))
    assert first is second
    gate.set()
    assert first.result(timeout=2)["ready"] is True
    assert calls == 1
    assert coordinator.status()["metrics"]["warmup_requests_deduplicated"] == 1


def test_one_symbol_failure_does_not_block_another() -> None:
    repo = _Repo()

    def fetch(requirement, missing):
        if requirement.symbol == "BAD":
            raise RuntimeError("permanent provider rejection")
        return _bars(50)

    coordinator = HistoricalDataCoordinator(
        repository=repo,
        fetch_bars=fetch,
        resolver=RequirementResolver(_Registry(), providers=(_FeatureProvider,)),
        expected_bar=lambda *_: True,
    )
    bad = coordinator.request("BAD", "US", applicable_strategy_ids=("slow",))
    good = coordinator.request("GOOD", "US", applicable_strategy_ids=("slow",))
    try:
        bad.result(timeout=2)
    except RuntimeError:
        pass
    assert good.result(timeout=2)["ready"] is True


def test_adaptive_concurrency_uses_aimd() -> None:
    control = AdaptiveConcurrencyController(hard_limit=8)
    initial = control.current
    control.record(latency_seconds=10, throttled=True, failed=False)
    assert control.current <= initial
    for _ in range(20):
        control.record(latency_seconds=0.01, throttled=False, failed=False)
    assert control.current >= 1
    assert control.current <= control.maximum


def test_reconnect_preserves_repository_and_marks_only_symbols_stale() -> None:
    repo = _Repo(_bars(50))
    coordinator = HistoricalDataCoordinator(
        repository=repo,
        fetch_bars=lambda *_: (),
        resolver=RequirementResolver(_Registry()),
        expected_bar=lambda *_: True,
    )
    before = tuple(repo.bars)
    coordinator.reconnect("US", ("ABC",), NOW)
    assert tuple(repo.bars) == before
    row = coordinator.status()["readiness"]["symbols"]["US:ABC"]
    assert row["state"] == SymbolReadiness.STALE.value


def test_kis_history_parser_normalizes_timestamp_and_rejects_invalid_ohlc() -> None:
    rows = _parse_rows(
        (
            {
                "stck_bsop_date": "20260825",
                "stck_cntg_hour": "093000",
                "stck_oprc": "100",
                "stck_hgpr": "103",
                "stck_lwpr": "99",
                "stck_prpr": "102",
                "cntg_vol": "50",
            },
            {
                "stck_bsop_date": "20260825",
                "stck_cntg_hour": "093100",
                "stck_oprc": "100",
                "stck_hgpr": "98",
                "stck_lwpr": "99",
                "stck_prpr": "102",
            },
        ),
        symbol="005930",
        market=MarketGroup.KR,
        exchange="KRX",
        venue=Venue.KRX,
        tr_id="test",
        default_day=NOW,
        zone=ZoneInfo("Asia/Seoul"),
    )
    assert len(rows) == 1
    assert rows[0].minute_start == datetime(2026, 8, 25, 0, 30, tzinfo=timezone.utc)
    assert rows[0].meta.is_tradeable is False
