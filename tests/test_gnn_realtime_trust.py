from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timedelta, timezone
import json
import sqlite3
import threading
import time

from app.routing.gnn_realtime_trust import (
    GnnRealtimeTrustEvaluator,
    _tail_text_lines,
    _tail_text_lines_across_rotations,
)
from app.strategy.exit_geometry import exit_geometry


def test_tail_reader_streams_only_requested_physical_lines(tmp_path) -> None:
    path = tmp_path / "large.jsonl"
    rows = [f"row-{index}-" + ("x" * 300_000) for index in range(5)]
    path.write_text("\n".join(rows) + "\n", encoding="utf-8")

    tail, has_earlier_rows = _tail_text_lines(path, max_lines=2)

    assert not isinstance(tail, (list, tuple))
    assert list(tail) == rows[-2:]
    assert has_earlier_rows is True


def test_tail_reader_reports_when_file_is_smaller_than_window(tmp_path) -> None:
    path = tmp_path / "small.jsonl"
    path.write_text("one\ntwo", encoding="utf-8")

    tail, has_earlier_rows = _tail_text_lines(path, max_lines=5)

    assert list(tail) == ["one", "two"]
    assert has_earlier_rows is False


def test_tail_reader_continues_across_rotated_journals(tmp_path) -> None:
    path = tmp_path / "shadow.jsonl"
    path.with_name("shadow.jsonl.1").write_text(
        "old-1\nold-2\nold-3\n",
        encoding="utf-8",
    )
    path.write_text("new-1\nnew-2\n", encoding="utf-8")

    tail, has_earlier_rows = _tail_text_lines_across_rotations(
        path,
        max_lines=4,
    )

    assert list(tail) == ["old-2", "old-3", "new-1", "new-2"]
    assert has_earlier_rows is True


def test_forward_live_outcomes_can_authorize_gnn_execution(tmp_path) -> None:
    log_path = tmp_path / "shadow.jsonl"
    database = tmp_path / "realtime.sqlite3"
    metadata_path = tmp_path / "model.json"
    metadata_path.write_text(
        json.dumps({"checkpoint_hash": "checkpoint-v3"}),
        encoding="utf-8",
    )
    base = datetime(2026, 7, 30, 0, 0, tzinfo=timezone.utc)
    with sqlite3.connect(database) as connection:
        connection.execute(
            """
            create table realtime_ticks (
                symbol text not null,
                received_at text not null,
                price real not null
            )
            """
        )
        rows = []
        payloads = []
        for index in range(10):
            observed = base + timedelta(seconds=index * 31)
            rows.extend(
                (
                    ("005930", observed.isoformat(), 100.0),
                    (
                        "005930",
                        (observed + timedelta(seconds=30)).isoformat(),
                        100.5,
                    ),
                )
            )
            payloads.append(
                {
                    "as_of": observed.isoformat(),
                    "symbol": "005930",
                    "decisions": [
                        {
                            "path": "cpu_gnn",
                            "action": "ACTIVATE_STRATEGY",
                            "strategy_id": "intraday_momentum",
                            "probability_success": 0.8,
                            "expected_net_return_bps": 19.0,
                            "total_uncertainty": 0.1,
                            "expected_cost_bps": 1.0,
                            "ontology_compatibility": 0.8,
                            "checkpoint_hash": "checkpoint-v3",
                        }
                    ],
                }
            )
        connection.executemany(
            "insert into realtime_ticks(symbol, received_at, price) values (?, ?, ?)",
            rows,
        )
    log_path.write_text(
        "".join(json.dumps(payload) + "\n" for payload in payloads),
        encoding="utf-8",
    )
    evaluator = GnnRealtimeTrustEvaluator(
        comparison_path=log_path,
        database_path=database,
        checkpoint_metadata_path=metadata_path,
        horizon_seconds=30,
        minimum_samples=10,
        window_samples=20,
        cache_seconds=1,
    )

    result = evaluator.evaluate(base + timedelta(seconds=400))

    assert result.passed
    assert result.sample_count == 10
    assert result.positive_net_rate == 1.0
    assert result.mean_realized_net_bps is not None
    assert result.mean_realized_net_bps > 0
    assert result.strategy_sample_counts == {"intraday_momentum": 10}
    assert result.trusted_strategy_ids == ("intraday_momentum",)
    assert result.trusted_strategy_markets == {"intraday_momentum": ("KRX",)}
    assert result.strategy_market_metrics["intraday_momentum"]["KRX"][
        "entry_authorized"
    ] is True
    assert result.strategy_metrics["intraday_momentum"]["passed"] is True


def test_realtime_trust_only_samples_actionable_ontology_admissible_routes(
    tmp_path,
) -> None:
    log_path = tmp_path / "shadow.jsonl"
    metadata_path = tmp_path / "model.json"
    metadata_path.write_text(
        json.dumps({"checkpoint_hash": "active-checkpoint"}),
        encoding="utf-8",
    )
    base = datetime(2026, 7, 30, 0, 0, tzinfo=timezone.utc)

    def decision(
        *,
        action: str = "ACTIVATE_STRATEGY",
        compatibility: float = 0.8,
        checkpoint_hash: str = "active-checkpoint",
        validation_strategy: str = "intraday_momentum",
    ) -> dict:
        return {
            "path": "cpu_gnn",
            "action": action,
            "strategy_id": "intraday_momentum",
            "validation_strategy_id": validation_strategy,
            "probability_success": 0.7,
            "expected_net_return_bps": 12.0,
            "total_uncertainty": 0.2,
            "expected_cost_bps": 2.0,
            "ontology_compatibility": compatibility,
            "checkpoint_hash": checkpoint_hash,
        }

    payloads = (
        {
            "as_of": base.isoformat(),
            "symbol": "005930",
            "decisions": [decision()],
        },
        {
            "as_of": (base + timedelta(seconds=31)).isoformat(),
            "symbol": "000660",
            "decisions": [decision(action="NO_TRADE")],
        },
        {
            "as_of": (base + timedelta(seconds=62)).isoformat(),
            "symbol": "035420",
            "decisions": [decision(compatibility=0.0)],
        },
        {
            "as_of": (base + timedelta(seconds=93)).isoformat(),
            "symbol": "051910",
            "decisions": [decision(validation_strategy="event_momentum")],
        },
        {
            "as_of": (base + timedelta(seconds=124)).isoformat(),
            "symbol": "068270",
            "decisions": [decision(checkpoint_hash="retired-checkpoint")],
        },
    )
    log_path.write_text(
        "".join(json.dumps(payload) + "\n" for payload in payloads),
        encoding="utf-8",
    )
    evaluator = GnnRealtimeTrustEvaluator(
        comparison_path=log_path,
        database_path=tmp_path / "unused.sqlite3",
        checkpoint_metadata_path=metadata_path,
        horizon_seconds=30,
        minimum_samples=10,
        allow_checkpoint_history=False,
    )

    candidates = evaluator._prediction_candidates(
        base + timedelta(seconds=300)
    )

    assert len(candidates) == 1
    assert candidates[0]["symbol"] == "005930"


def test_realtime_trust_fails_closed_before_enough_mature_samples(tmp_path) -> None:
    evaluator = GnnRealtimeTrustEvaluator(
        comparison_path=tmp_path / "missing.jsonl",
        database_path=tmp_path / "missing.sqlite3",
        minimum_samples=10,
    )

    result = evaluator.evaluate(datetime.now(timezone.utc))

    assert not result.passed
    assert "GNN_TRUST_LOG_MISSING" in result.reason_codes


def test_realtime_trust_refresh_is_single_flight_across_callers(tmp_path) -> None:
    evaluator = GnnRealtimeTrustEvaluator(
        comparison_path=tmp_path / "unused.jsonl",
        database_path=tmp_path / "unused.sqlite3",
        cache_seconds=30,
    )
    calls = 0

    def evaluate_once(now):
        nonlocal calls
        calls += 1
        time.sleep(0.05)
        return evaluator._empty(now, ("TEST_RESULT",))

    evaluator._evaluate_uncached = evaluate_once
    now = datetime.now(timezone.utc)
    with ThreadPoolExecutor(max_workers=8) as pool:
        results = list(pool.map(lambda _: evaluator.evaluate(now), range(8)))

    assert calls == 1
    assert all(result.reason_codes == ("TEST_RESULT",) for result in results)


def test_realtime_trust_can_return_stale_while_refreshing_in_background(
    tmp_path,
) -> None:
    evaluator = GnnRealtimeTrustEvaluator(
        comparison_path=tmp_path / "unused.jsonl",
        database_path=tmp_path / "unused.sqlite3",
        cache_seconds=30,
        stale_while_refresh=True,
    )
    refreshed = threading.Event()

    def evaluate_once(now):
        time.sleep(0.05)
        refreshed.set()
        return evaluator._empty(now, ("REFRESHED",))

    evaluator._evaluate_uncached = evaluate_once
    now = datetime.now(timezone.utc)

    pending = evaluator.evaluate(now)

    assert pending.reason_codes == ("GNN_TRUST_REFRESH_PENDING",)
    assert refreshed.wait(timeout=1.0)
    deadline = time.monotonic() + 1.0
    while evaluator._cached is None and time.monotonic() < deadline:
        time.sleep(0.01)
    assert evaluator.evaluate(now).reason_codes == ("REFRESHED",)


def test_realtime_trust_uses_executable_exit_geometry_for_new_strategies(
    tmp_path,
) -> None:
    evaluator = GnnRealtimeTrustEvaluator(
        comparison_path=tmp_path / "unused.jsonl",
        database_path=tmp_path / "unused.sqlite3",
    )

    assert evaluator._strategy_horizon(
        "ofi_microprice_exhaustion_reversal"
    ) == exit_geometry("ofi_microprice_exhaustion_reversal").max_holding_seconds


def test_validation_only_negative_forecast_is_retained_as_calibration_sample(
    tmp_path,
) -> None:
    log_path = tmp_path / "shadow.jsonl"
    metadata_path = tmp_path / "model.json"
    metadata_path.write_text(
        json.dumps({"checkpoint_hash": "active-checkpoint"}),
        encoding="utf-8",
    )
    observed = datetime(2026, 7, 30, 0, 0, tzinfo=timezone.utc)
    log_path.write_text(
        json.dumps(
            {
                "as_of": observed.isoformat(),
                "symbol": "SOFI",
                "decisions": [{"path": "cpu_gnn", "action": "NO_TRADE"}],
                "validation_candidates": [
                    {
                        "path": "cpu_gnn_validation",
                        "action": "VALIDATE_ONLY",
                        "strategy_id": "vwap_mean_reversion",
                        "validation_strategy_id": "vwap_mean_reversion",
                        "probability_success": 0.3,
                        "expected_net_return_bps": -12.0,
                        "expected_cost_bps": 50.0,
                        "total_uncertainty": 0.4,
                        "ontology_compatibility": 0.7,
                        "checkpoint_hash": "active-checkpoint",
                    }
                ],
            }
        )
        + "\n",
        encoding="utf-8",
    )
    evaluator = GnnRealtimeTrustEvaluator(
        comparison_path=log_path,
        database_path=tmp_path / "unused.sqlite3",
        checkpoint_metadata_path=metadata_path,
        horizon_seconds=30,
        minimum_samples=10,
    )

    candidates = evaluator._prediction_candidates(
        observed + timedelta(seconds=60)
    )

    assert len(candidates) == 1
    assert candidates[0]["strategy_id"] == "vwap_mean_reversion"
    assert candidates[0]["expected_net_return_bps"] == -12.0


def test_valid_samples_are_not_evicted_by_invalid_raw_log_tail(tmp_path) -> None:
    log_path = tmp_path / "shadow.jsonl"
    metadata_path = tmp_path / "model.json"
    metadata_path.write_text(
        json.dumps({"checkpoint_hash": "active-checkpoint"}),
        encoding="utf-8",
    )
    base = datetime(2026, 7, 30, 0, 0, tzinfo=timezone.utc)
    payloads = []
    for index in range(10):
        observed = base + timedelta(seconds=index * 31)
        payloads.append(
            {
                "as_of": observed.isoformat(),
                "symbol": f"KR{index}",
                "validation_candidates": [
                    {
                        "path": "cpu_gnn_validation",
                        "action": "VALIDATE_ONLY",
                        "strategy_id": "intraday_momentum",
                        "validation_strategy_id": "intraday_momentum",
                        "probability_success": 0.7,
                        "expected_net_return_bps": 12.0,
                        "expected_cost_bps": 2.0,
                        "total_uncertainty": 0.2,
                        "ontology_compatibility": 0.8,
                        "checkpoint_hash": "active-checkpoint",
                    }
                ],
            }
        )
    # More than the evaluator's initial 2,000-line physical tail. These rows
    # model a catalog-mismatched checkpoint flooding NO_TRADE diagnostics.
    for index in range(2_500):
        payloads.append(
            {
                "as_of": (base + timedelta(hours=1, seconds=index)).isoformat(),
                "symbol": "NO_VALIDATION",
                "decisions": [
                    {
                        "path": "cpu_gnn",
                        "action": "NO_TRADE",
                        "reason_codes": ["GNN_STRATEGY_CATALOG_MISMATCH"],
                    }
                ],
            }
        )
    log_path.write_text(
        "".join(json.dumps(payload) + "\n" for payload in payloads),
        encoding="utf-8",
    )
    evaluator = GnnRealtimeTrustEvaluator(
        comparison_path=log_path,
        database_path=tmp_path / "unused.sqlite3",
        checkpoint_metadata_path=metadata_path,
        horizon_seconds=30,
        minimum_samples=10,
        window_samples=20,
    )

    candidates = evaluator._prediction_candidates(
        base + timedelta(hours=2)
    )

    assert len(candidates) == 10
    assert {candidate["symbol"] for candidate in candidates} == {
        f"KR{index}" for index in range(10)
    }


def test_horizon_bucket_retains_first_actionable_positive_forecast(
    tmp_path,
) -> None:
    log_path = tmp_path / "shadow.jsonl"
    metadata_path = tmp_path / "model.json"
    metadata_path.write_text(
        json.dumps({"checkpoint_hash": "active-checkpoint"}),
        encoding="utf-8",
    )
    observed = datetime(2026, 7, 30, 0, 0, tzinfo=timezone.utc)

    def payload(at: datetime, expected_net: float) -> dict:
        return {
            "as_of": at.isoformat(),
            "symbol": "SOFI",
            "validation_candidates": [
                {
                    "path": "cpu_gnn_validation",
                    "action": "VALIDATE_ONLY",
                    "strategy_id": "vwap_mean_reversion",
                    "validation_strategy_id": "vwap_mean_reversion",
                    "probability_success": 0.7,
                    "expected_net_return_bps": expected_net,
                    "expected_cost_bps": 50.0,
                    "total_uncertainty": 0.2,
                    "ontology_compatibility": 0.8,
                    "checkpoint_hash": "active-checkpoint",
                }
            ],
        }

    log_path.write_text(
        "\n".join(
            json.dumps(item)
            for item in (
                payload(observed, -8.0),
                payload(observed + timedelta(seconds=10), 15.0),
                payload(observed + timedelta(seconds=20), 25.0),
            )
        )
        + "\n",
        encoding="utf-8",
    )
    evaluator = GnnRealtimeTrustEvaluator(
        comparison_path=log_path,
        database_path=tmp_path / "unused.sqlite3",
        checkpoint_metadata_path=metadata_path,
        horizon_seconds=30,
        minimum_samples=10,
    )

    candidates = evaluator._prediction_candidates(
        observed + timedelta(seconds=60)
    )

    assert len(candidates) == 1
    assert candidates[0]["as_of"] == observed + timedelta(seconds=10)
    assert candidates[0]["expected_net_return_bps"] == 15.0


def test_realtime_outcome_uses_first_strategy_exit_before_horizon_reversal(
    tmp_path,
) -> None:
    database = tmp_path / "realtime.sqlite3"
    observed = datetime(2026, 7, 30, 0, 0, tzinfo=timezone.utc)
    with sqlite3.connect(database) as connection:
        connection.execute(
            """
            create table realtime_ticks (
                symbol text not null,
                received_at text not null,
                price real not null
            )
            """
        )
        connection.executemany(
            "insert into realtime_ticks(symbol, received_at, price) values (?, ?, ?)",
            (
                ("SOFI", observed.isoformat(), 100.0),
                ("SOFI", (observed + timedelta(seconds=10)).isoformat(), 102.0),
                ("SOFI", (observed + timedelta(seconds=30)).isoformat(), 99.5),
            ),
        )
        evaluator = GnnRealtimeTrustEvaluator(
            comparison_path=tmp_path / "unused.jsonl",
            database_path=database,
            horizon_seconds=30,
            minimum_samples=10,
        )
        outcome = evaluator._outcome(
            connection,
            {
                "symbol": "SOFI",
                "as_of": observed,
                "probability": 0.8,
                "uncertainty": 0.1,
                "cost_bps": 49.0,
                "expected_net_return_bps": 30.0,
                "strategy_id": "vwap_mean_reversion",
                "horizon_seconds": 30,
            },
        )

    assert outcome is not None
    assert outcome[0] > 0.0


def test_realtime_outcome_replays_live_trailing_exit(tmp_path) -> None:
    database = tmp_path / "realtime.sqlite3"
    observed = datetime(2026, 7, 30, 0, 0, tzinfo=timezone.utc)
    with sqlite3.connect(database) as connection:
        connection.execute(
            "create table realtime_ticks "
            "(symbol text not null, received_at text not null, price real not null)"
        )
        connection.executemany(
            "insert into realtime_ticks(symbol, received_at, price) values (?, ?, ?)",
            (
                ("005930", observed.isoformat(), 100.0),
                ("005930", (observed + timedelta(seconds=10)).isoformat(), 101.0),
                # 30bps giveback from the favourable 101 watermark is 100.697.
                # The live session exits here, before the later horizon loss.
                ("005930", (observed + timedelta(seconds=20)).isoformat(), 100.6),
                ("005930", (observed + timedelta(seconds=30)).isoformat(), 99.0),
            ),
        )
        evaluator = GnnRealtimeTrustEvaluator(
            comparison_path=tmp_path / "unused.jsonl",
            database_path=database,
            horizon_seconds=30,
            minimum_samples=10,
        )
        outcome = evaluator._outcome(
            connection,
            {
                "symbol": "005930",
                "as_of": observed,
                "probability": 0.8,
                "uncertainty": 0.1,
                "cost_bps": 5.0,
                "expected_net_return_bps": 30.0,
                "strategy_id": "intraday_momentum",
                "position_direction": "LONG",
                "horizon_seconds": 30,
            },
        )

    assert outcome is not None
    assert outcome[0] > 0.0


def test_realtime_outcome_is_direction_aware_for_short_strategy(tmp_path) -> None:
    database = tmp_path / "realtime.sqlite3"
    observed = datetime(2026, 7, 30, 0, 0, tzinfo=timezone.utc)
    with sqlite3.connect(database) as connection:
        connection.execute(
            "create table realtime_ticks "
            "(symbol text not null, received_at text not null, price real not null)"
        )
        connection.executemany(
            "insert into realtime_ticks(symbol, received_at, price) values (?, ?, ?)",
            (
                ("005930", observed.isoformat(), 100.0),
                ("005930", (observed + timedelta(seconds=10)).isoformat(), 98.0),
                ("005930", (observed + timedelta(seconds=30)).isoformat(), 101.0),
            ),
        )
        evaluator = GnnRealtimeTrustEvaluator(
            comparison_path=tmp_path / "unused.jsonl",
            database_path=database,
            horizon_seconds=30,
            minimum_samples=10,
        )
        outcome = evaluator._outcome(
            connection,
            {
                "symbol": "005930",
                "as_of": observed,
                "probability": 0.8,
                "uncertainty": 0.1,
                "cost_bps": 5.0,
                "expected_net_return_bps": 30.0,
                "strategy_id": "opening_range_breakdown",
                "position_direction": "SHORT",
                "horizon_seconds": 30,
            },
        )

    assert outcome is not None
    assert outcome[0] > 0.0


def test_busy_strategy_does_not_evict_another_strategys_positive_edge_samples(
    tmp_path,
) -> None:
    """The trust window is bounded per strategy, not across all of them.

    Every threshold is per strategy, so a global newest-N cut let one
    high-frequency strategy starve another's positive-edge evidence. Observed on
    2026-08-06: three of five strategies reported
    ``CALIBRATED_AWAITING_POSITIVE_EDGE`` — "still gathering evidence" — when the
    evidence existed and said validation had FAILED.
    """
    log_path = tmp_path / "shadow.jsonl"
    database = tmp_path / "realtime.sqlite3"
    metadata_path = tmp_path / "model.json"
    metadata_path.write_text(
        json.dumps({"checkpoint_hash": "active-checkpoint"}),
        encoding="utf-8",
    )
    base = datetime(2026, 8, 6, 2, 0, tzinfo=timezone.utc)
    payloads = []
    rows = []

    def sample(symbol: str, at: datetime, strategy: str, expected_net: float):
        rows.extend(
            (
                (symbol, at.isoformat(), 100.0),
                (symbol, (at + timedelta(seconds=30)).isoformat(), 99.0),
            )
        )
        payloads.append(
            {
                "as_of": at.isoformat(),
                "symbol": symbol,
                "validation_candidates": [
                    {
                        "path": "cpu_gnn_validation",
                        "action": "VALIDATE_ONLY",
                        "strategy_id": strategy,
                        "validation_strategy_id": strategy,
                        "probability_success": 0.6,
                        "expected_net_return_bps": expected_net,
                        "expected_cost_bps": 5.0,
                        "total_uncertainty": 0.2,
                        "ontology_compatibility": 0.8,
                        "checkpoint_hash": "active-checkpoint",
                    }
                ],
            }
        )

    # Six positive-edge forecasts, older than the flood below. Enough to clear
    # minimum_positive_prediction_samples (5) and force a real verdict.
    for index in range(6):
        sample(f"KR{index}", base + timedelta(seconds=index * 31),
               "intraday_momentum", 20.0)
    # A second strategy quoting far more often, occupying the newest samples.
    for index in range(40):
        sample(f"US{index}", base + timedelta(seconds=400 + index * 31),
               "liquidity_shock_reversal", -30.0)

    with sqlite3.connect(database) as connection:
        connection.execute(
            """
            create table realtime_ticks (
                symbol text not null,
                received_at text not null,
                price real not null
            )
            """
        )
        connection.executemany(
            "insert into realtime_ticks(symbol, received_at, price) values (?, ?, ?)",
            rows,
        )
    log_path.write_text(
        "".join(json.dumps(payload) + "\n" for payload in payloads),
        encoding="utf-8",
    )
    evaluator = GnnRealtimeTrustEvaluator(
        comparison_path=log_path,
        database_path=database,
        checkpoint_metadata_path=metadata_path,
        horizon_seconds=30,
        minimum_samples=10,
        window_samples=20,
        cache_seconds=1,
    )

    candidates = evaluator._prediction_candidates(
        base + timedelta(seconds=3_000)
    )
    per_strategy: dict[str, int] = {}
    for candidate in candidates:
        strategy = candidate["strategy_id"]
        per_strategy[strategy] = per_strategy.get(strategy, 0) + 1

    # A global newest-20 cut kept only the busy strategy and dropped all six.
    assert per_strategy["intraday_momentum"] == 6
    assert per_strategy["liquidity_shock_reversal"] == 20

    result = evaluator.evaluate(base + timedelta(seconds=3_000))
    metrics = result.strategy_metrics["intraday_momentum"]

    assert metrics["trade_sample_count"] == 6
    # The verdict is a real failure, not "warming up" — realized net was
    # negative on exactly the forecasts the model called positive.
    assert metrics["execution_validation_stage"] == "POSITIVE_EDGE_VALIDATION_FAILED"
    assert metrics["entry_authorized"] is False
    assert result.trusted_strategy_ids == ()


def test_negative_calibration_does_not_grant_positive_entry_authority(
    tmp_path,
) -> None:
    log_path = tmp_path / "shadow.jsonl"
    database = tmp_path / "realtime.sqlite3"
    metadata_path = tmp_path / "model.json"
    metadata_path.write_text(
        json.dumps({"checkpoint_hash": "checkpoint-v4"}),
        encoding="utf-8",
    )
    base = datetime(2026, 7, 30, 0, 0, tzinfo=timezone.utc)
    payloads = []
    rows = []
    for index in range(10):
        observed = base + timedelta(seconds=index * 31)
        rows.extend(
            (
                ("SOFI", observed.isoformat(), 100.0),
                (
                    "SOFI",
                    (observed + timedelta(seconds=30)).isoformat(),
                    99.5,
                ),
            )
        )
        payloads.append(
            {
                "as_of": observed.isoformat(),
                "symbol": "SOFI",
                "validation_candidates": [
                    {
                        "path": "cpu_gnn_validation",
                        "action": "VALIDATE_ONLY",
                        "strategy_id": "breakout_volume",
                        "validation_strategy_id": "breakout_volume",
                        "probability_success": 0.1,
                        "expected_net_return_bps": -40.0,
                        "expected_cost_bps": 5.0,
                        "total_uncertainty": 0.1,
                        "ontology_compatibility": 0.8,
                        "checkpoint_hash": "checkpoint-v4",
                    }
                ],
            }
        )
    with sqlite3.connect(database) as connection:
        connection.execute(
            """
            create table realtime_ticks (
                symbol text not null,
                received_at text not null,
                price real not null
            )
            """
        )
        connection.executemany(
            "insert into realtime_ticks(symbol, received_at, price) values (?, ?, ?)",
            rows,
        )
    log_path.write_text(
        "".join(json.dumps(payload) + "\n" for payload in payloads),
        encoding="utf-8",
    )
    evaluator = GnnRealtimeTrustEvaluator(
        comparison_path=log_path,
        database_path=database,
        checkpoint_metadata_path=metadata_path,
        horizon_seconds=30,
        minimum_samples=10,
        window_samples=20,
        cache_seconds=1,
    )

    result = evaluator.evaluate(base + timedelta(seconds=400))

    assert result.passed is True
    assert result.calibrated_strategy_ids == ("breakout_volume",)
    assert result.trusted_strategy_ids == ()
    metrics = result.strategy_metrics["breakout_volume"]
    assert metrics["calibration_passed"] is True
    assert metrics["entry_authorized"] is False
    assert metrics["execution_validation_stage"] == (
        "CALIBRATED_AWAITING_POSITIVE_EDGE"
    )


def test_trust_payload_distinguishes_untaught_upside_from_awaiting_samples(
    tmp_path,
    monkeypatch,
) -> None:
    """The dashboard needs "retrain me" and "wait for me" to look different.

    Both render as ``CALIBRATED_AWAITING_POSITIVE_EDGE`` with 0 positive samples,
    but a suppressed upside head never resolves on its own. The API has to say
    which strategies were taught an upside before the UI can show it.
    """
    from app import web

    metadata = {
        "strategy_ids": ["intraday_momentum", "breakout_volume"],
        "minimum_upside_supervision_rows": 20,
        "strategy_supervision": {
            "intraday_momentum": {"upside_rows": 25},
            "breakout_volume": {"upside_rows": 4},
        },
        "label_outcomes_by_market": {
            "KRX": {
                "intraday_momentum": {
                    "filled": 30,
                    "mean_net_return_bps_when_filled": 8.5,
                },
                "breakout_volume": {
                    "filled": 30,
                    "mean_net_return_bps_when_filled": -12.0,
                },
            },
            "US": {
                "intraday_momentum": {
                    "filled": 40,
                    "mean_net_return_bps_when_filled": -35.0,
                }
            },
        },
    }
    path = tmp_path / "rgcn_shadow.json"
    path.write_text(json.dumps(metadata), encoding="utf-8")
    monkeypatch.setattr(
        web._gnn_realtime_trust_evaluator, "checkpoint_metadata_path", path
    )

    payload = web._with_upside_supervision(
        {
            "score": 0.0,
            "sample_count": 0,
            "reason_codes": ["GNN_TRUST_INSUFFICIENT_REALTIME_SAMPLES"],
            "strategy_metrics": {
                "intraday_momentum": {"trade_sample_count": 0},
                "breakout_volume": {"trade_sample_count": 0},
            }
        }
    )

    assert payload["upside_supervised_strategy_ids"] == ["intraday_momentum"]
    assert payload["score_available"] is False
    assert payload["checkpoint_live_authorized"] is False
    assert payload["trust_state"] == "CHECKPOINT_NOT_PROMOTED"
    assert "GNN_CHECKPOINT_NOT_LIVE_AUTHORIZED" in payload["reason_codes"]
    assert payload["minimum_upside_supervision_rows"] == 20
    assert payload["upside_authorized_strategy_markets"] == {
        "intraday_momentum": ["KRX"]
    }
    assert payload["training_strategy_market_metrics"]["intraday_momentum"][
        "KRX"
    ]["mean_net_return_bps_when_filled"] == 8.5
    taught = payload["strategy_metrics"]["intraday_momentum"]
    untaught = payload["strategy_metrics"]["breakout_volume"]
    assert taught["upside_supervised"] is True
    assert taught["upside_training_rows"] == 25
    assert taught["upside_authorized_markets"] == ["KRX"]
    assert untaught["upside_supervised"] is False
    assert untaught["upside_training_rows"] == 4


def test_unreadable_checkpoint_metadata_leaves_the_trust_payload_untouched(
    tmp_path,
    monkeypatch,
) -> None:
    """A diagnostic surface must not be able to break the gate it reports on."""
    from app import web

    monkeypatch.setattr(
        web._gnn_realtime_trust_evaluator,
        "checkpoint_metadata_path",
        tmp_path / "missing.json",
    )
    original = {"passed": True, "strategy_metrics": {"a": {"score": 1.0}}}

    payload = web._with_upside_supervision(dict(original))

    assert payload == original


def test_gnn_runtime_observability_separates_inference_from_validation(
    monkeypatch,
) -> None:
    from app import web

    monkeypatch.setattr(
        web,
        "_live_shadow_state",
        {
            "enabled": True,
            "last_attempt_at": "2026-08-13T14:38:10+00:00",
            "last_success_at": "2026-08-13T14:38:11+00:00",
            "last_symbol": "INTC",
            "generated": 321,
            "errors": {},
        },
    )

    payload = web._with_gnn_runtime_observability(
        {"sample_count": 202, "evaluated_at": "2026-08-13T14:38:12+00:00"},
        now=datetime(2026, 8, 13, 14, 38, 14, tzinfo=timezone.utc),
    )

    assert payload["inference_input_last_received_at"] == "2026-08-13T14:38:10+00:00"
    assert payload["inference_last_completed_at"] == "2026-08-13T14:38:11+00:00"
    assert payload["inference_active"] is True
    assert payload["prediction_persisted_count"] == 321
    assert payload["validation_count"] == 202
    assert payload["validation_count_as_of"] == "2026-08-13T14:38:12+00:00"
