"""Walk-forward ablation harness: leakage detection, arms, metrics and slices."""

from __future__ import annotations

import math
import random
from datetime import datetime, timedelta, timezone

import pytest

from app.evaluation.context_ablation import (
    ABLATION_ARMS,
    AblationRow,
    LeakageError,
    assert_no_future_leakage,
    filter_features,
    ml_metrics,
    run_ablation,
    trade_metrics,
)
from app.evaluation.purged_walk_forward import (
    PurgedWalkForwardSplit,
    purged_walk_forward_splits,
)
from app.evaluation.walk_forward import walk_forward_splits

START = datetime(2026, 6, 1, 0, 30, tzinfo=timezone.utc)
DAYS = ("MON", "TUE", "WED", "THU", "FRI")
SESSIONS = ("OPENING", "MORNING_TREND", "MIDDAY", "AFTERNOON", "CLOSING")


def _rows(count: int = 240, *, label_minutes: int = 5, seed: int = 5) -> list[AblationRow]:
    """Rows whose label depends only on the GLOBAL feature.

    Constructed that way on purpose: an ablation that removes the global layer must lose
    almost all of its skill, and every other arm must keep it. A harness that cannot show
    that is not measuring what it claims to.
    """
    rng = random.Random(seed)
    rows: list[AblationRow] = []
    for index in range(count):
        as_of = START + timedelta(minutes=10 * index)
        signal = rng.gauss(0.0, 1.0)
        noise = rng.gauss(0.0, 1.0)
        label = 1 if signal > 0 else 0
        rows.append(
            AblationRow(
                as_of=as_of,
                label_end=as_of + timedelta(minutes=label_minutes),
                features={
                    "global_direction": signal,
                    "domestic_direction": noise,
                    "sector_return": rng.gauss(0.0, 1.0),
                    "session_progress": rng.random(),
                    "day_of_week": float(index % 5),
                    "seasonality_z": rng.gauss(0.0, 1.0),
                    "ontology_suitability": rng.random(),
                    "gnn_trade_quality": rng.random(),
                    "own_return": rng.gauss(0.0, 1.0),
                },
                label=label,
                net_return_bps=(30.0 if label else -25.0),
                turnover=0.01,
                slippage_bps=3.0,
                day=DAYS[index % 5],
                session=SESSIONS[index % 5],
                regime="TREND_UP" if label else "TREND_DOWN",
                sector="semiconductor" if index % 2 else "bio",
            )
        )
    return rows


def _predictor(train, test, removed):
    """A predictor that can only see ``global_direction`` when it is present."""
    return [
        1.0 / (1.0 + math.exp(-2.0 * row.features.get("global_direction", 0.0)))
        for row in test
    ]


# --------------------------------------------------------------------------- #
# Leakage
# --------------------------------------------------------------------------- #
def test_purged_splits_pass_the_leakage_check() -> None:
    rows = _rows(label_minutes=5)
    splits = purged_walk_forward_splits(rows, train_size=80, test_size=20, embargo_count=5)
    assert_no_future_leakage(rows, splits)


def test_a_naive_chronological_split_leaks_and_is_caught() -> None:
    """The failure a plain time split does not catch, and purging exists to remove.

    Labels here resolve 60 minutes after the observation while rows arrive every 10, so
    the last six training rows of every naive split are still resolving inside the test
    window.
    """
    rows = _rows(count=120, label_minutes=60)
    naive = walk_forward_splits(rows, train_size=40, test_size=10)
    naive_as_purged = [
        PurgedWalkForwardSplit(
            train_indices=split.train_indices,
            test_indices=split.test_indices,
            purged_indices=(),
            embargoed_indices=(),
        )
        for split in naive
    ]
    with pytest.raises(LeakageError):
        assert_no_future_leakage(rows, naive_as_purged)

    # The purged split over the same rows survives the check.
    assert_no_future_leakage(
        rows, purged_walk_forward_splits(rows, train_size=40, test_size=10)
    )


def test_the_leakage_message_names_the_offending_row() -> None:
    rows = _rows(count=40, label_minutes=60)
    leaking = [
        PurgedWalkForwardSplit(
            train_indices=(0, 1, 2),
            test_indices=(3, 4),
            purged_indices=(),
            embargoed_indices=(),
        )
    ]
    with pytest.raises(LeakageError) as excinfo:
        assert_no_future_leakage(rows, leaking)
    message = str(excinfo.value)
    assert rows[0].as_of.isoformat() in message
    assert rows[0].label_end.isoformat() in message
    assert rows[3].as_of.isoformat() in message


def test_run_ablation_scores_only_leakage_free_splits() -> None:
    """A long label window is purged away; what survives is scored."""
    rows = _rows(count=200, label_minutes=60)
    result = run_ablation(rows, _predictor, train_size=60, test_size=20, embargo_count=6)
    assert result.split_count > 0
    assert result.by_name()["FULL"].ml.sample_count > 0


# --------------------------------------------------------------------------- #
# Arms
# --------------------------------------------------------------------------- #
def test_every_declared_arm_is_scored() -> None:
    result = run_ablation(
        _rows(), _predictor, train_size=80, test_size=20, embargo_count=5
    )
    assert {arm.name for arm in result.arms} == set(ABLATION_ARMS)
    assert result.split_count > 0


def test_removing_the_layer_that_carries_the_signal_destroys_the_score() -> None:
    result = run_ablation(
        _rows(), _predictor, train_size=80, test_size=20, embargo_count=5
    )
    arms = result.by_name()
    assert arms["FULL"].ml.f1 > 0.8
    assert arms["GLOBAL"].ml.f1 < arms["FULL"].ml.f1
    assert arms["BASE"].ml.f1 < arms["FULL"].ml.f1
    # Layers the signal does not live in must not change the score.
    assert arms["SECTOR"].ml.f1 == pytest.approx(arms["FULL"].ml.f1)
    assert arms["DAY"].ml.f1 == pytest.approx(arms["FULL"].ml.f1)


def test_lift_over_base_is_reported_per_arm() -> None:
    result = run_ablation(
        _rows(), _predictor, train_size=80, test_size=20, embargo_count=5
    )
    lift = result.lift_over("BASE")
    assert lift["FULL"] > 0.0
    assert "BASE" not in lift


def test_filter_features_removes_exactly_one_layer() -> None:
    features = {
        "global_direction": 1.0,
        "domestic_direction": 1.0,
        "session_progress": 1.0,
        "day_of_week": 1.0,
        "own_return": 1.0,
    }
    assert set(filter_features(features, ("global",))) == {
        "domestic_direction",
        "session_progress",
        "day_of_week",
        "own_return",
    }
    assert set(filter_features(features, ())) == set(features)


def test_unknown_arm_is_rejected() -> None:
    with pytest.raises(ValueError):
        run_ablation(_rows(), _predictor, train_size=80, test_size=20, arms=("NONSENSE",))


def test_predictor_returning_the_wrong_count_is_rejected() -> None:
    with pytest.raises(ValueError):
        run_ablation(
            _rows(),
            lambda train, test, removed: [0.5],
            train_size=80,
            test_size=20,
            embargo_count=5,
            arms=("FULL",),
        )


# --------------------------------------------------------------------------- #
# Metrics
# --------------------------------------------------------------------------- #
def test_ml_metrics_on_a_perfect_predictor() -> None:
    metrics = ml_metrics([1, 0, 1, 0], [1.0, 0.0, 1.0, 0.0])
    assert metrics.precision == 1.0
    assert metrics.recall == 1.0
    assert metrics.f1 == 1.0
    assert metrics.brier == 0.0
    assert metrics.calibration_error == 0.0


def test_calibration_error_catches_overconfidence() -> None:
    # Always says 0.9; right half the time.
    metrics = ml_metrics([1, 0] * 50, [0.9] * 100)
    assert metrics.calibration_error == pytest.approx(0.4, abs=1e-6)


def test_trade_metrics_cover_the_declared_set() -> None:
    rows = _rows(count=20)
    metrics = trade_metrics(rows, [True] * len(rows))
    payload = metrics.as_dict()
    assert set(payload) == {
        "return",
        "win_rate",
        "profit_factor",
        "expectancy",
        "sharpe",
        "sortino",
        "MDD",
        "turnover",
        "slippage",
        "trade_count",
    }
    assert metrics.trade_count == len(rows)
    assert metrics.max_drawdown_bps >= 0.0


def test_profit_factor_is_infinite_rather_than_a_flattering_number() -> None:
    rows = [
        AblationRow(
            as_of=START,
            label_end=START,
            features={},
            label=1,
            net_return_bps=10.0,
        )
    ]
    assert trade_metrics(rows, [True]).profit_factor == float("inf")


def test_no_trades_taken_reports_zeros_not_an_error() -> None:
    rows = _rows(count=10)
    metrics = trade_metrics(rows, [False] * len(rows))
    assert metrics.trade_count == 0
    assert metrics.total_return_bps == 0.0


# --------------------------------------------------------------------------- #
# Slices
# --------------------------------------------------------------------------- #
def test_metrics_are_sliced_by_day_session_regime_and_sector() -> None:
    result = run_ablation(
        _rows(), _predictor, train_size=80, test_size=20, embargo_count=5
    )
    full = result.by_name()["FULL"]
    assert set(full.slices) == {"day", "session", "regime", "sector"}
    assert set(full.slices["sector"]) == {"semiconductor", "bio"}
    for buckets in full.slices.values():
        for payload in buckets.values():
            assert "ml" in payload and "trade" in payload


def test_result_serialises_for_a_report() -> None:
    import json

    payload = run_ablation(
        _rows(), _predictor, train_size=80, test_size=20, embargo_count=5
    ).as_dict()
    text = json.dumps(payload, allow_nan=True)
    assert json.loads(text)["arms"]
