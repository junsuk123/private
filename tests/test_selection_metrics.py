"""Selection metrics must detect ranking skill that accuracy cannot see.

The v4/v5 model cards reported ``success_direction_accuracy`` against a majority
baseline. On a grid that is ~97% trivial negatives, a constant "not a success"
predictor beats a model that ranks well but sits on the wrong side of its
threshold — so the card read as "no predictive power" while the head was
separating winners from losers at AUC 0.74. These tests pin the instrument that
tells the two apart.
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

import numpy as np
import pytest

from app.evaluation.stored_counterfactual import CounterfactualLabel
from app.models.strategy_utility.training import (
    _Snapshot,
    _market_authorization_verdicts,
    _market_purged_split,
    _expected_net_from_raw,
    _rank_auc,
    _selection_metrics,
    _skill_verdict,
    _target_mask,
)
from app.strategy.catalog import STRATEGY_IDS


def _snapshot(symbol: str, day: int) -> _Snapshot:
    as_of = datetime(2026, 8, 1, tzinfo=timezone.utc) + timedelta(days=day)
    return _Snapshot(
        as_of=as_of,
        label_end=as_of + timedelta(hours=1),
        symbol=symbol,
        context=np.zeros(2, dtype=np.float32),
        targets=np.zeros((1, 1), dtype=np.float32),
        target_masks=np.ones((1, 1), dtype=np.float32),
        attractive=False,
        nets=np.zeros(1, dtype=np.float32),
        filled=np.zeros(1, dtype=bool),
    )


def test_rank_auc_matches_hand_computed_values() -> None:
    assert _rank_auc(np.array([1.0, 2.0, 3.0]), np.array([False, False, True])) == 1.0
    assert _rank_auc(np.array([3.0, 2.0, 1.0]), np.array([False, False, True])) == 0.0
    # All scores tied: every pair is a coin flip.
    assert _rank_auc(np.array([1.0, 1.0, 1.0]), np.array([True, False, False])) == 0.5
    # Undefined without both classes.
    assert _rank_auc(np.array([1.0, 2.0]), np.array([True, True])) is None


def test_authorization_score_uses_payoff_not_win_probability_alone() -> None:
    raw = np.zeros((1, len(STRATEGY_IDS), 11), dtype=np.float32)
    high_probability_tiny_payoff = STRATEGY_IDS.index("intraday_momentum")
    lower_probability_large_payoff = STRATEGY_IDS.index("breakout_volume")

    raw[0, high_probability_tiny_payoff, 0] = 2.0
    raw[0, high_probability_tiny_payoff, 4] = -6.0
    raw[0, lower_probability_large_payoff, 0] = 0.0
    raw[0, lower_probability_large_payoff, 4] = 3.0

    scores = _expected_net_from_raw(raw)

    assert (
        scores[0, lower_probability_large_payoff]
        > scores[0, high_probability_tiny_payoff]
    )


def test_perfect_ranking_is_detected_where_accuracy_would_fail() -> None:
    """A head whose threshold is wrong but whose ORDER is right.

    Every score sits below any sensible decision threshold, so a thresholded
    accuracy sees nothing; the ordering is nonetheless perfect.
    """
    rng = np.random.default_rng(3)
    symbols = np.array([f"S{index % 12}" for index in range(240)])
    scores = np.linspace(-9.0, -8.0, 240)          # uniformly "negative"
    nets = np.where(np.argsort(np.argsort(scores)) >= 120, 40.0, -40.0)
    scores = scores + rng.normal(0, 1e-6, scores.size)

    metrics = _selection_metrics(scores, nets, symbols, draws=200)

    assert metrics["selection_auc"] == pytest.approx(1.0, abs=1e-6)
    assert metrics["selection_auc_permutation_p"] < 0.05
    assert metrics["selection_top_decile_net_bps"] == pytest.approx(40.0)


def test_pure_noise_does_not_look_like_skill() -> None:
    rng = np.random.default_rng(5)
    symbols = np.array([f"S{index % 12}" for index in range(240)])
    scores = rng.normal(size=240)
    nets = rng.normal(size=240) * 20.0

    metrics = _selection_metrics(scores, nets, symbols, draws=300)

    assert 0.35 < metrics["selection_auc"] < 0.65
    assert metrics["selection_auc_ci_low"] < 0.5 < metrics["selection_auc_ci_high"]
    assert not _skill_verdict(metrics)["selection_ranking_skill_established"]


def test_between_symbol_separation_is_not_reported_as_timing_skill() -> None:
    """The null must be the within-symbol permutation, never a flat 0.5.

    Here the score carries NO information about which rows of a symbol win — it
    only identifies the symbol, and some symbols win more often. Pooled AUC is
    well above 0.5, and a 0.5 null would call that skill. Shuffling outcomes
    inside each symbol reproduces the same separation, so the permutation null
    lands near the observed value and the p-value stays unimpressed.
    """
    rng = np.random.default_rng(7)
    scores, nets, symbols = [], [], []
    for index in range(14):
        win_rate = 0.05 + 0.9 * (index / 13.0)
        for _ in range(40):
            symbols.append(f"S{index}")
            scores.append(float(index))          # constant within a symbol
            nets.append(30.0 if rng.random() < win_rate else -30.0)
    metrics = _selection_metrics(
        np.asarray(scores), np.asarray(nets), np.asarray(symbols), draws=400
    )

    assert metrics["selection_auc"] > 0.65
    assert metrics["selection_auc_within_symbol_null"] == pytest.approx(
        metrics["selection_auc"], abs=0.05
    )
    assert metrics["selection_auc_permutation_p"] > 0.05
    assert not _skill_verdict(metrics)["selection_ranking_skill_established"]


def test_clustered_bootstrap_is_wider_than_treating_rows_as_independent() -> None:
    """Rows of one symbol are one observation's worth of evidence, not forty.

    Resampling rows would shrink the interval until an unestablished edge looked
    established.
    """
    rng = np.random.default_rng(9)
    scores, nets, symbols = [], [], []
    for index in range(10):
        offset = rng.normal(0, 30.0)             # symbol-level common shock
        for _ in range(40):
            symbols.append(f"S{index}")
            score = rng.normal()
            scores.append(score)
            nets.append(offset + 8.0 * score + rng.normal(0, 5.0))
    scores_a = np.asarray(scores)
    nets_a = np.asarray(nets)

    clustered = _selection_metrics(
        scores_a, nets_a, np.asarray(symbols), draws=400
    )
    pretend_independent = _selection_metrics(
        scores_a,
        nets_a,
        np.asarray([f"row{index}" for index in range(len(scores))]),
        draws=400,
    )

    clustered_width = (
        clustered["selection_top_decile_net_ci_high"]
        - clustered["selection_top_decile_net_ci_low"]
    )
    naive_width = (
        pretend_independent["selection_top_decile_net_ci_high"]
        - pretend_independent["selection_top_decile_net_ci_low"]
    )
    assert clustered_width > naive_width


def test_verdict_separates_ranking_skill_from_a_profitable_edge() -> None:
    established_ranking_only = {
        "selection_auc_ci_low": 0.55,
        "selection_auc_within_symbol_null": 0.65,
        "selection_auc_permutation_p": 0.001,
        "selection_top_decile_net_p_nonpositive": 0.48,
    }
    verdict = _skill_verdict(established_ranking_only)
    assert verdict["selection_ranking_skill_established"] is True
    # Ranks better than chance, but not beyond the symbol-preference null...
    assert verdict["selection_ranking_clears_within_symbol_null"] is False
    # ...and the money question is still a coin flip.
    assert verdict["selection_net_edge_established"] is False

    profitable = dict(established_ranking_only)
    profitable["selection_top_decile_net_p_nonpositive"] = 0.01
    assert _skill_verdict(profitable)["selection_net_edge_established"] is True


def test_verdict_is_false_when_metrics_are_absent() -> None:
    verdict = _skill_verdict({})
    assert verdict["selection_ranking_skill_established"] is False
    assert verdict["selection_net_edge_established"] is False


def test_market_holdouts_are_independent_when_us_tape_is_newer() -> None:
    snapshots = [
        *[_snapshot("005930", day) for day in range(10)],
        *[_snapshot("AAPL", 30 + day) for day in range(10)],
    ]

    train, validation, _purged = _market_purged_split(snapshots)

    assert {item.symbol for item in validation} == {"005930", "AAPL"}
    assert sum(item.symbol == "005930" for item in validation) == 2
    assert sum(item.symbol == "AAPL" for item in validation) == 2
    assert {item.symbol for item in train} == {"005930", "AAPL"}


def test_market_authority_does_not_cross_authorize() -> None:
    passing = {
        "selection_rows": 100,
        "selection_auc_ci_low": 0.60,
        "selection_auc_within_symbol_null": 0.52,
        "selection_auc_permutation_p": 0.01,
        "selection_top_decile_net_p_nonpositive": 0.01,
    }
    failing = {
        **passing,
        "selection_auc_ci_low": 0.45,
        "selection_auc_permutation_p": 0.50,
    }
    metrics = {
        **{f"krx_{key}": value for key, value in passing.items()},
        **{f"us_{key}": value for key, value in failing.items()},
    }

    checks, markets = _market_authorization_verdicts(metrics, base_ready=True)

    assert markets == ["KRX"]
    assert checks["KRX"]["live_authorized"] is True
    assert checks["US"]["live_authorized"] is False


def test_untriggered_strategy_is_not_mislabeled_as_a_realized_loss() -> None:
    at = datetime(2026, 8, 1, tzinfo=timezone.utc)
    untriggered = CounterfactualLabel(
        as_of=at,
        label_end=at + timedelta(minutes=5),
        symbol="AAPL",
        strategy_id="intraday_momentum",
        triggered=False,
        filled=False,
        net_return_bps=0.0,
        cost_bps=50.0,
        exit_reason="NO_TRIGGER",
    )
    realized_loss = CounterfactualLabel(
        as_of=at,
        label_end=at + timedelta(minutes=5),
        symbol="AAPL",
        strategy_id="intraday_momentum",
        triggered=True,
        filled=True,
        net_return_bps=-25.0,
        cost_bps=50.0,
        exit_reason="INITIAL_STOP",
    )

    assert _target_mask(untriggered)[0] == 0.0
    assert _target_mask(realized_loss)[0] == 1.0


def test_constant_and_weak_context_flags_are_reported() -> None:
    """A flag the training window never varied has a single-valued weight.

    Serving can still vary it, and that regime is untrained — not merely rare.
    The card has to say so, because the field looks perfectly healthy from the
    outside.
    """
    from types import SimpleNamespace

    from app.features.strategy_graph_context import (
        STRATEGY_GRAPH_CONTEXT_FIELDS,
        STRATEGY_GRAPH_CONTEXT_SCHEMA,
    )
    from app.models.strategy_utility.training import _context_field_coverage

    fields = list(STRATEGY_GRAPH_CONTEXT_FIELDS)
    always_on = fields.index("rvgi_available")
    rare = fields.index("box_available")
    healthy = fields.index("return_1m_scaled")

    rows = []
    for index in range(1000):
        values = [0.0] * len(fields)
        values[always_on] = 1.0                       # never varies
        values[rare] = 1.0 if index else 0.0          # 0.1% minority support
        values[healthy] = float(index % 7) / 7.0      # genuinely varies
        rows.append(
            SimpleNamespace(symbol="005930", as_of=index, features=tuple(values))
        )

    coverage = _context_field_coverage(tuple(rows), STRATEGY_GRAPH_CONTEXT_SCHEMA)

    assert "rvgi_available" in coverage["context_fields_constant_in_training"]
    assert "box_available" in coverage["context_flags_below_minimum_support"]
    assert "return_1m_scaled" not in coverage["context_fields_constant_in_training"]
    assert "return_1m_scaled" not in coverage["context_flags_below_minimum_support"]


def test_context_coverage_is_skipped_for_legacy_schemas() -> None:
    from app.models.strategy_utility.training import _context_field_coverage

    assert _context_field_coverage((), "realtime_strategy_graph_v4_market") == {}
