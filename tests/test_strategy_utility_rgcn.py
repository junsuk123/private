from __future__ import annotations

from datetime import datetime, timedelta, timezone

import numpy as np
import pytest

from app.evaluation.stored_counterfactual import CounterfactualLabel
from app.models.strategy_utility import (
    FixedShapeStrategyUtilityModel,
    StrategyUtilityModelConfig,
)
from app.models.strategy_utility.training import (
    _MINIMUM_UPSIDE_SUPERVISION_ROWS,
    _label_outcome_summary,
    _strategy_supervision,
)


def _inputs():
    x = np.ones((1, 3, 4, 5), dtype=np.float32)
    adjacency = np.zeros((1, 3, 2, 4, 4), dtype=np.float32)
    adjacency[:, :, :, np.arange(4), np.arange(4)] = 1
    node_mask = np.ones((1, 3, 4), dtype=np.float32)
    strategy_mask = np.ones((1, 4, 7), dtype=np.float32)
    return x, adjacency, node_mask, strategy_mask


def _model():
    return FixedShapeStrategyUtilityModel(
        StrategyUtilityModelConfig(1, 3, 4, 5, 2, 7, seed=11)
    )


def test_fixed_shape_model_outputs_stock_strategy_utility() -> None:
    output = _model().infer(*_inputs())
    assert output.utility.shape == (1, 4, 7)
    assert output.no_trade_probability.shape == (1, 4)
    assert np.isfinite(output.utility).all()
    assert np.all((output.probability_success >= 0) & (output.probability_success <= 1))


def test_hard_masks_make_candidate_unselectable() -> None:
    inputs = list(_inputs())
    inputs[3][0, 2, 4] = 0
    output = _model().infer(*inputs)
    assert output.utility[0, 2, 4] == -np.inf


def test_padding_has_no_trade_probability_one() -> None:
    inputs = list(_inputs())
    inputs[2][:, :, 3] = 0
    output = _model().infer(*inputs)
    assert np.all(output.utility[0, 3] == -np.inf)
    assert output.no_trade_probability[0, 3] == 1


def test_dynamic_shape_is_rejected() -> None:
    x, adjacency, node_mask, strategy_mask = _inputs()
    with pytest.raises(ValueError, match="fixed shape"):
        _model().infer(x[:, :2], adjacency, node_mask, strategy_mask)


def test_same_seed_and_inputs_are_deterministic() -> None:
    first = _model().infer(*_inputs()).utility
    second = _model().infer(*_inputs()).utility
    np.testing.assert_array_equal(first, second)


def test_checkpoint_roundtrip_preserves_inference(tmp_path) -> None:
    model = _model()
    path = model.save_checkpoint(tmp_path / "rgcn.npz")
    restored = FixedShapeStrategyUtilityModel.load_checkpoint(path)
    np.testing.assert_array_equal(
        model.infer(*_inputs()).utility,
        restored.infer(*_inputs()).utility,
    )


def test_strategy_outcome_summary_separates_gross_edge_from_cost_drag() -> None:
    start = datetime(2026, 8, 3, tzinfo=timezone.utc)
    rows = tuple(
        CounterfactualLabel(
            as_of=start + timedelta(minutes=index),
            label_end=start + timedelta(minutes=index + 1),
            symbol="005930",
            strategy_id="intraday_momentum",
            triggered=True,
            filled=True,
            net_return_bps=-10.0,
            cost_bps=30.0,
            exit_reason="TIME",
        )
        for index in range(20)
    )

    summary = _label_outcome_summary(rows)["intraday_momentum"]

    assert summary["mean_gross_return_bps_when_filled"] == 20.0
    assert summary["mean_cost_bps_when_filled"] == 30.0
    assert summary["performance_diagnosis"] == "EXECUTION_COST_EXCEEDS_GROSS_EDGE"


def _label(index: int, *, net_bps: float, strategy_id: str) -> CounterfactualLabel:
    start = datetime(2026, 8, 3, tzinfo=timezone.utc)
    return CounterfactualLabel(
        as_of=start + timedelta(minutes=index),
        label_end=start + timedelta(minutes=index + 1),
        symbol="005930",
        strategy_id=strategy_id,
        triggered=True,
        filled=True,
        net_return_bps=net_bps,
        cost_bps=30.0,
        exit_reason="TIME",
    )


def test_supervision_counts_the_rows_that_actually_train_each_channel() -> None:
    """MAE trains on losing fills, MFE on winning ones -- count them apart.

    The decoder makes MFE the only positive term in a net-edge forecast, so a
    strategy with many losing fills and two winning ones has a well-taught
    downside and an untaught upside. One number for "samples" hides exactly the
    asymmetry that matters.
    """
    rows = tuple(
        [_label(i, net_bps=-20.0, strategy_id="intraday_momentum") for i in range(40)]
        + [_label(100 + i, net_bps=15.0, strategy_id="intraday_momentum") for i in range(2)]
    )

    supervision = _strategy_supervision(rows)["intraday_momentum"]

    assert supervision["realized_rows"] == 42
    assert supervision["downside_rows"] == 40
    assert supervision["upside_rows"] == 2
    assert supervision["upside_supervised"] is False


def test_upside_supervision_requires_the_documented_minimum() -> None:
    minimum = _MINIMUM_UPSIDE_SUPERVISION_ROWS
    just_under = tuple(
        _label(i, net_bps=15.0, strategy_id="breakout_volume")
        for i in range(minimum - 1)
    )
    exactly_enough = tuple(
        _label(i, net_bps=15.0, strategy_id="breakout_volume")
        for i in range(minimum)
    )

    assert _strategy_supervision(just_under)["breakout_volume"]["upside_supervised"] is False
    assert _strategy_supervision(exactly_enough)["breakout_volume"]["upside_supervised"] is True


def test_unreachable_strategies_report_no_supervision_not_full_coverage() -> None:
    """The bug this replaced: coverage was counted per SNAPSHOT, so a strategy
    that never triggered still certified 1,615 rows and passed a 500-row gate."""
    rows = tuple(
        _label(i, net_bps=15.0, strategy_id="intraday_momentum") for i in range(30)
    )

    supervision = _strategy_supervision(rows)

    assert supervision["intraday_momentum"]["upside_supervised"] is True
    assert supervision["event_momentum"]["labels"] == 0
    assert supervision["event_momentum"]["upside_rows"] == 0
    assert supervision["event_momentum"]["upside_supervised"] is False
