"""Node activation must be a measurement, not a timer.

The 3D panel used to sweep its four layers on a 3.6-second wall-clock carousel:
`phaseIndex = floor((now % 3600) / 900)`. It looked like staged inference while
carrying one bit of real information (whether the log was fresh), and it lit the
encoder-input and hidden-message layers, which nothing instruments at all. The
renderer now scales glow and pulse amplitude by the values below, so a node the
pass never evaluated does not move.
"""

from __future__ import annotations

from app.gnn_visualization import _activation_map
from app.strategy.catalog import STRATEGY_IDS


def test_every_catalogue_strategy_gets_a_state() -> None:
    activation = _activation_map({}, active=True)

    assert set(activation["strategies"]) == set(STRATEGY_IDS)


def test_reason_codes_separate_a_shut_gate_from_a_measured_loss() -> None:
    decision = {
        "action": "NO_TRADE",
        "reason_codes": [
            "NON_POSITIVE_NET_EDGE:intraday_momentum",
            "ONTOLOGY_BLOCKED:vwap_mean_reversion",
        ],
    }

    strategies = _activation_map(decision, active=True)["strategies"]

    # Evaluated and found unprofitable: the model looked and said no.
    assert strategies["intraday_momentum"]["state"] == "EVALUATED_NON_POSITIVE"
    # Gate shut before the model was consulted: nothing was measured.
    assert strategies["vwap_mean_reversion"]["state"] == "GATE_BLOCKED"
    # An arm that appears in no reason code was not part of this pass at all, and
    # zero intensity is what stops the renderer animating it.
    assert strategies["rvgi_box_breakout"]["state"] == "UNEVALUATED"
    assert strategies["rvgi_box_breakout"]["intensity"] == 0.0
    assert (
        strategies["intraday_momentum"]["intensity"]
        > strategies["vwap_mean_reversion"]["intensity"]
    )


def test_a_selected_arm_that_still_declined_is_not_shown_as_a_win() -> None:
    elected = _activation_map(
        {"strategy_id": "intraday_momentum", "action": "NO_TRADE"}, active=True
    )["strategies"]["intraday_momentum"]
    traded = _activation_map(
        {"strategy_id": "intraday_momentum", "action": "BUY"}, active=True
    )["strategies"]["intraday_momentum"]

    assert elected["state"] == "ELECTED_NO_TRADE"
    assert traded["state"] == "SELECTED"
    assert traded["intensity"] > elected["intensity"]


def test_stale_log_damps_everything_rather_than_freezing_the_last_frame() -> None:
    decision = {"strategy_id": "intraday_momentum", "action": "BUY"}

    live = _activation_map(decision, active=True)["strategies"]["intraday_momentum"]
    stale = _activation_map(decision, active=False)["strategies"]["intraday_momentum"]

    # A stale record keeps its shape but must not glow like a live one.
    assert stale["intensity"] < live["intensity"]
    assert stale["intensity"] <= 0.12


def test_uninstrumented_layers_say_so_instead_of_taking_a_turn() -> None:
    layers = _activation_map({"reason_codes": ["ONTOLOGY_BLOCKED:gap_context"]}, active=True)["layers"]

    assert layers["input"]["observed"] is False
    assert layers["input"]["reason"] == "ENCODER_INPUT_NOT_LOGGED"
    assert layers["message_passing"]["observed"] is False
    assert layers["strategy_election"]["observed"] is True
    assert layers["strategy_election"]["evaluated"] == 1


def test_only_decoded_channels_are_reported() -> None:
    # A NO_TRADE decision logs every head channel as null; counting keys instead
    # of values had the output layer claiming three decoded channels with nothing
    # in them.
    empty = _activation_map(
        {
            "action": "NO_TRADE",
            "probability_success": None,
            "expected_cost_bps": None,
            "total_uncertainty": None,
        },
        active=True,
    )

    assert empty["channels"] == {}
    assert empty["layers"]["output_decode"] == {"observed": False, "channels": 0}


def test_gross_return_is_reconstructed_from_net_and_cost() -> None:
    activation = _activation_map(
        {
            "action": "BUY",
            "strategy_id": "intraday_momentum",
            "probability_success": 0.61,
            "expected_net_return_bps": 42.0,
            "expected_cost_bps": 28.0,
            "total_uncertainty": 0.4,
        },
        active=True,
    )

    channels = activation["channels"]
    assert channels["gross_return_bps"] == 70.0
    assert channels["probability_success"] == 0.61
    assert activation["layers"]["output_decode"]["channels"] == 4
    assert activation["selected_strategy_id"] == "intraday_momentum"
