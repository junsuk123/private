"""The micro layer and the execution catalogue must speak a translatable language.

Measured defect: the micro/ontology layer reports a METHODOLOGY as a
``SelectedStrategy`` enum value (momentum / breakout / mean_reversion /
vwap_reversion) while the execution layer needs a CATALOGUED strategy id to resolve
an algorithm, an exit geometry and a deployment authorisation. The two sets had ZERO
overlap, so every ontology-elected candidate arrived at ``_deployment_authorized`` as
e.g. "momentum", ``get_algorithm("momentum")`` returned None, and the election was
rejected. 0 of 13 catalogued strategies were reachable through the ontology path —
which is why ontology strategy selection produced nothing at all.
"""

from __future__ import annotations

import pytest

from app.graph.macro_micro_common import SelectedStrategy
from app.graph.micro_reasoner import _METHODOLOGY_TO_STRATEGY
from app.strategy.catalog import (
    METHODOLOGY_STRATEGY_ALIASES,
    NON_TRADABLE_MICRO_STRATEGIES,
    STRATEGY_IDS,
    is_known_strategy,
    resolve_strategy_id,
)
from app.technical.strategy_algorithms import get_algorithm


# --------------------------------------------------------------------------- #
# The bridge must be complete and must land on real catalogue entries.         #
# --------------------------------------------------------------------------- #
def test_every_alias_target_is_catalogued() -> None:
    """An alias pointing at a non-catalogued id would reintroduce the whole bug."""
    for source, target in METHODOLOGY_STRATEGY_ALIASES.items():
        assert is_known_strategy(target), f"{source} -> {target} is not catalogued"
        assert get_algorithm(target) is not None, f"{target} has no algorithm"


def test_every_tradable_micro_verdict_can_be_resolved() -> None:
    """Every buy-thesis value of the enum must translate; the rest must not."""
    for member in SelectedStrategy:
        value = member.value
        resolved = resolve_strategy_id(value)
        if value in NON_TRADABLE_MICRO_STRATEGIES:
            assert resolved is None, f"{value} must not resolve to a tradable id"
        else:
            assert resolved is not None, f"{value} has no catalogued target"
            assert is_known_strategy(resolved)


def test_micro_reasoners_own_mapping_is_covered() -> None:
    """Whatever the micro reasoner can emit must be translatable downstream."""
    for emitted in set(_METHODOLOGY_TO_STRATEGY.values()):
        if emitted in NON_TRADABLE_MICRO_STRATEGIES:
            continue
        assert resolve_strategy_id(emitted) is not None, emitted


# --------------------------------------------------------------------------- #
# Resolution semantics                                                         #
# --------------------------------------------------------------------------- #
def test_catalogued_ids_pass_through_unchanged() -> None:
    """The GNN already speaks catalogue ids; translation must not disturb them."""
    for strategy_id in STRATEGY_IDS:
        assert resolve_strategy_id(strategy_id) == strategy_id


@pytest.mark.parametrize("verdict", sorted(NON_TRADABLE_MICRO_STRATEGIES))
def test_non_tradable_verdicts_resolve_to_nothing(verdict: str) -> None:
    """hold / sell / reduce_risk are not buy theses and must never become one."""
    assert resolve_strategy_id(verdict) is None


def test_unknown_name_resolves_to_none_not_a_default() -> None:
    """Defaulting to the first catalogue entry would turn a generic ontology gate
    into an executable trade — the exact failure the election code warns about."""
    assert resolve_strategy_id("no_such_methodology") is None
    assert resolve_strategy_id("") is None
    assert resolve_strategy_id(None) is None


def test_resolution_is_case_and_whitespace_tolerant() -> None:
    assert resolve_strategy_id("  Momentum  ") == "intraday_momentum"


# --------------------------------------------------------------------------- #
# The reachability property that was zero.                                     #
# --------------------------------------------------------------------------- #
def test_ontology_path_can_now_reach_catalogued_strategies() -> None:
    reachable = {
        resolve_strategy_id(value)
        for value in (member.value for member in SelectedStrategy)
        if resolve_strategy_id(value) is not None
    }
    assert reachable, "the ontology path must be able to elect something"
    # Every reachable target must be a real, algorithm-backed strategy.
    for strategy_id in reachable:
        assert is_known_strategy(strategy_id)
    # Regression guard on the measured number: it was 0.
    assert len(reachable) >= 3


def test_aliases_are_mapped_by_thesis_not_collapsed_to_one() -> None:
    """A table that mapped everything to a single strategy would destroy the
    distinction the micro layer spent its work establishing."""
    targets = set(METHODOLOGY_STRATEGY_ALIASES.values())
    assert len(targets) >= 3, f"aliases collapsed to {targets}"
