"""A balanced tape is a measurement, not a missing one.

``aggressor_imbalance_5s`` is exactly ``0.0`` whenever the five-second window contained no
trade that could be direction-classified — 46.7% of frames on the live KRX feed. The flow
checks used to read ``(f.aggressor_imbalance_5s or -1.0) < threshold``, and because ``0.0``
is falsy that rewrote every balanced tape to maximally sell-side.

It only bit the strategies whose threshold is at or below zero, which are exactly the
mean-reversion and exhaustion arms — the ones whose whole purpose is to fire when the
momentum arms stand down. So in a neutral tape nothing could trade: momentum correctly
declined, and its counterpart was rejected on a reading it was designed to accept.
"""

from __future__ import annotations

import pytest

from app.technical.strategy_algorithms import build_algorithm_registry

#: Threshold <= 0.0 — a neutral or mildly sell-side tape is deliberately admissible.
NEUTRAL_TOLERANT = (
    "liquidity_shock_reversal",
    "ofi_microprice_exhaustion_reversal",
    "market_intraday_momentum",
)

#: Threshold > 0.0 — a balanced tape genuinely fails these, and must keep failing.
FLOW_DEMANDING = (
    "breakout_volume",
    "vwap_mean_reversion",
    "intraday_momentum",
    "overnight_gap_carry",
    "event_momentum",
)


@pytest.fixture(scope="module")
def below():
    registry = build_algorithm_registry()
    return next(iter(registry.values()))._below_minimum


def test_a_missing_reading_is_refused(below) -> None:
    """None means the window produced nothing; a strategy must not fire on a guess."""
    assert below(None, -0.35) is True
    assert below(None, 0.0) is True
    assert below(None, 0.15) is True


def test_a_balanced_tape_clears_a_threshold_that_allows_it(below) -> None:
    assert below(0.0, -0.35) is False
    assert below(0.0, -0.25) is False
    assert below(0.0, 0.0) is False


def test_a_balanced_tape_still_fails_a_threshold_that_demands_buy_side(below) -> None:
    """The fix must not loosen a strategy that legitimately requires positive flow."""
    assert below(0.0, 0.05) is True
    assert below(0.0, 0.15) is True


def test_a_genuinely_sell_side_tape_is_still_refused(below) -> None:
    assert below(-0.40, -0.35) is True
    assert below(-0.01, 0.0) is True


def test_inclusive_thresholds_reject_the_boundary(below) -> None:
    assert below(0.0, 0.0, inclusive=True) is True
    assert below(0.1, 0.0, inclusive=True) is False


# --------------------------------------------------------------------------- #
# The thresholds themselves are the evidence for which arms this was silencing
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("strategy_id", NEUTRAL_TOLERANT)
def test_counter_trend_arms_admit_a_neutral_tape(strategy_id) -> None:
    algorithm = build_algorithm_registry()[strategy_id]
    threshold = algorithm.p("min_aggressor_imbalance")
    assert threshold <= 0.0, "this arm is only interesting because it tolerates flat flow"
    assert algorithm._below_minimum(0.0, threshold) is False


@pytest.mark.parametrize("strategy_id", FLOW_DEMANDING)
def test_momentum_arms_are_unchanged_by_the_fix(strategy_id) -> None:
    algorithm = build_algorithm_registry()[strategy_id]
    threshold = algorithm.p("min_aggressor_imbalance")
    assert threshold > 0.0
    # Same verdict as the old falsy expression produced, for every input that matters.
    for reading in (None, 0.0, -0.5, threshold - 0.001):
        legacy = (reading or -1.0) < threshold
        assert algorithm._below_minimum(reading, threshold) is legacy
    assert algorithm._below_minimum(threshold, threshold) is False


def test_no_strategy_still_uses_the_falsy_idiom() -> None:
    """Guard against the pattern coming back in a new algorithm.

    Scans executable tokens only. The docstring on ``_below_minimum`` quotes the old
    expression on purpose, and a naive text search flags that explanation as the defect
    it is warning about.
    """
    import io
    import re
    import tokenize
    from pathlib import Path

    from app.technical import strategy_algorithms

    source = Path(strategy_algorithms.__file__).read_text()
    code_only: list[str] = []
    for token in tokenize.generate_tokens(io.StringIO(source).readline):
        if token.type in (tokenize.STRING, tokenize.COMMENT):
            continue
        code_only.append(token.string)
    # ``\bor`` so the tail of an identifier does not count: ``price / anchor - 1.0``
    # ends in "or" followed by a negative number and is entirely innocent.
    offenders = re.findall(r"\bor\s*-\s*[0-9]", " ".join(code_only))
    assert offenders == [], f"falsy-zero comparison reintroduced: {offenders}"
