from __future__ import annotations

from datetime import datetime, timedelta, timezone

from app.trading.domestic_universe import UniverseDecision, resolve_universe

DAY1 = datetime(2026, 8, 10, 1, 0, tzinfo=timezone.utc)
DAY2 = DAY1 + timedelta(days=1)


def _state(decision: UniverseDecision) -> dict:
    return {"session_date": decision.session_date, "symbols": list(decision.symbols)}


def test_universe_is_locked_for_the_session() -> None:
    """The bug this replaces re-picked every 30 seconds, so nothing survived long
    enough to accumulate the 20 bars a symbol needs before it can be evaluated."""
    first = resolve_universe([f"{i:06d}" for i in range(1, 40)], now=DAY1, size=5)

    # Ranking churns completely, mid-session.
    second = resolve_universe(
        [f"{i:06d}" for i in range(900, 940)], now=DAY1, state=_state(first), size=5
    )

    assert second.symbols == first.symbols
    assert second.source == "session_locked"


def test_a_new_session_may_reselect() -> None:
    first = resolve_universe([f"{i:06d}" for i in range(1, 40)], now=DAY1, size=5)

    second = resolve_universe(
        [f"{i:06d}" for i in range(900, 940)], now=DAY2, state=_state(first), size=5
    )

    assert second.source == "reselected"
    assert second.symbols != first.symbols


def test_hysteresis_keeps_an_incumbent_that_slipped_just_past_the_cut() -> None:
    """A name oscillating around the boundary must not be traded in and out; its
    accumulated history is worth more than a marginal ranking difference."""
    first = resolve_universe(["000001", "000002", "000003"], now=DAY1, size=3)

    # 000003 slips to rank 5 next session but is still inside the widened band.
    ranked = ["000001", "000002", "000009", "000008", "000003"]
    second = resolve_universe(ranked, now=DAY2, state=_state(first), size=3)

    assert "000003" in second.symbols
    assert second.retained


def test_an_incumbent_that_falls_far_out_is_dropped() -> None:
    first = resolve_universe(["000001", "000002", "000003"], now=DAY1, size=3)
    ranked = ["000001", "000002", "000009", "000008", "000007", "000006", "000003"]

    second = resolve_universe(ranked, now=DAY2, state=_state(first), size=3)

    assert "000003" not in second.symbols
    assert "000003" in second.dropped


def test_a_failed_ranking_keeps_the_previous_universe() -> None:
    """A stale universe still has depth; an empty one has nothing. Losing the
    ranking API must not cost the history already accumulated."""
    first = resolve_universe(["000001", "000002"], now=DAY1, size=2)

    second = resolve_universe([], now=DAY2, state=_state(first), size=2)

    assert second.symbols == first.symbols
    assert second.source == "carried_over"


def test_no_ranking_and_no_history_is_empty_not_invented() -> None:
    decision = resolve_universe([], now=DAY1, size=5)

    assert decision.symbols == ()
    assert decision.source == "empty"


def test_selection_is_capped_and_ordered_by_rank() -> None:
    decision = resolve_universe([f"{i:06d}" for i in range(1, 100)], now=DAY1, size=30)

    assert len(decision.symbols) == 30
    assert decision.symbols[0] == "000001"
