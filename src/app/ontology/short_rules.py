"""Closed-world short-side ontology rules and the directional permission mask.

Two responsibilities, both advisory (this module can block; it never authorises an
order — RiskManager / ProfitabilityGate / FinalTradeGate remain the real gates):

1. :func:`evaluate_short_facts` — the CLOSED-WORLD rule. A new short arm is
   executable only when every required fact is explicitly true. Anything unstated is
   false.
2. :func:`permitted_arms` — which ``strategy:DIRECTION`` arms a given macro regime
   admits, including the regimes that admit *neither* direction.

Why the closed world is inverted here
-------------------------------------
The long-side ontology is deliberately open-world in places: an unanswerable
permission check returns ``None`` and must not be read as a refusal, because
withholding a trade on missing metadata is its own cost. The short side is the
opposite. A missing borrow fact does not cost a skipped trade, it costs an order the
broker rejects — or worse, accepts and then force-closes at a price it chooses.

So on this side there is no "probably borrowable". Unstated is false, and there is no
combination of absent fields that yields an executable short.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable, Mapping, Sequence

from app.ontology.trading_domain_ontology import (
    NO_NEW_ENTRY_REGIMES,
    SHORT_AUTHORIZATION_FACTS,
    SHORT_REQUIRED_FACTS,
    IntentType,
    ReasonCodes as RC,
)
from app.ontology.trading_fact_builder import TradingFacts


@dataclass(frozen=True)
class ShortFactVerdict:
    """Which short facts hold, and what that implies for executability."""

    established: tuple[str, ...]
    missing: tuple[str, ...]
    reason_codes: tuple[str, ...]
    # True only when every fact in SHORT_REQUIRED_FACTS is established. Deployment
    # authorization is reported separately, because "this order could be executed"
    # and "this strategy is allowed to try" are different questions and conflating
    # them is how a validated-execution-path short trades before it is validated.
    execution_facts_satisfied: bool
    live_probe_authorized: bool
    live_authorized: bool

    @property
    def executable(self) -> bool:
        """Executable == every required fact AND at least probe authorization."""
        return self.execution_facts_satisfied and (
            self.live_probe_authorized or self.live_authorized
        )

    def as_dict(self) -> dict[str, Any]:
        return {
            "established": list(self.established),
            "missing": list(self.missing),
            "reason_codes": list(self.reason_codes),
            "execution_facts_satisfied": self.execution_facts_satisfied,
            "live_probe_authorized": self.live_probe_authorized,
            "live_authorized": self.live_authorized,
            "executable": self.executable,
        }


def evaluate_short_facts(
    facts: TradingFacts,
    *,
    max_borrow_snapshot_age_seconds: float = 120.0,
    min_hours_before_recall: float = 24.0,
    min_liquidity_score: float = 0.45,
    max_days_to_cover: float = 5.0,
) -> ShortFactVerdict:
    """Establish each short fact from ``facts``. Unstated == false.

    Returns the full missing list rather than the first failure: an operator asking
    "why can this short not trade" wants every reason, and one-at-a-time reporting
    turns a single diagnosis into as many evaluation cycles as there are gates.
    """
    established: list[str] = []
    missing: list[str] = []
    reasons: list[str] = []

    def _record(name: str, holds: bool, code: str | None = None) -> None:
        if holds:
            established.append(name)
        else:
            missing.append(name)
            if code:
                reasons.append(code)

    _record(
        "ShortSalePermitted",
        bool(facts.short_sale_permitted),
        RC.SHORT_SALE_NOT_PERMITTED,
    )
    # Availability requires a FRESH observation. A stale locate is not a locate:
    # borrow availability moves intraday and recall waves cluster exactly when these
    # strategies want to be short.
    age = facts.borrow_snapshot_age_seconds
    fresh = age is not None and 0.0 <= age <= max_borrow_snapshot_age_seconds
    _record(
        "BorrowAvailable",
        bool(facts.borrow_available) and fresh,
        RC.SHORT_BORROW_UNAVAILABLE,
    )
    required_quantity = max(1, int(facts.borrow_quantity_required or 1))
    _record(
        "BorrowQuantitySufficient",
        int(facts.borrow_available_quantity or 0) >= required_quantity,
        RC.SHORT_BORROW_QUANTITY_INSUFFICIENT,
    )
    # An unknown fee is unacceptable, not free. Pricing an unpriced borrow at zero is
    # how a negative-expectancy short passes a cost gate.
    fee = facts.borrow_fee_bps_annualised
    _record(
        "BorrowCostAcceptable",
        fee is not None and fee <= facts.max_borrow_fee_bps_annualised,
        RC.SHORT_BORROW_COST_UNACCEPTABLE,
    )
    # Recall risk. An ABSENT deadline is acceptable — an open-ended loan is the normal
    # case, and treating "no deadline" as "due now" would refuse every short. A
    # deadline that is *close* is not acceptable, because the lender then controls the
    # exit timing rather than the strategy.
    remaining = facts.hours_to_recall_deadline
    recall_ok = remaining is None or remaining >= min_hours_before_recall
    if facts.short_squeeze_risk:
        recall_ok = False
    if facts.days_to_cover is not None and facts.days_to_cover > max_days_to_cover:
        recall_ok = False
    _record("RecallRiskAcceptable", recall_ok, RC.SHORT_RECALL_RISK_HIGH)
    # Execution liquidity. A short EXITS by buying, under pressure, so a thin or wide
    # book is worse here than for a long.
    liquidity_ok = (
        facts.has_orderbook
        and facts.orderbook_fresh
        and facts.best_bid > 0
        and facts.best_ask >= facts.best_bid
    )
    _record(
        "ShortExecutionLiquiditySufficient",
        liquidity_ok,
        RC.SHORT_EXECUTION_LIQUIDITY_INSUFFICIENT,
    )
    _record(
        "ShortStrategyShadowValidated",
        bool(facts.short_strategy_shadow_validated),
        RC.SHORT_NOT_SHADOW_VALIDATED,
    )

    probe = bool(facts.short_strategy_live_probe_authorized)
    live = bool(facts.short_strategy_live_authorized)
    if probe:
        established.append("ShortStrategyLiveProbeAuthorized")
    if live:
        established.append("ShortStrategyLiveAuthorized")
    if not probe and not live:
        missing.append("ShortStrategyLiveProbeAuthorized")
        reasons.append(RC.SHORT_NOT_LIVE_PROBE_AUTHORIZED)

    satisfied = not [name for name in SHORT_REQUIRED_FACTS if name in missing]
    if satisfied and (probe or live):
        reasons.append(RC.SHORT_EXECUTABLE)
    return ShortFactVerdict(
        established=tuple(dict.fromkeys(established)),
        missing=tuple(dict.fromkeys(missing)),
        reason_codes=tuple(dict.fromkeys(reasons)),
        execution_facts_satisfied=satisfied,
        live_probe_authorized=probe,
        live_authorized=live,
    )


# --------------------------------------------------------------------------- #
# Directional permission mask                                                  #
# --------------------------------------------------------------------------- #
# Which arms each macro regime admits, as ``strategy_id:DIRECTION``.
#
# Two design points worth stating:
#
# * TREND_DOWN admits LONG cash-equity paths. ``vwap_mean_reversion`` covers a
#   confirmed recovery; ``residual_relative_strength`` covers the stricter case
#   where an individual low-beta stock is rising in absolute terms despite weak
#   breadth. A falling index alone never grants either entry.
# * HIGH_VOL_DISLOCATED admits NO new entry in EITHER direction, only closes. A
#   dislocated book is not a short opportunity; it is a market whose prices are not
#   information, and the correct response to "I cannot read the tape" is not "so I
#   will bet the other way".
REGIME_PERMITTED_ARMS: Mapping[str, tuple[str, ...]] = {
    "TREND_UP": (
        "intraday_momentum:LONG",
        "opening_range_breakout:LONG",
        "residual_relative_strength:LONG",
        "market_intraday_momentum:LONG",
        # Beta-neutral, so it stays valid even while the index rises: it shorts the
        # stock-specific offer rather than the market.
        "residual_relative_weakness:SHORT",
    ),
    "TREND_DOWN": (
        "market_intraday_momentum_short:SHORT",
        "opening_range_breakdown:SHORT",
        "residual_relative_weakness:SHORT",
        "vwap_mean_reversion:LONG",
        # Long-only bear-market path: the algorithm additionally requires
        # positive absolute trend, positive market/sector-neutral residuals,
        # weak breadth and low beta. It is not permission to buy a mere
        # "least-bad decliner".
        "residual_relative_strength:LONG",
    ),
    "HIGH_VOL_TRENDING_DOWN": (
        "market_intraday_momentum_short:SHORT",
        "opening_range_breakdown:SHORT",
        "residual_relative_strength:LONG",
    ),
    "HIGH_VOL_TRENDING_UP": (
        "intraday_momentum:LONG",
        "opening_range_breakout:LONG",
    ),
    "RANGE": (
        "vwap_mean_reversion:LONG",
        "adaptive_anchored_vwap_reversion:LONG",
        "residual_relative_strength:LONG",
        "residual_relative_weakness:SHORT",
    ),
    # No new entries at all. CLOSE_LONG / CLOSE_SHORT remain available because exiting
    # must never be blocked.
    "HIGH_VOL_DISLOCATED": (),
}

# Actions always available regardless of regime. Exiting an open position is never
# gated by a regime rule: a position we cannot exit is an unbounded liability.
ALWAYS_PERMITTED_ACTIONS: tuple[str, ...] = ("CLOSE_LONG", "CLOSE_SHORT", "NO_TRADE")


def permits_new_entry(macro_regime: str | None) -> bool:
    """May ANY new entry be opened in this regime, in either direction?"""
    return str(macro_regime or "").strip().upper() not in NO_NEW_ENTRY_REGIMES


def permitted_arms(macro_regime: str | None) -> tuple[str, ...]:
    """``strategy_id:DIRECTION`` arms this regime admits for a NEW entry.

    An unknown regime returns an empty tuple — fail closed. Returning "everything" for
    an unrecognised regime name would make a typo in the regime classifier silently
    open every arm in both directions.
    """
    regime = str(macro_regime or "").strip().upper()
    if not permits_new_entry(regime):
        return ()
    return REGIME_PERMITTED_ARMS.get(regime, ())


def arm_permitted(
    strategy_id: str, direction: str, macro_regime: str | None
) -> bool | None:
    """Is this arm permitted in this regime?

    ``None`` means unanswerable — the regime is not in the table and no claim is being
    made — which callers must not read as a refusal. That mirrors
    ``macro_strategy_permitted``'s existing convention on the long side.

    ``False`` is a real refusal and is returned whenever the regime IS known.
    """
    regime = str(macro_regime or "").strip().upper()
    if not regime:
        return None
    if not permits_new_entry(regime):
        return False
    if regime not in REGIME_PERMITTED_ARMS:
        return None
    arm = f"{str(strategy_id or '').strip().lower()}:{str(direction or 'LONG').strip().upper()}"
    return arm in REGIME_PERMITTED_ARMS[regime]


def short_intent_for(effect: str) -> IntentType:
    return (
        IntentType.OPEN_SHORT
        if str(effect or "OPEN").strip().upper() == "OPEN"
        else IntentType.CLOSE_SHORT
    )


def blocked_directions(macro_regime: str | None) -> tuple[str, ...]:
    """Directions with no permitted new-entry arm in this regime.

    Used by the dashboard's entry-blockade view to answer "why is nothing trading"
    with "this regime permits no long arm" rather than with a per-candidate reason.
    """
    arms = permitted_arms(macro_regime)
    if not arms:
        return ("LONG", "SHORT")
    directions = {arm.rsplit(":", 1)[-1] for arm in arms}
    return tuple(sorted({"LONG", "SHORT"} - directions))
