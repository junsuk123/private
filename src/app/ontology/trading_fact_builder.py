"""Convert live quote / orderbook / account / model / cost / validation inputs into
normalized ontology facts (:class:`TradingFacts`) the rules reason over.

The builder is defensive: every field has a safe default so a partial/degraded input
never raises, and missing evidence is represented explicitly (e.g. ``has_orderbook``
False, ``validation_state`` UNVALIDATED) rather than assumed favourable.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from app.ontology.trading_domain_ontology import DataTier, ValidationState


@dataclass(frozen=True)
class TradingFacts:
    symbol: str
    side: str = "BUY"                      # BUY / SELL
    is_domestic: bool = True

    # --- execution feasibility ---
    reference_price: float = 0.0           # decision reference (last_price) — NOT executable
    best_bid: float = 0.0
    best_ask: float = 0.0
    has_orderbook: bool = False
    orderbook_fresh: bool = True
    spread_rate: float = 0.0
    exchange: str = ""
    exchange_known: bool = True            # resolved to a real routing venue

    # --- cost-adjusted edge ---
    gross_expected_return: float = 0.0
    net_expected_return: float = 0.0
    required_min_net_return: float = 0.0
    all_in_cost_rate: float = 0.0

    # --- signal / micro ---
    model_score: float = 0.0
    model_confidence: float = 0.0
    signal_family: str = ""                # momentum | mean_reversion | breakout | sentiment | model | ...
    primary_data_tier: str = DataTier.T5_DERIVED
    exit_reason: str = ""                  # SELL only (free-form decision-engine string)

    # --- risk / inventory ---
    held_quantity: float = 0.0
    position_unrealized_rate: float = 0.0
    position_age_seconds: float = 0.0
    inventory_weight: float = 0.0          # position weight of total equity
    downside_risk: float = 0.0
    principal_floor_distance: float = 1.0  # fraction of cushion remaining (1 = far from floor)

    # --- validation evidence ---
    backtest_expectancy: float | None = None
    paper_expectancy: float | None = None
    live_expectancy: float | None = None
    sample_size: int = 0
    oos_positive: bool | None = None       # out-of-sample edge confirmed
    parameter_count: int = 0

    diagnostics: dict[str, Any] = field(default_factory=dict)

    @property
    def is_buy(self) -> bool:
        return str(self.side).upper() == "BUY"

    @property
    def net_edge_positive(self) -> bool:
        return self.net_expected_return > self.required_min_net_return

    @property
    def gross_positive(self) -> bool:
        return self.gross_expected_return > 0.0


def _num(value: Any, default: float = 0.0) -> float:
    try:
        if value is None:
            return default
        return float(value)
    except (TypeError, ValueError):
        return default


def build_trading_facts(
    symbol: str,
    side: str = "BUY",
    *,
    is_domestic: bool | None = None,
    reference_price: Any = 0.0,
    best_bid: Any = 0.0,
    best_ask: Any = 0.0,
    orderbook_fresh: bool = True,
    exchange: str = "",
    exchange_known: bool = True,
    profitability: dict[str, Any] | None = None,
    model_score: Any = 0.0,
    model_confidence: Any = 0.0,
    signal_family: str = "",
    primary_data_tier: str = DataTier.T5_DERIVED,
    exit_reason: str = "",
    held_quantity: Any = 0.0,
    position_unrealized_rate: Any = 0.0,
    position_age_seconds: Any = 0.0,
    inventory_weight: Any = 0.0,
    downside_risk: Any = 0.0,
    principal_floor_distance: Any = 1.0,
    validation: dict[str, Any] | None = None,
    diagnostics: dict[str, Any] | None = None,
) -> TradingFacts:
    """Assemble :class:`TradingFacts` from heterogeneous live inputs.

    ``profitability`` accepts a ProfitabilityGate / cost-engine decision dict
    (keys: gross_expected_return, net_expected_return, required_min_net_return,
    all_in_cost_rate, spread_rate). ``validation`` accepts a metrics dict
    (backtest_expectancy, paper_expectancy, live_expectancy, sample_size,
    oos_positive, parameter_count).
    """
    prof = profitability or {}
    val = validation or {}
    bid = _num(best_bid)
    ask = _num(best_ask)
    has_ob = bid > 0 and ask > 0 and ask >= bid
    spread_rate = _num(prof.get("spread_rate"))
    if spread_rate <= 0 and has_ob:
        mid = (bid + ask) / 2.0
        spread_rate = (ask - bid) / mid if mid > 0 else 0.0
    if is_domestic is None:
        s = str(symbol or "").strip().upper()
        is_domestic = s.isdigit() and len(s) == 6

    return TradingFacts(
        symbol=str(symbol or "").upper(),
        side=str(side or "BUY").upper(),
        is_domestic=bool(is_domestic),
        reference_price=_num(reference_price),
        best_bid=bid,
        best_ask=ask,
        has_orderbook=has_ob,
        orderbook_fresh=bool(orderbook_fresh),
        spread_rate=spread_rate,
        exchange=str(exchange or "").upper(),
        exchange_known=bool(exchange_known),
        gross_expected_return=_num(prof.get("gross_expected_return")),
        net_expected_return=_num(prof.get("net_expected_return")),
        required_min_net_return=_num(prof.get("required_min_net_return")),
        all_in_cost_rate=_num(prof.get("all_in_cost_rate")),
        model_score=_num(model_score),
        model_confidence=_num(model_confidence),
        signal_family=str(signal_family or "").lower(),
        primary_data_tier=str(primary_data_tier or DataTier.T5_DERIVED),
        exit_reason=str(exit_reason or ""),
        held_quantity=_num(held_quantity),
        position_unrealized_rate=_num(position_unrealized_rate),
        position_age_seconds=_num(position_age_seconds),
        inventory_weight=_num(inventory_weight),
        downside_risk=_num(downside_risk),
        principal_floor_distance=_num(principal_floor_distance, 1.0),
        backtest_expectancy=None if val.get("backtest_expectancy") is None else _num(val.get("backtest_expectancy")),
        paper_expectancy=None if val.get("paper_expectancy") is None else _num(val.get("paper_expectancy")),
        live_expectancy=None if val.get("live_expectancy") is None else _num(val.get("live_expectancy")),
        sample_size=int(_num(val.get("sample_size"))),
        oos_positive=val.get("oos_positive"),
        parameter_count=int(_num(val.get("parameter_count"))),
        diagnostics=dict(diagnostics or {}),
    )


def classify_validation_state(facts: TradingFacts, *, min_sample_size: int, overfit_param_count: int) -> ValidationState:
    """Derive the strategy's validation state from its evidence (spec validation ontology)."""
    # Negative net expectancy over a sufficient sample => disabled.
    for expectancy in (facts.live_expectancy, facts.paper_expectancy, facts.backtest_expectancy):
        if expectancy is not None and facts.sample_size >= min_sample_size and expectancy <= 0.0:
            return ValidationState.NEGATIVE_EXPECTANCY
    if facts.live_expectancy is not None and facts.live_expectancy > 0 and facts.sample_size >= min_sample_size:
        return ValidationState.LIVE_VALIDATED
    if facts.paper_expectancy is not None and facts.paper_expectancy > 0 and facts.sample_size >= min_sample_size:
        return ValidationState.PAPER_VALIDATED
    # Overfit suspicion: many parameters and weak/absent out-of-sample evidence.
    if facts.parameter_count >= overfit_param_count and not facts.oos_positive:
        return ValidationState.OVERFIT_SUSPECTED
    if facts.backtest_expectancy is not None:
        return ValidationState.BACKTEST_ONLY
    return ValidationState.UNVALIDATED
