from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime

from app.cost import TradingCostEngine
from app.trading.contracts import Bar, TradePlan


@dataclass(frozen=True)
class CounterfactualOutcome:
    as_of: datetime
    symbol: str
    strategy_id: str
    action: str
    filled: bool
    entry_time: datetime | None
    exit_time: datetime | None
    entry_price: float | None
    exit_price: float | None
    exit_reason: str
    gross_return_bps: float
    cost_bps: float
    net_return_bps: float
    mae_bps: float
    mfe_bps: float
    holding_seconds: float


class EventDrivenFillSimulator:
    """Causal long-only simulator with conservative same-bar barrier ordering."""

    def __init__(self, cost_engine: TradingCostEngine | None = None) -> None:
        self.cost_engine = cost_engine or TradingCostEngine()

    def simulate(
        self,
        plan: TradePlan,
        bars: tuple[Bar, ...],
        *,
        as_of: datetime,
        venue: str = "KRX",
        market: str = "KR",
        instrument_type: str = "domestic_stock",
    ) -> CounterfactualOutcome:
        future = tuple(
            sorted(
                (
                    bar
                    for bar in bars
                    if bar.symbol == plan.symbol and bar.start_time >= as_of
                ),
                key=lambda bar: bar.start_time,
            )
        )
        limit = float(
            plan.entry_price_policy.get(
                "limit", plan.entry_price_policy.get("reference", 0.0)
            )
        )
        entry_index = next(
            (
                index
                for index, bar in enumerate(future)
                if bar.start_time <= plan.expires_at and bar.low <= limit <= bar.high
            ),
            None,
        )
        if entry_index is None or limit <= 0:
            return _empty_outcome(plan, as_of, "ENTRY_NOT_FILLED")
        entry_bar = future[entry_index]
        stop = float(plan.initial_stop["price"])
        target = float(plan.profit_policy["price"])
        deadline = entry_bar.start_time.timestamp() + plan.max_holding_seconds
        exit_price = entry_bar.close
        exit_time = entry_bar.end_time
        exit_reason = "MAX_HOLDING_TIME"
        lows: list[float] = []
        highs: list[float] = []
        for bar in future[entry_index:]:
            if bar.start_time.timestamp() > deadline:
                break
            lows.append(bar.low)
            highs.append(bar.high)
            # When both barriers occur inside one OHLC bar, choose the adverse
            # barrier because intrabar ordering is unknown.
            if bar.low <= stop:
                exit_price, exit_time, exit_reason = stop, bar.end_time, "INITIAL_STOP"
                break
            if bar.high >= target:
                exit_price, exit_time, exit_reason = target, bar.end_time, "PROFIT_TARGET"
                break
            exit_price, exit_time = bar.close, bar.end_time
        gross = exit_price / limit - 1
        cost = self.cost_engine.estimate(
            symbol=plan.symbol,
            market=market,
            venue=venue,
            instrument_type=instrument_type,
            entry_price=limit,
            expected_exit_price=exit_price,
            quantity=plan.proposed_quantity,
        )
        mae = min((low / limit - 1 for low in lows), default=0.0)
        mfe = max((high / limit - 1 for high in highs), default=0.0)
        return CounterfactualOutcome(
            as_of=as_of,
            symbol=plan.symbol,
            strategy_id=plan.strategy_id,
            action="TRADE",
            filled=True,
            entry_time=entry_bar.start_time,
            exit_time=exit_time,
            entry_price=limit,
            exit_price=exit_price,
            exit_reason=exit_reason,
            gross_return_bps=gross * 10_000,
            cost_bps=cost.total_cost_rate * 10_000,
            net_return_bps=cost.net_expected_return * 10_000,
            mae_bps=mae * 10_000,
            mfe_bps=mfe * 10_000,
            holding_seconds=max(
                0.0, (exit_time - entry_bar.start_time).total_seconds()
            ),
        )

    def counterfactual_matrix(
        self,
        plans: tuple[TradePlan, ...],
        bars: tuple[Bar, ...],
        *,
        as_of: datetime,
    ) -> tuple[CounterfactualOutcome, ...]:
        outcomes = tuple(self.simulate(plan, bars, as_of=as_of) for plan in plans)
        return outcomes + (
            CounterfactualOutcome(
                as_of=as_of,
                symbol=plans[0].symbol if plans else "",
                strategy_id="NO_TRADE",
                action="NO_TRADE",
                filled=False,
                entry_time=None,
                exit_time=None,
                entry_price=None,
                exit_price=None,
                exit_reason="NO_TRADE_BASELINE",
                gross_return_bps=0,
                cost_bps=0,
                net_return_bps=0,
                mae_bps=0,
                mfe_bps=0,
                holding_seconds=0,
            ),
        )


def _empty_outcome(
    plan: TradePlan, as_of: datetime, reason: str
) -> CounterfactualOutcome:
    return CounterfactualOutcome(
        as_of=as_of,
        symbol=plan.symbol,
        strategy_id=plan.strategy_id,
        action="TRADE",
        filled=False,
        entry_time=None,
        exit_time=None,
        entry_price=None,
        exit_price=None,
        exit_reason=reason,
        gross_return_bps=0,
        cost_bps=0,
        net_return_bps=0,
        mae_bps=0,
        mfe_bps=0,
        holding_seconds=0,
    )
