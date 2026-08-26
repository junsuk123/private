from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime

from app.cost import TradingCostEngine
from app.cost.round_trip import all_in_round_trip_bps
from app.trading.contracts import Bar, TradePlan
from app.trading.directional import (
    PositionDirection,
    favourable_watermark,
    gross_return_bps,
    parse_direction,
    stop_breached,
    target_reached,
    trailing_breached,
    trailing_price,
)
from app.cost.cost_coverage import minimum_trailing_net_bps


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
        spread_bps: float | None = None,
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
        direction = parse_direction(plan.position_direction)
        trailing_rate = max(
            0.0, float(plan.trailing_policy.get("bps", 0.0)) / 10_000.0
        )
        baseline_cost = self.cost_engine.estimate(
            symbol=plan.symbol,
            market=market,
            venue=venue,
            instrument_type=instrument_type,
            entry_price=limit,
            expected_exit_price=limit,
            quantity=plan.proposed_quantity,
            orderbook_snapshot=_orderbook_from_spread(limit, spread_bps),
        )
        baseline_cost_bps = all_in_round_trip_bps(
            plan.symbol,
            spread_bps=spread_bps,
            fallback_bps=baseline_cost.total_cost_rate * 10_000.0,
        )
        target_gross_bps = gross_return_bps(limit, target, direction)
        trailing_activation_bps = baseline_cost_bps + minimum_trailing_net_bps(
            target_gross_bps,
            baseline_cost_bps,
            configured_floor_bps=float(
                plan.trailing_policy.get("minimum_net_bps", 5.0)
            ),
        )
        deadline = entry_bar.start_time.timestamp() + plan.max_holding_seconds
        exit_price = entry_bar.close
        exit_time = entry_bar.end_time
        exit_reason = "MAX_HOLDING_TIME"
        lows: list[float] = []
        highs: list[float] = []
        watermark = limit
        for bar in future[entry_index:]:
            if bar.start_time.timestamp() > deadline:
                break
            lows.append(bar.low)
            highs.append(bar.high)
            # Use only the PRIOR bar's favourable watermark for the trailing
            # stop. Creating a new high and then claiming the same bar crossed
            # the resulting trail assumes an intrabar order OHLC cannot prove.
            resolved_trailing = trailing_price(
                watermark, trailing_rate, direction
            )
            adverse_price = (
                bar.low if direction is PositionDirection.LONG else bar.high
            )
            favourable_price = (
                bar.high if direction is PositionDirection.LONG else bar.low
            )
            # When stop and target occur inside one OHLC bar, choose the adverse
            # barrier because intrabar ordering is unknown.
            if stop_breached(adverse_price, stop, direction):
                exit_price, exit_time, exit_reason = stop, bar.end_time, "INITIAL_STOP"
                break
            if target_reached(favourable_price, target, direction):
                exit_price, exit_time, exit_reason = target, bar.end_time, "PROFIT_TARGET"
                break
            trailing_locks_profit = (
                gross_return_bps(limit, resolved_trailing, direction)
                >= trailing_activation_bps
            )
            if (
                trailing_rate > 0
                and trailing_locks_profit
                and trailing_breached(
                    adverse_price, resolved_trailing, limit, direction
                )
            ):
                exit_price, exit_time, exit_reason = (
                    resolved_trailing,
                    bar.end_time,
                    "TRAILING_STOP",
                )
                break
            watermark = favourable_watermark(
                watermark, favourable_price, direction
            )
            exit_price, exit_time = bar.close, bar.end_time
        if (
            exit_reason == "MAX_HOLDING_TIME"
            and exit_time.timestamp() < deadline
        ):
            # The stored series ended or crossed a session gap before the
            # strategy's own clock expired. Treating that last close as a time
            # exit invents an outcome the live strategy would not have taken.
            exit_reason = "FUTURE_WINDOW_CENSORED"
        gross_bps = gross_return_bps(limit, exit_price, direction)
        cost = self.cost_engine.estimate(
            symbol=plan.symbol,
            market=market,
            venue=venue,
            instrument_type=instrument_type,
            entry_price=limit,
            expected_exit_price=exit_price,
            quantity=plan.proposed_quantity,
            orderbook_snapshot=_orderbook_from_spread(limit, spread_bps),
        )
        if direction is PositionDirection.LONG:
            mae_bps = min((low / limit - 1 for low in lows), default=0.0) * 10_000
            mfe_bps = max((high / limit - 1 for high in highs), default=0.0) * 10_000
        else:
            mae_bps = min((1 - high / limit for high in highs), default=0.0) * 10_000
            mfe_bps = max((1 - low / limit for low in lows), default=0.0) * 10_000
        # Labels must pay the same point-in-time all-in round trip used by live
        # selection. The fee engine alone neither applies the venue-tape floor nor
        # knows the stored spread unless it is supplied explicitly.
        cost_bps = all_in_round_trip_bps(
            plan.symbol,
            spread_bps=spread_bps,
            fallback_bps=cost.total_cost_rate * 10_000.0,
        )
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
            gross_return_bps=gross_bps,
            cost_bps=cost_bps,
            net_return_bps=gross_bps - cost_bps,
            mae_bps=mae_bps,
            mfe_bps=mfe_bps,
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


def _orderbook_from_spread(
    price: float,
    spread_bps: float | None,
) -> dict[str, float] | None:
    """Reconstruct the book shape the cost engine expects from a stored spread."""
    if price <= 0.0 or spread_bps is None or spread_bps <= 0.0:
        return None
    half_spread = price * float(spread_bps) / 20_000.0
    bid = price - half_spread
    ask = price + half_spread
    if bid <= 0.0 or ask <= bid:
        return None
    return {"bid_price": bid, "ask_price": ask}
