from __future__ import annotations

import hashlib
import math
from dataclasses import dataclass, field
from datetime import datetime
from statistics import mean, median, pstdev
from typing import Iterable, Sequence

from app.cost import CostBreakdown, TradingCostEngine
from app.evaluation.walk_forward import WalkForwardSplit, walk_forward_splits


@dataclass(frozen=True)
class StrategyTradeObservation:
    strategy_name: str
    ticker: str
    entry_time: datetime
    exit_time: datetime
    entry_price: float
    exit_price: float
    quantity: int = 1
    split: str | None = None
    venue: str = "KRX"
    market: str = "KR"
    instrument_type: str = "domestic_stock"


@dataclass(frozen=True)
class EvaluatedTrade:
    observation: StrategyTradeObservation
    cost_breakdown: CostBreakdown
    gross_return: float
    net_return: float
    gross_profit: float
    net_profit: float
    fee_converted_loss: bool
    break_even_failure: bool


@dataclass(frozen=True)
class StrategyValidationReport:
    validation_id: str
    strategy_name: str
    generated_at: datetime
    train_size: int
    test_size: int
    walk_forward_splits: tuple[WalkForwardSplit, ...]
    evaluated_trades: tuple[EvaluatedTrade, ...]
    gross_total_return: float
    net_total_return: float
    gross_win_rate: float
    net_win_rate: float
    average_cost_per_trade: float
    average_net_profit_per_trade: float
    break_even_failure_ratio: float
    fee_converted_loss_ratio: float
    cost_to_alpha_ratio_mean: float
    cost_to_alpha_ratio_median: float
    out_of_sample_net_return: float
    out_of_sample_sharpe: float
    max_drawdown_after_cost: float
    reality_check_p_value: float | None
    passed: bool
    ontology_tags: tuple[str, ...]
    metadata: dict[str, float | int | str] = field(default_factory=dict)


@dataclass(frozen=True)
class StrategyParameterAdjustment:
    strategy_name: str
    validation_id: str
    passed: bool
    suggested_parameters: dict[str, float | bool | str]
    reason: str


@dataclass(frozen=True)
class RealityCheckConfig:
    train_size: int = 20
    test_size: int = 10
    step_size: int | None = None
    target_net_return: float = 0.0
    max_fee_converted_loss_ratio: float = 0.30
    max_break_even_failure_ratio: float = 0.50
    max_reality_check_p_value: float = 0.10
    bootstrap_iterations: int = 200
    bootstrap_block_size: int = 5
    min_test_trades: int = 1
    conservative_target_step: float = 0.001
    min_target_net_return: float = 0.003
    max_target_net_return: float = 0.02
    min_max_spread_rate: float = 0.0005


class RealityCheckValidator:
    def __init__(
        self,
        config: RealityCheckConfig | None = None,
        cost_engine: TradingCostEngine | None = None,
    ) -> None:
        self.config = config or RealityCheckConfig()
        self.cost_engine = cost_engine or TradingCostEngine()

    def validate(
        self,
        trades: Sequence[StrategyTradeObservation],
        *,
        strategy_name: str | None = None,
        generated_at: datetime | None = None,
    ) -> StrategyValidationReport:
        ordered = tuple(sorted(trades, key=lambda trade: (trade.entry_time, trade.exit_time)))
        if not ordered:
            raise ValueError("RealityCheckValidator requires at least one trade observation")
        selected_strategy = strategy_name or ordered[0].strategy_name
        strategy_trades = tuple(trade for trade in ordered if trade.strategy_name == selected_strategy)
        if not strategy_trades:
            raise ValueError(f"No trades found for strategy {selected_strategy}")

        evaluated = tuple(self._evaluate_trade(trade) for trade in strategy_trades)
        splits = walk_forward_splits(
            evaluated,
            train_size=min(self.config.train_size, max(1, len(evaluated) - self.config.test_size)),
            test_size=min(self.config.test_size, max(1, len(evaluated) // 3 or 1)),
            step_size=self.config.step_size,
        ) if len(evaluated) >= 2 else ()
        test_indices = _out_of_sample_indices(evaluated, splits, self.config)
        out_of_sample = tuple(evaluated[index] for index in test_indices)
        if not out_of_sample:
            out_of_sample = evaluated

        gross_returns = [trade.gross_return for trade in evaluated]
        net_returns = [trade.net_return for trade in evaluated]
        out_net_returns = [trade.net_return for trade in out_of_sample]
        cost_ratios = [trade.cost_breakdown.cost_to_alpha_ratio for trade in evaluated]
        gross_total_return = _compound_return(gross_returns)
        net_total_return = _compound_return(net_returns)
        out_of_sample_net_return = _compound_return(out_net_returns)
        out_of_sample_sharpe = _sharpe(out_net_returns)
        fee_converted_loss_ratio = _ratio(trade.fee_converted_loss for trade in evaluated)
        break_even_failure_ratio = _ratio(trade.break_even_failure for trade in evaluated)
        p_value = _block_bootstrap_p_value(
            out_net_returns,
            iterations=self.config.bootstrap_iterations,
            block_size=self.config.bootstrap_block_size,
        )
        passed = (
            len(out_of_sample) >= self.config.min_test_trades
            and out_of_sample_net_return > self.config.target_net_return
            and out_of_sample_sharpe > 0
            and fee_converted_loss_ratio < self.config.max_fee_converted_loss_ratio
            and break_even_failure_ratio < self.config.max_break_even_failure_ratio
            and (p_value is None or p_value < self.config.max_reality_check_p_value)
        )
        validation_id = _validation_id(selected_strategy, evaluated, net_total_return, out_of_sample_net_return)
        return StrategyValidationReport(
            validation_id=validation_id,
            strategy_name=selected_strategy,
            generated_at=generated_at or datetime.now(tz=strategy_trades[-1].exit_time.tzinfo),
            train_size=self.config.train_size,
            test_size=self.config.test_size,
            walk_forward_splits=splits,
            evaluated_trades=evaluated,
            gross_total_return=gross_total_return,
            net_total_return=net_total_return,
            gross_win_rate=_win_rate(gross_returns),
            net_win_rate=_win_rate(net_returns),
            average_cost_per_trade=mean([trade.cost_breakdown.total_cost for trade in evaluated]),
            average_net_profit_per_trade=mean([trade.net_profit for trade in evaluated]),
            break_even_failure_ratio=break_even_failure_ratio,
            fee_converted_loss_ratio=fee_converted_loss_ratio,
            cost_to_alpha_ratio_mean=mean(cost_ratios),
            cost_to_alpha_ratio_median=median(cost_ratios),
            out_of_sample_net_return=out_of_sample_net_return,
            out_of_sample_sharpe=out_of_sample_sharpe,
            max_drawdown_after_cost=_max_drawdown(net_returns),
            reality_check_p_value=p_value,
            passed=passed,
            ontology_tags=("RealityCheckPassed",) if passed else ("NoOutOfSampleValidation", "DataSnoopingRisk"),
            metadata={
                "out_of_sample_trade_count": len(out_of_sample),
                "evaluated_trade_count": len(evaluated),
            },
        )

    def _evaluate_trade(self, trade: StrategyTradeObservation) -> EvaluatedTrade:
        cost = self.cost_engine.estimate(
            symbol=trade.ticker,
            market=trade.market,
            venue=trade.venue,
            instrument_type=trade.instrument_type,
            entry_price=trade.entry_price,
            expected_exit_price=trade.exit_price,
            quantity=trade.quantity,
            target_net_return=self.config.target_net_return,
        )
        gross_profit = cost.gross_expected_profit
        net_profit = cost.net_expected_profit
        gross_return = cost.gross_expected_return
        net_return = cost.net_expected_return
        return EvaluatedTrade(
            observation=trade,
            cost_breakdown=cost,
            gross_return=gross_return,
            net_return=net_return,
            gross_profit=gross_profit,
            net_profit=net_profit,
            fee_converted_loss=gross_profit > 0 and net_profit <= 0,
            break_even_failure=cost.gross_expected_return < cost.break_even_return,
        )


class StrategyParameterReestimator:
    """Conservative parameter re-estimation from after-cost validation reports."""

    def __init__(self, config: RealityCheckConfig | None = None) -> None:
        self.config = config or RealityCheckConfig()

    def reestimate(
        self,
        report: StrategyValidationReport,
        current_parameters: dict[str, float | bool | str] | None = None,
    ) -> StrategyParameterAdjustment:
        current = dict(current_parameters or {})
        current_target = float(current.get("target_net_return", self.config.min_target_net_return))
        current_spread = float(current.get("max_spread_rate", 0.0015))
        suggested: dict[str, float | bool | str] = {
            "requires_reality_check_passed": True,
            "last_validation_id": report.validation_id,
        }

        if not report.passed:
            suggested["enabled"] = False
            suggested["target_net_return"] = min(
                self.config.max_target_net_return,
                max(current_target + self.config.conservative_target_step, self.config.min_target_net_return),
            )
            suggested["max_spread_rate"] = max(self.config.min_max_spread_rate, current_spread * 0.8)
            return StrategyParameterAdjustment(
                strategy_name=report.strategy_name,
                validation_id=report.validation_id,
                passed=False,
                suggested_parameters=suggested,
                reason="Reality check failed; keep strategy disabled for live use and tighten cost gates.",
            )

        average_oos_return = _average_return(
            trade.net_return for trade in _out_of_sample_trades(report)
        )
        suggested["enabled"] = bool(current.get("enabled", True))
        suggested["target_net_return"] = min(
            self.config.max_target_net_return,
            max(current_target, self.config.min_target_net_return, average_oos_return * 0.5),
        )
        if report.fee_converted_loss_ratio > 0:
            suggested["max_spread_rate"] = max(self.config.min_max_spread_rate, current_spread * 0.9)
        else:
            suggested["max_spread_rate"] = current_spread
        return StrategyParameterAdjustment(
            strategy_name=report.strategy_name,
            validation_id=report.validation_id,
            passed=True,
            suggested_parameters=suggested,
            reason="Reality check passed; update validation id and keep after-cost target conservative.",
        )

    def reestimate_many(
        self,
        reports: Sequence[StrategyValidationReport],
        current_config: dict[str, dict[str, float | bool | str]] | None = None,
    ) -> dict[str, dict[str, float | bool | str]]:
        current_config = current_config or {}
        overrides: dict[str, dict[str, float | bool | str]] = {}
        for report in reports:
            adjustment = self.reestimate(report, current_config.get(report.strategy_name, {}))
            overrides[report.strategy_name] = adjustment.suggested_parameters
        return overrides


def _out_of_sample_indices(
    evaluated: Sequence[EvaluatedTrade],
    splits: Sequence[WalkForwardSplit],
    config: RealityCheckConfig,
) -> tuple[int, ...]:
    if splits:
        return tuple(dict.fromkeys(index for split in splits for index in split.test_indices))
    split_at = max(0, len(evaluated) - min(config.test_size, len(evaluated)))
    return tuple(range(split_at, len(evaluated)))


def _out_of_sample_trades(report: StrategyValidationReport) -> tuple[EvaluatedTrade, ...]:
    indices = _out_of_sample_indices(report.evaluated_trades, report.walk_forward_splits, RealityCheckConfig(test_size=report.test_size))
    return tuple(report.evaluated_trades[index] for index in indices) if indices else report.evaluated_trades


def _average_return(returns: Iterable[float]) -> float:
    values = list(returns)
    return mean(values) if values else 0.0


def _compound_return(returns: Sequence[float]) -> float:
    value = 1.0
    for item in returns:
        value *= 1 + item
    return value - 1


def _win_rate(returns: Sequence[float]) -> float:
    return sum(1 for item in returns if item > 0) / len(returns) if returns else 0.0


def _ratio(flags: Sequence[bool]) -> float:
    values = list(flags)
    return sum(1 for item in values if item) / len(values) if values else 0.0


def _sharpe(returns: Sequence[float]) -> float:
    if len(returns) < 2:
        return 0.0
    deviation = pstdev(returns)
    if deviation <= 0:
        return 0.0
    return mean(returns) / deviation * math.sqrt(len(returns))


def _max_drawdown(returns: Sequence[float]) -> float:
    equity = 1.0
    peak = 1.0
    max_drawdown = 0.0
    for item in returns:
        equity *= 1 + item
        peak = max(peak, equity)
        drawdown = equity / peak - 1
        max_drawdown = min(max_drawdown, drawdown)
    return max_drawdown


def _block_bootstrap_p_value(
    returns: Sequence[float],
    *,
    iterations: int,
    block_size: int,
) -> float | None:
    if len(returns) < 3 or iterations <= 0:
        return None
    observed = mean(returns)
    centered = [item - observed for item in returns]
    block = max(1, min(block_size, len(centered)))
    better_or_equal = 0
    for iteration in range(iterations):
        sample: list[float] = []
        cursor = iteration % len(centered)
        while len(sample) < len(centered):
            for offset in range(block):
                sample.append(centered[(cursor + offset) % len(centered)])
                if len(sample) == len(centered):
                    break
            cursor = (cursor + block + iteration + 1) % len(centered)
        if mean(sample) >= observed:
            better_or_equal += 1
    return (better_or_equal + 1) / (iterations + 1)


def _validation_id(
    strategy_name: str,
    trades: Sequence[EvaluatedTrade],
    net_total_return: float,
    out_of_sample_net_return: float,
) -> str:
    payload = "|".join(
        [
            strategy_name,
            str(len(trades)),
            f"{net_total_return:.10f}",
            f"{out_of_sample_net_return:.10f}",
            trades[0].observation.entry_time.isoformat(),
            trades[-1].observation.exit_time.isoformat(),
        ]
    )
    return f"reality-{hashlib.sha256(payload.encode('utf-8')).hexdigest()[:16]}"
