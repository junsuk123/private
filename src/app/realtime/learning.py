from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any

from app.schemas.domain import OrderAction, StrategySignal, TimeSynchronizedTickerFrame
from app.storage import ModelArtifactStore


@dataclass(frozen=True)
class RealtimeSupervisedExample:
    ticker: str
    as_of: datetime
    predicted_action: str
    predicted_score: float
    realized_pnl: float
    realized_return: float
    label: int
    feature_snapshot: dict[str, float | int]


@dataclass(frozen=True)
class HypotheticalTradeResult:
    ticker: str
    entry_time: datetime
    exit_time: datetime
    side: str
    entry_price: float
    exit_price: float
    quantity: int
    realized_pnl: float


def build_realtime_supervised_examples(
    frames: tuple[TimeSynchronizedTickerFrame, ...],
    signals: tuple[StrategySignal, ...],
) -> tuple[RealtimeSupervisedExample, ...]:
    signal_by_ticker = {signal.ticker: signal for signal in signals}
    examples: list[RealtimeSupervisedExample] = []
    for previous, current in _adjacent_frames_by_ticker(frames):
        signal = signal_by_ticker.get(previous.ticker)
        if signal is None:
            continue
        previous_price = _frame_price(previous)
        current_price = _frame_price(current)
        if previous_price <= 0 or current_price <= 0:
            continue
        direction = _action_direction(signal.action)
        realized_return = (current_price - previous_price) / previous_price * direction
        realized_pnl = (current_price - previous_price) * direction
        examples.append(
            RealtimeSupervisedExample(
                ticker=previous.ticker,
                as_of=previous.bucket_end,
                predicted_action=str(signal.action.value if hasattr(signal.action, "value") else signal.action),
                predicted_score=float(signal.score),
                realized_pnl=round(realized_pnl, 6),
                realized_return=round(realized_return, 8),
                label=int(realized_pnl > 0),
                feature_snapshot={
                    "impact_score": previous.impact_score,
                    "event_count": len(previous.events),
                    "quote_count": len(previous.realtime_quotes),
                    "execution_count": len(previous.realtime_executions),
                    "macro_count": len(previous.macro_metrics),
                    "signal_confidence": float(signal.confidence),
                    "signal_score": float(signal.score),
                },
            )
        )
    return tuple(examples)


def run_hypothetical_realtime_test(
    frames: tuple[TimeSynchronizedTickerFrame, ...],
    signals: tuple[StrategySignal, ...],
    *,
    quantity: int = 1,
) -> dict[str, Any]:
    signal_by_ticker = {signal.ticker: signal for signal in signals}
    trades: list[HypotheticalTradeResult] = []
    for previous, current in _adjacent_frames_by_ticker(frames):
        signal = signal_by_ticker.get(previous.ticker)
        if signal is None or signal.action not in {OrderAction.BUY, OrderAction.SELL, OrderAction.REDUCE}:
            continue
        entry = _frame_price(previous)
        exit_ = _frame_price(current)
        if entry <= 0 or exit_ <= 0:
            continue
        direction = _action_direction(signal.action)
        side = "BUY" if direction >= 0 else "SELL"
        trades.append(
            HypotheticalTradeResult(
                ticker=previous.ticker,
                entry_time=previous.bucket_end,
                exit_time=current.bucket_end,
                side=side,
                entry_price=entry,
                exit_price=exit_,
                quantity=quantity,
                realized_pnl=round((exit_ - entry) * direction * quantity, 6),
            )
        )
    total_pnl = round(sum(trade.realized_pnl for trade in trades), 6)
    winning = sum(1 for trade in trades if trade.realized_pnl > 0)
    return {
        "mode": "testing",
        "orders_submitted": 0,
        "hypothetical_trades": trades,
        "trade_count": len(trades),
        "winning_trades": winning,
        "win_rate": winning / len(trades) if trades else 0.0,
        "realized_pnl": total_pnl,
        "generated_at": datetime.now(timezone.utc),
    }


def update_realtime_model_artifacts(
    store: ModelArtifactStore,
    examples: tuple[RealtimeSupervisedExample, ...],
    test_result: dict[str, Any] | None = None,
) -> dict[str, str]:
    saved: dict[str, str] = {}
    saved["realtime_supervised"] = str(
        store.save_json(
            "realtime_supervised:trade_timing",
            {
                "examples": examples,
                "example_count": len(examples),
                "positive_labels": sum(example.label for example in examples),
            },
            model_family="realtime_supervised",
            metadata={"label_source": "inference_action_plus_realtime_realized_pnl"},
        )
    )
    if test_result is not None:
        saved["hypothetical_testing"] = str(
            store.save_json(
                "hypothetical_testing:pnl_report",
                test_result,
                model_family="hypothetical_testing",
                metadata={"orders_submitted": 0},
            )
        )
    return saved


def _adjacent_frames_by_ticker(
    frames: tuple[TimeSynchronizedTickerFrame, ...],
) -> tuple[tuple[TimeSynchronizedTickerFrame, TimeSynchronizedTickerFrame], ...]:
    by_ticker: dict[str, list[TimeSynchronizedTickerFrame]] = {}
    for frame in sorted(frames, key=lambda item: (item.ticker, item.bucket_start)):
        by_ticker.setdefault(frame.ticker, []).append(frame)
    pairs: list[tuple[TimeSynchronizedTickerFrame, TimeSynchronizedTickerFrame]] = []
    for ticker_frames in by_ticker.values():
        pairs.extend(zip(ticker_frames, ticker_frames[1:]))
    return tuple(pairs)


def _frame_price(frame: TimeSynchronizedTickerFrame) -> float:
    if frame.realtime_quotes:
        return float(frame.realtime_quotes[-1].last_price)
    if frame.market_snapshot is not None:
        return float(frame.market_snapshot.last_price)
    return 0.0


def _action_direction(action: OrderAction | str) -> int:
    action_value = action.value if hasattr(action, "value") else str(action)
    if action_value in {"SELL", "REDUCE"}:
        return -1
    return 1
