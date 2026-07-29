from __future__ import annotations

import hashlib
import json
import sqlite3
from collections import defaultdict
from dataclasses import asdict, dataclass
from datetime import datetime, timedelta
from pathlib import Path
from statistics import fmean
from typing import Sequence

from app.backtesting.event_simulator import EventDrivenFillSimulator
from app.evaluation.purged_walk_forward import purged_walk_forward_splits
from app.strategy.experts import ALL_EXPERT_TYPES, ExpertContext
from app.trading.contracts import Bar


@dataclass(frozen=True)
class EvaluationConfig:
    history_bars: int = 30
    horizon_bars: int = 15
    stride_bars: int = 5
    minimum_symbol_bars: int = 100
    quantity: int = 1
    venue: str = "NASD"
    market: str = "US"
    instrument_type: str = "overseas_stock"
    feature_schema_name: str = "counterfactual_quantiles_v1"


@dataclass(frozen=True)
class CounterfactualLabel:
    as_of: datetime
    label_end: datetime
    symbol: str
    strategy_id: str
    triggered: bool
    filled: bool
    net_return_bps: float
    cost_bps: float
    exit_reason: str
    features: tuple[float, ...] = ()


@dataclass(frozen=True)
class _Interval:
    as_of: datetime
    label_end: datetime


def causal_percentile(value: float, history: Sequence[float]) -> float:
    """Empirical percentile using prior observations only."""
    if not history:
        return 0.5
    return sum(item <= value for item in history) / len(history)


def load_minute_bars(database: Path) -> dict[str, tuple[Bar, ...]]:
    by_symbol: dict[str, list[Bar]] = defaultdict(list)
    connection = sqlite3.connect(database)
    try:
        rows = connection.execute(
            """
            SELECT symbol, minute_start, open, high, low, close, volume
            FROM realtime_minute_bars
            ORDER BY symbol, minute_start
            """
        )
        for symbol, start, open_, high, low, close, volume in rows:
            start_time = datetime.fromisoformat(start)
            by_symbol[symbol].append(
                Bar(
                    symbol=symbol,
                    venue="NASD",
                    interval="1m",
                    start_time=start_time,
                    end_time=start_time + timedelta(minutes=1),
                    open=float(open_),
                    high=float(high),
                    low=float(low),
                    close=float(close),
                    volume=float(volume),
                )
            )
    finally:
        connection.close()
    return {symbol: tuple(bars) for symbol, bars in by_symbol.items()}


def build_labels(
    bars_by_symbol: dict[str, tuple[Bar, ...]],
    config: EvaluationConfig,
) -> tuple[CounterfactualLabel, ...]:
    simulator = EventDrivenFillSimulator()
    labels: list[CounterfactualLabel] = []
    experts = tuple(expert_type() for expert_type in ALL_EXPERT_TYPES)
    for symbol, bars in sorted(bars_by_symbol.items()):
        if len(bars) < config.minimum_symbol_bars:
            continue
        returns = [0.0]
        returns.extend(bars[i].close / bars[i - 1].close - 1 for i in range(1, len(bars)))
        volumes = [bar.volume for bar in bars]
        spreads = [max(0.0, (bar.high - bar.low) / bar.close) for bar in bars]
        for index in range(
            config.history_bars,
            len(bars) - config.horizon_bars,
            config.stride_bars,
        ):
            current = bars[index]
            history_start = index - config.history_bars
            return_history = returns[history_start:index]
            volume_history = volumes[history_start:index]
            spread_history = spreads[history_start:index]
            prior_high = max(bar.high for bar in bars[max(0, index - 10) : index])
            vwap_proxy = sum(
                bar.close * max(bar.volume, 1.0) for bar in bars[history_start:index]
            ) / sum(max(bar.volume, 1.0) for bar in bars[history_start:index])
            current_return = returns[index]
            quantiles = {
                "return": causal_percentile(current_return, return_history),
                "volume": causal_percentile(current.volume, volume_history),
                "breakout": causal_percentile(
                    current.close / prior_high - 1,
                    [
                        bars[j].close / max(bar.high for bar in bars[max(0, j - 10) : j]) - 1
                        for j in range(max(10, history_start), index)
                    ],
                ),
                "vwap_deviation": causal_percentile(
                    current.close / vwap_proxy - 1,
                    [bar.close / vwap_proxy - 1 for bar in bars[history_start:index]],
                ),
                "reversion": causal_percentile(current_return, return_history),
                "liquidity_shock": causal_percentile(spreads[index], spread_history),
                "price_drop": causal_percentile(-current_return, [-item for item in return_history]),
                "recovery": causal_percentile(current_return, return_history),
                "liquidity": causal_percentile(current.volume, volume_history),
                # No point-in-time event, sector membership, or exchange-session
                # calendar is present in this store. Closed-world evaluation must
                # not manufacture these features.
                "event_relevance": 0.0,
                "event_direction": 0.0,
                "relative_strength": 0.0,
                "gap": 0.0,
                "opening_confirmation": 0.0,
            }
            as_of = current.end_time
            future = bars[index + 1 : index + 1 + config.horizon_bars]
            context = ExpertContext(
                symbol=symbol,
                as_of=as_of,
                price=current.close,
                proposed_quantity=config.quantity,
                feature_snapshot_id=f"stored:{symbol}:{as_of.isoformat()}",
                utility_evidence_id="counterfactual",
                quantiles={
                    **{name: 1.0 for name in quantiles},
                    "vwap_deviation": 0.0,
                },
            )
            for expert in experts:
                plan = expert.propose(context)
                if plan is None:  # all-one quantiles admit all current experts
                    continue
                triggered = expert.admissible(
                    ExpertContext(
                        symbol=symbol,
                        as_of=as_of,
                        price=current.close,
                        proposed_quantity=config.quantity,
                        feature_snapshot_id=context.feature_snapshot_id,
                        utility_evidence_id=context.utility_evidence_id,
                        quantiles=quantiles,
                    )
                )
                outcome = simulator.simulate(
                    plan,
                    future,
                    as_of=as_of,
                    venue=config.venue,
                    market=config.market,
                    instrument_type=config.instrument_type,
                )
                labels.append(
                    CounterfactualLabel(
                        as_of=as_of,
                        label_end=future[-1].end_time,
                        symbol=symbol,
                        strategy_id=expert.strategy_id,
                        triggered=triggered,
                        filled=outcome.filled,
                        net_return_bps=outcome.net_return_bps,
                        cost_bps=outcome.cost_bps,
                        exit_reason=outcome.exit_reason,
                        features=(
                            _realtime_microstructure_proxy_features(
                                current=current,
                                current_return=current_return,
                                return_history=return_history,
                                vwap_proxy=vwap_proxy,
                            )
                            if config.feature_schema_name == "realtime_microstructure_v1"
                            else (
                                quantiles["return"],
                                quantiles["volume"],
                                quantiles["breakout"],
                                quantiles["vwap_deviation"],
                                quantiles["reversion"],
                                quantiles["liquidity_shock"],
                                quantiles["price_drop"],
                                quantiles["recovery"],
                                quantiles["liquidity"],
                                quantiles["event_relevance"],
                                quantiles["relative_strength"],
                                quantiles["gap"],
                            )
                        ),
                    )
                )
    return tuple(labels)


def _realtime_microstructure_proxy_features(
    *,
    current: Bar,
    current_return: float,
    return_history: Sequence[float],
    vwap_proxy: float,
) -> tuple[float, ...]:
    """Causal minute-bar proxy for the 12-field live slow-intelligence schema.

    Historical minute bars do not contain L2 depth.  Price, spread, signed-flow,
    VWAP, and volatility fields retain the live units while unavailable depth
    fields use neutral closed-world values.  Metadata records this proxy
    provenance so authorization remains limited to order-free shadow inference.
    """

    price_scale = 100_000.0
    bar_range = max(0.0, current.high - current.low)
    spread_bps = (bar_range / current.close * 10_000.0) if current.close else 0.0
    close_location = (
        ((current.close - current.low) - (current.high - current.close)) / bar_range
        if bar_range
        else 0.0
    )
    mean_return = fmean(return_history) if return_history else 0.0
    realized_volatility = (
        (
            sum((item - mean_return) ** 2 for item in return_history)
            / max(1, len(return_history))
        )
        ** 0.5
        if return_history
        else 0.0
    )
    signed_flow = current.volume * (1.0 if current_return > 0 else -1.0 if current_return < 0 else 0.0)
    return (
        current.close / price_scale,
        current.close / price_scale,
        spread_bps / 100.0,
        current.close / price_scale,
        max(-1.0, min(1.0, close_location)),
        signed_flow / 10_000.0,
        max(-1.0, min(1.0, current_return * 10_000.0)),
        vwap_proxy / price_scale,
        realized_volatility * 100.0,
        1.0,
        0.0,
        1.0,
    )


def build_report(
    database: Path,
    *,
    config: EvaluationConfig | None = None,
) -> dict[str, object]:
    selected_config = config or EvaluationConfig()
    bars_by_symbol = load_minute_bars(database)
    labels = build_labels(bars_by_symbol, selected_config)
    config_json = json.dumps(asdict(selected_config), sort_keys=True, separators=(",", ":"))
    data_digest = hashlib.sha256()
    for row in labels:
        data_digest.update(repr(row).encode())
    code_digest = hashlib.sha256()
    for source in (
        Path(__file__),
        Path(__file__).parents[1] / "backtesting" / "event_simulator.py",
        Path(__file__).parents[1] / "strategy" / "experts.py",
    ):
        code_digest.update(source.read_bytes())

    by_strategy: dict[str, list[CounterfactualLabel]] = defaultdict(list)
    for row in labels:
        by_strategy[row.strategy_id].append(row)
    metrics = {}
    for strategy_id, rows in sorted(by_strategy.items()):
        fired = [row for row in rows if row.triggered]
        filled = [row for row in fired if row.filled]
        metrics[strategy_id] = {
            "labels": len(rows),
            "triggered": len(fired),
            "filled": len(filled),
            "fill_rate_when_triggered": len(filled) / len(fired) if fired else None,
            "mean_net_return_bps_when_filled": fmean(
                row.net_return_bps for row in filled
            )
            if filled
            else None,
            "positive_net_rate_when_filled": sum(row.net_return_bps > 0 for row in filled)
            / len(filled)
            if filled
            else None,
        }

    snapshots: dict[tuple[datetime, str], list[CounterfactualLabel]] = defaultdict(list)
    for row in labels:
        snapshots[(row.as_of, row.symbol)].append(row)
    ordered_keys = sorted(snapshots)
    intervals = [
        _Interval(key[0], max(row.label_end for row in snapshots[key])) for key in ordered_keys
    ]
    train_size = max(1, int(len(intervals) * 0.5))
    test_size = max(1, int(len(intervals) * 0.2))
    splits = (
        purged_walk_forward_splits(
            intervals,
            train_size=train_size,
            test_size=test_size,
            embargo_count=max(1, int(len(intervals) * 0.01)),
            step_size=test_size,
        )
        if len(intervals) >= train_size + test_size
        else ()
    )
    selected_returns: list[float] = []
    selected_trades = 0
    for split in splits:
        training_rows = [
            row
            for index in split.train_indices
            for row in snapshots[ordered_keys[index]]
            if row.triggered and row.filled
        ]
        strategy_means = {
            strategy_id: fmean(row.net_return_bps for row in training_rows if row.strategy_id == strategy_id)
            for strategy_id in {row.strategy_id for row in training_rows}
        }
        for index in split.test_indices:
            candidates = [
                row
                for row in snapshots[ordered_keys[index]]
                if row.triggered and row.filled
            ]
            candidate = max(
                candidates,
                key=lambda row: strategy_means.get(row.strategy_id, float("-inf")),
                default=None,
            )
            expected = (
                strategy_means.get(candidate.strategy_id, float("-inf"))
                if candidate is not None
                else float("-inf")
            )
            if candidate is not None and expected > 0:
                selected_returns.append(candidate.net_return_bps)
                selected_trades += 1
            else:
                selected_returns.append(0.0)

    unavailable = {
        "event_momentum": "point-in-time event feed absent",
        "cross_sectional_relative_strength": "point-in-time sector graph absent",
        "gap_context": "authoritative exchange session calendar absent",
        "legacy": "historical legacy decisions were not journaled against these snapshots",
        "temporal_rgcn": "no trained/calibrated checkpoint exists",
    }
    dense_symbols = sum(len(bars) >= selected_config.minimum_symbol_bars for bars in bars_by_symbol.values())
    observed_dates = {
        bar.start_time.date().isoformat() for bars in bars_by_symbol.values() for bar in bars
    }
    promotion_eligible = (
        len(observed_dates) >= 60
        and dense_symbols >= 100
        and not unavailable
        and selected_trades >= 100
    )
    return {
        "status": "NOT_PROMOTED" if not promotion_eligible else "ELIGIBLE_FOR_REVIEW",
        "promotion_eligible": promotion_eligible,
        "configuration": asdict(selected_config),
        "reproducibility": {
            "label_data_sha256": data_digest.hexdigest(),
            "config_sha256": hashlib.sha256(config_json.encode()).hexdigest(),
            "code_sha256": code_digest.hexdigest(),
        },
        "coverage": {
            "bars": sum(map(len, bars_by_symbol.values())),
            "symbols": len(bars_by_symbol),
            "symbols_meeting_minimum": dense_symbols,
            "distinct_utc_dates": len(observed_dates),
            "first_event": min(
                (bar.start_time.isoformat() for bars in bars_by_symbol.values() for bar in bars),
                default=None,
            ),
            "last_event": max(
                (bar.start_time.isoformat() for bars in bars_by_symbol.values() for bar in bars),
                default=None,
            ),
        },
        "labels": {"snapshots": len(snapshots), "strategy_labels": len(labels)},
        "leakage_audit": {
            "feature_cutoff_precedes_label_window": True,
            "rolling_quantiles_use_prior_slice_only": True,
            "walk_forward_splits": len(splits),
            "purging_enabled": True,
            "embargo_enabled": True,
        },
        "strategy_metrics": metrics,
        "walk_forward_tabular_baseline": {
            "policy": "train-window strategy mean; select only positive expected net utility",
            "observations": len(selected_returns),
            "selected_trades": selected_trades,
            "mean_net_return_bps": fmean(selected_returns) if selected_returns else None,
        },
        "mandatory_system_comparison": {
            "ontology_only": "NoTrade: required event/sector/session facts fail closed",
            "tabular_baseline": "evaluated above",
            "legacy": "UNAVAILABLE",
            "temporal_rgcn_cpu": "UNAVAILABLE_UNTRAINED",
            "temporal_rgcn_npu": "UNAVAILABLE_UNTRAINED_AND_NPU_BENCHMARK_REJECTED",
        },
        "unavailable_evidence": unavailable,
        "limitations": [
            "The local minute store is concentrated in a few US sessions, not the target KRX/NXT market.",
            "Minute OHLC cannot recover tick-level queue position or intrabar barrier ordering.",
            "Missing point-in-time events, sector graph, session calendar, and historical legacy decisions prevent the mandatory seven-strategy comparison.",
        ],
    }


def write_report(report: dict[str, object], output: Path) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
