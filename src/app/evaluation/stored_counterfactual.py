from __future__ import annotations

import hashlib
import json
import sqlite3
from collections import defaultdict
from dataclasses import asdict, dataclass, replace
from datetime import datetime, timedelta
from pathlib import Path
from statistics import fmean
from typing import Sequence

from app.backtesting.event_simulator import EventDrivenFillSimulator
from app.evaluation.purged_walk_forward import purged_walk_forward_splits
from app.strategy.experts import ALL_EXPERT_TYPES, ExpertContext
from app.trading.contracts import Bar
from app.features.schemas import OHLCVBar
from app.technical import indicators as ti


@dataclass(frozen=True)
class EvaluationConfig:
    history_bars: int = 30
    # US BanKIS round-trip expenses are roughly 40bp before spread/slippage.
    # A 4-10 minute label mostly teaches "cost exceeds move"; use a one-hour
    # causal window so strategies can learn moves large enough to clear costs.
    horizon_bars: int = 60
    stride_bars: int = 5
    minimum_symbol_bars: int = 100
    maximum_bar_gap_seconds: float = 120.0
    minimum_active_history_fraction: float = 0.10
    minimum_future_bars: int = 5
    quantity: int = 1
    venue: str = "NASD"
    market: str = "US"
    instrument_type: str = "overseas_stock"
    feature_schema_name: str = "counterfactual_quantiles_v1"
    infer_market_from_symbol: bool = True


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
            normalized_symbol = str(symbol).upper()
            venue = (
                "KRX"
                if normalized_symbol.isdigit() and len(normalized_symbol) == 6
                else "NASD"
            )
            by_symbol[symbol].append(
                Bar(
                    symbol=symbol,
                    venue=venue,
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
        venue, market, instrument_type = _execution_market_for_symbol(
            symbol,
            config,
        )
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
            history_window = bars[history_start : index + 1]
            if not _bars_are_contiguous(
                history_window,
                maximum_gap_seconds=config.maximum_bar_gap_seconds,
            ):
                continue
            active_history = sum(
                bar.high > bar.low
                or (
                    position > 0
                    and bar.close != history_window[position - 1].close
                )
                for position, bar in enumerate(history_window)
            )
            if (
                active_history / max(1, len(history_window))
                < config.minimum_active_history_fraction
            ):
                continue
            return_history = returns[history_start:index]
            volume_history = volumes[history_start:index]
            spread_history = spreads[history_start:index]
            prior_high = max(bar.high for bar in bars[max(0, index - 10) : index])
            vwap_proxy = sum(
                bar.close * max(bar.volume, 1.0) for bar in bars[history_start:index]
            ) / sum(max(bar.volume, 1.0) for bar in bars[history_start:index])
            current_return = returns[index]
            completed = tuple(
                OHLCVBar(
                    ticker=bar.symbol,
                    as_of=bar.end_time,
                    open=bar.open,
                    high=bar.high,
                    low=bar.low,
                    close=bar.close,
                    volume=bar.volume,
                )
                for bar in bars[: index + 1]
            )
            rvgi_result = ti.rvgi(completed, 10)
            box = ti.causal_box_geometry(completed, 20)
            rvgi_diff = (
                rvgi_result.main - rvgi_result.signal
                if rvgi_result.ok
                and rvgi_result.main is not None
                and rvgi_result.signal is not None
                else None
            )
            box_breakout = bool(
                box.ok and box.high is not None and current.close > box.high
            )
            volume_confirmed = (
                causal_percentile(current.volume, volume_history) >= 0.65
            )
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
                "rvgi_diff": (
                    max(0.0, min(1.0, 0.5 + rvgi_diff * 5.0))
                    if rvgi_diff is not None
                    else 0.0
                ),
                "rvgi_cross": 1.0 if rvgi_result.bullish_cross else 0.0,
                "box_position": float(box.position or 0.0) if box.ok else 0.0,
                "false_breakout_risk": (
                    0.0 if box_breakout and volume_confirmed else 1.0
                ),
            }
            as_of = current.end_time
            future = _contiguous_future_bars(
                bars[index + 1 : index + 1 + config.horizon_bars],
                expected_start=current.end_time,
                maximum_gap_seconds=config.maximum_bar_gap_seconds,
            )
            required_future_bars = max(
                1,
                min(config.minimum_future_bars, config.horizon_bars),
            )
            if len(future) < required_future_bars:
                continue
            context = ExpertContext(
                symbol=symbol,
                as_of=as_of,
                price=current.close,
                proposed_quantity=config.quantity,
                feature_snapshot_id=f"stored:{symbol}:{as_of.isoformat()}",
                utility_evidence_id="counterfactual",
                quantiles=quantiles,
            )
            for expert in experts:
                plan = expert.propose(context)
                triggered = plan is not None
                # A strategy that did not fire is a useful classification
                # negative, but it is not a hypothetical filled trade.  The
                # previous implementation forced every expert to propose with
                # synthetic all-one quantiles, then simulated P&L even when
                # the real point-in-time context was inadmissible.  That
                # contaminated the return/cost heads with fabricated losses.
                if plan is not None:
                    baseline_cost = simulator.cost_engine.estimate(
                        symbol=symbol,
                        market=market,
                        venue=venue,
                        instrument_type=instrument_type,
                        entry_price=current.close,
                        expected_exit_price=current.close,
                        quantity=config.quantity,
                    )
                    fee_policy = simulator.cost_engine.policy_for(
                        venue=venue,
                        instrument_type=instrument_type,
                    )
                    cost_bps = baseline_cost.total_cost_rate * 10_000.0
                    # Target a net reward comparable to the unavoidable
                    # round-trip cost. A cost+8bp target had very poor
                    # reward/risk after a normal adverse move.
                    minimum_gross_bps = (
                        cost_bps
                        + fee_policy.safety_margin_rate * 10_000.0
                        + max(25.0, cost_bps)
                    )
                    configured_gross_bps = float(
                        plan.profit_policy.get("bps", 0.0)
                    )
                    target_bps = max(configured_gross_bps, minimum_gross_bps)
                    plan = replace(
                        plan,
                        profit_policy={
                            **plan.profit_policy,
                            "bps": target_bps,
                            "price": current.close * (1.0 + target_bps / 10_000.0),
                        },
                    )
                outcome = (
                    simulator.simulate(
                        plan,
                        future,
                        as_of=as_of,
                        venue=venue,
                        market=market,
                        instrument_type=instrument_type,
                    )
                    if plan is not None
                    else None
                )
                labels.append(
                    CounterfactualLabel(
                        as_of=as_of,
                        label_end=future[-1].end_time,
                        symbol=symbol,
                        strategy_id=expert.strategy_id,
                        triggered=triggered,
                        filled=outcome.filled if outcome is not None else False,
                        net_return_bps=(
                            outcome.net_return_bps if outcome is not None else 0.0
                        ),
                        cost_bps=outcome.cost_bps if outcome is not None else 0.0,
                        exit_reason=(
                            outcome.exit_reason
                            if outcome is not None
                            else "STRATEGY_NOT_TRIGGERED"
                        ),
                        features=_label_features(
                            config.feature_schema_name,
                            current=current,
                            current_return=current_return,
                            return_history=return_history,
                            vwap_proxy=vwap_proxy,
                            rvgi_result=rvgi_result,
                            box=box,
                            quantiles=quantiles,
                        ),
                    )
                )
    return tuple(labels)


def _bars_are_contiguous(
    bars: Sequence[Bar],
    *,
    maximum_gap_seconds: float,
) -> bool:
    return all(
        0.0
        <= (current.start_time - previous.start_time).total_seconds()
        <= maximum_gap_seconds
        for previous, current in zip(bars, bars[1:])
    )


def _contiguous_future_bars(
    bars: Sequence[Bar],
    *,
    expected_start: datetime,
    maximum_gap_seconds: float,
) -> tuple[Bar, ...]:
    selected: list[Bar] = []
    previous_start = expected_start - timedelta(minutes=1)
    for bar in bars:
        gap = (bar.start_time - previous_start).total_seconds()
        if gap < 0.0 or gap > maximum_gap_seconds:
            break
        selected.append(bar)
        previous_start = bar.start_time
    return tuple(selected)


def _execution_market_for_symbol(
    symbol: str,
    config: EvaluationConfig,
) -> tuple[str, str, str]:
    normalized = str(symbol or "").upper().strip()
    if config.infer_market_from_symbol and normalized.isdigit() and len(normalized) == 6:
        return "KRX", "KR", "domestic_stock"
    if config.infer_market_from_symbol and normalized and normalized[0].isalpha():
        return "NASD", "US", "overseas_stock"
    return config.venue, config.market, config.instrument_type


def _label_features(
    schema_name: str,
    *,
    current,
    current_return: float,
    return_history: Sequence[float],
    vwap_proxy: float,
    rvgi_result,
    box,
    quantiles: dict[str, float],
) -> tuple[float, ...]:
    micro = _realtime_microstructure_proxy_features(
        current=current,
        current_return=current_return,
        return_history=return_history,
        vwap_proxy=vwap_proxy,
    )
    if schema_name == "realtime_microstructure_v1":
        return micro
    if schema_name in {
        "realtime_strategy_context_v2",
        "realtime_strategy_graph_v3",
    }:
        return (*micro, *_rvgi_box_proxy_features(current, rvgi_result, box))
    if schema_name == "realtime_strategy_graph_v4_market":
        is_krx = (
            str(getattr(current, "symbol", "") or "").isdigit()
            and len(str(getattr(current, "symbol", "") or "")) == 6
        )
        return (
            *micro,
            *_rvgi_box_proxy_features(current, rvgi_result, box),
            1.0 if is_krx else 0.0,
        )
    return (
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


def _rvgi_box_proxy_features(current, rvgi_result, box) -> tuple[float, ...]:
    scale = max(float(current.close), 1e-12)
    rvgi_available = 1.0 if rvgi_result.ok else 0.0
    box_available = 1.0 if box.ok else 0.0
    main = float(rvgi_result.main or 0.0)
    signal = float(rvgi_result.signal or 0.0)
    return (
        rvgi_available,
        main,
        signal,
        main - signal if rvgi_result.ok else 0.0,
        float(rvgi_result.slope or 0.0),
        1.0 if rvgi_result.bullish_cross else 0.0,
        box_available,
        float(box.high or 0.0) / scale,
        float(box.low or 0.0) / scale,
        float(box.mid or 0.0) / scale,
        float(box.width_pct or 0.0),
        float(box.position or 0.0),
        ((float(current.close) / float(box.high) - 1.0) * 100.0)
        if box.ok and box.high
        else 0.0,
        0.0,
        1.0 if box.source_timestamp is not None else 0.0,
    )


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
