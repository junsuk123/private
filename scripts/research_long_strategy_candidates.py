from __future__ import annotations

"""Screen pre-declared long-only strategy theses on causal stored snapshots.

The rules below are deliberately specified in code before outcomes are inspected.
They are not threshold-optimised.  Every fill enters at the next completed bar's
open, pays the system's all-in round-trip estimate, and resolves stop before target
when both occur in one bar.  Results remain research-only because the current store
contains too few distinct market days for promotion.
"""

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path
from statistics import fmean

from app.cost.round_trip import all_in_round_trip_bps
from app.data.investor_flow_store import InvestorFlowStore
from app.evaluation.stored_counterfactual import (
    EvaluationConfig,
    build_labels,
    load_minute_bars,
    load_minute_microstructure,
    load_news_sentiment,
)
from app.features.strategy_graph_context import (
    STRATEGY_GRAPH_CONTEXT_FIELDS,
    STRATEGY_GRAPH_CONTEXT_SCHEMA,
)

INDEX = {name: i for i, name in enumerate(STRATEGY_GRAPH_CONTEXT_FIELDS)}


def _m(row, name: str) -> float:
    return float(row.features[INDEX[name]])


def _market(symbol: str) -> str:
    return "KRX" if symbol.isdigit() and len(symbol) == 6 else "US"


def _rules(row) -> tuple[str, ...]:
    rules: list[str] = []
    trend = _m(row, "trend_available") >= 1.0
    oscillator = _m(row, "oscillator_available") >= 1.0
    micro = _m(row, "microstructure_available") >= 1.0
    liquid = (not micro) or _m(row, "liquidity_score") >= 0.45
    tight = (not micro) or _m(row, "spread_bps_scaled") <= 0.35

    # Established trend, but entry is on a shallow pullback to EMA rather than
    # after a fully extended persistence/volume burst (the losing current rule).
    if (
        trend
        and oscillator
        and liquid
        and tight
        and _m(row, "supertrend_direction") > 0.0
        and _m(row, "dmi_spread_scaled") >= 0.0
        and 0.0 <= _m(row, "ema_separation_pct") <= 1.00
        and -0.30 <= _m(row, "ema_fast_distance_pct") <= 0.25
        and 0.0 <= _m(row, "distance_from_vwap") <= 0.015
        and 0.35 <= _m(row, "rsi_scaled") <= 0.70
        and _m(row, "return_1m_scaled") > 0.0
    ):
        rules.append("trend_pullback_reentry")

    # A downside VWAP displacement is only bought after price, DMI and the last
    # completed bar all turn.  This separates a reclaim from a falling knife.
    if (
        trend
        and oscillator
        and liquid
        and tight
        and -0.020 <= _m(row, "distance_from_vwap") <= -0.0025
        and _m(row, "return_1m_scaled") >= 0.10
        and _m(row, "ema_fast_distance_pct") >= -0.20
        and _m(row, "dmi_spread_scaled") >= -0.05
        and 0.28 <= _m(row, "rsi_scaled") <= 0.55
    ):
        rules.append("vwap_reclaim_reversal")

    # Compression breakout that is not yet far from the fast EMA.  Existing
    # breakout arms buy expansion regardless of extension and repeatedly stop.
    if (
        trend
        and oscillator
        and liquid
        and tight
        and _m(row, "keltner_available") >= 1.0
        and 1.0 <= _m(row, "keltner_position") <= 1.6
        and _m(row, "dmi_spread_scaled") >= 0.08
        and _m(row, "adx_scaled") >= 0.22
        and 1.2 <= _m(row, "volume_spike_ratio") <= 4.0
        and 0.0 <= _m(row, "ema_fast_distance_pct") <= 0.35
        and _m(row, "return_1m_scaled") > 0.0
    ):
        rules.append("nonextended_keltner_breakout")
    return tuple(rules)


def _simulate(entry, future, cost_bps: float, horizon_minutes: int, atr_fraction: float):
    stop_bps = max(60.0, min(180.0, atr_fraction * 10_000.0 * 1.2))
    target_bps = cost_bps + 1.5 * (stop_bps + cost_bps)
    stop = entry * (1.0 - stop_bps / 10_000.0)
    target = entry * (1.0 + target_bps / 10_000.0)
    deadline = future[0].start_time.timestamp() + horizon_minutes * 60
    lows, highs = [], []
    exit_price = future[0].open
    reason = "MAX_HOLDING_TIME"
    for bar in future:
        if bar.start_time.timestamp() > deadline:
            break
        lows.append(bar.low)
        highs.append(bar.high)
        if bar.low <= stop:
            exit_price, reason = stop, "INITIAL_STOP"
            break
        if bar.high >= target:
            exit_price, reason = target, "PROFIT_TARGET"
            break
        exit_price = bar.close
    gross = (exit_price / entry - 1.0) * 10_000.0
    return {
        "net_bps": gross - cost_bps,
        "gross_bps": gross,
        "cost_bps": cost_bps,
        "mfe_bps": (max(highs, default=entry) / entry - 1.0) * 10_000.0,
        "mae_bps": (min(lows, default=entry) / entry - 1.0) * 10_000.0,
        "exit_reason": reason,
    }


def _summary(rows):
    if not rows:
        return {"rows": 0, "mean_net_bps": None, "positive_rate": None}
    return {
        "rows": len(rows),
        "symbols": len({row["symbol"] for row in rows}),
        "mean_net_bps": fmean(row["net_bps"] for row in rows),
        "positive_rate": sum(row["net_bps"] > 0 for row in rows) / len(rows),
        "mean_gross_bps": fmean(row["gross_bps"] for row in rows),
        "mean_cost_bps": fmean(row["cost_bps"] for row in rows),
        "mean_mfe_bps": fmean(row["mfe_bps"] for row in rows),
        "mean_mae_bps": fmean(row["mae_bps"] for row in rows),
        "exit_reasons": dict(Counter(row["exit_reason"] for row in rows)),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--database", type=Path, default=Path("data/store/realtime_market_data.sqlite3"))
    parser.add_argument("--output", type=Path, default=Path("data/reports/long_strategy_candidates.json"))
    args = parser.parse_args()
    bars_by_symbol = load_minute_bars(args.database)
    micro_by_symbol = load_minute_microstructure(args.database)
    labels = build_labels(
        bars_by_symbol,
        EvaluationConfig(feature_schema_name=STRATEGY_GRAPH_CONTEXT_SCHEMA, align_strategy_horizons=True),
        microstructure_by_symbol=micro_by_symbol,
        investor_flow_by_symbol=InvestorFlowStore().load_all(),
        news_by_ticker=load_news_sentiment(),
    )
    snapshots = {}
    for row in labels:
        snapshots.setdefault((row.symbol, row.as_of), row)
    bar_index = {
        symbol: {bar.end_time: index for index, bar in enumerate(bars)}
        for symbol, bars in bars_by_symbol.items()
    }
    outcomes = defaultdict(list)
    for (symbol, as_of), row in sorted(snapshots.items(), key=lambda item: item[0][1]):
        rules = _rules(row)
        if not rules:
            continue
        bars = bars_by_symbol[symbol]
        index = bar_index[symbol].get(as_of)
        if index is None or index + 2 >= len(bars):
            continue
        future = bars[index + 1 :]
        entry = float(future[0].open)
        if entry <= 0:
            continue
        current_micro = micro_by_symbol.get(symbol, {}).get(bars[index].start_time)
        spread = current_micro.spread_bps if current_micro is not None else None
        cost_bps = all_in_round_trip_bps(symbol, spread_bps=spread)
        atr_fraction = max(0.0, _m(row, "atr_pct"))
        horizon = 180 if _market(symbol) == "US" else 60
        # A full future clock is mandatory; crossing a session gap is not silently
        # treated as a completed intraday outcome.
        required = horizon + 1
        candidate_future = future[:required]
        if len(candidate_future) < required:
            continue
        if any(
            (later.start_time - earlier.start_time).total_seconds() > 120
            for earlier, later in zip(candidate_future, candidate_future[1:])
        ):
            continue
        simulated = _simulate(entry, candidate_future, cost_bps, horizon, atr_fraction)
        for rule in rules:
            outcomes[(_market(symbol), rule)].append(
                {"as_of": as_of.isoformat(), "symbol": symbol, **simulated}
            )

    report = {"protocol": "predeclared rules; 50/25/25 chronological; next-bar-open; all-in costs", "groups": {}}
    for key, rows in sorted(outcomes.items()):
        ordered = sorted(rows, key=lambda row: (row["as_of"], row["symbol"]))
        first, second = int(len(ordered) * 0.5), int(len(ordered) * 0.75)
        report["groups"][f"{key[0]}:{key[1]}"] = {
            "all": _summary(ordered),
            "train": _summary(ordered[:first]),
            "validation": _summary(ordered[first:second]),
            "test": _summary(ordered[second:]),
        }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps({"output": str(args.output), "groups": len(report["groups"])}))


if __name__ == "__main__":
    main()
