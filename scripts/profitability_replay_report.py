#!/usr/bin/env python3
"""Before/after profitability replay report.

Measurement instrument for the profitability refactor. Reads the live order journal
(logs/live-orders.jsonl) and reports:

  * order-flow outcomes (submitted / error / blocked / amended / canceled / statuses)
  * block/rejection reason distribution
  * cost-aware realized metrics on matched round-trips (FIFO per ticker):
      trade count, gross PnL, NET PnL after costs, win rate, avg win, avg loss,
      payoff ratio, expectancy, max drawdown

Round-trip PnL uses the LIMIT price of FILLED/ACCEPTED orders as a fill-price proxy
(limit orders fill at or better than the limit), so realized numbers are APPROXIMATE.
Net PnL applies the same TradingCostEngine the live gate uses.

How to use it for a before/after comparison
-------------------------------------------
1. BEFORE: run this on the current (pre-refactor) journal -> baseline numbers.
2. Run the refactored engine (paper or live) to accumulate a new journal.
3. AFTER: run this again -> compare. The success criterion is improved NET expectancy
   and fewer negative-cost trades, NOT more trades.

Usage:
    PYTHONPATH=src python scripts/profitability_replay_report.py [--journal PATH] [--markdown OUT.md]
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter, defaultdict, deque
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from app.cost import TradingCostEngine  # noqa: E402


def _iter_events(path: Path):
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError:
                continue


def _venue_instrument(market: str, ticker: str) -> tuple[str, str]:
    m = str(market or "").upper()
    if str(ticker or "").isdigit() or m in {"KR", "KRX", "KOSPI", "KOSDAQ", "KONEX"}:
        return "KRX", "domestic_stock"
    if "NYSE" in m:
        return "NYSE", "overseas_stock"
    if "AMEX" in m:
        return "AMEX", "overseas_stock"
    return "NASD", "overseas_stock"


def analyze(path: Path) -> dict[str, Any]:
    event_types: Counter = Counter()
    statuses: Counter = Counter()
    block_reasons: Counter = Counter()
    # Per-ticker FIFO of accepted BUY lots: (price, qty, market)
    open_buys: dict[str, deque] = defaultdict(deque)
    round_trips: list[dict[str, Any]] = []
    cost_engine = TradingCostEngine()

    def _accepted_order(payload: dict[str, Any]) -> dict[str, Any] | None:
        order = payload.get("order")
        if not isinstance(order, dict):
            return None
        return order

    for event in _iter_events(path):
        etype = event.get("event_type", "?")
        event_types[etype] += 1
        payload = event.get("payload", {}) or {}
        status = payload.get("status") or payload.get("order_status")
        if status:
            statuses[str(status)] += 1
        if etype in {"live_order_blocked", "live_order_amend_blocked"}:
            for code in payload.get("reason_codes", ()) or ():
                block_reasons[str(code)] += 1

        # Treat submitted/accepted/filled orders as executed lots for PnL matching.
        if etype == "live_order_submitted" or (status in {"FILLED", "ACCEPTED"}):
            order = _accepted_order(payload)
            if not order:
                continue
            side = str(order.get("side", "")).upper()
            ticker = str(order.get("ticker", ""))
            price = float(order.get("limit_price", 0.0) or 0.0)
            qty = int(order.get("quantity", 0) or 0)
            market = str(order.get("market", ""))
            if price <= 0 or qty <= 0 or not ticker:
                continue
            if side == "BUY":
                open_buys[ticker].append((price, qty, market))
            elif side in {"SELL", "REDUCE"}:
                remaining = qty
                while remaining > 0 and open_buys[ticker]:
                    buy_price, buy_qty, buy_market = open_buys[ticker][0]
                    matched = min(remaining, buy_qty)
                    venue, instrument = _venue_instrument(buy_market or market, ticker)
                    cost = cost_engine.estimate(
                        symbol=ticker,
                        market=buy_market or market or "KR",
                        venue=venue,
                        instrument_type=instrument,
                        entry_price=buy_price,
                        expected_exit_price=price,
                        quantity=matched,
                    )
                    gross_return = (price - buy_price) / buy_price if buy_price else 0.0
                    round_trips.append(
                        {
                            "ticker": ticker,
                            "gross_return": gross_return,
                            "net_return": cost.net_expected_return,
                            "gross_pnl": (price - buy_price) * matched,
                            "net_pnl": cost.net_expected_profit,
                            "qty": matched,
                        }
                    )
                    remaining -= matched
                    if matched >= buy_qty:
                        open_buys[ticker].popleft()
                    else:
                        open_buys[ticker][0] = (buy_price, buy_qty - matched, buy_market)

    return {
        "event_types": dict(event_types),
        "statuses": dict(statuses),
        "block_reasons": dict(block_reasons.most_common()),
        "round_trips": round_trips,
    }


def _metrics(round_trips: list[dict[str, Any]]) -> dict[str, Any]:
    n = len(round_trips)
    if n == 0:
        return {"round_trips": 0}
    net = [r["net_return"] for r in round_trips]
    wins = [x for x in net if x > 0]
    losses = [x for x in net if x <= 0]
    avg_win = sum(wins) / len(wins) if wins else 0.0
    avg_loss = sum(losses) / len(losses) if losses else 0.0
    win_rate = len(wins) / n
    payoff = (avg_win / abs(avg_loss)) if avg_loss else float("inf")
    expectancy = win_rate * avg_win - (1 - win_rate) * abs(avg_loss)
    # Max drawdown on cumulative net PnL.
    cum = 0.0
    peak = 0.0
    max_dd = 0.0
    for r in round_trips:
        cum += r["net_pnl"]
        peak = max(peak, cum)
        max_dd = min(max_dd, cum - peak)
    return {
        "round_trips": n,
        "gross_pnl": round(sum(r["gross_pnl"] for r in round_trips), 2),
        "net_pnl": round(sum(r["net_pnl"] for r in round_trips), 2),
        "win_rate": round(win_rate, 4),
        "avg_win_net": round(avg_win, 5),
        "avg_loss_net": round(avg_loss, 5),
        "payoff_ratio": round(payoff, 3) if payoff != float("inf") else "inf",
        "expectancy_net": round(expectancy, 5),
        "negative_net_trades": len(losses),
        "max_drawdown_net_pnl": round(max_dd, 2),
    }


def render_markdown(result: dict[str, Any], metrics: dict[str, Any], journal: Path) -> str:
    lines = [
        "# Profitability Replay Report",
        "",
        "![Before/after net-profitability gate](diagrams/profitability_before_after.svg)",
        "",
        f"Source journal: `{journal}`",
        "",
        "> Realized PnL uses limit-price as a fill-price proxy and is APPROXIMATE.",
        "> Net PnL/return apply the live TradingCostEngine. Compare BEFORE vs AFTER the",
        "> refactor; the success criterion is improved NET expectancy and fewer",
        "> negative-cost trades, not more trades.",
        "",
        "## Order-flow outcomes",
        "",
        "| event_type | count |",
        "|---|---|",
    ]
    for k, v in sorted(result["event_types"].items(), key=lambda kv: -kv[1]):
        lines.append(f"| {k} | {v} |")
    lines += ["", "## Broker statuses", "", "| status | count |", "|---|---|"]
    for k, v in sorted(result["statuses"].items(), key=lambda kv: -kv[1]):
        lines.append(f"| {k} | {v} |")
    lines += ["", "## Block / rejection reasons", "", "| reason | count |", "|---|---|"]
    for k, v in result["block_reasons"].items():
        lines.append(f"| {k} | {v} |")
    lines += ["", "## Cost-aware realized metrics (matched round-trips)", "", "| metric | value |", "|---|---|"]
    for k, v in metrics.items():
        lines.append(f"| {k} | {v} |")
    lines.append("")
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--journal", default="logs/live-orders.jsonl")
    parser.add_argument("--markdown", default="data/reports/profitability_replay.md")
    args = parser.parse_args()

    journal = Path(args.journal)
    if not journal.exists():
        print(f"Journal not found: {journal}")
        return 1

    result = analyze(journal)
    metrics = _metrics(result["round_trips"])
    report = render_markdown(result, metrics, journal)
    print(report)
    if args.markdown:
        out = Path(args.markdown)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(report, encoding="utf-8")
        print(f"\nWrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
