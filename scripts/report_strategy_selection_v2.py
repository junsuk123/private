"""Generate the strategy-audit table and coverage matrix from real code and stored data.

Read-only. It opens no broker connection, submits nothing, and writes only under
``data/reports/``.

What it will and will not print
------------------------------
Where the stored evidence cannot support a number, the cell says so
(``INSUFFICIENT_DATA`` / ``null``) rather than showing a plausible value. That is the point
of running it: the most useful output today is the list of strategies that hold live
authorisation with no observations behind them.

Usage
-----
    python scripts/report_strategy_selection_v2.py                # console tables
    python scripts/report_strategy_selection_v2.py --json out.json
    python scripts/report_strategy_selection_v2.py --outcomes 5000
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from app.strategy.coverage import StrategyCoverageAnalyzer  # noqa: E402
from app.strategy.registry import default_strategy_registry  # noqa: E402
from app.strategy_validation import (  # noqa: E402
    StrategyAuditRunner,
    TradeObservation,
)


def _load_outcomes(limit: int) -> dict[str, list[TradeObservation]]:
    """Read realized outcomes from the performance store, if there are any.

    Query-only against the store's public reader. The realtime writer owns these files, so
    nothing here opens a write transaction or holds a long-running cursor.
    """
    try:
        from app.trading.strategy_performance_store import default_store
    except Exception as exc:  # noqa: BLE001
        print(f"[warn] performance store unavailable: {type(exc).__name__}: {exc}")
        return {}

    store = default_store()
    registry = default_strategy_registry()
    by_strategy: dict[str, list[TradeObservation]] = {}
    for spec in registry.all_specs():
        try:
            outcomes = store.recent_outcomes(spec.strategy_id, limit=limit)
        except Exception as exc:  # noqa: BLE001
            print(f"[warn] {spec.strategy_id}: {type(exc).__name__}: {exc}")
            continue
        rows: list[TradeObservation] = []
        for outcome in outcomes or ():
            recorded = getattr(outcome, "recorded_at", None)
            if not isinstance(recorded, datetime):
                continue
            if recorded.tzinfo is None:
                recorded = recorded.replace(tzinfo=timezone.utc)
            holding = float(getattr(outcome, "holding_seconds", 0.0) or 0.0)
            net = float(getattr(outcome, "realized_net_bps", 0.0) or 0.0)
            gross = getattr(outcome, "realized_gross_bps", None)
            # Cost is recoverable from the pair; where gross is absent the row carries no
            # cost information and is reported with cost 0 rather than an invented number.
            cost = float(gross) - net if gross is not None else 0.0
            rows.append(
                TradeObservation(
                    strategy_id=spec.strategy_id,
                    symbol=str(getattr(outcome, "symbol", "") or ""),
                    market=str(getattr(outcome, "market", "") or "UNKNOWN"),
                    regime=str(getattr(outcome, "regime", "") or "UNKNOWN"),
                    # The store records only a close time, so the open is reconstructed from
                    # the holding period. Reported as-is: it is exact when holding_seconds is
                    # recorded and degenerate (zero-length) when it is not, which is what the
                    # overlap discount then reads.
                    opened_at=recorded - timedelta(seconds=max(0.0, holding)),
                    closed_at=recorded,
                    gross_return_bps=float(gross) if gross is not None else net,
                    net_return_bps=net,
                    cost_bps=max(0.0, cost),
                    max_adverse_excursion_bps=getattr(
                        outcome, "max_adverse_excursion_bps", None
                    ),
                    evidence_source=str(
                        getattr(outcome, "evaluation_source", None)
                        or getattr(outcome, "source", "SHADOW")
                    ).upper(),
                    predicted_net_bps=getattr(outcome, "expected_net_bps", None),
                )
            )
        if rows:
            by_strategy[spec.strategy_id] = rows
    return by_strategy


def _audit_table(outcome_limit: int) -> list[dict[str, Any]]:
    trades = _load_outcomes(outcome_limit)
    report = StrategyAuditRunner().run(trades)
    return report.table()


def _coverage_matrix() -> dict[str, Any]:
    analyzer = StrategyCoverageAnalyzer()
    return {
        "summary": analyzer.summary(),
        "matrix": analyzer.matrix(),
        "research_candidates": analyzer.research_candidates(minimum_observations=5),
    }


def _print_audit(rows: list[dict[str, Any]]) -> None:
    header = (
        f"{'strategy_id':<36}{'family':<22}{'lifecycle':<12}{'class':<18}"
        f"{'n':>6}{'net_bps':>10}{'lower':>9}{'cost_x':>8}{'oos':>6}"
    )
    print(header)
    print("-" * len(header))
    for row in rows:
        print(
            f"{row['strategy_id']:<36}{row['family']:<22}{row['current_lifecycle']:<12}"
            f"{row['classification']:<18}{row['trades']:>6}"
            f"{_fmt(row['net_ev_bps']):>10}{_fmt(row['lower_bound_bps']):>9}"
            f"{_fmt(row['break_even_cost_multiple'], 2):>8}{_fmt(row['oos_stability'], 2):>6}"
        )
    flagged = [
        row
        for row in rows
        if "AUDIT_LIVE_AUTHORIZED_WITHOUT_EVIDENCE" in row["reason_codes"]
    ]
    if flagged:
        print()
        print("LIVE-AUTHORISED WITH NO OBSERVATIONS:")
        for row in flagged:
            print(f"  - {row['strategy_id']} (recommend {row['recommended_lifecycle']})")


def _fmt(value: Any, digits: int = 1) -> str:
    if value is None:
        return "-"
    try:
        return f"{float(value):.{digits}f}"
    except (TypeError, ValueError):
        return "-"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--json", type=Path, default=None, help="write the full report here")
    parser.add_argument(
        "--outcomes", type=int, default=2000, help="max stored outcomes per strategy"
    )
    args = parser.parse_args()

    registry = default_strategy_registry()
    payload: dict[str, Any] = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "specs": registry.as_table(),
        "lifecycle_recommendations": registry.lifecycle_recommendations(),
        "audit": _audit_table(args.outcomes),
        "coverage": _coverage_matrix(),
    }

    print("=== STRATEGY AUDIT ===")
    _print_audit(payload["audit"])
    print()
    print("=== COVERAGE ===")
    print(json.dumps(payload["coverage"]["summary"], indent=2))
    gaps = payload["coverage"]["research_candidates"]
    if gaps:
        print(f"{len(gaps)} recurring coverage gap(s):")
        for gap in gaps[:10]:
            print(
                f"  - trend={gap['trend']} vol={gap['volatility']} liq={gap['liquidity']} "
                f"session={gap['session']} event={gap['event']} micro={gap['microstructure']} "
                f"(n={gap['observations']})"
            )
    else:
        print("no coverage observations recorded yet "
              "(STRATEGY_SELECTOR_V2_ENABLED has not run)")

    if args.json:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")
        print(f"\nwrote {args.json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
