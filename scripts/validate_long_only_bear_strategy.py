from __future__ import annotations

import argparse
import json
from pathlib import Path

from app.strategy_validation.bear_cash_equity import (
    BearValidationConfig,
    build_bear_cash_equity_report,
)


def main() -> int:
    parser = argparse.ArgumentParser(description="Validate the long-only bear cash-equity submode; no orders.")
    parser.add_argument("--database", type=Path, default=Path("data/store/realtime_market_data.sqlite3"))
    parser.add_argument("--output", type=Path, default=Path("data/reports/long_only_bear_validation.json"))
    args = parser.parse_args()
    report = build_bear_cash_equity_report(BearValidationConfig(database=args.database))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, default=str), encoding="utf-8")
    print(json.dumps({"output": str(args.output), "promotion_eligible": report["promotion_eligible"], "no_orders": True}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

