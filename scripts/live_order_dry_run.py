from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from app.schemas.domain import FinalOrder, OrderSide, OrderType


def main() -> int:
    parser = argparse.ArgumentParser(description="Create a no-submit FinalOrder dry-run report.")
    parser.add_argument("--symbols", nargs="+", default=["005930"])
    parser.add_argument("--no-submit", action="store_true", required=True)
    args = parser.parse_args()

    symbols = []
    for raw_arg in args.symbols:
        for item in str(raw_arg).split(","):
            symbol = item.strip()
            if symbol:
                symbols.append(symbol.zfill(6) if symbol.isdigit() else symbol)
    orders = [
        FinalOrder(
            ticker=symbol,
            market="KR",
            order_type=OrderType.LIMIT,
            side=OrderSide.BUY,
            quantity=1,
            limit_price=1.0,
            manual_approval_required=True,
        )
        for symbol in symbols
    ]
    report = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "no_submit": True,
        "orders": [order.__dict__ for order in orders],
        "status": "DRY_RUN_ONLY",
    }
    path = _write_report(report)
    print(json.dumps({"ok": True, "report_path": str(path), **report}, indent=2, default=str))
    return 0


def _write_report(report: dict) -> Path:
    output_dir = Path("data/reports")
    output_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    path = output_dir / f"live_order_dry_run_{stamp}.json"
    path.write_text(json.dumps(report, indent=2, sort_keys=True, default=str), encoding="utf-8")
    return path


if __name__ == "__main__":
    raise SystemExit(main())
