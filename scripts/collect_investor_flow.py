"""Backfill / refresh daily investor flow for the KRX symbols in the bar store.

The server refreshes this on its own schedule (see
``web._start_investor_flow_refresher``); this script is the manual entry point for
a first backfill or an out-of-band top-up. Both call the SAME routine in
``app.data.investor_flow_collector`` so the two paths cannot drift.

Read-only against KIS (``inquire-investor``); it never touches an order endpoint.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from app.data.investor_flow_collector import (  # noqa: E402
    DEFAULT_BAR_DATABASE,
    refresh_investor_flow,
)
from app.data.investor_flow_store import InvestorFlowStore  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--database", type=Path, default=Path(DEFAULT_BAR_DATABASE))
    parser.add_argument("--store", type=Path, default=None)
    parser.add_argument(
        "--minimum-bars",
        type=int,
        default=100,
        help="Skip symbols the labeller would discard anyway.",
    )
    parser.add_argument("--limit", type=int, default=0, help="0 == no limit")
    parser.add_argument("--delay", type=float, default=0.3)
    args = parser.parse_args()

    result = refresh_investor_flow(
        store=InvestorFlowStore(args.store) if args.store else None,
        database=args.database,
        minimum_bars=args.minimum_bars,
        limit=args.limit,
        delay_seconds=args.delay,
    )
    print(json.dumps(result.as_dict(), ensure_ascii=False, indent=2, sort_keys=True))
    if result.failures:
        # Non-zero exit so a scripted backfill cannot silently half-succeed.
        sys.exit(1)


if __name__ == "__main__":
    main()
