from __future__ import annotations

import argparse
import json
from pathlib import Path

from app.evaluation.stored_counterfactual import build_report, write_report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--database",
        type=Path,
        default=Path("data/store/realtime_market_data.sqlite3"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("data/reports/refactor_counterfactual_evaluation.json"),
    )
    args = parser.parse_args()
    report = build_report(args.database)
    write_report(report, args.output)
    print(json.dumps({"output": str(args.output), "status": report["status"]}))


if __name__ == "__main__":
    main()

