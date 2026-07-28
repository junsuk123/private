from __future__ import annotations

import argparse
import json
from pathlib import Path

from app.evaluation.stored_counterfactual import EvaluationConfig, build_labels, load_minute_bars
from app.models.strategy_utility.training import train_counterfactual_checkpoint


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--database", type=Path, default=Path("data/store/realtime_market_data.sqlite3"))
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("data/models/strategy_utility/rgcn_shadow.npz"),
    )
    args = parser.parse_args()
    labels = build_labels(load_minute_bars(args.database), EvaluationConfig())
    report = train_counterfactual_checkpoint(labels, args.output)
    print(json.dumps(report, ensure_ascii=False, sort_keys=True))


if __name__ == "__main__":
    main()
