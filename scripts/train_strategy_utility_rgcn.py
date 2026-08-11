from __future__ import annotations

import argparse
import json
from pathlib import Path

from app.data.investor_flow_store import InvestorFlowStore
from app.features.strategy_graph_context import STRATEGY_GRAPH_CONTEXT_SCHEMA
from app.evaluation.stored_counterfactual import (
    EvaluationConfig,
    build_labels,
    load_minute_bars,
    load_minute_microstructure,
    load_news_sentiment,
)
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
    labels = build_labels(
        load_minute_bars(args.database),
        EvaluationConfig(
            feature_schema_name=STRATEGY_GRAPH_CONTEXT_SCHEMA,
            align_strategy_horizons=True,
        ),
        microstructure_by_symbol=load_minute_microstructure(args.database),
        investor_flow_by_symbol=InvestorFlowStore().load_all(),
        news_by_ticker=load_news_sentiment(),
    )
    report = train_counterfactual_checkpoint(
        labels,
        args.output,
        input_feature_schema=STRATEGY_GRAPH_CONTEXT_SCHEMA,
        authorize_live_shadow=True,
    )
    print(json.dumps(report, ensure_ascii=False, sort_keys=True))


if __name__ == "__main__":
    main()
