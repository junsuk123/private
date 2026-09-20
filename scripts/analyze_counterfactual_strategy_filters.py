from __future__ import annotations

"""Diagnose whether causal context filters improve after-cost strategy outcomes.

This is a research diagnostic, not an optimiser or a deployment tool.  It uses the
same point-in-time labels and cost model as strategy-utility training, then keeps
the final chronological quarter untouched while candidate one-feature filters are
chosen on the first half and checked on the middle quarter.
"""

import argparse
import json
import math
from collections import Counter, defaultdict
from dataclasses import asdict
from pathlib import Path
from statistics import fmean

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


def _market(symbol: str) -> str:
    return "KRX" if symbol.isdigit() and len(symbol) == 6 else "US"


def _summary(rows) -> dict[str, object]:
    if not rows:
        return {"rows": 0, "mean_net_bps": None, "positive_rate": None}
    return {
        "rows": len(rows),
        "mean_net_bps": fmean(row.net_return_bps for row in rows),
        "positive_rate": sum(row.net_return_bps > 0.0 for row in rows) / len(rows),
        "mean_gross_bps": fmean(row.gross_return_bps for row in rows),
        "mean_cost_bps": fmean(row.cost_bps for row in rows),
        "mean_mfe_bps": fmean(row.mfe_bps for row in rows),
        "mean_mae_bps": fmean(row.mae_bps for row in rows),
        "exit_reasons": dict(Counter(row.exit_reason for row in rows)),
        "symbols": len({row.symbol for row in rows}),
    }


def _quantile(values: list[float], fraction: float) -> float:
    ordered = sorted(values)
    if not ordered:
        return 0.0
    return ordered[min(len(ordered) - 1, int((len(ordered) - 1) * fraction))]


def _split(rows):
    ordered = sorted(rows, key=lambda row: (row.as_of, row.symbol))
    first = max(1, int(len(ordered) * 0.50))
    second = max(first + 1, int(len(ordered) * 0.75))
    return ordered[:first], ordered[first:second], ordered[second:]


def _feature(row, index: int) -> float:
    return float(row.features[index])


def _candidate_filters(train, minimum_rows: int):
    candidates = []
    for index, field in enumerate(STRATEGY_GRAPH_CONTEXT_FIELDS):
        values = [_feature(row, index) for row in train]
        if not values or max(values) - min(values) <= 1e-12:
            continue
        for fraction in (0.20, 0.35, 0.50, 0.65, 0.80):
            threshold = _quantile(values, fraction)
            for operator in ("ge", "le"):
                selected = [
                    row
                    for row in train
                    if (_feature(row, index) >= threshold)
                    == (operator == "ge")
                ]
                if len(selected) < minimum_rows:
                    continue
                summary = _summary(selected)
                candidates.append(
                    {
                        "field": field,
                        "index": index,
                        "operator": operator,
                        "threshold": threshold,
                        "train": summary,
                    }
                )
    return candidates


def _apply(rows, candidate):
    index = int(candidate["index"])
    threshold = float(candidate["threshold"])
    ge = candidate["operator"] == "ge"
    return [
        row
        for row in rows
        if (_feature(row, index) >= threshold) == ge
    ]


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
        default=Path("data/reports/strategy_filter_diagnosis.json"),
    )
    args = parser.parse_args()

    config = EvaluationConfig(
        feature_schema_name=STRATEGY_GRAPH_CONTEXT_SCHEMA,
        align_strategy_horizons=True,
    )
    labels = build_labels(
        load_minute_bars(args.database),
        config,
        microstructure_by_symbol=load_minute_microstructure(args.database),
        investor_flow_by_symbol=InvestorFlowStore().load_all(),
        news_by_ticker=load_news_sentiment(),
    )
    grouped = defaultdict(list)
    for row in labels:
        if row.triggered and row.filled and row.outcome_observed:
            grouped[(row.strategy_id, _market(row.symbol))].append(row)

    result: dict[str, object] = {
        "configuration": asdict(config),
        "feature_schema": STRATEGY_GRAPH_CONTEXT_SCHEMA,
        "selection_protocol": (
            "choose one-feature filter on first 50%; require non-negative middle "
            "25%; report final 25% untouched"
        ),
        "groups": {},
    }
    output_groups = result["groups"]
    assert isinstance(output_groups, dict)
    for (strategy_id, market), rows in sorted(grouped.items()):
        train, validation, test = _split(rows)
        minimum_train = max(3, min(12, int(len(train) * 0.30)))
        candidates = _candidate_filters(train, minimum_train)
        viable = []
        for candidate in candidates:
            validation_summary = _summary(_apply(validation, candidate))
            if (
                int(validation_summary["rows"]) >= 2
                and validation_summary["mean_net_bps"] is not None
                and float(validation_summary["mean_net_bps"]) >= 0.0
            ):
                item = dict(candidate)
                item["validation"] = validation_summary
                item["test"] = _summary(_apply(test, candidate))
                viable.append(item)
        viable.sort(
            key=lambda item: (
                float(item["validation"]["mean_net_bps"] or -math.inf),
                int(item["validation"]["rows"]),
                float(item["train"]["mean_net_bps"] or -math.inf),
            ),
            reverse=True,
        )
        output_groups[f"{market}:{strategy_id}"] = {
            "baseline": _summary(rows),
            "train": _summary(train),
            "validation": _summary(validation),
            "test": _summary(test),
            "best_filter": viable[0] if viable else None,
            "viable_filter_count": len(viable),
        }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    print(json.dumps({"output": str(args.output), "groups": len(output_groups)}))


if __name__ == "__main__":
    main()
