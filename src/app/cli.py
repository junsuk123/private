from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict, is_dataclass
from datetime import datetime, time
from pathlib import Path
from typing import Any

from app.audit import AuditLogger
from app.backtesting import run_accelerated_demo
from app.execution import PaperOrderExecutor
from app.pipeline import build_analysis_context
from app.research import ResearchService
from app.research import ResearchRunResult
from app.runtime import DataEnvironment
from app.simulation import (
    MarketCalendar,
    SyntheticScenarioConfig,
    generate_synthetic_training_bundle,
    generate_synthetic_training_corpus,
)
from app.storage import LocalResearchStore


def run_demo(research_result: ResearchRunResult | None = None) -> dict[str, Any]:
    store = LocalResearchStore()
    context = build_analysis_context(research_result, store.load())

    paper_executor = PaperOrderExecutor()
    paper_receipts = tuple(
        paper_executor.submit(result.final_order)
        for result in context.risk_results
        if result.approved and result.final_order is not None
    )

    audit = AuditLogger(Path("logs/audit.jsonl"))
    audit.record("portfolio_report", context.report)
    audit.record("strategy_signals", context.signals)
    audit.record("order_intents", context.intents)
    audit.record("risk_results", context.risk_results)
    audit.record("paper_receipts", paper_receipts)

    return {
        "portfolio_report": context.report,
        "graph_triples": context.graph.triples(),
        "classified_events": context.events,
        "reasoning_paths": context.reasoning_paths,
        "strategy_signals": context.signals,
        "order_intents": context.intents,
        "risk_results": context.risk_results,
        "paper_receipts": paper_receipts,
        "audit_log": "logs/audit.jsonl",
        "store_summary": store.summary(),
    }


def main() -> None:
    _configure_stdout()
    parser = argparse.ArgumentParser(description="Personal investment analysis system CLI")
    subparsers = parser.add_subparsers(dest="command", required=True)
    subparsers.add_parser("demo", help="Run the safe local sample pipeline")
    accelerated_parser = subparsers.add_parser(
        "accelerated-demo",
        help="Run an accelerated paper-trading simulation in minute-based steps over 50 symbols",
    )
    accelerated_parser.add_argument("--target-return", type=float, default=0.02)
    accelerated_parser.add_argument("--period-minutes", type=int, default=390)
    accelerated_parser.add_argument("--initial-cash", type=float, default=10_000_000)
    accelerated_parser.add_argument("--seed", type=int, default=42)
    accelerated_parser.add_argument("--output-dir", default="data/reports")
    research_parser = subparsers.add_parser("research", help="Run public research collectors")
    research_parser.add_argument(
        "--config",
        default="config/research_sources.example.json",
        help="Path to research source config JSON",
    )
    sim_parser = subparsers.add_parser(
        "generate-sim-data",
        help="Disabled: learning and testing now use realtime data only",
    )
    sim_parser.add_argument("--exchange", choices=("US", "KRX"), default="US")
    sim_parser.add_argument("--scenarios", type=int, default=1)
    sim_parser.add_argument("--ticker-count", type=int, default=20)
    sim_parser.add_argument("--trading-days", type=int, default=15)
    sim_parser.add_argument("--interval-minutes", type=int, default=5)
    sim_parser.add_argument("--seed", type=int, default=20260613)
    sim_parser.add_argument("--randomness-scale", type=float, default=1.25)
    sim_parser.add_argument("--shock-probability", type=float, default=0.015)
    sim_parser.add_argument("--volume-spike-probability", type=float, default=0.025)
    sim_parser.add_argument("--news-events-per-ticker", type=int, default=8)
    args = parser.parse_args()

    if args.command == "demo":
        print(json.dumps(_to_jsonable(run_demo()), indent=2, ensure_ascii=False, sort_keys=True))
    elif args.command == "accelerated-demo":
        result = run_accelerated_demo(
            target_return_rate=args.target_return,
            period_minutes=args.period_minutes,
            initial_cash=args.initial_cash,
            output_dir=Path(args.output_dir),
            seed=args.seed,
        )
        print(json.dumps(_to_jsonable(result), indent=2, ensure_ascii=False, sort_keys=True))
    elif args.command == "research":
        result = ResearchService().run_from_config(Path(args.config))
        stored = LocalResearchStore().save_research_result(result)
        payload = {"research": result, "stored_new_records": stored}
        print(json.dumps(_to_jsonable(payload), indent=2, ensure_ascii=False, sort_keys=True))
    elif args.command == "generate-sim-data":
        raise SystemExit("Synthetic data generation is disabled for realtime-only learning/testing.")
        calendar = MarketCalendar.krx() if args.exchange == "KRX" else MarketCalendar.us()
        if args.scenarios > 1:
            corpus = generate_synthetic_training_corpus(
                DataEnvironment.simulation(),
                scenarios=args.scenarios,
                ticker_count=args.ticker_count,
                trading_days=args.trading_days,
                interval_minutes=args.interval_minutes,
                calendar=calendar,
                seed=args.seed,
                randomness_scale=args.randomness_scale,
            )
            print(json.dumps(_to_jsonable(corpus), indent=2, ensure_ascii=False, sort_keys=True))
        else:
            bundle = generate_synthetic_training_bundle(
                DataEnvironment.simulation(),
                tickers=None,
                ticker_count=args.ticker_count,
                trading_days=args.trading_days,
                interval_minutes=args.interval_minutes,
                calendar=calendar,
                seed=args.seed,
                scenario_config=SyntheticScenarioConfig(
                    randomness_scale=args.randomness_scale,
                    shock_probability=args.shock_probability,
                    volume_spike_probability=args.volume_spike_probability,
                    news_events_per_ticker=args.news_events_per_ticker,
                ),
            )
            print(json.dumps(_to_jsonable(bundle), indent=2, ensure_ascii=False, sort_keys=True))


def _to_jsonable(value: Any) -> Any:
    if is_dataclass(value):
        return _to_jsonable(asdict(value))
    if isinstance(value, dict):
        return {str(key): _to_jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_to_jsonable(item) for item in value]
    if isinstance(value, datetime):
        return value.isoformat()
    if isinstance(value, time):
        return value.isoformat()
    if isinstance(value, Path):
        return str(value)
    return value


def _configure_stdout() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")


if __name__ == "__main__":
    main()
