from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict, is_dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

from app.audit import AuditLogger
from app.execution import PaperOrderExecutor
from app.pipeline import build_analysis_context
from app.research import ResearchService
from app.research import ResearchRunResult
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
    research_parser = subparsers.add_parser("research", help="Run public research collectors")
    research_parser.add_argument(
        "--config",
        default="config/research_sources.example.json",
        help="Path to research source config JSON",
    )
    args = parser.parse_args()

    if args.command == "demo":
        print(json.dumps(_to_jsonable(run_demo()), indent=2, ensure_ascii=False, sort_keys=True))
    elif args.command == "research":
        result = ResearchService().run_from_config(Path(args.config))
        stored = LocalResearchStore().save_research_result(result)
        payload = {"research": result, "stored_new_records": stored}
        print(json.dumps(_to_jsonable(payload), indent=2, ensure_ascii=False, sort_keys=True))


def _to_jsonable(value: Any) -> Any:
    if is_dataclass(value):
        return _to_jsonable(asdict(value))
    if isinstance(value, dict):
        return {str(key): _to_jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_to_jsonable(item) for item in value]
    if isinstance(value, datetime):
        return value.isoformat()
    return value


def _configure_stdout() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")


if __name__ == "__main__":
    main()
