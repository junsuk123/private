"""Why did no strategy get selected? Aggregate the answer per market and strategy.

Reads what the running system already writes -- the live entry-blockade endpoint
when the server is up, and the decision/shadow journals when it is not -- and
prints the selection funnel, the first-failing-stage distribution, and the mean
rule/cost/model/fused decomposition.

This script measures; it never trades and never mutates state.

    python scripts/diagnose_strategy_selection.py                  # table
    python scripts/diagnose_strategy_selection.py --json           # machine form
    python scripts/diagnose_strategy_selection.py --source journal
"""

from __future__ import annotations

import argparse
import json
import sys
import urllib.error
import urllib.request
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from app.technical.selection_diagnostics import STAGE_ORDER  # noqa: E402

DEFAULT_ENDPOINT = "http://127.0.0.1:8010"
#: Journals that can carry per-candidate decision records.
JOURNAL_CANDIDATES = (
    ROOT / "logs" / "decision-log.jsonl",
    ROOT / "logs" / "refactor-shadow-comparison.jsonl",
)


def _fetch(url: str, timeout: float) -> dict[str, Any] | None:
    try:
        with urllib.request.urlopen(url, timeout=timeout) as response:  # noqa: S310
            return json.loads(response.read().decode("utf-8"))
    except (urllib.error.URLError, TimeoutError, OSError, json.JSONDecodeError):
        return None


def from_live(endpoint: str, timeout: float) -> dict[str, Any]:
    """The authoritative live answer: the blockade chain plus engine counters."""
    out: dict[str, Any] = {"source": "live", "endpoint": endpoint}
    blockade = _fetch(f"{endpoint}/api/realtime-trading/entry-blockade", timeout)
    if blockade is None:
        out["error"] = "endpoint unreachable"
        return out
    out["trading_possible"] = blockade.get("trading_possible")
    out["blocking_stage"] = blockade.get("blocking_stage")
    out["blocking_detail"] = blockade.get("blocking_detail")
    chain: list[dict[str, Any]] = []
    for step in blockade.get("chain") or ():
        if not isinstance(step, dict):
            continue
        data = step.get("data") if isinstance(step.get("data"), dict) else {}
        chain.append(
            {
                "stage": step.get("stage"),
                "ok": step.get("ok"),
                "detail": step.get("detail"),
                "blocking_reason_codes": list(data.get("blocking_reason_codes") or ()),
            }
        )
    out["chain"] = chain

    status = _fetch(f"{endpoint}/api/realtime-trading/status", timeout)
    if isinstance(status, dict):
        engine = status.get("status") or {}
        summary = engine.get("last_summary") or {}
        session = status.get("strategy_session") or {}
        out["engine"] = {
            "cycles": engine.get("cycles"),
            "buy_enabled": status.get("buy_enabled"),
            "buy_disabled_reason": status.get("buy_disabled_reason"),
            "submitted": engine.get("submitted"),
            "buy_submit_attempted": engine.get("buy_submit_attempted"),
            "buy_evaluated": summary.get("buy_evaluated"),
            "buy_rejected": summary.get("buy_rejected"),
            "buy_candidate_count": summary.get("buy_candidate_count"),
            "rejections": list(summary.get("rejections") or ()),
        }
        out["session"] = {
            "phase": session.get("phase"),
            "macro_regime": session.get("macro_regime"),
            "last_reason": session.get("last_reason"),
            "bandit_reason_codes": list(session.get("bandit_reason_codes") or ()),
            "cost_coverage_ratio": session.get("cost_coverage_ratio"),
            "expected_cost_bps": session.get("expected_cost_bps"),
            "expected_net_return_bps": session.get("expected_net_return_bps"),
        }
    diagnostics = _fetch(f"{endpoint}/api/strategy-selection/diagnostics", timeout)
    if isinstance(diagnostics, dict) and not diagnostics.get("error"):
        out["selection_diagnostics"] = diagnostics
    return out


def _iter_journal(path: Path, limit: int) -> Iterable[dict[str, Any]]:
    """Tail ``limit`` parseable JSON objects without loading a multi-GB file."""
    if not path.exists():
        return ()
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8", errors="replace") as handle:
        for line in handle:
            line = line.strip()
            if not line or not line.startswith("{"):
                continue
            try:
                payload = json.loads(line)
            except json.JSONDecodeError:
                continue
            if isinstance(payload, dict):
                rows.append(payload)
                if len(rows) > limit:
                    rows.pop(0)
    return rows


def from_journal(limit: int) -> dict[str, Any]:
    """Offline fallback: count reason codes per market/strategy from journals."""
    stage_counts: Counter = Counter()
    reason_counts: Counter = Counter()
    per_market: dict[str, Counter] = defaultdict(Counter)
    per_strategy: dict[str, Counter] = defaultdict(Counter)
    scanned = 0
    used: list[str] = []

    for path in JOURNAL_CANDIDATES:
        rows = list(_iter_journal(path, limit))
        if not rows:
            continue
        used.append(str(path.relative_to(ROOT)))
        for row in rows:
            scanned += 1
            market = str(row.get("market") or row.get("market_group") or "UNKNOWN")
            strategy = str(
                row.get("strategy_id") or row.get("selected_strategy") or "UNKNOWN"
            )
            codes = row.get("reason_codes") or row.get("rejections") or ()
            if isinstance(codes, str):
                codes = [codes]
            for code in codes:
                text = str(code)
                reason_counts[text] += 1
                per_market[market][text] += 1
                per_strategy[strategy][text] += 1
            stage = row.get("stage") or row.get("blocking_stage")
            if stage:
                stage_counts[str(stage)] += 1

    return {
        "source": "journal",
        "journals": used,
        "rows_scanned": scanned,
        "stage_counts": {
            stage.value: stage_counts.get(stage.value, 0) for stage in STAGE_ORDER
        },
        "reason_code_counts": dict(reason_counts.most_common()),
        "by_market": {k: dict(v.most_common(12)) for k, v in per_market.items()},
        "by_strategy": {k: dict(v.most_common(12)) for k, v in per_strategy.items()},
    }


def _print_table(report: dict[str, Any]) -> None:
    def line(char: str = "-") -> None:
        print(char * 78)

    line("=")
    print(f"STRATEGY SELECTION DIAGNOSTIC  ({report.get('source')})")
    line("=")

    if report.get("error"):
        print(f"  error: {report['error']}")

    if report.get("chain"):
        print(f"\ntrading_possible : {report.get('trading_possible')}")
        print(f"blocking_stage   : {report.get('blocking_stage')}")
        print(f"blocking_detail  : {report.get('blocking_detail')}")
        print("\nCHAIN (first not-ok stage is the cause)")
        line()
        for step in report["chain"]:
            flag = "OK " if step.get("ok") else "STOP"
            print(f"  [{flag}] {str(step.get('stage')):<22} {step.get('detail')}")
            for code in step.get("blocking_reason_codes") or ():
                print(f"         - {code}")

    engine = report.get("engine")
    if engine:
        print("\nENGINE COUNTERS")
        line()
        for key, value in engine.items():
            if key == "rejections":
                continue
            print(f"  {key:<22} {value}")
        for rejection in engine.get("rejections") or ():
            print(f"  rejection              {rejection}")

    session = report.get("session")
    if session:
        print("\nSTRATEGY SESSION")
        line()
        for key, value in session.items():
            print(f"  {key:<24} {value}")

    diagnostics = report.get("selection_diagnostics")
    if diagnostics:
        funnel = diagnostics.get("funnel") or {}
        if funnel:
            print("\nFUNNEL")
            line()
            for key, value in funnel.items():
                print(f"  {key:<18} {value}")
        decomposition = diagnostics.get("edge_decomposition") or {}
        if decomposition:
            print("\nEDGE DECOMPOSITION (mean, per first-failing stage)")
            line()
            header = (
                f"  {'stage':<28}{'n':>5}{'gross':>9}{'cost':>9}"
                f"{'rule':>9}{'model':>9}{'w':>6}{'fused':>9}{'req':>8}"
            )
            print(header)
            for stage, stats in decomposition.items():
                def fmt(name: str, width: int = 9, digits: int = 1) -> str:
                    value = stats.get(name)
                    return f"{value:>{width}.{digits}f}" if isinstance(value, (int, float)) else f"{'-':>{width}}"

                print(
                    f"  {stage:<28}{int(stats.get('count') or 0):>5}"
                    f"{fmt('rule_gross_bps')}{fmt('all_in_cost_bps')}"
                    f"{fmt('rule_net_bps')}{fmt('model_net_bps')}"
                    f"{fmt('model_weight', 6, 2)}{fmt('fused_net_bps')}"
                    f"{fmt('required_net_bps', 8)}"
                )

    counts = report.get("stage_counts")
    if counts and any(counts.values()):
        print("\nSTAGE COUNTS")
        line()
        for stage, value in counts.items():
            if value:
                print(f"  {stage:<30} {value}")

    reasons = report.get("reason_code_counts")
    if reasons:
        print("\nREASON CODES (most frequent first)")
        line()
        for code, value in list(reasons.items())[:25]:
            print(f"  {code:<46} {value}")

    for key, title in (("by_market", "BY MARKET"), ("by_strategy", "BY STRATEGY")):
        grouped = report.get(key)
        if not grouped:
            continue
        print(f"\n{title}")
        line()
        for name, codes in grouped.items():
            print(f"  {name}")
            for code, value in codes.items():
                print(f"      {code:<44} {value}")

    print()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--source",
        choices=("auto", "live", "journal"),
        default="auto",
        help="auto tries the running server first, then falls back to journals",
    )
    parser.add_argument("--endpoint", default=DEFAULT_ENDPOINT)
    parser.add_argument("--timeout", type=float, default=120.0)
    parser.add_argument(
        "--limit", type=int, default=5000, help="journal rows to tail per file"
    )
    parser.add_argument("--json", action="store_true", help="emit JSON only")
    args = parser.parse_args()

    report: dict[str, Any]
    if args.source == "journal":
        report = from_journal(args.limit)
    else:
        report = from_live(args.endpoint, args.timeout)
        if args.source == "auto" and report.get("error"):
            fallback = from_journal(args.limit)
            fallback["live_error"] = report["error"]
            report = fallback

    if args.json:
        print(json.dumps(report, ensure_ascii=False, indent=2, default=str))
    else:
        _print_table(report)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
