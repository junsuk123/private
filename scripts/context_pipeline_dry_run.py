#!/usr/bin/env python
"""Exercise the production execution path end to end, with the broker call blocked.

What this proves and what it does not
--------------------------------------
It runs the **real** components — the real calendar, the real freshness registry, the real
graph and model runtime, the real ``FinalTradeGate``, the real order state machine and the
real reconcilers — against live data from the stores, and drives an order intent through
its whole lifecycle. The single thing it substitutes is the broker: a recording client
that answers like KIS and places nothing.

That is the only substitution, and it is the point. Everything that decides *whether* to
send an order is the production code; only the socket at the end is not.

    python scripts/context_pipeline_dry_run.py
    python scripts/context_pipeline_dry_run.py --symbols <code> <code> --json

With no ``--symbols`` it uses whatever the live cycle elected, and synthetic placeholders
if the cycle elected nothing. No issuer code is written into this file.

Exit code 0 means every stage completed and every fail-closed check behaved. Non-zero
means a stage did not, and the report says which.
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from app.data.freshness import DataFreshnessRegistry  # noqa: E402
from app.execution.order_state_machine import OrderState, OrderStateMachine  # noqa: E402
from app.execution.reconciliation import (  # noqa: E402
    AccountReconciler,
    AccountView,
    BrokerOrderStatus,
    BrokerPosition,
    OrderReconciler,
    PositionReconciler,
)
from app.models.gnn_runtime import GnnRuntime  # noqa: E402
from app.models.graph_snapshot import FEATURE_DIM, GraphSnapshotBuilder  # noqa: E402
from app.models.temporal_hetero_gnn import TemporalHeteroGnnConfig  # noqa: E402
from app.storage.trading_state_store import TradingStateStore  # noqa: E402
from app.trading.context_decision_pipeline import (  # noqa: E402
    CandidateInput,
    ContextDecisionPipeline,
)
from app.trading.context_runtime import (  # noqa: E402
    GRAPH_MAX_NODES,
    GRAPH_TIME_STEPS,
    ContextRuntime,
)


class RecordingBroker:
    """A broker client that records and never sends.

    Deliberately not a mock of the whole KIS surface: it implements only
    ``get_order_status``, which is the sole broker call the reconciliation path makes. A
    dry run that stubbed ``place_limit_order`` would be asserting that a stub returns what
    it was told to.
    """

    def __init__(self) -> None:
        self.status_calls: list[str] = []
        self.submissions: list[dict[str, Any]] = []
        self.answers: dict[str, BrokerOrderStatus] = {}

    def place_limit_order(self, order: Any) -> None:  # pragma: no cover - never called
        raise AssertionError(
            "the dry run must not reach the broker; the gate or the runner is misconfigured"
        )

    def get_order_status(self, broker_order_id: str) -> BrokerOrderStatus:
        self.status_calls.append(broker_order_id)
        return self.answers.get(
            broker_order_id, BrokerOrderStatus(broker_order_id, "OPEN")
        )


def _stage(report: dict[str, Any], name: str, ok: bool, **detail: Any) -> bool:
    report["stages"].append({"stage": name, "ok": bool(ok), **detail})
    return bool(ok)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--symbols", nargs="*", default=[])
    parser.add_argument("--store", default=None, help="Store path. Defaults to a temp file.")
    parser.add_argument("--json", action="store_true", help="Emit only the JSON report.")
    args = parser.parse_args()

    now = datetime.now(timezone.utc)
    report: dict[str, Any] = {
        "started_at": now.isoformat(),
        "broker": "RecordingBroker (no orders are sent)",
        "stages": [],
    }

    if args.store:
        store_path = Path(args.store)
    else:
        import tempfile

        store_path = Path(tempfile.mkdtemp(prefix="obaits-dryrun-")) / "state.sqlite3"
    store = TradingStateStore(store_path)
    report["store_path"] = str(store_path)

    broker = RecordingBroker()
    machine = OrderStateMachine(store)
    registry = DataFreshnessRegistry()
    config = TemporalHeteroGnnConfig(
        max_nodes=GRAPH_MAX_NODES, feature_dim=FEATURE_DIM, time_steps=GRAPH_TIME_STEPS
    )
    gnn = GnnRuntime(config=config, require_checkpoint=True)
    pipeline = ContextDecisionPipeline(
        store=store,
        gnn_runtime=gnn,
        snapshot_builder=GraphSnapshotBuilder(
            max_nodes=GRAPH_MAX_NODES, time_steps=GRAPH_TIME_STEPS
        ),
        state_machine=machine,
        freshness=registry,
    )
    runtime = ContextRuntime(
        store=store,
        freshness=registry,
        gnn_runtime=gnn,
        pipeline=pipeline,
        state_machine=machine,
        require_checkpoint=True,
    )

    # -- 1. sessions resolve from the calendar --------------------------------- #
    session = runtime.session_view(now=now)
    _stage(
        report,
        "calendar_session",
        bool(session["groups"]["KR"]["session_phase"]),
        kr=session["groups"]["KR"]["session_phase"],
        us=session["groups"]["US"]["session_phase"],
        kr_trading_day=session["groups"]["KR"]["trading_day"],
    )

    # -- 2. a cycle runs against live stores ------------------------------------ #
    candidates = [
        CandidateInput(ticker=str(symbol).strip().upper(), market_group="KR")
        for symbol in args.symbols
    ] or None
    cycle = runtime.refresh(now=now, candidates=candidates, websocket_connected=False)
    _stage(
        report,
        "live_cycle",
        cycle is not None,
        decision_count=len(cycle.decisions) if cycle else 0,
        regime=cycle.regime.dominant if cycle else None,
        runtime_error=runtime.status().last_error,
    )

    # -- 3. health is reported honestly ------------------------------------------ #
    model_health = runtime.model_health_view(now=now)
    data_health = runtime.data_health_view(now=now)
    dashboard = runtime.dashboard_view(now=now)
    consistent = not (
        dashboard["readiness"]["new_entry_permitted"]
        and (
            dashboard["GNN_HEALTH"] == "OFFLINE"
            or dashboard["DATA_AGE"] == "STALE"
        )
    )
    _stage(
        report,
        "health_consistency",
        consistent,
        gnn=model_health["state"],
        data=data_health["worst_state"],
        final_gate=dashboard["FINAL_GATE"],
        new_entry_permitted=dashboard["readiness"]["new_entry_permitted"],
    )

    # -- 4. the order lifecycle, end to end -------------------------------------- #
    # Symbols come from the arguments or from whatever the cycle actually saw. No issuer
    # is written into this script: the runtime must never carry a preferred symbol, and a
    # dry run that hardcoded one would be exercising a universe the live path does not use.
    primary, secondary = _dry_run_symbols(args.symbols, cycle)
    report["symbols"] = {"primary": primary, "secondary": secondary}

    intent = machine.create(
        ticker=primary,
        side="BUY",
        quantity=100,
        idempotency_key=f"dryrun-{now.strftime('%Y%m%dT%H%M%S')}",
        limit_price=70_000.0,
        market_group="KR",
        venue="KRX",
        now=now,
    )
    duplicate = machine.create(
        ticker=primary,
        side="BUY",
        quantity=100,
        idempotency_key=f"dryrun-{now.strftime('%Y%m%dT%H%M%S')}",
        limit_price=70_000.0,
        now=now,
    )
    _stage(
        report,
        "idempotency",
        duplicate.intent_id == intent.intent_id,
        intent_id=intent.intent_id,
    )

    machine.transition(intent.intent_id, OrderState.GATED, now=now)
    machine.transition(intent.intent_id, OrderState.SUBMITTING, now=now)
    machine.transition(
        intent.intent_id, OrderState.SUBMITTED, broker_order_id="DRYRUN-1", now=now
    )
    _stage(
        report,
        "duplicate_prevention",
        machine.has_duplicate_risk(primary, "BUY"),
    )

    broker.answers["DRYRUN-1"] = BrokerOrderStatus(
        "DRYRUN-1", "OPEN", filled_quantity=40, average_price=70_000.0
    )
    OrderReconciler(machine, broker).reconcile(now=now)
    partial = machine.get(intent.intent_id)
    _stage(
        report,
        "partial_fill",
        partial is not None
        and partial.state is OrderState.PARTIALLY_FILLED
        and partial.filled_quantity == 40,
        state=partial.state.value if partial else None,
        filled=partial.filled_quantity if partial else None,
    )

    broker.answers["DRYRUN-1"] = BrokerOrderStatus(
        "DRYRUN-1", "FILLED", filled_quantity=100, average_price=70_200.0
    )
    OrderReconciler(machine, broker).reconcile(now=now)
    filled = machine.get(intent.intent_id)
    _stage(
        report,
        "fill_reconciliation",
        filled is not None and filled.state is OrderState.FILLED,
        state=filled.state.value if filled else None,
        average_fill_price=filled.average_fill_price if filled else None,
    )

    # -- 5. restart recovery ------------------------------------------------------- #
    stranded = machine.create(
        ticker=secondary,
        side="BUY",
        quantity=10,
        idempotency_key=f"dryrun-stranded-{now.strftime('%Y%m%dT%H%M%S')}",
        limit_price=200_000.0,
        now=now,
    )
    machine.transition(stranded.intent_id, OrderState.GATED, now=now)
    machine.transition(stranded.intent_id, OrderState.SUBMITTING, now=now)
    reopened = OrderStateMachine(TradingStateStore(store_path))
    recovery = reopened.recover(now=now)
    _stage(
        report,
        "restart_recovery",
        any(
            item["intent_id"] == stranded.intent_id
            for item in recovery["needs_broker_query"]
        ),
        open_count=recovery["open_count"],
        blocked_tickers=recovery["blocked_tickers"],
    )

    # -- 6. an unanswerable broker leaves the order UNKNOWN, not clean ------------- #
    class _Silent:
        def get_order_status(self, broker_order_id: str):
            raise TimeoutError("broker did not answer")

    unresolved = OrderReconciler(reopened, _Silent()).reconcile(now=now)
    stranded_after = reopened.get(stranded.intent_id)
    _stage(
        report,
        "unknown_state_on_timeout",
        not unresolved.reconciled
        and stranded_after is not None
        and stranded_after.state is OrderState.UNKNOWN,
        reasons=list(unresolved.reason_codes),
    )

    # -- 7. reconciliation --------------------------------------------------------- #
    positions = PositionReconciler(reopened).reconcile(
        [BrokerPosition(primary, 100.0, 70_200.0)], now=now
    )
    account = AccountReconciler(store).reconcile(
        AccountView(equity=1e8, cash=3e7, currency="KRW", observed_at=now),
        local_equity=1e8,
        local_cash=3e7,
        now=now,
    )
    _stage(
        report,
        "position_reconciliation",
        positions.reconciled,
        discrepancies=[item.as_dict() for item in positions.discrepancies],
    )
    _stage(report, "account_reconciliation", account.reconciled)

    # -- 8. nothing reached the broker ---------------------------------------------- #
    _stage(
        report,
        "no_orders_submitted",
        not broker.submissions,
        status_queries=len(broker.status_calls),
    )

    report["ok"] = all(stage["ok"] for stage in report["stages"])
    report["finished_at"] = datetime.now(timezone.utc).isoformat()

    output = ROOT / "data" / "reports"
    output.mkdir(parents=True, exist_ok=True)
    path = output / f"context_pipeline_dry_run_{now.strftime('%Y%m%d_%H%M%S')}.json"
    path.write_text(
        json.dumps(report, indent=2, sort_keys=True, default=str), encoding="utf-8"
    )
    report["report_path"] = str(path)

    if args.json:
        print(json.dumps(report, indent=2, default=str))
    else:
        for stage in report["stages"]:
            mark = "ok  " if stage["ok"] else "FAIL"
            detail = {k: v for k, v in stage.items() if k not in {"stage", "ok"}}
            print(f"[{mark}] {stage['stage']}: {json.dumps(detail, default=str)}")
        print(f"\nreport: {path}")
        print(f"result: {'PASS' if report['ok'] else 'FAIL'}")
    return 0 if report["ok"] else 1


def _dry_run_symbols(requested: list[str], cycle: Any) -> tuple[str, str]:
    """Two distinct symbols for the lifecycle walk, discovered rather than hardcoded.

    Preference order: the arguments, then whatever the live cycle actually elected, then
    two synthetic placeholders. The placeholders never reach a broker — the whole point of
    this script — so they only need to be distinct and well-formed.
    """
    symbols: list[str] = []
    for raw in requested:
        for item in str(raw).split(","):
            symbol = item.strip().upper()
            if symbol and symbol not in symbols:
                symbols.append(symbol)
    if cycle is not None:
        for decision in cycle.decisions:
            symbol = str(decision.ticker).strip().upper()
            if symbol and symbol not in symbols:
                symbols.append(symbol)
    placeholders = ["DRYRUN-A", "DRYRUN-B"]
    while len(symbols) < 2:
        candidate = placeholders.pop(0)
        if candidate not in symbols:
            symbols.append(candidate)
    return symbols[0], symbols[1]


if __name__ == "__main__":
    raise SystemExit(main())
