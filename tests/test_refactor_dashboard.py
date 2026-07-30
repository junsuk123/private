from __future__ import annotations

import json
import sqlite3
from pathlib import Path

from fastapi import FastAPI
from fastapi.testclient import TestClient

from app.refactor_dashboard import (
    _second_level_market_series,
    build_refactor_dashboard,
    build_strategy_market_stream,
    build_strategy_market_view,
)
from app.web_account_routes import create_account_router


def _write_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value), encoding="utf-8")


def test_second_series_uses_real_quote_updates_without_fake_trade_volume() -> None:
    with sqlite3.connect(":memory:") as connection:
        connection.row_factory = sqlite3.Row
        connection.executescript(
            """
            create table ticks (
                exchange_timestamp text, price real, volume integer,
                trade_direction text
            );
            create table books (
                exchange_timestamp text, best_bid real, best_ask real,
                spread_bps real, imbalance real
            );
            insert into ticks values
                ('2026-07-28T01:00:00+00:00', 100, 4, 'BUY');
            insert into books values
                ('2026-07-28T01:00:00+00:00', 99, 101, 200, 0.1),
                ('2026-07-28T01:00:01+00:00', 100, 102, 198, 0.2);
            """
        )
        ticks = connection.execute("select * from ticks").fetchall()
        books = connection.execute("select * from books").fetchall()

    bars, micro = _second_level_market_series(ticks, books)

    assert [bar["bar_source"] for bar in bars] == ["trade", "quote_mid"]
    assert bars[1]["close"] == 101
    assert bars[1]["volume"] == 0
    assert micro["trade_bar_count"] == 1
    assert micro["quote_bar_count"] == 1


def test_refactor_dashboard_is_fail_closed_and_surfaces_evidence(tmp_path: Path) -> None:
    _write_json(
        tmp_path / "config/refactor_profile.example.json",
        {
            "mode": "shadow",
            "broker_submission_enabled": False,
            "maximum_order_notional": 0,
            "allowed_symbols": [],
            "flags": {
                "legacy_vote_path": True,
                "websocket_market_data": True,
                "local_chart_engine": True,
                "ontology_router": True,
                "gnn_shadow": True,
                "gnn_rerank": False,
                "npu_inference": False,
                "strategy_owned_execution": False,
                "live_enabled": False,
            },
        },
    )
    _write_json(
        tmp_path / "data/reports/refactor_counterfactual_evaluation.json",
        {
            "status": "NOT_PROMOTED",
            "promotion_eligible": False,
            "coverage": {"distinct_utc_dates": 11, "symbols": 255},
            "labels": {"snapshots": 4980, "strategy_labels": 34860},
            "walk_forward_tabular_baseline": {
                "observations": 1992,
                "selected_trades": 0,
            },
            "strategy_metrics": {"intraday_momentum": {"triggered": 3, "filled": 2}},
        },
    )
    _write_json(
        tmp_path / "data/reports/strategy_utility_openvino.json",
        {
            "cpu": {"p95_ms": 0.66},
            "npu": {"p95_ms": 1.4},
            "promotion_eligible": False,
        },
    )

    result = build_refactor_dashboard(tmp_path)

    assert result["mode"] == "shadow"
    assert result["live_order_capable"] is False
    assert result["evaluation"]["strategy_labels"] == 34860
    assert result["evaluation"]["selected_trades"] == 0
    assert result["devices"]["selected"] == "CPU"
    assert any(not gate["passed"] for gate in result["promotion_gates"])


def test_refactor_dashboard_reads_strategy_ownership_without_mutation(tmp_path: Path) -> None:
    _write_json(
        tmp_path / "config/refactor_profile.example.json",
        {
            "mode": "shadow",
            "broker_submission_enabled": False,
            "maximum_order_notional": 0,
            "allowed_symbols": [],
            "flags": {
                "legacy_vote_path": True,
                "websocket_market_data": False,
                "local_chart_engine": False,
                "ontology_router": False,
                "gnn_shadow": False,
                "gnn_rerank": False,
                "npu_inference": False,
                "strategy_owned_execution": False,
                "live_enabled": False,
            },
        },
    )
    database = tmp_path / "data/store/trading-lifecycle.sqlite3"
    database.parent.mkdir(parents=True)
    with sqlite3.connect(database) as connection:
        connection.executescript(
            """
            create table strategy_instances (
                strategy_instance_id text, status text
            );
            create table positions (
                symbol text, quantity integer, origin_strategy_id text,
                strategy_instance_id text, opened_at text
            );
            insert into strategy_instances values ('instance-1', 'OPEN');
            insert into positions values (
                '005930', 2, 'intraday_momentum', 'instance-1',
                '2026-07-27T00:00:00+00:00'
            );
            """
        )

    result = build_refactor_dashboard(tmp_path)

    assert result["lifecycle"]["instances"] == 1
    assert result["lifecycle"]["open_positions"] == 1
    assert result["lifecycle"]["positions"][0]["strategy_id"] == "intraday_momentum"


def test_account_page_and_refactor_api_expose_new_console() -> None:
    app = FastAPI()
    observed_symbols: list[str] = []
    app.include_router(
        create_account_router(
            refactor_provider=lambda: {
                "mode": "shadow",
                "live_order_capable": False,
            },
            market_view_provider=lambda symbol, limit: {
                "symbol": symbol or "005930",
                "limit": limit,
            },
            market_stream_provider=lambda symbol, limit: {
                "symbol": symbol,
                "limit": limit,
                "stream": True,
            },
            market_stream_observer=observed_symbols.append,
        )
    )
    client = TestClient(app)

    page = client.get("/account")
    payload = client.get("/api/refactor/dashboard")

    assert page.status_code == 200
    assert "Strategy Trading Terminal" in page.text
    assert 'id="price-chart"' in page.text
    assert 'id="asset-overview-title"' in page.text
    assert 'id="asset-total"' in page.text
    assert 'id="execution-track"' in page.text
    assert 'id="ops-overview"' in page.text
    assert 'id="ops-gate-grid"' in page.text
    assert 'id="ops-gnn-samples"' in page.text
    assert 'id="ops-current-reason"' in page.text
    assert "operations_overview.js" in page.text
    assert "account_dashboard.js" not in page.text
    assert payload.status_code == 200
    assert payload.json()["mode"] == "shadow"

    market = client.get("/api/refactor/market-view?symbol=005930&limit=90")
    assert market.status_code == 200
    assert market.json() == {"symbol": "005930", "limit": 90}
    stream = client.get("/api/refactor/market-stream?symbol=005930&limit=30")
    assert stream.status_code == 200
    assert stream.json() == {"symbol": "005930", "limit": 30, "stream": True}
    assert observed_symbols == ["005930"]


def test_strategy_market_view_returns_candles_and_execution_stages(tmp_path: Path) -> None:
    _write_json(
        tmp_path / "config/refactor_profile.example.json",
        {
            "mode": "shadow",
            "broker_submission_enabled": False,
            "maximum_order_notional": 0,
            "allowed_symbols": [],
            "flags": {
                "legacy_vote_path": True,
                "websocket_market_data": False,
                "local_chart_engine": False,
                "ontology_router": False,
                "gnn_shadow": False,
                "gnn_rerank": False,
                "npu_inference": False,
                "strategy_owned_execution": False,
                "live_enabled": False,
            },
        },
    )
    database = tmp_path / "data/store/realtime_market_data.sqlite3"
    database.parent.mkdir(parents=True)
    with sqlite3.connect(database) as connection:
        connection.executescript(
            """
            create table realtime_minute_bars (
                symbol text, minute_start text, open real, high real, low real,
                close real, volume real, vwap real, trade_count integer,
                spread_bps real, orderbook_imbalance real, liquidity_score real,
                volatility real, last_update_age_ms real
            );
            create table realtime_ticks (
                symbol text, exchange_timestamp text, received_at text,
                price real, volume real, trade_direction text, latency_ms real
            );
            create table realtime_orderbook (
                symbol text, exchange_timestamp text, received_at text, source text,
                best_bid real, best_ask real,
                spread_bps real, total_bid_volume real, total_ask_volume real,
                imbalance real, latency_ms real, levels_json text
            );
            insert into realtime_minute_bars values (
                '005930', '2026-07-27T00:00:00+00:00',
                100, 103, 99, 102, 1000, 101, 5, 2, 0.2, 0.8, 0.01, 10
            );
            insert into realtime_ticks values (
                '005930', '2026-07-27T00:00:30+00:00',
                '2026-07-27T00:00:31+00:00', 102, 10, 'BUY', 4
            );
            insert into realtime_orderbook values (
                '005930', '2026-07-27T00:00:30+00:00',
                '2026-07-27T00:00:31+00:00', 'kis_realtime_websocket',
                101, 102, 2, 100, 80, 0.1, 3,
                '[{"bid_price":101,"bid_size":100,"ask_price":102,"ask_size":80}]'
            );
            """
        )
    shadow_path = tmp_path / "logs/refactor-shadow-comparison.jsonl"
    shadow_path.parent.mkdir(parents=True)
    shadow_path.write_text(
        json.dumps(
            {
                "symbol": "005930",
                "as_of": "2026-07-27T00:00:30+00:00",
                "action_agreement": False,
                "strategy_agreement": False,
                "decisions": [
                    {
                        "path": "cpu_gnn",
                        "action": "BUY",
                        "strategy_id": "intraday_momentum",
                        "utility": 1.25,
                        "reason_codes": ["ONTOLOGY_ALLOWED"],
                    }
                ],
            }
        )
        + "\n",
        encoding="utf-8",
    )
    journal_path = tmp_path / "data/store/causal-order-journal.jsonl"
    journal_path.write_text(
        json.dumps(
            {
                "event_type": "order_intent_persisted",
                "payload": {
                    "symbol": "005930",
                    "intent_id": "intent-1",
                    "action": "BUY",
                    "quantity": 1,
                },
            }
        )
        + "\n",
        encoding="utf-8",
    )

    result = build_strategy_market_view("005930", root=tmp_path)

    assert result["symbol"] == "005930"
    assert result["market"]["bars"][0]["close"] == 102
    assert result["market"]["latest_orderbook"]["best_bid"] == 101
    assert result["market"]["latest_orderbook"]["levels"][0]["ask_size"] == 80
    assert result["market"]["latest_orderbook"]["stale"] is True
    assert result["market"]["second_bars"][0]["close"] == 102
    assert result["market"]["second_bars"][0]["vwap"] == 102
    assert len(result["execution"]["stages"]) == 6
    assert result["selection"]["strategy_id"] == "intraday_momentum"
    assert result["algorithm"]["visual_indicators"] == ["MA5", "MA20", "VWAP", "Volume"]
    trace = result["decision_ontology"]
    assert trace["provenance"]["decision"] == "logs/refactor-shadow-comparison.jsonl"
    assert {source["id"] for source in trace["sources"]} >= {
        "minute_bars",
        "orderbook",
        "event_feed",
    }
    assert {indicator["id"] for indicator in trace["indicators"]} >= {
        "return",
        "volume",
        "vwap_deviation",
    }
    assert len(trace["algorithms"]) == 8
    assert next(
        algorithm
        for algorithm in trace["algorithms"]
        if algorithm["id"] == "intraday_momentum"
    )["final_selected"] is True
    assert trace["final_decision"]["action"] == "BUY"
    assert result["execution"]["stages"][2]["status"] == "complete"
    assert result["live_order_capable"] is False

    stream = build_strategy_market_stream("005930", root=tmp_path)
    assert stream["symbol"] == "005930"
    assert stream["market"]["second_bars"][0]["close"] == 102
    assert stream["market"]["microstructure"]["tick_count_1s"] == 1


def test_market_view_keeps_ontology_admissible_when_gnn_says_no_trade(tmp_path: Path) -> None:
    _write_json(
        tmp_path / "config/refactor_profile.example.json",
        {
            "mode": "shadow",
            "broker_submission_enabled": False,
            "maximum_order_notional": 0,
            "allowed_symbols": [],
            "flags": {
                "legacy_vote_path": True,
                "websocket_market_data": True,
                "local_chart_engine": True,
                "ontology_router": True,
                "gnn_shadow": True,
                "gnn_rerank": False,
                "npu_inference": False,
                "strategy_owned_execution": False,
                "live_enabled": False,
            },
        },
    )
    shadow_path = tmp_path / "logs/refactor-shadow-comparison.jsonl"
    shadow_path.parent.mkdir(parents=True)
    shadow_path.write_text(
        json.dumps(
            {
                "symbol": "005930",
                "as_of": "2026-07-27T00:00:30+00:00",
                "decisions": [
                    {
                        "path": "ontology",
                        "action": "ADMISSIBLE",
                        "strategy_id": "intraday_momentum",
                        "reason_codes": [],
                    },
                    {
                        "path": "cpu_gnn",
                        "action": "NO_TRADE",
                        "strategy_id": None,
                        "reason_codes": ["NON_POSITIVE_NET_EDGE"],
                    },
                ],
            }
        )
        + "\n",
        encoding="utf-8",
    )

    result = build_strategy_market_view("005930", root=tmp_path)

    assert result["selection"]["action"] == "NO_TRADE"
    assert result["selection"]["path"] == "cpu_gnn"
    assert result["selection"]["ontology_allowed"] is True
    assert result["selection"]["ontology_action"] == "ADMISSIBLE"
    assert result["selection"]["ontology_strategy_id"] == "intraday_momentum"
    assert result["algorithm"]["strategy_id"] == "intraday_momentum"
    trace = result["decision_ontology"]
    assert trace["ontology_selection"] == {
        "strategy_id": "intraday_momentum",
        "allowed": True,
        "action": "ADMISSIBLE",
    }
    assert trace["final_decision"]["action"] == "NO_TRADE"
    assert trace["final_decision"]["reason_codes"] == ["NON_POSITIVE_NET_EDGE"]
    assert next(
        algorithm
        for algorithm in trace["algorithms"]
        if algorithm["id"] == "intraday_momentum"
    )["ontology_selected"] is True
    candidate = result["candidates"][0]
    assert candidate["ontology_allowed"] is True
    assert candidate["action"] == "ADMISSIBLE"
    assert candidate["strategy_id"] == "intraday_momentum"
