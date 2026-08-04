from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from fastapi import FastAPI
from fastapi.testclient import TestClient

from app.gnn_visualization import build_strategy_gnn_state, build_strategy_gnn_visualization
from app.web_account_routes import create_account_router


def _checkpoint(tmp_path: Path) -> tuple[Path, Path]:
    checkpoint = tmp_path / "rgcn.npz"
    np.savez_compressed(
        checkpoint,
        config=np.asarray([1, 1, 3, 5, 3, 3, 4, 17], dtype=np.int64),
        relation_weights=np.ones((3, 5, 4), dtype=np.float32),
        self_weight=np.ones((5, 4), dtype=np.float32),
        strategy_heads=np.ones((3, 4, 8), dtype=np.float32),
        no_trade_head=np.ones((4,), dtype=np.float32),
        temporal_weights=np.ones((1,), dtype=np.float32),
    )
    metadata = tmp_path / "rgcn.json"
    metadata.write_text(
        json.dumps(
            {
                "checkpoint": str(checkpoint),
                "checkpoint_hash": "test-hash",
                "config": {"strategy_count": 3},
                "strategy_ids": ["intraday_momentum", "event_momentum", "breakout_volume"],
                "relation_names": [
                    "same_methodology_family",
                    "confirming_methodology",
                    "contrasting_methodology",
                ],
                "rows": 30,
                "snapshots": 10,
                "validation_metrics": {"success_direction_accuracy": 0.75},
            }
        ),
        encoding="utf-8",
    )
    return metadata, checkpoint


def test_visualization_uses_checkpoint_topology_not_market_ontology(tmp_path: Path) -> None:
    metadata, _checkpoint_path = _checkpoint(tmp_path)
    inference = tmp_path / "inference.jsonl"
    inference.write_text(
        json.dumps(
            {
                "as_of": "2026-08-02T00:00:00+00:00",
                "symbol": "005930",
                "decisions": [
                    {
                        "path": "cpu_gnn",
                        "strategy_id": "intraday_momentum",
                        "utility": 1.25,
                        "probability_success": 0.7,
                        "expected_net_return_bps": 8.2,
                        "reason_codes": [],
                    }
                ],
            }
        )
        + "\n",
        encoding="utf-8",
    )

    payload = build_strategy_gnn_visualization(
        metadata_path=metadata,
        inference_path=inference,
    )

    assert payload["source"]["not_ontology_fact_graph"] is True
    assert payload["counts"]["strategy_nodes"] == 3
    assert payload["counts"]["strategy_links"] == 6
    assert payload["counts"]["nodes"] == 36
    assert payload["counts"]["links"] == 206
    assert payload["counts"]["parameter_links"] == 200
    assert sum(node["kind"] == "feature" for node in payload["nodes"]) == 5
    assert sum(node["kind"] == "hidden" for node in payload["nodes"]) == 4
    assert sum(node["kind"] == "output" for node in payload["nodes"]) == 24
    assert payload["inference"]["successful_decisions"] == 1
    momentum = next(node for node in payload["nodes"] if node["id"] == "intraday_momentum")
    assert momentum["active"] is True
    assert momentum["latest_utility"] == 1.25
    assert "GNN_HEAD_SCHEMA_MISMATCH" in payload["model"]["runtime_reasons"]


def test_account_route_exposes_dedicated_gnn_graph() -> None:
    app = FastAPI()
    app.include_router(
        create_account_router(
            gnn_graph_provider=lambda: {
                "schema": "strategy_rgcn_visualization_v1",
                "nodes": [{"id": "strategy"}],
                "links": [],
            },
            gnn_state_provider=lambda: {"state": "INFERENCE_RUNNING", "active": True},
        )
    )
    client = TestClient(app)

    response = client.get("/api/account/gnn-graph")
    state = client.get("/api/account/gnn-state")
    page = client.get("/account")

    assert response.status_code == 200
    assert response.json()["schema"] == "strategy_rgcn_visualization_v1"
    assert state.json()["active"] is True
    assert 'id="gnn-model-canvas"' in page.text
    assert 'id="gnn-inference-live"' in page.text
    assert 'id="gnn-visualization-toggle"' in page.text
    assert 'aria-pressed="false"' in page.text
    # ONE cache-bust marker for the terminal bundle, bumped alongside the version
    # in web_account_routes.py. The point is that the page cannot ship a stale
    # strategy_terminal.js against a changed payload contract. There were two
    # markers here from successive features; the older one only recorded which
    # release last touched the file, and every bump broke it.
    assert "20260804-gnn-activation-1" in page.text


def test_gnn_state_marks_recent_log_as_active(tmp_path: Path) -> None:
    inference = tmp_path / "inference.jsonl"
    inference.write_text(
        json.dumps(
            {
                "as_of": "2026-08-02T00:00:00+00:00",
                "symbol": "AAPL",
                "decisions": [
                    {
                        "path": "cpu_gnn",
                        "action": "NO_TRADE",
                        "strategy_id": "intraday_momentum",
                        "utility": 0.4,
                        "reason_codes": [],
                    }
                ],
            }
        )
        + "\n",
        encoding="utf-8",
    )

    state = build_strategy_gnn_state(inference_path=inference, active_window_seconds=10)

    assert state["state"] == "INFERENCE_RUNNING"
    assert state["active"] is True
    assert state["symbol"] == "AAPL"
    # The old static ``phases`` list was what let the renderer sweep the four
    # layers on a timer. It is replaced by measured activation: this record
    # elected an arm without trading, and instruments no encoder or hidden layer.
    activation = state["activation"]
    assert activation["selected_strategy_id"] == "intraday_momentum"
    assert activation["strategies"]["intraday_momentum"]["state"] == "ELECTED_NO_TRADE"
    assert activation["layers"]["strategy_election"]["observed"] is True
    assert activation["layers"]["input"]["observed"] is False
