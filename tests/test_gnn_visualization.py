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
    assert momentum["edge_count"] > 0
    assert 0.0 <= momentum["connectivity"] <= 1.0
    assert momentum["relation_diversity"] >= 1
    assert payload["dynamics"]["mapping"] == "checkpoint_weighted_orbital_physics_v2"
    assert all("dynamic_weight" in link for link in payload["links"])
    assert all(link["polarity"] in {-1, 1} for link in payload["links"])
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
    assert 'id="gnn-auto-rotation-toggle"' in page.text
    assert 'id="gnn-model-fullscreen"' in page.text
    assert 'id="gnn-system-health"' in page.text
    assert 'id="gnn-health-physics"' in page.text
    assert 'aria-pressed="false"' in page.text
    # ONE cache-bust marker for the terminal bundle, bumped alongside the version
    # in web_account_routes.py. The point is that the page cannot ship a stale
    # strategy_terminal.js against a changed payload contract. There were two
    # markers here from successive features; the older one only recorded which
    # release last touched the file, and every bump broke it.
    assert "20260821-multi-trade-layers-v1" in page.text


def test_gnn_auto_rotation_is_default_on_persistent_and_pauses_for_manual_control() -> None:
    script = (
        Path(__file__).parents[1] / "src" / "app" / "static" / "strategy_terminal.js"
    ).read_text(encoding="utf-8")

    assert "strategy-terminal-gnn-auto-rotation-v1" in script
    assert "return saved === null ? true : saved === 'true'" in script
    assert "localStorage.setItem(GNN_AUTO_ROTATION_STORAGE_KEY" in script
    assert "gnnAutoRotationEnabled && !dragging && now >= autoRotateResumeAt" in script
    assert "rotationY = (rotationY + dt * GNN3D_AUTO_ROTATE_RAD_PER_SECOND)" in script
    assert "autoRotateResumeAt = performance.now() + GNN3D_AUTO_ROTATE_RESUME_DELAY_MS" in script
    assert "pointercancel', resumeAutoRotationAfterManualControl" in script


def test_gnn_model_panel_has_dedicated_fullscreen_control_and_canvas_resize() -> None:
    root = Path(__file__).parents[1]
    script = (root / "src" / "app" / "static" / "strategy_terminal.js").read_text(
        encoding="utf-8"
    )
    styles = (root / "src" / "app" / "static" / "strategy_terminal.css").read_text(
        encoding="utf-8"
    )

    assert "function bindGnnModelFullscreen()" in script
    assert "document.fullscreenElement === panel" in script
    assert "panel.requestFullscreen" in script
    assert "panel.classList.add('is-viewport-fullscreen')" in script
    assert "window.dispatchEvent(new Event('resize'))" in script
    assert "body.gnn-model-fullscreen" in styles
    assert ".gnn-model-panel:fullscreen .gnn-model-stage" in styles


def test_graph_polling_updates_existing_scene_instead_of_recreating_it() -> None:
    script = (
        Path(__file__).parents[1] / "src" / "app" / "static" / "strategy_terminal.js"
    ).read_text(encoding="utf-8")

    assert "gnn3dState.updateData(data)" in script
    assert "requestId !== terminalState.gnnGraphRequestId" in script
    assert "terminalState.ontologySignature === graphSignature" in script
    # A new inference is dynamic state, not a topology change. Including it in
    # the scene signature caused the WebGL canvas to be torn down every poll.
    topology_signature = script.split("function prepareGnnGraph(data)", 1)[1].split(
        "function bindGnnControls", 1
    )[0]
    signature_assignment = topology_signature.split("const signature =", 1)[1].split(";", 1)[0]
    assert "latest_at" not in signature_assignment


def test_graph_physics_uses_live_microstructure_without_faking_inference() -> None:
    script = (
        Path(__file__).parents[1] / "src" / "app" / "static" / "strategy_terminal.js"
    ).read_text(encoding="utf-8")

    assert "micro.return_5s" in script
    assert "micro.aggressor_imbalance_5s" in script
    assert "micro.spread_change_5s_bps" in script
    assert "refreshGnnMarketForces();" in script
    # The current rope solver normalises raw live forces once and then applies
    # gravity through the spring's learned mass/tension.  The former baseSag
    # expression belonged to the retired straight-chord solver.
    assert "const market = gnn3dMarketRopeTerms(forces)" in script
    assert "market.gravity * (1 + spring.mass * .34)" in script
    assert "ambientEnergy * (.16 + Number(link.learned_strength || 0) * .24)" in script
    assert "(!link.kind || link.kind === 'topology')" in script


def test_graph_motion_and_emphasis_are_driven_by_measured_graph_data() -> None:
    root = Path(__file__).parents[1]
    script = (root / "src" / "app" / "static" / "strategy_terminal.js").read_text(
        encoding="utf-8"
    )
    styles = (root / "src" / "app" / "static" / "strategy_terminal.css").read_text(
        encoding="utf-8"
    )

    assert "const relationshipSpeed = (.72 + coupling * .72) / (.82 + inertia * .34)" in script
    assert "const liveBoost = 1 + activation * 1.45" in script
    assert "orbit.ascending = (orbit.ascending + Number(orbit.precession || 0)" in script
    assert "meshOrbits[index].liveActivation = nodeIntensity[index] || 0" in script
    assert "Every compute kind gets an additive halo" in script
    assert "Math.min(.92" in script
    assert "function renderGnnSystemHealth()" in script
    assert "authorizationBlocked ? 'SHADOW ONLY' : 'LIVE AUTHORIZED'" in script
    assert "checkpoint_weighted_orbital_physics_v2" in (
        root / "src" / "app" / "gnn_visualization.py"
    ).read_text(encoding="utf-8")
    assert ".gnn-system-health" in styles


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


def test_recent_schema_mismatch_is_active_transport_but_blocked_inference(
    tmp_path: Path,
) -> None:
    inference = tmp_path / "inference.jsonl"
    inference.write_text(
        json.dumps({
            "symbol": "005930",
            "decisions": [{
                "path": "cpu_gnn",
                "action": "NO_TRADE",
                "strategy_id": None,
                "reason_codes": ["GNN_FEATURE_SCHEMA_MISMATCH"],
            }],
        }) + "\n",
        encoding="utf-8",
    )

    state = build_strategy_gnn_state(inference_path=inference, active_window_seconds=10)

    assert state["active"] is True
    assert state["inference_usable"] is False
    assert state["state"] == "BLOCKED"
    assert state["model_id"] == "strategy_utility_rgcn"


def test_unpromoted_checkpoint_is_usable_for_validation_not_orders(
    tmp_path: Path,
) -> None:
    inference = tmp_path / "inference.jsonl"
    inference.write_text(
        json.dumps({
            "symbol": "INTC",
            "decisions": [{
                "path": "cpu_gnn",
                "action": "NO_TRADE",
                "strategy_id": None,
                "reason_codes": ["GNN_CHECKPOINT_NOT_LIVE_AUTHORIZED"],
            }],
        }) + "\n",
        encoding="utf-8",
    )

    state = build_strategy_gnn_state(inference_path=inference, active_window_seconds=10)

    assert state["state"] == "INFERENCE_RUNNING"
    assert state["inference_usable"] is True
    assert state["order_authorized"] is False
    assert state["phase"] == "validation_only"


def test_strategy_nodes_publish_upside_supervision_next_to_label_count() -> None:
    """``training_labels`` counts snapshots, so it is high even for a strategy that
    never triggered — it was the number that used to certify coverage for heads with
    zero supervision. The graph must carry the rows that actually trained the upside
    beside it, or it implies evidence the checkpoint does not have."""
    from app.gnn_visualization import build_strategy_gnn_visualization

    payload = build_strategy_gnn_visualization()
    nodes = [node for node in payload["nodes"] if node.get("kind") == "strategy"]
    if not nodes:
        return

    for node in nodes:
        assert "training_upside_rows" in node
        assert "training_filled_rows" in node
        assert "minimum_upside_rows" in node
        assert "upside_supervised" in node
        rows = node["training_upside_rows"]
        if rows is None:
            assert node["upside_supervised"] is None
        else:
            assert node["upside_supervised"] == (
                rows >= node["minimum_upside_rows"]
            )
            # An upside row is a realized profitable fill, so it can never exceed
            # the fills it came from.
            assert rows <= node["training_filled_rows"]
