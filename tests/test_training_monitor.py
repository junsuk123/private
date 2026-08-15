from __future__ import annotations

import json
import re
import sys
from pathlib import Path

from fastapi.testclient import TestClient

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from app import web as web_module
from app.web import app


def _write_artifact(
    root: Path,
    *,
    stamp: str,
    minute: int,
    rows: int,
    new_rows: int,
    auc: float,
    promoted: bool = False,
) -> None:
    artifact_id = f"live_short_horizon.{stamp}"
    (root / f"{artifact_id}.json").write_text(
        json.dumps(
            {
                "artifact_id": artifact_id,
                "created_at": f"2026-07-28T12:{minute:02d}:00+00:00",
                "metrics": {
                    "auc": auc,
                    "precision_at_k": 0.4,
                    "avg_forward_net_return_bps_top_k": 2.5,
                    "example_count": rows,
                    "positive_labels": rows // 2,
                    "negative_labels": rows - rows // 2,
                },
                "training_data": {
                    "row_count": rows,
                    "fresh_row_count": rows - 2,
                    "new_materialized_row_count": new_rows,
                },
                "live_eligible": promoted,
                "deployment": {
                    "promoted": promoted,
                    "reason": "PROMOTED" if promoted else "NOT_LIVE_ELIGIBLE",
                },
            }
        ),
        encoding="utf-8",
    )


def test_training_history_reports_real_cycle_change(tmp_path: Path) -> None:
    _write_artifact(
        tmp_path,
        stamp="20260728T120100000000Z",
        minute=1,
        rows=100,
        new_rows=10,
        auc=0.55,
    )
    _write_artifact(
        tmp_path,
        stamp="20260728T120600000000Z",
        minute=6,
        rows=125,
        new_rows=25,
        auc=0.61,
        promoted=True,
    )

    result = web_module._live_training_history(
        root=tmp_path,
        limit=10,
        use_cache=False,
    )

    assert len(result["points"]) == 2
    assert result["status"] == "promoted"
    assert result["latest"]["training_rows"] == 125
    assert result["change"]["training_rows"] == 25
    assert round(result["change"]["auc"], 2) == 0.06
    assert result["optimizer"]["classification_learning_rate"] == 0.08
    assert result["rows_per_hour"] == 300.0


def test_strategy_terminal_includes_training_monitor() -> None:
    response = TestClient(app).get("/account")

    assert response.status_code == 200
    assert 'id="training-performance-chart"' in response.text
    assert 'id="training-data-chart"' in response.text
    # The terminal's CSS and JS must ship under the SAME cache-bust version, which
    # is the property that actually matters: a page serving new JS against old CSS
    # (or vice versa) is the stale-bundle bug this guarded. Asserting the literal
    # version instead meant every asset bump broke this test in three separate
    # files; test_gnn_visualization keeps one deliberate literal as the deploy
    # marker, and this one checks the invariant.
    css_version = re.search(r"strategy_terminal\.css\?v=([\w.-]+)", response.text)
    js_version = re.search(r"strategy_terminal\.js\?v=([\w.-]+)", response.text)
    assert css_version and js_version
    assert css_version.group(1) == js_version.group(1)


def test_diagnostics_score_is_labeled_as_infrastructure_not_tradability() -> None:
    response = TestClient(app).get("/account")
    root = Path(__file__).parents[1]
    script = (root / "src" / "app" / "static" / "strategy_terminal.js").read_text(
        encoding="utf-8"
    )

    assert response.status_code == 200
    assert 'aria-label="인프라 준비도"' in response.text
    assert "인프라 신뢰도 기준" in script
    assert "실거래 승격 기준" not in script


def test_operations_overview_requires_checkpoint_and_trusted_strategy_for_execution() -> None:
    script = (
        Path(__file__).parents[1]
        / "src"
        / "app"
        / "static"
        / "operations_overview.js"
    ).read_text(encoding="utf-8")

    assert "gnn?.checkpoint_live_authorized === true" in script
    assert "trusted.length > 0" in script


def test_decision_ontology_graph_has_graph_only_fullscreen_control() -> None:
    response = TestClient(app).get("/account")
    root = Path(__file__).parents[1]
    script = (root / "src" / "app" / "static" / "strategy_terminal.js").read_text(
        encoding="utf-8"
    )
    styles = (root / "src" / "app" / "static" / "strategy_terminal.css").read_text(
        encoding="utf-8"
    )

    assert response.status_code == 200
    assert 'id="decision-ontology-panel"' in response.text
    assert 'id="decision-ontology-fullscreen"' in response.text
    assert 'aria-controls="decision-ontology-panel"' in response.text
    assert "panel.requestFullscreen()" in script
    assert "document.exitFullscreen()" in script
    assert "fullscreenchange" in script
    assert "event.key !== 'Escape'" in script
    assert ".decision-ontology-panel:fullscreen" in styles
    assert ".decision-ontology-panel.is-viewport-fullscreen" in styles


def test_decision_ontology_empty_active_path_uses_a_connected_gate() -> None:
    root = Path(__file__).parents[1]
    script = (root / "src" / "app" / "static" / "strategy_terminal.js").read_text(
        encoding="utf-8"
    )

    assert "id: 'decision_gate'" in script
    assert "if (algorithm.gate)" in script
    assert "from: `algorithm:${algorithm.id}`" in script
    assert "to: decisionId" in script
    assert "sources.filter((source) => source.available).slice" not in script
    assert "indicators.filter((item) => item.available).slice" not in script
