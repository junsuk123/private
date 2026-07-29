from __future__ import annotations

import json
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
    assert "20260729-live-candidate-fix-3" in response.text
