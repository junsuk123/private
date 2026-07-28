from __future__ import annotations

import json
import sys
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from app import web as web_module
from app.features.feature_schema import LIVE_SHORT_HORIZON_SCHEMA


def _ready_snapshot() -> dict:
    return {
        "score": 1.0,
        "threshold": 0.9,
        "ready": True,
        "reasons": [],
        "components": {},
        "active_markets": ["US"],
        "evaluated_at": datetime.now(timezone.utc).isoformat(),
    }


def test_reliability_evaluator_requires_every_hard_gate() -> None:
    policy = SimpleNamespace(conflicts=lambda: [])
    with (
        patch.object(
            web_module,
            "_cached_kis_connection_probe",
            return_value={"ok": True, "account_checked": True, "actual_equity": 200_000},
        ),
        patch.object(
            web_module,
            "evaluate_live_runtime_gates",
            return_value=SimpleNamespace(ok=True, failures=()),
        ),
        patch.object(
            web_module,
            "load_short_horizon_strategy_config",
            return_value={"execution": {"live_trading_enabled": True}},
        ),
        patch.object(web_module, "_env_flag", return_value=True),
        patch.object(web_module.TradingPolicySnapshot, "from_environment", return_value=policy),
        patch.object(web_module, "_latest_model_reliability", return_value={"ok": True}),
        patch.object(web_module, "_auto_market_health", return_value={"ok": True}),
        patch.object(web_module, "_active_live_market_groups", return_value=("US",)),
    ):
        result = web_module._evaluate_auto_reliability()

    assert result["ready"] is True
    assert result["score"] == 1.0


def test_model_reliability_uses_active_champion_not_rejected_challenger() -> None:
    now = datetime.now(timezone.utc)
    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        active = {
            "artifact_id": "active-champion",
            "feature_schema_hash": LIVE_SHORT_HORIZON_SCHEMA.schema_hash,
            "live_eligible": True,
            "metrics": {"auc": 0.72, "precision_at_k": 0.43},
            "reason_codes": [],
        }
        challenger = {
            "artifact_id": "rejected-challenger",
            "feature_schema_hash": LIVE_SHORT_HORIZON_SCHEMA.schema_hash,
            "live_eligible": False,
            "metrics": {"auc": 0.60, "precision_at_k": 0.22},
            "reason_codes": ["METRICS_BELOW_LIVE_THRESHOLDS"],
        }
        (root / "latest.json").write_text(json.dumps(active), encoding="utf-8")
        (root / "live_short_horizon.rejected.json").write_text(
            json.dumps(challenger),
            encoding="utf-8",
        )
        with (
            patch.object(web_module, "Path", return_value=root),
            patch.dict(
                "os.environ",
                {
                    "AUTO_RELIABILITY_MODEL_MAX_AGE_SECONDS": "1800",
                    "AUTO_RELIABILITY_ACTIVE_MODEL_MAX_AGE_SECONDS": "2592000",
                },
            ),
        ):
            result = web_module._latest_model_reliability(now)

    assert result["ok"] is True
    assert result["artifact_id"] == "active-champion"
    assert result["metrics"]["auc"] == 0.72
    assert result["latest_challenger"]["artifact_id"] == "rejected-challenger"
    assert result["latest_challenger"]["live_eligible"] is False


def test_reliability_step_promotes_only_after_sustained_passes() -> None:
    now = datetime.now(timezone.utc)
    with web_module._live_lock:
        web_module._auto_reliability_state["ready_streak"] = 0
        web_module._auto_reliability_state["unready_streak"] = 0
    with (
        patch.dict(
            "os.environ",
            {
                "AUTO_RELIABILITY_PROMOTE_CONSECUTIVE": "2",
                "AUTO_RELIABILITY_DEMOTE_CONSECUTIVE": "2",
            },
        ),
        patch.object(web_module, "_evaluate_auto_reliability", return_value=_ready_snapshot()),
        patch.object(web_module, "_active_operation_mode", side_effect=["learning", "learning"]),
        patch.object(web_module, "_auto_reliability_enter_learning"),
        patch.object(web_module, "_auto_reliability_learning_maintenance"),
        patch.object(
            web_module,
            "_auto_reliability_transition_to_live",
            return_value={"ok": True, "status": "started", "live_trading_status": "armed"},
        ) as promote,
    ):
        first = web_module._auto_reliability_step(now)
        second = web_module._auto_reliability_step(now)

    assert first["mode"] == "learning"
    assert second["mode"] == "live_trading"
    promote.assert_called_once()


def test_live_mode_is_demoted_immediately_on_broker_failure() -> None:
    failed = {
        **_ready_snapshot(),
        "score": 0.8,
        "ready": False,
        "reasons": ["BROKER_NOT_READY"],
    }
    with (
        patch.object(web_module, "_evaluate_auto_reliability", return_value=failed),
        patch.object(web_module, "_active_operation_mode", return_value="live_trading"),
        patch.object(web_module, "_auto_reliability_enter_learning"),
        patch.object(web_module, "_auto_reliability_transition_to_learning") as demote,
        patch.object(web_module, "_auto_reliability_learning_maintenance"),
    ):
        result = web_module._auto_reliability_step()

    assert result["mode"] == "learning"
    demote.assert_called_once()
