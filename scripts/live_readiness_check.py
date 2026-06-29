from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from app.config import LiveConfigError, load_live_trading_safety_config, load_order_execution_config
from app.execution.kis_auth import build_kis_client, run_kis_health_check, validate_live_secret_file
from app.features.feature_schema import LIVE_SHORT_HORIZON_SCHEMA
from app.models.model_artifact_registry import ModelArtifactRegistry
from app.trading.live_runtime_guard import evaluate_live_runtime_gates


def main() -> int:
    parser = argparse.ArgumentParser(description="Run fail-closed live trading readiness checks.")
    parser.add_argument("--dry-run", action="store_true", help="Skip KIS network checks.")
    parser.add_argument("--check", default="", help="Comma-separated subset, e.g. kis-auth,kis-account.")
    parser.add_argument("--no-orders", action="store_true", help="Document that no orders may be submitted.")
    args = parser.parse_args()

    selected = {item.strip() for item in args.check.split(",") if item.strip()}
    report = {
        "checked_at": datetime.now(timezone.utc).isoformat(),
        "dry_run": args.dry_run,
        "no_orders": True,
        "gates": {},
        "failures": {},
    }

    def record(name: str, ok: bool, reason: str | None = None) -> None:
        report["gates"][name] = ok
        if not ok and reason:
            report["failures"][name] = reason

    try:
        load_live_trading_safety_config(allow_example=args.dry_run)
        record("live_trading_safety_config", True)
    except LiveConfigError as exc:
        record("live_trading_safety_config", False, str(exc))
    try:
        load_order_execution_config(allow_example=args.dry_run)
        record("order_execution_config", True)
    except LiveConfigError as exc:
        record("order_execution_config", False, str(exc))

    try:
        artifact = ModelArtifactRegistry().load_latest_live_eligible()
        model_ok = artifact.feature_schema_hash == LIVE_SHORT_HORIZON_SCHEMA.schema_hash
        record("live_eligible_model", model_ok, None if model_ok else "feature schema mismatch")
    except Exception as exc:
        if args.dry_run:
            report["gates"]["live_eligible_model_diagnostic"] = False
            report.setdefault("diagnostics", {})["live_eligible_model"] = exc.__class__.__name__
        else:
            record("live_eligible_model", False, exc.__class__.__name__)

    secrets = validate_live_secret_file()
    secret_ok = all(secrets.values()) if not args.dry_run else True
    record("kis_secret_file", bool(secret_ok), "missing KIS secret file or required keys")

    runtime = evaluate_live_runtime_gates(require_manual_arming=False)
    if args.dry_run:
        report["gates"]["live_flags_diagnostic"] = runtime.ok
        if runtime.failures:
            report.setdefault("diagnostics", {})["live_flags"] = list(runtime.failures)
    else:
        record("live_flags", runtime.ok, ",".join(runtime.failures) if runtime.failures else None)

    should_check_kis = not args.dry_run and (not selected or selected & {"kis-auth", "kis-account", "kis-websocket"})
    if should_check_kis:
        try:
            client = build_kis_client()
            health = run_kis_health_check(
                client,
                include_account=not selected or "kis-account" in selected,
                include_websocket=not selected or "kis-websocket" in selected,
            )
            record("kis_health", health.ok, ",".join(f"{k}:{v}" for k, v in health.failures.items()))
        except Exception as exc:  # noqa: BLE001 - readiness should report all hard failures.
            record("kis_health", False, exc.__class__.__name__)
    elif args.dry_run:
        record("kis_health", True)

    report_path = _write_report(report)
    print(json.dumps({"ok": not report["failures"], "report_path": str(report_path), **report}, indent=2))
    return 0 if not report["failures"] else 1


def _write_report(report: dict) -> Path:
    output_dir = Path("data/reports")
    output_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    path = output_dir / f"live_readiness_{stamp}.json"
    path.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    return path


if __name__ == "__main__":
    raise SystemExit(main())
