from __future__ import annotations

import json
from datetime import datetime, timezone

from app.routing.shadow_comparison import ShadowComparisonRecorder, ShadowDecision


def test_shadow_comparison_records_disagreement_without_execution(tmp_path) -> None:
    recorder = ShadowComparisonRecorder(tmp_path / "shadow.jsonl")
    comparison = recorder.compare(
        correlation_id="decision-1",
        symbol="005930",
        as_of=datetime(2026, 7, 27, tzinfo=timezone.utc),
        decisions=(
            ShadowDecision("legacy", "BUY", None, None, ("VOTE",)),
            ShadowDecision("ontology", "NO_TRADE", None, 0, ("COST",)),
            ShadowDecision("cpu_gnn", "ACTIVATE", "momentum", 2, ()),
            ShadowDecision("npu_gnn", "ACTIVATE", "momentum", 2.01, ()),
        ),
        validation_candidates=(
            ShadowDecision(
                "cpu_gnn_validation",
                "VALIDATE_ONLY",
                "momentum",
                -1.0,
                ("ORDER_PERMISSION_NOT_GRANTED",),
                expected_net_return_bps=-2.0,
            ),
        ),
    )
    assert not comparison.action_agreement
    assert not comparison.strategy_agreement
    assert comparison.utility_spread == 2.01
    payload = json.loads((tmp_path / "shadow.jsonl").read_text().strip())
    assert len(payload["decisions"]) == 4
    assert len(payload["validation_candidates"]) == 1
    assert payload["validation_candidates"][0]["action"] == "VALIDATE_ONLY"
    assert not hasattr(recorder, "broker")


def test_shadow_comparison_rotates_oversized_log(tmp_path) -> None:
    path = tmp_path / "shadow.jsonl"
    path.write_text("old telemetry\n", encoding="utf-8")
    recorder = ShadowComparisonRecorder(path, max_bytes=1, backup_count=2)

    recorder.compare(
        correlation_id="rotation-check",
        symbol="005930",
        as_of=datetime(2026, 8, 10, tzinfo=timezone.utc),
        decisions=(
            ShadowDecision(
                path="cpu_gnn",
                action="HOLD",
                strategy_id=None,
                utility=None,
                reason_codes=("NO_EDGE",),
            ),
        ),
    )

    rotated = tuple(tmp_path.glob("shadow.jsonl.r*"))
    assert len(rotated) == 1
    assert rotated[0].read_text(encoding="utf-8") == "old telemetry\n"
    assert "rotation-check" in path.read_text(encoding="utf-8")


def test_recorders_for_same_path_share_rotation_lock(tmp_path) -> None:
    path = tmp_path / "shadow.jsonl"
    first = ShadowComparisonRecorder(path, max_bytes=400, backup_count=100)
    second = ShadowComparisonRecorder(path, max_bytes=400, backup_count=100)
    errors: list[Exception] = []

    def write(recorder: ShadowComparisonRecorder, prefix: str) -> None:
        try:
            for index in range(20):
                recorder.compare(
                    correlation_id=f"{prefix}-{index}",
                    symbol="005930",
                    as_of=datetime(2026, 8, 10, tzinfo=timezone.utc),
                    decisions=(
                        ShadowDecision(
                            path="cpu_gnn",
                            action="HOLD",
                            strategy_id=None,
                            utility=None,
                            reason_codes=("NO_EDGE",),
                        ),
                    ),
                )
        except Exception as exc:  # pragma: no cover - assertion captures the race.
            errors.append(exc)

    import threading

    threads = (
        threading.Thread(target=write, args=(first, "a")),
        threading.Thread(target=write, args=(second, "b")),
    )
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()

    assert errors == []
    rows = []
    for segment in (*tmp_path.glob("shadow.jsonl.r*"), path):
        rows.extend(
            json.loads(line)
            for line in segment.read_text(encoding="utf-8").splitlines()
        )
    assert len(rows) == 40
