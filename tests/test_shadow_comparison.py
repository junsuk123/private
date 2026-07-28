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
    )
    assert not comparison.action_agreement
    assert not comparison.strategy_agreement
    assert comparison.utility_spread == 2.01
    payload = json.loads((tmp_path / "shadow.jsonl").read_text().strip())
    assert len(payload["decisions"]) == 4
    assert not hasattr(recorder, "broker")
