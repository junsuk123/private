from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from app.trading_pipeline import (  # noqa: E402
    _ontology_filter_1_python_loop,
    build_lightweight_market_snapshots,
    ontology_filter_1,
    universe_from_tickers,
)


def test_vectorized_screening_matches_legacy_cpu_rules(monkeypatch) -> None:
    monkeypatch.setenv("ONTOLOGY_NPU_ENABLED", "false")
    snapshots = build_lightweight_market_snapshots(
        universe_from_tickers(tuple(f"SIM{i:04d}" for i in range(512))),
        seed=17,
    )

    legacy = _ontology_filter_1_python_loop(snapshots, target_count=40)
    vectorized = ontology_filter_1(snapshots, target_count=40)

    assert vectorized.candidate_stocks == legacy.candidate_stocks
    assert vectorized.rejected_stocks == legacy.rejected_stocks
    assert vectorized.full_universe_count == legacy.full_universe_count
    assert vectorized.chart_fetch_scope == vectorized.candidate_stocks
    assert vectorized.metrics["backend"] == "python_numpy_vectorized"


def test_vectorized_screening_materializes_only_compact_traces(monkeypatch) -> None:
    monkeypatch.setenv("ONTOLOGY_NPU_ENABLED", "false")
    snapshots = build_lightweight_market_snapshots(
        universe_from_tickers(tuple(f"SIM{i:04d}" for i in range(4096))),
        seed=23,
    )

    result = ontology_filter_1(snapshots, target_count=50)

    assert len(result.candidate_stocks) <= 50
    assert len(result.traces) <= len(result.candidate_stocks) + 24
    assert result.metrics["candidate_count_input"] == 4096
    assert result.metrics["candidate_count_after_hard_filter"] + len(result.rejected_stocks) == 4096
