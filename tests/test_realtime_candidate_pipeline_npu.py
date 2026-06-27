from __future__ import annotations

from app.trading_pipeline import build_lightweight_market_snapshots, ontology_filter_1, universe_from_tickers


def test_candidate_pipeline_records_npu_topk_metrics(monkeypatch) -> None:
    monkeypatch.setenv("ONTOLOGY_NPU_ENABLED", "true")
    monkeypatch.setenv("ONTOLOGY_NPU_TOP_K", "12")
    snapshots = build_lightweight_market_snapshots(universe_from_tickers(tuple(f"SIM{i:04d}" for i in range(80))))

    result = ontology_filter_1(snapshots, target_count=40)

    assert len(result.candidate_stocks) <= 12
    assert result.chart_fetch_scope == result.candidate_stocks
    assert result.metrics["candidate_count_input"] == 80
    assert result.metrics["candidate_count_after_npu_topk"] == len(result.candidate_stocks)


def test_candidate_pipeline_cpu_rules_fallback_when_disabled(monkeypatch) -> None:
    monkeypatch.setenv("ONTOLOGY_NPU_ENABLED", "false")
    snapshots = build_lightweight_market_snapshots(universe_from_tickers(tuple(f"SIM{i:04d}" for i in range(80))))

    result = ontology_filter_1(snapshots, target_count=10)

    assert len(result.candidate_stocks) <= 10
    assert result.metrics["device"] == "CPU_RULES"
