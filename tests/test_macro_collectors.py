from __future__ import annotations

from dataclasses import dataclass

from app.data.public_collectors import FredMacroCollector


@dataclass
class _FakeFredClient:
    def get_csv_rows(self, url, params):
        assert params == {"id": "FEDFUNDS"}
        return [
            {"observation_date": "2026-05-01", "FEDFUNDS": "4.25"},
            {"observation_date": "2026-06-01", "FEDFUNDS": "."},
            {"observation_date": "2026-07-01", "FEDFUNDS": "4.10"},
        ]


def test_fred_uses_latest_valid_public_csv_observation_without_api_key(monkeypatch) -> None:
    monkeypatch.delenv("FRED_API_KEY", raising=False)

    metric = FredMacroCollector(client=_FakeFredClient()).collect_latest(
        "FEDFUNDS",
        "us_federal_funds_rate",
    )

    assert metric is not None
    assert metric.value == 4.10
    assert metric.observed_at.isoformat() == "2026-07-01T00:00:00+00:00"
    assert metric.source.source_name == "fred_public_csv"
    assert metric.source.source_id == "fred:FEDFUNDS"
