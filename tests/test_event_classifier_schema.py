from __future__ import annotations

from app.data.event_classifier import build_event_classifier_from_env
from app.data.event_classifier_keyword import KeywordEventClassifier
from app.data.event_classifier_openvino import OpenVinoEventClassifier


def test_keyword_event_classifier_output_schema() -> None:
    result = KeywordEventClassifier().classify(
        "AAPL earnings beat and revenue growth",
        "Analyst upgrade cites strong guidance.",
        ticker="AAPL",
    )

    assert 0.0 <= result.ticker_relevance <= 1.0
    assert -1.0 <= result.sentiment_score <= 1.0
    assert result.event_labels
    assert 0.0 <= result.confidence <= 1.0
    assert result.provider == "keyword"


def test_openvino_missing_model_falls_back_to_keyword(tmp_path) -> None:
    classifier = OpenVinoEventClassifier(str(tmp_path / "missing.xml"))

    result = classifier.classify("Lawsuit risk rises", "Regulator penalty risk", ticker="XYZ")

    assert result.provider == "keyword"
    assert "lawsuit" in result.event_labels or "regulation" in result.event_labels


def test_build_event_classifier_default_provider(monkeypatch) -> None:
    monkeypatch.delenv("EVENT_CLASSIFIER_PROVIDER", raising=False)

    assert build_event_classifier_from_env().provider == "keyword"
