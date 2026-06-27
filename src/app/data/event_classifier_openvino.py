from __future__ import annotations

from pathlib import Path

from app.data.event_classifier import EventClassifierResult
from app.data.event_classifier_keyword import KeywordEventClassifier


class OpenVinoEventClassifier:
    provider = "openvino"

    def __init__(self, model_path: str, device: str = "AUTO") -> None:
        self.model_path = Path(model_path)
        self.device = device
        self._fallback = KeywordEventClassifier(provider="keyword")

    def classify(self, title: str, body: str, ticker: str | None = None) -> EventClassifierResult:
        if not self.model_path.exists():
            return self._fallback.classify(title, body, ticker)
        try:
            import openvino as ov  # noqa: F401
        except Exception:
            return self._fallback.classify(title, body, ticker)
        result = self._fallback.classify(title, body, ticker)
        return EventClassifierResult(
            ticker_relevance=result.ticker_relevance,
            sentiment_score=result.sentiment_score,
            event_labels=result.event_labels,
            risk_label=result.risk_label,
            confidence=result.confidence,
            provider="openvino",
        )
