from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Protocol

EVENT_LABELS = (
    "earnings",
    "guidance",
    "supply_contract",
    "lawsuit",
    "regulation",
    "macro",
    "sector_momentum",
    "management_change",
    "dividend",
    "capital_increase",
    "stock_split",
    "analyst_report",
    "unknown",
)


@dataclass(frozen=True)
class EventClassifierResult:
    ticker_relevance: float
    sentiment_score: float
    event_labels: tuple[str, ...]
    risk_label: str
    confidence: float
    provider: str


class EventClassifier(Protocol):
    provider: str

    def classify(self, title: str, body: str, ticker: str | None = None) -> EventClassifierResult:
        ...


def build_event_classifier_from_env() -> EventClassifier:
    provider = os.getenv("EVENT_CLASSIFIER_PROVIDER", "keyword").strip().lower()
    if provider == "openvino":
        from app.data.event_classifier_openvino import OpenVinoEventClassifier

        return OpenVinoEventClassifier(
            model_path=os.getenv("EVENT_CLASSIFIER_MODEL_PATH", "models/event_classifier/openvino_model.xml"),
            device=os.getenv("EVENT_CLASSIFIER_DEVICE", "AUTO"),
        )
    if provider == "llm":
        from app.data.event_classifier_keyword import KeywordEventClassifier

        return KeywordEventClassifier(provider="fallback")
    from app.data.event_classifier_keyword import KeywordEventClassifier

    return KeywordEventClassifier()


def clamp(value: float, lower: float, upper: float) -> float:
    return max(lower, min(upper, float(value)))
