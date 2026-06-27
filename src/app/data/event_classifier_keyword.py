from __future__ import annotations

from app.data.event_classifier import EVENT_LABELS, EventClassifierResult, clamp


class KeywordEventClassifier:
    provider = "keyword"

    def __init__(self, provider: str = "keyword") -> None:
        self.provider = provider

    def classify(self, title: str, body: str, ticker: str | None = None) -> EventClassifierResult:
        text = f"{title} {body}".lower()
        labels = [label for label, keywords in _LABEL_KEYWORDS.items() if any(item in text for item in keywords)]
        if not labels:
            labels = ["unknown"]
        positive_hits = sum(1 for word in _POSITIVE if word in text)
        negative_hits = sum(1 for word in _NEGATIVE if word in text)
        sentiment = clamp((positive_hits - negative_hits) / max(1, positive_hits + negative_hits), -1.0, 1.0)
        risk_label = "high" if negative_hits >= 2 or any(label in labels for label in ("lawsuit", "regulation")) else "normal"
        relevance = 0.35
        if ticker and ticker.lower() in text:
            relevance = 1.0
        elif labels != ["unknown"]:
            relevance = 0.65
        confidence = clamp(0.35 + 0.12 * len([label for label in labels if label in EVENT_LABELS]), 0.0, 0.92)
        return EventClassifierResult(
            ticker_relevance=relevance,
            sentiment_score=sentiment,
            event_labels=tuple(labels[:5]),
            risk_label=risk_label,
            confidence=confidence,
            provider=self.provider,
        )


_LABEL_KEYWORDS = {
    "earnings": ("earnings", "profit", "revenue", "margin", "실적", "매출", "이익"),
    "guidance": ("guidance", "outlook", "forecast", "가이던스", "전망"),
    "supply_contract": ("contract", "supply", "deal", "수주", "공급"),
    "lawsuit": ("lawsuit", "litigation", "suit", "소송"),
    "regulation": ("regulation", "regulator", "penalty", "fine", "규제", "과징금"),
    "macro": ("rate", "inflation", "fx", "fed", "금리", "환율", "물가"),
    "sector_momentum": ("sector", "momentum", "업종", "섹터"),
    "management_change": ("ceo", "cfo", "management", "resign", "appoint"),
    "dividend": ("dividend", "배당"),
    "capital_increase": ("capital increase", "offering", "유상증자"),
    "stock_split": ("stock split", "액면분할"),
    "analyst_report": ("analyst", "upgrade", "downgrade", "target price"),
}
_POSITIVE = ("beat", "growth", "upgrade", "positive", "strong", "surge", "증가", "호조", "상향")
_NEGATIVE = ("miss", "cut", "downgrade", "negative", "weak", "risk", "fall", "하락", "부진", "리스크")
