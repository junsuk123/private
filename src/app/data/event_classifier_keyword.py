from __future__ import annotations

import re

from app.data.event_classifier import EVENT_LABELS, EventClassifierResult, clamp


class KeywordEventClassifier:
    provider = "keyword"

    def __init__(self, provider: str = "keyword") -> None:
        self.provider = provider

    def classify(self, title: str, body: str, ticker: str | None = None) -> EventClassifierResult:
        text = f"{title} {body}".lower()
        labels = [label for label, patterns in _LABEL_PATTERNS.items() if any(pattern.search(text) for pattern in patterns)]
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


def _patterns(*values: str) -> tuple[re.Pattern[str], ...]:
    return tuple(re.compile(value, re.IGNORECASE) for value in values)


# English tokens use word boundaries and semantic phrases.  The old substring
# matcher treated ``contract`` anywhere in a regulator's boilerplate as a supply
# award and even matched ``rate`` inside unrelated words.  These labels feed the
# ontology, so false positives are more harmful than returning ``unknown``.
_LABEL_PATTERNS = {
    "earnings": _patterns(r"\bearnings\b", r"\bprofit\b", r"\brevenue\b", r"\bmargin\b", r"실적", r"매출", r"이익"),
    "guidance": _patterns(r"\bguidance\b", r"\boutlook\b", r"\bforecast\b", r"가이던스", r"전망"),
    "supply_contract": _patterns(
        r"\bsupply\s+(?:contract|agreement|deal)\b",
        r"\b(?:wins?|won|awarded|secures?|signed)\s+(?:a\s+)?(?:major\s+|material\s+)?(?:supply\s+)?contract\b",
        r"\bcontract\s+award\b",
        r"수주",
        r"공급\s*(?:계약|협약)",
    ),
    "lawsuit": _patterns(r"\blawsuit\b", r"\blitigation\b", r"\blegal\s+suit\b", r"소송"),
    "regulation": _patterns(r"\bregulation\b", r"\bregulator(?:y)?\b", r"\bpenalt(?:y|ies)\b", r"\bfine[ds]?\b", r"\benforcement\s+action", r"규제", r"과징금", r"제재"),
    "macro": _patterns(r"\binterest\s+rates?\b", r"\binflation\b", r"\bforeign\s+exchange\b", r"\bfomc\b", r"\bfederal\s+reserve\b", r"금리", r"환율", r"물가", r"통화정책"),
    "sector_momentum": _patterns(r"\bsector\b", r"\bmomentum\b", r"업종", r"섹터"),
    "management_change": _patterns(r"\bceo\b", r"\bcfo\b", r"\bmanagement\s+change\b", r"\bresign(?:s|ed|ation)?\b", r"\bappoint(?:s|ed|ment)?\b", r"대표이사\s*(?:변경|선임|사임)"),
    "dividend": _patterns(r"\bdividend\b", r"배당"),
    "capital_increase": _patterns(r"\bcapital\s+increase\b", r"\bsecurities\s+offering\b", r"유상증자"),
    "stock_split": _patterns(r"\bstock\s+split\b", r"액면분할"),
    "analyst_report": _patterns(r"\banalyst\b", r"\bupgrade[ds]?\b", r"\bdowngrade[ds]?\b", r"\btarget\s+price\b"),
}
_POSITIVE = ("beat", "growth", "upgrade", "positive", "strong", "surge", "증가", "호조", "상향")
_NEGATIVE = ("miss", "cut", "downgrade", "negative", "weak", "risk", "fall", "하락", "부진", "리스크")
