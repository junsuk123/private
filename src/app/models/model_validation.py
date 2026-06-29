from __future__ import annotations


def validate_training_dataset(
    rows: list[dict],
    *,
    minimum_examples: int = 30,
    minimum_positive_labels: int = 5,
    minimum_negative_labels: int = 5,
) -> tuple[bool, tuple[str, ...]]:
    reasons: list[str] = []
    if len(rows) < minimum_examples:
        reasons.append("INSUFFICIENT_EXAMPLES")
    positives = sum(1 for row in rows if int(row["label"]) == 1)
    negatives = sum(1 for row in rows if int(row["label"]) == 0)
    if positives < minimum_positive_labels:
        reasons.append("INSUFFICIENT_POSITIVE_LABELS")
    if negatives < minimum_negative_labels:
        reasons.append("INSUFFICIENT_NEGATIVE_LABELS")
    return not reasons, tuple(reasons)


def auc_like_score(labels: list[int], scores: list[float]) -> float:
    positives = [(score, label) for score, label in zip(scores, labels, strict=True) if label == 1]
    negatives = [(score, label) for score, label in zip(scores, labels, strict=True) if label == 0]
    if not positives or not negatives:
        return 0.5
    wins = 0.0
    total = 0
    for pos_score, _ in positives:
        for neg_score, _ in negatives:
            total += 1
            if pos_score > neg_score:
                wins += 1
            elif pos_score == neg_score:
                wins += 0.5
    return wins / max(1, total)
