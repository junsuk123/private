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
    """Exact tie-aware rank AUC in O(n log n), rather than all positive/negative pairs."""
    import math

    pairs = sorted(zip(scores, labels, strict=True))
    if any(not math.isfinite(score) for score, _ in pairs):
        raise ValueError("AUC scores must be finite")
    positives = sum(label == 1 for _, label in pairs)
    negatives = sum(label == 0 for _, label in pairs)
    if not positives or not negatives:
        return 0.5
    wins = 0.0
    lower_negatives = 0
    start = 0
    while start < len(pairs):
        end = start + 1
        while end < len(pairs) and pairs[end][0] == pairs[start][0]:
            end += 1
        tied_positive = sum(label == 1 for _, label in pairs[start:end])
        tied_negative = sum(label == 0 for _, label in pairs[start:end])
        wins += tied_positive * (lower_negatives + 0.5 * tied_negative)
        lower_negatives += tied_negative
        start = end
    return wins / (positives * negatives)
