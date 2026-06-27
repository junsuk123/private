from __future__ import annotations

from unittest.mock import patch

import numpy as np

from app.graph.npu_classifier import OntologyNpuLinearScorer


def test_score_candidates_returns_top_k_without_full_dict() -> None:
    scorer = OntologyNpuLinearScorer(batch_size=512)
    tickers = tuple(f"T{i}" for i in range(17))
    features = np.zeros((17, 8), dtype=np.float32)
    features[:, 1] = np.arange(17, dtype=np.float32) / 10.0

    result = scorer.score_candidates(tickers, features, top_k=5)

    assert len(result.tickers) == 5
    assert result.tickers[0] == "T16"
    assert result.scores.shape == (5, 6)
    assert result.profile["input_count"] == 17
    assert result.profile["batch_bucket"] == 512


def test_cpu_fallback_normalizes_nan_and_inf() -> None:
    scorer = OntologyNpuLinearScorer(batch_size=512)
    features = np.array([[np.nan, np.inf, -np.inf, 0, 0, 0, 0, 1]], dtype=np.float32)
    with patch.dict("sys.modules", {"openvino": None}):
        result = scorer.score_candidates(("BAD",), features, top_k=1)

    assert np.isfinite(result.scores).all()
    assert result.profile["device"] == "CPU_NUMPY"


def test_batch_buckets_cover_requested_sizes() -> None:
    scorer = OntologyNpuLinearScorer(batch_size="auto")
    for count, expected in ((1, 512), (17, 512), (512, 512), (1024, 1024), (2049, 4096)):
        features = np.ones((count, 8), dtype=np.float32)
        result = scorer.score_candidates(tuple(str(i) for i in range(count)), features, top_k=3)
        assert result.profile["batch_bucket"] == expected
