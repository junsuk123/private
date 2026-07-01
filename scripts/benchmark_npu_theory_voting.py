from __future__ import annotations

import argparse
import json
import os
import time

import numpy as np

from app.graph.npu_conflict_scorer import NpuConflictScorer
from app.graph.npu_evidence_cluster_compressor import NpuEvidenceClusterCompressor
from app.graph.npu_theory_vote_scorer import NpuTheoryVoteScorer
from app.npu.tensor_schemas import get_tensor_schema


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="CPU")
    parser.add_argument("--count", type=int, default=1024)
    args = parser.parse_args()
    os.environ["NPU_DEVICE_PREFERENCE"] = args.device
    rng = np.random.default_rng(42)

    theory_features = rng.normal(0.2, 0.1, size=(args.count, get_tensor_schema("theory_vote_features").feature_dim)).astype(np.float32)
    conflict_features = rng.random(size=(args.count, get_tensor_schema("conflict_features").feature_dim), dtype=np.float32)
    semantic_features = rng.random(size=(args.count, 8), dtype=np.float32)
    feature_names = (
        "PriceAboveMA20",
        "MA20AboveMA60",
        "MACDBullishCross",
        "BullishCandle",
        "VolumeBackedRise",
        "OpeningReturnStrength",
        "BreakoutConfirmed",
        "DrawdownRisk",
    )

    started = time.perf_counter()
    theory_scores, theory_status = NpuTheoryVoteScorer().score_matrix(theory_features)
    clusters = NpuEvidenceClusterCompressor().compress(semantic_features, feature_names=feature_names)
    conflicts = NpuConflictScorer().score_matrix(conflict_features)
    total_ms = (time.perf_counter() - started) * 1000.0

    print(
        json.dumps(
            {
                "candidate_count": args.count,
                "device": args.device,
                "total_ms": round(total_ms, 3),
                "theory_vote": theory_status.as_dict(),
                "evidence_cluster": clusters.status.as_dict(),
                "conflict": conflicts.status.as_dict(),
                "theory_score_shape": list(theory_scores.shape),
                "cluster_shape": list(clusters.clusters.shape),
                "conflict_shape": list(conflicts.penalties.shape),
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
