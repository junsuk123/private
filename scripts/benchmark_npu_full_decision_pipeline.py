from __future__ import annotations

import argparse
import json
import os
import time

import numpy as np

from app.execution.npu_execution_edge_scorer import NpuExecutionEdgeScorer
from app.graph.action_aggregator import ActionAggregator
from app.graph.npu_theory_vote_scorer import NpuTheoryVoteScorer
from app.graph.theory_vote import TheoryVote
from app.models.npu_short_horizon_predictor import NpuShortHorizonPredictor
from app.npu.tensor_schemas import get_tensor_schema


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="CPU")
    parser.add_argument("--count", type=int, default=512)
    args = parser.parse_args()
    os.environ["NPU_DEVICE_PREFERENCE"] = args.device
    rng = np.random.default_rng(7)

    votes = tuple(
        TheoryVote(
            ticker=f"T{i:04d}",
            theory_id="gao_2018_intraday_momentum",
            theory_family="intraday_momentum",
            style="continuation",
            action="BUY",
            horizon_bucket="late_intraday",
            expected_holding_minutes=180,
            raw_signal=0.5,
            confidence=0.7,
            validation_weight=0.5,
            evidence_cluster_id="momentum_cluster",
        )
        for i in range(args.count)
    )
    started = time.perf_counter()
    theory = NpuTheoryVoteScorer().score(votes, top_k=50)
    short = NpuShortHorizonPredictor().predict_matrix(rng.normal(0.0, 0.1, size=(args.count, 8)).astype(np.float32))
    execution = NpuExecutionEdgeScorer().score_matrix(
        rng.random(size=(args.count, get_tensor_schema("execution_edge_features").feature_dim), dtype=np.float32)
    )
    decision = ActionAggregator().decide("T0000", votes[:1], npu_profile=theory.status.as_dict())
    total_ms = (time.perf_counter() - started) * 1000.0
    print(
        json.dumps(
            {
                "candidate_count": args.count,
                "device": args.device,
                "total_ms": round(total_ms, 3),
                "top_k": len(theory.top_indices),
                "theory_vote": theory.status.as_dict(),
                "short_horizon": short.status.as_dict(),
                "execution_edge": execution.status.as_dict(),
                "sample_decision": decision.as_dict(),
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
