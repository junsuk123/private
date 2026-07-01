from __future__ import annotations

from dataclasses import dataclass
from math import sqrt
from typing import Mapping, Sequence

import numpy as np

from app.npu.runtime_manager import NpuModuleStatus, NpuRuntimeManager, get_npu_runtime_manager


DEFAULT_CLUSTER_NAMES = (
    "trend_cluster",
    "candle_cluster",
    "volume_cluster",
    "reversal_cluster",
    "momentum_cluster",
    "breakout_cluster",
    "microstructure_cluster",
    "risk_cluster",
)


@dataclass(frozen=True)
class EvidenceClusterCompressionResult:
    clusters: np.ndarray
    cluster_schema: tuple[str, ...]
    status: NpuModuleStatus


class NpuEvidenceClusterCompressor:
    def __init__(self, runtime: NpuRuntimeManager | None = None, *, enabled: bool = True) -> None:
        self.runtime = runtime or get_npu_runtime_manager()
        self.enabled = enabled

    def compress(
        self,
        feature_matrix: np.ndarray,
        *,
        feature_names: Sequence[str],
        cluster_map: Mapping[str, Sequence[str]] | None = None,
        mode: str = "sqrt_capped_sum",
    ) -> EvidenceClusterCompressionResult:
        matrix = np.nan_to_num(np.asarray(feature_matrix, dtype=np.float32), nan=0.0, posinf=0.0, neginf=0.0)
        if matrix.ndim != 2:
            raise ValueError("feature_matrix must be 2D")
        clusters = cluster_map or _default_cluster_map()
        weights = np.zeros((len(feature_names), len(clusters)), dtype=np.float32)
        counts = np.zeros((len(clusters),), dtype=np.float32)
        feature_index = {name: index for index, name in enumerate(feature_names)}
        for col, names in enumerate(clusters.values()):
            present = [feature_index[name] for name in names if name in feature_index]
            counts[col] = max(1, len(present))
            for row in present:
                if mode == "weighted_mean":
                    weights[row, col] = 1.0 / counts[col]
                elif mode == "capped_sum":
                    weights[row, col] = 1.0
                else:
                    weights[row, col] = 1.0 / sqrt(counts[col])
        output, status = self.runtime.run_linear(
            module_name="evidence_cluster_compressor",
            features=matrix,
            weights=weights,
            activation="relu",
            enabled=self.enabled,
        )
        if mode == "capped_sum":
            output = np.minimum(output, 1.0)
        return EvidenceClusterCompressionResult(output, tuple(clusters.keys()), status)


def _default_cluster_map() -> dict[str, tuple[str, ...]]:
    return {
        "trend_cluster": ("PriceAboveMA20", "MA20AboveMA60", "MACDBullishCross", "TrendContinuation"),
        "candle_cluster": ("BullishCandle", "BearishCandle", "CloseNearHigh", "CloseNearLow"),
        "volume_cluster": ("VolumeBackedRise", "VolumeBackedFall", "VolumeZScoreHigh", "MFIHigh"),
        "reversal_cluster": ("RecentNegativeReturnShock", "OversoldCondition", "ShortTermReversalCandidate"),
        "momentum_cluster": ("OpeningReturnStrength", "IntradayMomentumCandidate", "MarketDirectionAligned"),
        "breakout_cluster": ("BreakoutConfirmed", "RangeBreakout", "MovingAverageBreakout"),
        "microstructure_cluster": ("OrderFlowImbalance", "QueueImbalance", "MicropricePressure", "SpreadRegime", "DepthScore"),
        "risk_cluster": ("DrawdownRisk", "StopLossTriggered", "TakeProfitZone", "VolatilitySpike", "CostDominatesAlpha"),
    }
