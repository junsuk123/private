"""Causal, bounded time-aware inputs shared by R-GCN training and serving.

Time is elapsed wall time, never the row index. We pool three most recent
minute observations, decay irregular gaps, and never interpolate across missing
observations. This is a small time-aware R-GCN, not a reproduction of TGAT.
"""
from __future__ import annotations

from collections import OrderedDict
from dataclasses import dataclass
from datetime import datetime
import math

import numpy as np

from app.models.strategy_utility.strategy_graph import (
    STRATEGY_NODE_COUNT, RELATION_NAMES, strategy_node_features,
    strategy_relation_adjacency, strategy_market_mask,
)

ARCHITECTURE = "time_aware_rgcn_v1"
TIME_STEPS = 3
HALF_LIFE_SECONDS = 120.0
MAX_AGE_SECONDS = 600.0


@dataclass(frozen=True)
class GraphObservation:
    symbol: str
    as_of: datetime
    features: tuple[float, ...]


def causal_graph_inputs(
    current: GraphObservation,
    history: tuple[GraphObservation, ...] = (),
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return [T,N,F], [T,R,N,N], [T,N] with age-weighted masks.

    The current observation is authoritative for its minute. A different symbol,
    future timestamp, nonfinite vector or an observation older than ten minutes
    cannot contribute. Weights are normalised only over available observations.
    """
    if current.as_of.tzinfo is None:
        raise ValueError("graph timestamps must be timezone-aware")
    values = np.asarray(current.features, dtype=np.float32)
    if values.ndim != 1 or not np.isfinite(values).all():
        raise ValueError("graph features must be a finite vector")
    by_minute: dict[int, GraphObservation] = {}
    for item in (*history, current):
        if item.symbol != current.symbol or item.as_of.tzinfo is None:
            continue
        age = (current.as_of - item.as_of).total_seconds()
        vector = np.asarray(item.features, dtype=np.float32)
        if not (0.0 <= age <= MAX_AGE_SECONDS) or vector.shape != values.shape or not np.isfinite(vector).all():
            continue
        bucket = int(item.as_of.timestamp()) // 60
        previous = by_minute.get(bucket)
        if previous is None or item.as_of >= previous.as_of:
            by_minute[bucket] = item
    observations = sorted(by_minute.values(), key=lambda item: item.as_of)[-TIME_STEPS:]
    weights = np.asarray([
        math.exp2(-(current.as_of - item.as_of).total_seconds() / HALF_LIFE_SECONDS)
        for item in observations
    ], dtype=np.float32)
    weights /= weights.sum()
    width = len(current.features) + STRATEGY_NODE_COUNT
    x = np.zeros((TIME_STEPS, STRATEGY_NODE_COUNT, width), dtype=np.float32)
    adjacency = np.zeros((TIME_STEPS, len(RELATION_NAMES), STRATEGY_NODE_COUNT, STRATEGY_NODE_COUNT), dtype=np.float32)
    masks = np.zeros((TIME_STEPS, STRATEGY_NODE_COUNT), dtype=np.float32)
    market_mask = strategy_market_mask(current.symbol)
    topology = strategy_relation_adjacency(market=current.symbol)
    start = TIME_STEPS - len(observations)
    for index, (item, weight) in enumerate(zip(observations, weights), start):
        x[index] = strategy_node_features(item.features) * market_mask[:, None]
        adjacency[index] = topology
        masks[index] = market_mask * weight
    return x, adjacency, masks


class CausalGraphHistory:
    """Bounded per-symbol history; an out-of-order read never corrupts the clock."""
    def __init__(self, max_symbols: int = 256) -> None:
        self.max_symbols = max(1, int(max_symbols))
        self._history: OrderedDict[str, tuple[GraphObservation, ...]] = OrderedDict()

    def inputs(self, current: GraphObservation) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        history = self._history.get(current.symbol, ())
        result = causal_graph_inputs(current, history)
        if not history or current.as_of >= history[-1].as_of:
            minute = int(current.as_of.timestamp()) // 60
            retained = tuple(item for item in history if int(item.as_of.timestamp()) // 60 != minute
                             and 0 <= (current.as_of - item.as_of).total_seconds() <= MAX_AGE_SECONDS)
            self._history[current.symbol] = (*retained, current)[-TIME_STEPS:]
            self._history.move_to_end(current.symbol)
            while len(self._history) > self.max_symbols:
                self._history.popitem(last=False)
        return result
