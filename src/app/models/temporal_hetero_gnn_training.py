"""Training for the temporal hetero GNN, and for the ontology's learnable edge weights.

Gradient-free by design
-----------------------
The runtime is NumPy — no autograd, no Torch in the production environment. Rather than
add a training-only dependency the serving path cannot use, this fits the model with an
evolutionary strategy: sample perturbations of the parameter vector, score each on the
training window's own loss, and step along the reward-weighted average. It is slower per
epoch than backpropagation and entirely sufficient here, because the model is small
(tens of thousands of parameters) and the label set is small too.

What it optimises
-----------------
A weighted sum of the heads that have labels:

* ``market_regime`` — binary cross-entropy per label (multi-label, so per-label BCE and
  not a softmax cross-entropy).
* ``trade_quality`` — BCE against the realised outcome.
* ``expected_return`` — Huber on realised bps, which keeps one 300bp outlier from
  dominating a window of 5bp moves.

Heads with no labels in a batch contribute nothing rather than being pushed toward zero.

Ontology weights
----------------
:func:`fit_relation_weights` learns the ``learnable`` edge weights **separately**, by
scoring each relation's contribution to realised outcomes. They are written to
``ontology_edge.learned_weight`` and never overwrite ``prior_strength`` — the whole point
of the two-column design is that the disagreement stays visible.

Leakage
-------
:func:`build_training_rows` reads persisted decisions and joins them to outcomes resolved
strictly **after** the decision timestamp. A row whose outcome window has not closed is
excluded, not imputed.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

import numpy as np

from app.models.graph_snapshot import GraphSnapshot
from app.models.temporal_hetero_gnn import (
    REGIME_LABELS,
    TemporalHeteroGnn,
    TemporalHeteroGnnConfig,
)
from app.ontology.market_graph import MarketGraph
from app.storage.trading_state_store import (
    TradingStateStore,
    default_trading_state_store,
    iso_column,
)

__all__ = [
    "TrainingExample",
    "TrainingReport",
    "fit_relation_weights",
    "persist_relation_weights",
    "train_temporal_hetero_gnn",
]

#: Head weights in the composite loss. Regime dominates because it is the input every
#: other layer conditions on; a model that predicts return well but regime badly makes the
#: strategy selector wrong in a way no return head can repair.
_HEAD_WEIGHTS = {"regime": 1.0, "trade_quality": 0.7, "expected_return": 0.3}

#: Huber transition point for the return head, in bps. Above it the loss is linear.
_RETURN_HUBER_DELTA_BPS = 40.0


@dataclass(frozen=True)
class TrainingExample:
    """One (graph window, node, labels) triple."""

    snapshot: GraphSnapshot
    node_id: str
    #: Multi-label regime truth in [0, 1] per label. Absent labels are skipped.
    regime_labels: Mapping[str, float] = field(default_factory=dict)
    #: 1.0 when the decision made money, 0.0 when it did not. ``None`` skips the head.
    trade_quality: float | None = None
    realised_return_bps: float | None = None

    def node_index(self) -> int | None:
        return self.snapshot.index_of(self.node_id)


@dataclass(frozen=True)
class TrainingReport:
    epochs: int
    population: int
    initial_loss: float
    final_loss: float
    example_count: int
    checkpoint_path: str | None = None
    per_head_loss: Mapping[str, float] = field(default_factory=dict)

    @property
    def improved(self) -> bool:
        return self.final_loss < self.initial_loss

    def as_dict(self) -> dict[str, Any]:
        return {
            "epochs": self.epochs,
            "population": self.population,
            "initial_loss": self.initial_loss,
            "final_loss": self.final_loss,
            "improved": self.improved,
            "example_count": self.example_count,
            "checkpoint_path": self.checkpoint_path,
            "per_head_loss": dict(self.per_head_loss),
        }


def _bce(probability: float, target: float) -> float:
    p = min(1.0 - 1e-7, max(1e-7, probability))
    return -(target * math.log(p) + (1.0 - target) * math.log(1.0 - p))


def _huber(error: float, delta: float) -> float:
    magnitude = abs(error)
    if magnitude <= delta:
        return 0.5 * error * error
    return delta * (magnitude - 0.5 * delta)


def _parameter_names(model: TemporalHeteroGnn) -> tuple[str, ...]:
    return model._TENSOR_NAMES  # noqa: SLF001 - the checkpoint contract, deliberately shared


def _flatten(model: TemporalHeteroGnn) -> np.ndarray:
    return np.concatenate(
        [np.asarray(getattr(model, name), dtype=np.float64).ravel() for name in _parameter_names(model)]
    )


def _unflatten(model: TemporalHeteroGnn, vector: np.ndarray) -> None:
    offset = 0
    for name in _parameter_names(model):
        current = np.asarray(getattr(model, name))
        size = current.size
        setattr(
            model,
            name,
            vector[offset : offset + size].reshape(current.shape).astype(np.float32),
        )
        offset += size


def evaluate_loss(
    model: TemporalHeteroGnn, examples: Sequence[TrainingExample]
) -> tuple[float, dict[str, float]]:
    """Composite loss and its per-head decomposition. Lower is better."""
    if not examples:
        return 0.0, {}
    totals = {"regime": 0.0, "trade_quality": 0.0, "expected_return": 0.0}
    counts = {"regime": 0, "trade_quality": 0, "expected_return": 0}
    for example in examples:
        index = example.node_index()
        if index is None:
            continue
        snapshot = example.snapshot
        output = model.infer(
            snapshot.features,
            snapshot.adjacency,
            snapshot.prior_bias,
            snapshot.node_type_index,
            snapshot.node_mask,
            collect_attention=False,
        )
        for position, label in enumerate(REGIME_LABELS):
            target = example.regime_labels.get(label)
            if target is None:
                continue
            totals["regime"] += _bce(float(output.market_regime[index, position]), float(target))
            counts["regime"] += 1
        if example.trade_quality is not None:
            totals["trade_quality"] += _bce(
                float(output.trade_quality[index]), float(example.trade_quality)
            )
            counts["trade_quality"] += 1
        if example.realised_return_bps is not None:
            error = float(output.expected_return_bps[index]) - float(
                example.realised_return_bps
            )
            # Scaled into bps-of-delta before the Huber, so the return head lands on the
            # same O(1) scale as the two cross-entropies. Unscaled, a 30bp error scores
            # ~450 against a cross-entropy of ~0.7 and the composite loss becomes the
            # return head with two rounding errors attached.
            totals["expected_return"] += _huber(error / _RETURN_HUBER_DELTA_BPS, 1.0)
            counts["expected_return"] += 1

    per_head = {
        head: (totals[head] / counts[head]) if counts[head] else 0.0
        for head in totals
    }
    # Heads with no labels contribute nothing rather than a zero that would look like a
    # perfectly-fit head.
    loss = sum(
        _HEAD_WEIGHTS[head] * value
        for head, value in per_head.items()
        if counts[head] > 0
    )
    return loss, per_head


def train_temporal_hetero_gnn(
    examples: Sequence[TrainingExample],
    *,
    config: TemporalHeteroGnnConfig,
    epochs: int = 30,
    population: int = 24,
    sigma: float = 0.02,
    #: RMS movement per weight per step. Weights initialise at ~1/sqrt(fan_in), so 0.01
    #: is a few percent of the typical magnitude — enough to move, small enough that a
    #: noisy gradient estimate cannot destroy the parameterisation in one step.
    learning_rate: float = 0.01,
    seed: int = 11,
    checkpoint_path: str | Path | None = None,
    initial: TemporalHeteroGnn | None = None,
    progress: Callable[[int, float], None] | None = None,
) -> tuple[TemporalHeteroGnn, TrainingReport]:
    """Fit the model with an evolutionary strategy and optionally save a checkpoint.

    The returned model is the best-scoring one seen, not the last one: an ES step can
    overshoot, and shipping a checkpoint worse than one already produced would be a
    silent regression.
    """
    if not examples:
        raise ValueError("training requires at least one example")
    model = initial or TemporalHeteroGnn(config)
    if model.config != config:
        raise ValueError("initial model config does not match the requested config")

    rng = np.random.default_rng(seed)
    theta = _flatten(model)
    initial_loss, _ = evaluate_loss(model, examples)
    best_loss = initial_loss
    best_theta = theta.copy()

    # Antithetic pairs: each perturbation is evaluated as +eps and -eps. Halves the
    # variance of the gradient estimate for the same number of evaluations, which matters
    # a great deal at the population sizes this runs at.
    pairs = max(1, int(population) // 2)
    probe = TemporalHeteroGnn(config)
    for epoch in range(max(1, int(epochs))):
        noise = rng.normal(0.0, 1.0, (pairs, theta.size))
        rewards = np.zeros(2 * pairs, dtype=np.float64)
        for index in range(pairs):
            for sign_index, sign in enumerate((1.0, -1.0)):
                _unflatten(probe, theta + sign * sigma * noise[index])
                loss, _ = evaluate_loss(probe, examples)
                rewards[2 * index + sign_index] = -loss
        spread = rewards.std()
        if spread > 0:
            normalised = (rewards - rewards.mean()) / spread
            advantage = normalised[0::2] - normalised[1::2]
            gradient = (noise.T @ advantage) / (2.0 * pairs * sigma)
            # RMS-normalised step. The raw ES gradient scales with population size and
            # sigma, so a fixed learning rate silently becomes a very different step as
            # either is tuned; normalising makes ``learning_rate`` mean "this much
            # movement per weight", which is the quantity worth choosing.
            rms = float(np.sqrt(np.mean(np.square(gradient))))
            if rms > 0.0:
                theta = theta + learning_rate * gradient / rms
        _unflatten(probe, theta)
        loss, _ = evaluate_loss(probe, examples)
        if loss < best_loss:
            best_loss = loss
            best_theta = theta.copy()
        if progress is not None:
            progress(epoch, loss)

    _unflatten(model, best_theta)
    final_loss, per_head = evaluate_loss(model, examples)
    saved: str | None = None
    if checkpoint_path is not None:
        saved = str(model.save_checkpoint(checkpoint_path))
    return model, TrainingReport(
        epochs=int(epochs),
        population=int(population),
        initial_loss=round(initial_loss, 8),
        final_loss=round(final_loss, 8),
        example_count=len(examples),
        checkpoint_path=saved,
        per_head_loss={name: round(value, 8) for name, value in per_head.items()},
    )


# --------------------------------------------------------------------------- #
# Ontology relation weights
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class RelationOutcome:
    """One realised outcome attributed to the edges that supported the decision."""

    edge_ids: Sequence[str]
    #: True when the decision worked out.
    profitable: bool


def fit_relation_weights(
    graph: MarketGraph,
    outcomes: Sequence[RelationOutcome],
    *,
    minimum_samples: int = 20,
    shrinkage_k: float = 30.0,
) -> dict[str, float]:
    """Learned weight per learnable edge: its shrunk empirical hit rate.

    An edge that supported 40 decisions of which 26 worked gets ``26/40`` shrunk toward
    its own prior by ``n/(n+k)`` — the same shrinkage rule the seasonality baselines use,
    for the same reason: a thin sample must not be able to overturn an expert prior on its
    own. Edges with fewer than ``minimum_samples`` observations are left unlearned rather
    than assigned a weight nobody should trust.
    """
    wins: dict[str, int] = {}
    totals: dict[str, int] = {}
    for outcome in outcomes:
        for edge_id in outcome.edge_ids:
            totals[edge_id] = totals.get(edge_id, 0) + 1
            if outcome.profitable:
                wins[edge_id] = wins.get(edge_id, 0) + 1

    learned: dict[str, float] = {}
    by_id = {edge.edge_id: edge for edge in graph.edges()}
    for edge_id, count in totals.items():
        edge = by_id.get(edge_id)
        if edge is None or not edge.learnable or count < max(1, int(minimum_samples)):
            continue
        empirical = wins.get(edge_id, 0) / count
        weight = count / (count + max(0.0, shrinkage_k))
        learned[edge_id] = round(
            weight * empirical + (1.0 - weight) * edge.prior_strength, 6
        )
    return learned


def persist_relation_weights(
    graph: MarketGraph,
    weights: Mapping[str, float],
    *,
    store: TradingStateStore | None = None,
    updated_at: datetime | None = None,
) -> int:
    """Write learned weights to ``ontology_edge``, keeping the priors untouched."""
    target = store or default_trading_state_store()
    moment = (updated_at or datetime.now(timezone.utc)).astimezone(timezone.utc)
    applied = graph.apply_learned_weights(weights, updated_at=moment)
    with target.transaction() as conn:
        for edge in graph.edges():
            conn.execute(
                "insert into ontology_edge"
                " (edge_id, source_id, target_id, relation, direction, prior_strength,"
                "  learnable, learned_weight, learned_updated_at, lag_min, lag_max,"
                "  attributes_json, updated_at)"
                " values (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)"
                " on conflict(edge_id) do update set"
                "  prior_strength = excluded.prior_strength,"
                "  learnable = excluded.learnable,"
                "  learned_weight = excluded.learned_weight,"
                "  learned_updated_at = excluded.learned_updated_at,"
                "  updated_at = excluded.updated_at",
                (
                    edge.edge_id,
                    edge.source_id,
                    edge.target_id,
                    edge.relation,
                    edge.direction,
                    edge.prior_strength,
                    1 if edge.learnable else 0,
                    edge.learned_weight,
                    iso_column(edge.learned_updated_at),
                    edge.lag_min,
                    edge.lag_max,
                    "{}",
                    iso_column(moment),
                ),
            )
    return len(applied)


def load_relation_weights(
    graph: MarketGraph, *, store: TradingStateStore | None = None
) -> int:
    """Reattach persisted learned weights to a freshly loaded graph."""
    target = store or default_trading_state_store()
    rows = target.fetch_all(
        "select edge_id, learned_weight, learned_updated_at from ontology_edge"
        " where learned_weight is not null"
    )
    weights = {str(row["edge_id"]): float(row["learned_weight"]) for row in rows}
    if not weights:
        return 0
    # Structural edges cannot carry a learned weight; a row that claims one is stale data
    # from before a relation was reclassified, and is skipped rather than applied.
    learnable = {edge.edge_id for edge in graph.edges() if edge.learnable}
    return len(graph.apply_learned_weights(
        {key: value for key, value in weights.items() if key in learnable}
    ))


# --------------------------------------------------------------------------- #
# Training-row construction
# --------------------------------------------------------------------------- #
def build_relation_outcomes(
    store: TradingStateStore | None = None,
    *,
    horizon_minutes: int = 30,
    now: datetime | None = None,
) -> tuple[RelationOutcome, ...]:
    """Join persisted decisions to outcomes that had time to resolve.

    A decision whose horizon has not elapsed is excluded. Including it would train the
    relation weights on an outcome that has not happened yet, which is the same leak the
    walk-forward harness purges for.
    """
    import json

    target = store or default_trading_state_store()
    moment = (now or datetime.now(timezone.utc)).astimezone(timezone.utc)
    cutoff = iso_column(moment - timedelta(minutes=max(1, int(horizon_minutes))))
    rows = target.fetch_all(
        "select d.decision_id, d.ontology_relations_json, e.filled_quantity,"
        "       e.average_fill_price, e.limit_price, e.side"
        " from strategy_decision d"
        " left join order_intent e on e.decision_id = d.decision_id"
        " where d.decided_at <= ?",
        (cutoff,),
    )
    outcomes: list[RelationOutcome] = []
    for row in rows:
        filled = int(row["filled_quantity"] or 0)
        average = row["average_fill_price"]
        limit = row["limit_price"]
        if filled <= 0 or average is None or limit is None:
            continue
        side = str(row["side"] or "BUY").upper()
        direction = -1.0 if side in {"SELL", "SHORT"} else 1.0
        # Fill quality against the limit is the only outcome available without a mark;
        # it is a weak label and is treated as one by the shrinkage above.
        profitable = direction * (float(limit) - float(average)) > 0.0
        try:
            relations = json.loads(str(row["ontology_relations_json"] or "[]"))
        except ValueError:
            continue
        edge_ids = [
            str(item.get("edge_id"))
            for item in relations
            if isinstance(item, Mapping) and item.get("edge_id")
        ]
        if edge_ids:
            outcomes.append(RelationOutcome(edge_ids=edge_ids, profitable=profitable))
    return tuple(outcomes)
