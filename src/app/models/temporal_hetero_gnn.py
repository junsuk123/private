"""Temporal heterogeneous GNN: relation attention with an ontology prior, then a TCN.

Shape of the model
------------------
For a window of ``T`` graph snapshots over ``N`` nodes:

1. **Node-type encoders.** One projection per node type. A ``MacroFactor``'s feature
   vector and a ``Stock``'s are not the same kind of object, and a shared encoder forces
   them into one basis — which is how a graph model ends up learning that "feature 3"
   means volatility for some nodes and foreign flow for others.
2. **Relation-specific attention with a prior bias.** Per relation ``r``::

       e[r,i,j] = LeakyReLU(a_r . [W_r h_i || W_r h_j]) + scale * prior_strength[r,i,j]
       alpha[r,i,:] = softmax_j(e[r,i,:])            over declared edges only

   The ontology prior is a **soft bias on the logit**, not a mask and not a multiplier.
   Where the ontology declared no edge the bias is ``-inf``, so attention cannot invent a
   relation; among declared edges the prior only *ranks*, and enough evidence overturns
   it. Both the prior and the realised attention are returned in
   :attr:`InferenceTrace.relation_attention`.
3. **Causal TCN.** Dilated 1-D convolutions over time with kernel 2 and dilations
   ``(1, 2, 4, ...)``. Causal by construction: ``y[t]`` reads ``x[t]`` and ``x[t-d]`` and
   nothing later, so no future bar can reach a present prediction. This is the property
   the walk-forward evaluation depends on, and it is a structural fact about the arithmetic
   rather than a discipline the caller has to maintain.
4. **Heads.** ``market_regime`` (multi-label sigmoid over the regime catalogue),
   ``sector_strength``, ``expected_return`` (bps), ``volatility``,
   ``strategy_suitability`` (per strategy family), ``trade_quality``, ``uncertainty``.

What this model may not do
--------------------------
**It cannot place an order.** It has no import path to ``app.execution``,
``app.risk``, any broker client or any order type, and a test asserts that. Its output is
evidence handed to the strategy selector and the gate; those decide, and the gate's limits
are not something a model output can move.

Failure is fail-closed
----------------------
A checkpoint whose tensors do not match the configured shapes raises rather than loading
partially, and :class:`GnnRuntime` turns that into ``OFFLINE`` — new entries blocked,
existing positions still managed. A model that silently loads half its weights is worse
than one that is admittedly absent.

Implementation note
-------------------
NumPy, not Torch. Torch is an optional extra in this project and absent from the
production runtime; the existing ``FixedShapeStrategyUtilityModel`` is numpy for the same
reason. Fixed shapes with masking keep inference allocation-free enough for the realtime
loop and make the checkpoint contract checkable.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from app.ontology.market_graph import NODE_TYPES, RELATION_TYPES

__all__ = [
    "GNN_HEADS",
    "InferenceTrace",
    "REGIME_LABELS",
    "STRATEGY_FAMILIES",
    "TemporalHeteroGnn",
    "TemporalHeteroGnnConfig",
    "TemporalHeteroGnnOutput",
]

#: Multi-label regime catalogue. Order is part of the checkpoint contract: appending is
#: safe, reordering invalidates every stored model.
REGIME_LABELS: tuple[str, ...] = (
    "TREND_UP",
    "TREND_DOWN",
    "RANGE_LOW_VOL",
    "RANGE_HIGH_VOL",
    "BREAKOUT_UP",
    "BREAKDOWN",
    "RISK_ON",
    "RISK_OFF",
    "LIQUIDITY_STRESS",
    "INDEX_UP_BREADTH_DOWN",
    "INDEX_DOWN_BREADTH_UP",
    "EVENT_SHOCK",
    "TRANSITION",
)

#: Strategy families the suitability head scores. Same append-only rule.
STRATEGY_FAMILIES: tuple[str, ...] = (
    "TREND",
    "BREAKOUT",
    "MEAN_REVERSION",
    "RELATIVE_STRENGTH",
    "ORDER_FLOW",
    "GAP",
    "EVENT",
    "DEFENSIVE",
)

GNN_HEADS: tuple[str, ...] = (
    "market_regime",
    "sector_strength",
    "expected_return",
    "volatility",
    "strategy_suitability",
    "trade_quality",
    "uncertainty",
)

#: Slope of the negative branch of the attention LeakyReLU.
_LEAKY_SLOPE = 0.2

#: Expected-return head scale, in bps. The raw head output is a tanh, so predictions are
#: bounded to +-this. An unbounded return head can emit a number no cost model can
#: contradict, which is how a model output becomes a de-facto order.
_EXPECTED_RETURN_SCALE_BPS = 300.0


@dataclass(frozen=True)
class TemporalHeteroGnnConfig:
    max_nodes: int
    feature_dim: int
    time_steps: int
    hidden_dim: int = 24
    layer_count: int = 2
    relation_count: int = len(RELATION_TYPES)
    node_type_count: int = len(NODE_TYPES)
    regime_count: int = len(REGIME_LABELS)
    strategy_family_count: int = len(STRATEGY_FAMILIES)
    tcn_dilations: tuple[int, ...] = (1, 2, 4)
    seed: int = 17

    def __post_init__(self) -> None:
        for name in ("max_nodes", "feature_dim", "time_steps", "hidden_dim", "layer_count"):
            if getattr(self, name) < 1:
                raise ValueError(f"{name} must be >= 1")
        if not self.tcn_dilations:
            raise ValueError("tcn_dilations must not be empty")

    def as_array(self) -> np.ndarray:
        return np.asarray(
            [
                self.max_nodes,
                self.feature_dim,
                self.time_steps,
                self.hidden_dim,
                self.layer_count,
                self.relation_count,
                self.node_type_count,
                self.regime_count,
                self.strategy_family_count,
                self.seed,
                len(self.tcn_dilations),
                *self.tcn_dilations,
            ],
            dtype=np.int64,
        )

    @classmethod
    def from_array(cls, values: Sequence[int]) -> "TemporalHeteroGnnConfig":
        items = [int(value) for value in values]
        if len(items) < 11:
            raise ValueError("invalid temporal hetero GNN checkpoint config")
        dilation_count = items[10]
        dilations = tuple(items[11 : 11 + dilation_count])
        if len(dilations) != dilation_count:
            raise ValueError("invalid temporal hetero GNN checkpoint dilations")
        return cls(
            max_nodes=items[0],
            feature_dim=items[1],
            time_steps=items[2],
            hidden_dim=items[3],
            layer_count=items[4],
            relation_count=items[5],
            node_type_count=items[6],
            regime_count=items[7],
            strategy_family_count=items[8],
            seed=items[9],
            tcn_dilations=dilations,
        )


@dataclass(frozen=True)
class InferenceTrace:
    """Everything needed to explain one forward pass.

    ``relation_attention`` holds, per relation, the realised attention weights alongside
    the ontology prior that biased them, so a reviewer can see where the model followed
    the expert and where it did not.
    """

    node_ids: tuple[str, ...]
    relation_attention: Mapping[str, dict[str, Any]] = field(default_factory=dict)
    #: Per-relation total attention mass, a cheap "which relations mattered" summary.
    relation_mass: Mapping[str, float] = field(default_factory=dict)
    active_node_count: int = 0

    def top_relations(self, limit: int = 5) -> tuple[tuple[str, float], ...]:
        ordered = sorted(self.relation_mass.items(), key=lambda item: -item[1])
        return tuple(ordered[: max(0, int(limit))])

    def as_dict(self) -> dict[str, Any]:
        return {
            "node_ids": list(self.node_ids),
            "active_node_count": self.active_node_count,
            "relation_mass": dict(self.relation_mass),
            "top_relations": [
                {"relation": name, "mass": mass} for name, mass in self.top_relations()
            ],
            "relation_attention": {
                name: payload for name, payload in self.relation_attention.items()
            },
        }


@dataclass(frozen=True)
class TemporalHeteroGnnOutput:
    """Head outputs for every node in the graph, plus the trace that produced them."""

    #: ``[N, regime_count]`` independent probabilities — multi-label, NOT a softmax. A
    #: market can be RISK_OFF and RANGE_HIGH_VOL at once, and forcing them to sum to one
    #: would make the second unrepresentable.
    market_regime: np.ndarray
    sector_strength: np.ndarray
    expected_return_bps: np.ndarray
    volatility: np.ndarray
    #: ``[N, strategy_family_count]`` in [0, 1].
    strategy_suitability: np.ndarray
    trade_quality: np.ndarray
    uncertainty: np.ndarray
    node_mask: np.ndarray
    trace: InferenceTrace

    def for_node(self, index: int) -> dict[str, Any]:
        return {
            "market_regime": {
                label: float(self.market_regime[index, position])
                for position, label in enumerate(REGIME_LABELS)
            },
            "sector_strength": float(self.sector_strength[index]),
            "expected_return_bps": float(self.expected_return_bps[index]),
            "volatility": float(self.volatility[index]),
            "strategy_suitability": {
                family: float(self.strategy_suitability[index, position])
                for position, family in enumerate(STRATEGY_FAMILIES)
            },
            "trade_quality": float(self.trade_quality[index]),
            "uncertainty": float(self.uncertainty[index]),
        }


def _sigmoid(value: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-np.clip(value, -30.0, 30.0)))


def _softplus(value: np.ndarray) -> np.ndarray:
    return np.log1p(np.exp(np.clip(value, -30.0, 30.0)))


def _leaky_relu(value: np.ndarray) -> np.ndarray:
    return np.where(value >= 0.0, value, _LEAKY_SLOPE * value)


def _masked_softmax(logits: np.ndarray, axis: int) -> np.ndarray:
    """Softmax that yields all-zeros for a row with no finite entry.

    A node with no incoming edge under a relation must receive **no** message, not a
    uniform one. Ordinary softmax over ``-inf`` produces NaN; a fallback to uniform would
    invent a relation the ontology never declared.
    """
    finite = np.isfinite(logits)
    safe = np.where(finite, logits, -np.inf)
    maximum = np.max(np.where(finite, safe, -np.inf), axis=axis, keepdims=True)
    maximum = np.where(np.isfinite(maximum), maximum, 0.0)
    exponent = np.where(finite, np.exp(safe - maximum), 0.0)
    total = exponent.sum(axis=axis, keepdims=True)
    return np.divide(exponent, total, out=np.zeros_like(exponent), where=total > 0)


class TemporalHeteroGnn:
    """Relation-aware heterogeneous GNN with an ontology prior and a causal TCN."""

    #: Bumped whenever the tensor layout changes. A checkpoint from another version is
    #: refused rather than coerced.
    ARTIFACT_VERSION = 1

    def __init__(self, config: TemporalHeteroGnnConfig) -> None:
        self.config = config
        rng = np.random.default_rng(config.seed)
        hidden = config.hidden_dim

        def normal(*shape: int, fan_in: int) -> np.ndarray:
            return rng.normal(0.0, 1.0 / max(1, fan_in) ** 0.5, shape).astype(np.float32)

        # Node-type encoders: one [F, H] projection per type.
        self.type_encoders = normal(
            config.node_type_count, config.feature_dim, hidden, fan_in=config.feature_dim
        )
        self.type_bias = np.zeros((config.node_type_count, hidden), dtype=np.float32)

        # Per layer: relation transforms, attention vectors, relation gates, self weight.
        self.relation_weights = normal(
            config.layer_count, config.relation_count, hidden, hidden, fan_in=hidden
        )
        self.attention_source = normal(
            config.layer_count, config.relation_count, hidden, fan_in=hidden
        )
        self.attention_target = normal(
            config.layer_count, config.relation_count, hidden, fan_in=hidden
        )
        #: Learned per-relation scalar gate. This is the "relation weight is learnable"
        #: half of the contract; the ontology prior is the other half and stays separate.
        self.relation_gate = np.zeros(
            (config.layer_count, config.relation_count), dtype=np.float32
        )
        self.self_weights = normal(config.layer_count, hidden, hidden, fan_in=hidden)
        self.layer_bias = np.zeros((config.layer_count, hidden), dtype=np.float32)

        # Causal TCN over time.
        depth = len(config.tcn_dilations)
        self.tcn_current = normal(depth, hidden, hidden, fan_in=hidden)
        self.tcn_lagged = normal(depth, hidden, hidden, fan_in=hidden)
        self.tcn_bias = np.zeros((depth, hidden), dtype=np.float32)

        # Heads.
        self.head_regime = normal(hidden, config.regime_count, fan_in=hidden)
        self.head_regime_bias = np.zeros(config.regime_count, dtype=np.float32)
        self.head_suitability = normal(
            hidden, config.strategy_family_count, fan_in=hidden
        )
        self.head_suitability_bias = np.zeros(
            config.strategy_family_count, dtype=np.float32
        )
        #: sector_strength, expected_return, volatility, trade_quality, uncertainty.
        self.head_scalar = normal(hidden, 5, fan_in=hidden)
        self.head_scalar_bias = np.zeros(5, dtype=np.float32)

    # ------------------------------------------------------------------ #
    # inference
    # ------------------------------------------------------------------ #
    def infer(
        self,
        features: np.ndarray,
        adjacency: np.ndarray,
        prior_bias: np.ndarray,
        node_type_index: np.ndarray,
        node_mask: np.ndarray,
        *,
        node_ids: Sequence[str] = (),
        collect_attention: bool = True,
    ) -> TemporalHeteroGnnOutput:
        """One forward pass over a ``[T, N, F]`` window.

        ``adjacency`` and ``prior_bias`` are ``[R, N, N]`` and static across the window:
        the ontology's relations do not change within a few minutes, and rebuilding them
        per step would cost more than it could possibly express.
        """
        self._validate(features, adjacency, prior_bias, node_type_index, node_mask)
        mask = node_mask.astype(np.float32)[:, None]

        # -- 1. node-type encoders ---------------------------------------- #
        encoders = self.type_encoders[node_type_index]      # [N, F, H]
        biases = self.type_bias[node_type_index]            # [N, H]
        hidden = np.einsum("tnf,nfh->tnh", features, encoders, optimize=True) + biases
        hidden = np.maximum(hidden, 0.0) * mask[None, :, :]

        # -- 2. relation attention with the ontology prior ----------------- #
        attention_by_relation: dict[str, dict[str, Any]] = {}
        relation_mass: dict[str, float] = {}
        edge_exists = np.isfinite(prior_bias) & (adjacency > 0.0)
        for layer in range(self.config.layer_count):
            transformed = np.einsum(
                "tnh,rhk->trnk", hidden, self.relation_weights[layer], optimize=True
            )
            source_logit = np.einsum(
                "trnk,rk->trn", transformed, self.attention_source[layer], optimize=True
            )
            target_logit = np.einsum(
                "trnk,rk->trn", transformed, self.attention_target[layer], optimize=True
            )
            # e[t,r,i,j] = LeakyReLU(target_i + source_j) + prior_bias[r,i,j]
            logits = _leaky_relu(target_logit[:, :, :, None] + source_logit[:, :, None, :])
            logits = np.where(edge_exists[None, :, :, :], logits + prior_bias[None], -np.inf)
            attention = _masked_softmax(logits, axis=-1)
            messages = np.einsum("trij,trjk->trik", attention, transformed, optimize=True)
            gates = _sigmoid(self.relation_gate[layer])[None, :, None, None]
            aggregated = (messages * gates).sum(axis=1)
            self_part = np.einsum(
                "tnh,hk->tnk", hidden, self.self_weights[layer], optimize=True
            )
            hidden = np.maximum(aggregated + self_part + self.layer_bias[layer], 0.0)
            hidden = hidden * mask[None, :, :]

            if collect_attention and layer == self.config.layer_count - 1:
                attention_by_relation, relation_mass = self._trace_attention(
                    attention[-1], prior_bias, edge_exists, node_ids
                )

        # -- 3. causal TCN -------------------------------------------------- #
        temporal = self._tcn(hidden)                        # [N, H]
        temporal = temporal * mask

        # -- 4. heads -------------------------------------------------------- #
        regime = _sigmoid(temporal @ self.head_regime + self.head_regime_bias)
        suitability = _sigmoid(
            temporal @ self.head_suitability + self.head_suitability_bias
        )
        scalars = temporal @ self.head_scalar + self.head_scalar_bias
        sector_strength = np.tanh(scalars[:, 0])
        expected_return = np.tanh(scalars[:, 1]) * _EXPECTED_RETURN_SCALE_BPS
        volatility = _softplus(scalars[:, 2])
        trade_quality = _sigmoid(scalars[:, 3])
        uncertainty = _softplus(scalars[:, 4])

        # A masked node has no prediction. Zeroing the heads would read as "confidently
        # neutral"; the honest values are no suitability, no quality and full uncertainty.
        flat_mask = node_mask.astype(bool)
        regime[~flat_mask] = 0.0
        suitability[~flat_mask] = 0.0
        sector_strength[~flat_mask] = 0.0
        expected_return[~flat_mask] = 0.0
        volatility[~flat_mask] = 0.0
        trade_quality[~flat_mask] = 0.0
        uncertainty[~flat_mask] = 1.0

        return TemporalHeteroGnnOutput(
            market_regime=regime,
            sector_strength=sector_strength,
            expected_return_bps=expected_return,
            volatility=volatility,
            strategy_suitability=suitability,
            trade_quality=trade_quality,
            uncertainty=uncertainty,
            node_mask=node_mask.astype(np.float32),
            trace=InferenceTrace(
                node_ids=tuple(node_ids),
                relation_attention=attention_by_relation,
                relation_mass=relation_mass,
                active_node_count=int(flat_mask.sum()),
            ),
        )

    # ------------------------------------------------------------------ #
    # internals
    # ------------------------------------------------------------------ #
    def _tcn(self, hidden: np.ndarray) -> np.ndarray:
        """Dilated causal convolutions over the time axis; returns the last step.

        ``y[t] = relu(Wc x[t] + Wl x[t-d] + b) + x[t]``. Only ``t`` and ``t-d`` are read,
        so nothing after ``t`` can influence ``y[t]``. Steps before the window starts are
        padded with the earliest available step rather than with zeros: a zero-padded
        history claims the market was flat before the window, which is a fabricated
        observation, whereas repeating the first bar claims only that nothing is known.
        """
        current = hidden
        for depth, dilation in enumerate(self.config.tcn_dilations):
            steps = current.shape[0]
            shift = min(int(dilation), steps - 1) if steps > 1 else 0
            if shift > 0:
                lagged = np.concatenate(
                    (np.repeat(current[:1], shift, axis=0), current[:-shift]), axis=0
                )
            else:
                lagged = current
            projected = (
                np.einsum("tnh,hk->tnk", current, self.tcn_current[depth], optimize=True)
                + np.einsum("tnh,hk->tnk", lagged, self.tcn_lagged[depth], optimize=True)
                + self.tcn_bias[depth]
            )
            current = np.maximum(projected, 0.0) + current
        return current[-1]

    def _trace_attention(
        self,
        attention: np.ndarray,
        prior_bias: np.ndarray,
        edge_exists: np.ndarray,
        node_ids: Sequence[str],
    ) -> tuple[dict[str, dict[str, Any]], dict[str, float]]:
        payload: dict[str, dict[str, Any]] = {}
        mass: dict[str, float] = {}
        names = list(node_ids)
        for index, relation in enumerate(RELATION_TYPES[: self.config.relation_count]):
            present = edge_exists[index]
            if not present.any():
                continue
            weights = attention[index]
            total = float(weights.sum())
            mass[relation] = round(total, 6)
            targets, sources = np.nonzero(present)
            edges = []
            for target, source in zip(targets.tolist(), sources.tolist()):
                edges.append(
                    {
                        "source": names[source] if source < len(names) else int(source),
                        "target": names[target] if target < len(names) else int(target),
                        "attention": round(float(weights[target, source]), 6),
                        "ontology_prior_bias": round(
                            float(prior_bias[index, target, source]), 6
                        ),
                    }
                )
            edges.sort(key=lambda item: -item["attention"])
            payload[relation] = {"total_attention": round(total, 6), "edges": edges[:20]}
        return payload, mass

    def _validate(
        self,
        features: np.ndarray,
        adjacency: np.ndarray,
        prior_bias: np.ndarray,
        node_type_index: np.ndarray,
        node_mask: np.ndarray,
    ) -> None:
        config = self.config
        expected = {
            "features": (config.time_steps, config.max_nodes, config.feature_dim),
            "adjacency": (config.relation_count, config.max_nodes, config.max_nodes),
            "prior_bias": (config.relation_count, config.max_nodes, config.max_nodes),
            "node_type_index": (config.max_nodes,),
            "node_mask": (config.max_nodes,),
        }
        actual = {
            "features": features.shape,
            "adjacency": adjacency.shape,
            "prior_bias": prior_bias.shape,
            "node_type_index": node_type_index.shape,
            "node_mask": node_mask.shape,
        }
        for name, shape in expected.items():
            if actual[name] != shape:
                raise ValueError(f"{name} shape {actual[name]} != required {shape}")
        for name, value in (
            ("features", features),
            ("adjacency", adjacency),
            ("node_mask", node_mask),
        ):
            if not np.isfinite(value).all():
                raise ValueError(f"{name} must be finite")
        if node_type_index.min() < 0 or node_type_index.max() >= config.node_type_count:
            raise ValueError("node_type_index out of range")

    # ------------------------------------------------------------------ #
    # checkpoints
    # ------------------------------------------------------------------ #
    _TENSOR_NAMES = (
        "type_encoders",
        "type_bias",
        "relation_weights",
        "attention_source",
        "attention_target",
        "relation_gate",
        "self_weights",
        "layer_bias",
        "tcn_current",
        "tcn_lagged",
        "tcn_bias",
        "head_regime",
        "head_regime_bias",
        "head_suitability",
        "head_suitability_bias",
        "head_scalar",
        "head_scalar_bias",
    )

    def save_checkpoint(self, path: str | Path) -> Path:
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        temporary = target.with_name(f".{target.stem}.writing.npz")
        np.savez_compressed(
            temporary,
            artifact_version=np.asarray([self.ARTIFACT_VERSION], dtype=np.int64),
            config=self.config.as_array(),
            **{name: getattr(self, name) for name in self._TENSOR_NAMES},
        )
        # Never expose a half-written checkpoint to a live loader.
        os.replace(temporary, target)
        return target

    @classmethod
    def load_checkpoint(cls, path: str | Path) -> "TemporalHeteroGnn":
        """Load, or raise. A partially-valid checkpoint is never returned.

        Shape and finiteness are both checked: a tensor of the right shape full of NaN
        would pass a shape check and then produce NaN predictions that every downstream
        comparison silently treats as False.
        """
        source = Path(path)
        with np.load(source, allow_pickle=False) as data:
            version = int(np.asarray(data["artifact_version"]).ravel()[0])
            if version != cls.ARTIFACT_VERSION:
                raise ValueError(
                    f"temporal hetero GNN checkpoint version {version} != "
                    f"{cls.ARTIFACT_VERSION}"
                )
            model = cls(TemporalHeteroGnnConfig.from_array(data["config"].tolist()))
            for name in cls._TENSOR_NAMES:
                if name not in data:
                    raise ValueError(f"checkpoint is missing tensor {name}")
                value = np.asarray(data[name], dtype=np.float32)
                expected = getattr(model, name).shape
                if value.shape != expected:
                    raise ValueError(
                        f"checkpoint tensor {name}: {value.shape} != {expected}"
                    )
                if not np.isfinite(value).all():
                    raise ValueError(f"checkpoint tensor {name} contains non-finite values")
                setattr(model, name, value.copy())
        return model

    def parameter_count(self) -> int:
        return int(
            sum(int(np.asarray(getattr(self, name)).size) for name in self._TENSOR_NAMES)
        )
