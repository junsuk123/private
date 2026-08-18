"""The heterogeneous market graph: typed nodes, typed relations, and two weights.

Two weights, never one
----------------------
Every edge carries a ``prior_strength`` — what a person claimed — and, separately, a
``learned_weight`` — what training measured. They are stored in different columns, loaded
from different places (config versus the ``ontology_edge`` table) and both appear in every
inference trace. Blending them into one number at rest would destroy the only question
worth asking about this graph: *where does the model disagree with the expert, and was it
right to?*

The prior does not gate anything. It enters the GNN as a soft bias on the relation's
attention logit (:meth:`MarketGraph.prior_bias`), so a strongly-held prior needs strong
contrary evidence to overturn — and when it is overturned, the trace shows the prior, the
learned weight and the realised attention side by side.

Structural relations are exempt. ``BELONGS_TO``, ``TRADED_ON``, ``REQUIRES`` and
``INVALIDATES`` are marked ``learnable: false`` in the config because they are definitions
and hard constraints, not hypotheses. A model must not be able to learn that a stock is
75% a member of its sector, or that stale data only sometimes invalidates a thesis.

Relationship to the RDF/OWL ontology
------------------------------------
``app.ontology.trading_domain_ontology`` and the ``.ttl`` files remain the authority for
reasoning, SHACL validation and the eligibility masks. This module is the *numeric
projection* of the same ideas for the GNN: an adjacency tensor and a prior-bias tensor
indexed by relation. It does not duplicate the reasoner's rules and never authorises a
trade.
"""

from __future__ import annotations

import threading
from dataclasses import dataclass, field, replace
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

__all__ = [
    "MarketGraph",
    "MarketGraphError",
    "NODE_TYPES",
    "OntologyEdge",
    "OntologyNode",
    "RELATION_TYPES",
    "RelationSpec",
    "default_market_graph",
    "load_market_graph",
    "reset_market_graph_cache",
]

DEFAULT_CONFIG_PATH = Path("config/market_graph_ontology.yaml")

NODE_TYPES: tuple[str, ...] = (
    "TemporalEntity",
    "Venue",
    "Market",
    "MacroFactor",
    "Sector",
    "Industry",
    "Stock",
    "MarketRegime",
    "Strategy",
    "RiskCondition",
    "TradeIntent",
)

RELATION_TYPES: tuple[str, ...] = (
    "INFLUENCES",
    "LEADS",
    "LAGS",
    "CORRELATED_WITH",
    "CONFIRMS",
    "CONTRADICTS",
    "INCREASES_RISK",
    "DECREASES_RISK",
    "SUITABLE_FOR",
    "UNSUITABLE_FOR",
    "REQUIRES",
    "INVALIDATES",
    "BELONGS_TO",
    "TRADED_ON",
)

#: Relations that express a hard constraint rather than a graded belief. Never learnable,
#: and read directly by the gate as well as by the model.
STRUCTURAL_RELATIONS: frozenset[str] = frozenset(
    {"BELONGS_TO", "TRADED_ON", "REQUIRES", "INVALIDATES"}
)


class MarketGraphError(RuntimeError):
    """The ontology config is not a valid market graph."""


@dataclass(frozen=True)
class RelationSpec:
    name: str
    symmetric: bool = False
    default_prior: float = 0.5
    learnable: bool = True
    lag_min: int = 0
    lag_max: int = 0


@dataclass(frozen=True)
class OntologyNode:
    node_id: str
    node_type: str
    label: str = ""
    attributes: Mapping[str, Any] = field(default_factory=dict)

    def as_dict(self) -> dict[str, Any]:
        return {
            "node_id": self.node_id,
            "node_type": self.node_type,
            "label": self.label,
            "attributes": dict(self.attributes),
        }


@dataclass(frozen=True)
class OntologyEdge:
    """One typed relation, carrying the expert prior and the learned weight apart."""

    source_id: str
    target_id: str
    relation: str
    prior_strength: float
    learnable: bool
    lag_min: int = 0
    lag_max: int = 0
    direction: str = "FORWARD"
    #: ``None`` until training has produced one. Never defaulted to the prior: "not yet
    #: learned" and "learned to equal the prior" are different states.
    learned_weight: float | None = None
    learned_updated_at: datetime | None = None
    attributes: Mapping[str, Any] = field(default_factory=dict)

    @property
    def edge_id(self) -> str:
        return f"{self.source_id}|{self.relation}|{self.target_id}"

    @property
    def effective_weight(self) -> float:
        """The weight message passing actually uses.

        The learned weight when one exists and the relation is learnable; the prior
        otherwise. Callers that need to know *which* read :attr:`weight_source`.
        """
        if self.learnable and self.learned_weight is not None:
            return float(self.learned_weight)
        return float(self.prior_strength)

    @property
    def weight_source(self) -> str:
        if not self.learnable:
            return "structural"
        return "learned" if self.learned_weight is not None else "prior"

    @property
    def prior_learned_gap(self) -> float | None:
        """How far training moved this edge from its prior. ``None`` when unlearned."""
        if self.learned_weight is None:
            return None
        return round(float(self.learned_weight) - float(self.prior_strength), 6)

    def as_dict(self) -> dict[str, Any]:
        return {
            "edge_id": self.edge_id,
            "source": self.source_id,
            "target": self.target_id,
            "relation": self.relation,
            "direction": self.direction,
            "prior_strength": self.prior_strength,
            "learnable": self.learnable,
            "learned_weight": self.learned_weight,
            "learned_updated_at": (
                self.learned_updated_at.isoformat() if self.learned_updated_at else None
            ),
            "effective_weight": self.effective_weight,
            "weight_source": self.weight_source,
            "prior_learned_gap": self.prior_learned_gap,
            "lag_min": self.lag_min,
            "lag_max": self.lag_max,
            "attributes": dict(self.attributes),
        }


class MarketGraph:
    """Typed nodes and relations, plus the tensors the GNN indexes them by."""

    def __init__(
        self,
        nodes: Sequence[OntologyNode],
        edges: Sequence[OntologyEdge],
        *,
        relations: Mapping[str, RelationSpec] | None = None,
        prior_bias_scale: float = 1.0,
        source_path: str | None = None,
    ) -> None:
        self._nodes: dict[str, OntologyNode] = {}
        for node in nodes:
            if node.node_type not in NODE_TYPES:
                raise MarketGraphError(
                    f"node {node.node_id}: unknown node type {node.node_type!r}"
                )
            self._nodes[node.node_id] = node
        self._relations = dict(relations or _default_relation_specs())
        self._edges: dict[str, OntologyEdge] = {}
        for edge in edges:
            self._validate_edge(edge)
            self._edges[edge.edge_id] = edge
        self._prior_bias_scale = float(prior_bias_scale)
        self.source_path = source_path
        self._lock = threading.RLock()
        self._order = tuple(self._nodes)
        self._index = {node_id: position for position, node_id in enumerate(self._order)}

    # -- structure ---------------------------------------------------------- #
    @property
    def node_ids(self) -> tuple[str, ...]:
        return self._order

    @property
    def prior_bias_scale(self) -> float:
        return self._prior_bias_scale

    def node(self, node_id: str) -> OntologyNode | None:
        return self._nodes.get(str(node_id))

    def nodes_of_type(self, node_type: str) -> tuple[OntologyNode, ...]:
        return tuple(
            node for node in self._nodes.values() if node.node_type == node_type
        )

    def index_of(self, node_id: str) -> int | None:
        return self._index.get(str(node_id))

    def edges(
        self,
        *,
        relation: str | None = None,
        source: str | None = None,
        target: str | None = None,
    ) -> tuple[OntologyEdge, ...]:
        with self._lock:
            items = list(self._edges.values())
        return tuple(
            edge
            for edge in items
            if (relation is None or edge.relation == relation)
            and (source is None or edge.source_id == source)
            and (target is None or edge.target_id == target)
        )

    def relation_spec(self, relation: str) -> RelationSpec | None:
        return self._relations.get(str(relation))

    def _validate_edge(self, edge: OntologyEdge) -> None:
        if edge.relation not in RELATION_TYPES:
            raise MarketGraphError(f"unknown relation {edge.relation!r}")
        for node_id in (edge.source_id, edge.target_id):
            if node_id not in self._nodes:
                raise MarketGraphError(
                    f"edge {edge.edge_id} references unknown node {node_id!r}"
                )
        if not 0.0 <= edge.prior_strength <= 1.0:
            raise MarketGraphError(
                f"edge {edge.edge_id}: prior_strength must be within [0, 1]"
            )
        if edge.relation in STRUCTURAL_RELATIONS and edge.learnable:
            raise MarketGraphError(
                f"edge {edge.edge_id}: {edge.relation} is structural and must not be learnable"
            )

    # -- learned weights ----------------------------------------------------- #
    def apply_learned_weights(
        self,
        weights: Mapping[str, float],
        *,
        updated_at: datetime | None = None,
    ) -> tuple[str, ...]:
        """Attach learned weights by ``edge_id``. Returns the ids that were applied.

        Weights for structural (non-learnable) edges are refused rather than applied:
        silently ignoring them would let a training run believe it had moved a constraint
        it cannot move.
        """
        moment = (updated_at or datetime.now(timezone.utc)).astimezone(timezone.utc)
        applied: list[str] = []
        with self._lock:
            for edge_id, raw in weights.items():
                edge = self._edges.get(str(edge_id))
                if edge is None:
                    continue
                if not edge.learnable:
                    raise MarketGraphError(
                        f"edge {edge_id} is structural; its weight cannot be learned"
                    )
                self._edges[edge.edge_id] = replace(
                    edge,
                    learned_weight=float(raw),
                    learned_updated_at=moment,
                )
                applied.append(edge.edge_id)
        return tuple(applied)

    def clear_learned_weights(self) -> None:
        """Drop every learned weight, returning the graph to its priors.

        The fail-closed response to a checkpoint that no longer matches the graph: the
        priors are always usable, a stale learned weight is not.
        """
        with self._lock:
            for edge_id, edge in list(self._edges.items()):
                if edge.learned_weight is not None:
                    self._edges[edge_id] = replace(
                        edge, learned_weight=None, learned_updated_at=None
                    )

    # -- numeric projection --------------------------------------------------- #
    def adjacency(self, *, use_learned: bool = True):
        """``[R, N, N]`` float32 weights, row-normalised per relation.

        ``adjacency[r, target, source]`` is the weight of the message flowing from
        ``source`` to ``target`` under relation ``r`` — the same orientation
        ``app.models.strategy_utility.strategy_graph`` already uses, so the two tensors
        can be reasoned about together.
        """
        import numpy as np

        size = len(self._order)
        tensor = np.zeros((len(RELATION_TYPES), size, size), dtype=np.float32)
        for edge in self.edges():
            relation_index = RELATION_TYPES.index(edge.relation)
            source = self._index[edge.source_id]
            target = self._index[edge.target_id]
            weight = edge.effective_weight if use_learned else edge.prior_strength
            tensor[relation_index, target, source] = weight
            spec = self._relations.get(edge.relation)
            if spec is not None and spec.symmetric:
                tensor[relation_index, source, target] = weight
        degrees = tensor.sum(axis=-1, keepdims=True)
        return np.divide(tensor, degrees, out=np.zeros_like(tensor), where=degrees > 0)

    def prior_bias(self):
        """``[R, N, N]`` additive bias for the attention logits.

        ``bias = prior_bias_scale * prior_strength`` on declared edges and ``-inf`` where
        no edge exists, so softmax cannot route attention across a relation the ontology
        never declared. The prior only *ranks* the edges that do exist; it never blocks
        one, which is what keeps it a prior rather than a mask.
        """
        import numpy as np

        size = len(self._order)
        bias = np.full(
            (len(RELATION_TYPES), size, size), -np.inf, dtype=np.float32
        )
        for edge in self.edges():
            relation_index = RELATION_TYPES.index(edge.relation)
            source = self._index[edge.source_id]
            target = self._index[edge.target_id]
            value = np.float32(self._prior_bias_scale * edge.prior_strength)
            bias[relation_index, target, source] = value
            spec = self._relations.get(edge.relation)
            if spec is not None and spec.symmetric:
                bias[relation_index, source, target] = value
        return bias

    def node_type_indices(self) -> dict[str, tuple[int, ...]]:
        """Node positions grouped by type, for the type-specific encoders."""
        grouped: dict[str, list[int]] = {name: [] for name in NODE_TYPES}
        for node_id, position in self._index.items():
            grouped[self._nodes[node_id].node_type].append(position)
        return {name: tuple(values) for name, values in grouped.items() if values}

    # -- serialisation --------------------------------------------------------- #
    def trace_edges(
        self, node_ids: Iterable[str] | None = None
    ) -> tuple[dict[str, Any], ...]:
        """Edge records for a decision trace, prior and learned weight both present."""
        wanted = set(node_ids) if node_ids is not None else None
        return tuple(
            edge.as_dict()
            for edge in self.edges()
            if wanted is None
            or edge.source_id in wanted
            or edge.target_id in wanted
        )

    def as_dict(self) -> dict[str, Any]:
        return {
            "source_path": self.source_path,
            "prior_bias_scale": self._prior_bias_scale,
            "node_count": len(self._nodes),
            "edge_count": len(self._edges),
            "nodes": [node.as_dict() for node in self._nodes.values()],
            "edges": [edge.as_dict() for edge in self.edges()],
        }

    def summary(self) -> dict[str, Any]:
        edges = self.edges()
        learned = [edge for edge in edges if edge.learned_weight is not None]
        by_relation: dict[str, int] = {}
        for edge in edges:
            by_relation[edge.relation] = by_relation.get(edge.relation, 0) + 1
        return {
            "node_count": len(self._nodes),
            "edge_count": len(edges),
            "learnable_edge_count": sum(1 for edge in edges if edge.learnable),
            "learned_edge_count": len(learned),
            "structural_edge_count": sum(1 for edge in edges if not edge.learnable),
            "relations": dict(sorted(by_relation.items())),
            "node_types": {
                node_type: len(self.nodes_of_type(node_type))
                for node_type in NODE_TYPES
                if self.nodes_of_type(node_type)
            },
            "max_prior_learned_gap": max(
                (abs(edge.prior_learned_gap or 0.0) for edge in learned), default=0.0
            ),
        }


def _default_relation_specs() -> dict[str, RelationSpec]:
    return {
        name: RelationSpec(
            name=name,
            learnable=name not in STRUCTURAL_RELATIONS,
            default_prior=1.0 if name in STRUCTURAL_RELATIONS else 0.5,
        )
        for name in RELATION_TYPES
    }


def _mapping(raw: Any, name: str) -> Mapping[str, Any]:
    if raw is None:
        return {}
    if not isinstance(raw, Mapping):
        raise MarketGraphError(f"{name} must be a mapping")
    return raw


def load_market_graph(path: str | Path = DEFAULT_CONFIG_PATH) -> MarketGraph:
    """Read the ontology config into a :class:`MarketGraph`.

    A missing file is an error here, unlike the other configs in this project: an empty
    market graph would silently disable every ontology prior and every structural
    constraint, and the model would go on producing numbers that looked fine.
    """
    target = Path(path)
    if not target.exists():
        raise MarketGraphError(f"market graph ontology not found at {target}")
    try:
        import yaml

        raw = yaml.safe_load(target.read_text(encoding="utf-8")) or {}
    except MarketGraphError:
        raise
    except Exception as exc:  # noqa: BLE001 - surfaced, never swallowed.
        raise MarketGraphError(f"cannot parse {target}: {exc}") from exc

    document = _mapping(raw, str(target))
    relations = _default_relation_specs()
    for name, value in _mapping(document.get("relation_types"), "relation_types").items():
        if name not in RELATION_TYPES:
            raise MarketGraphError(f"unknown relation type {name!r} in {target}")
        entry = _mapping(value, f"relation_types.{name}")
        learnable = bool(entry.get("learnable", name not in STRUCTURAL_RELATIONS))
        if name in STRUCTURAL_RELATIONS and learnable:
            raise MarketGraphError(f"relation {name} is structural and cannot be learnable")
        relations[name] = RelationSpec(
            name=name,
            symmetric=bool(entry.get("symmetric", False)),
            default_prior=float(entry.get("default_prior", 0.5)),
            learnable=learnable,
            lag_min=int(entry.get("lag_min", 0)),
            lag_max=int(entry.get("lag_max", 0)),
        )

    nodes: list[OntologyNode] = []
    for entry in document.get("nodes") or ():
        item = _mapping(entry, "nodes[]")
        node_id = str(item.get("id") or "").strip()
        if not node_id:
            raise MarketGraphError("every ontology node needs an id")
        nodes.append(
            OntologyNode(
                node_id=node_id,
                node_type=str(item.get("type") or "").strip(),
                label=str(item.get("label") or ""),
                attributes=dict(_mapping(item.get("attributes"), "attributes")),
            )
        )

    edges: list[OntologyEdge] = []
    for entry in document.get("edges") or ():
        item = _mapping(entry, "edges[]")
        relation = str(item.get("relation") or "").strip()
        spec = relations.get(relation)
        if spec is None:
            raise MarketGraphError(f"unknown relation {relation!r} in {target}")
        edges.append(
            OntologyEdge(
                source_id=str(item.get("source") or "").strip(),
                target_id=str(item.get("target") or "").strip(),
                relation=relation,
                prior_strength=float(item.get("prior_strength", spec.default_prior)),
                learnable=bool(item.get("learnable", spec.learnable)),
                lag_min=int(item.get("lag_min", spec.lag_min)),
                lag_max=int(item.get("lag_max", spec.lag_max)),
                direction=str(item.get("direction") or "FORWARD"),
                attributes=dict(_mapping(item.get("attributes"), "attributes")),
            )
        )

    return MarketGraph(
        nodes,
        edges,
        relations=relations,
        prior_bias_scale=float(document.get("prior_bias_scale", 1.0)),
        source_path=str(target),
    )


_graph_cache: MarketGraph | None = None
_graph_lock = threading.Lock()


def default_market_graph() -> MarketGraph:
    global _graph_cache
    with _graph_lock:
        if _graph_cache is None:
            _graph_cache = load_market_graph()
        return _graph_cache


def reset_market_graph_cache() -> None:
    """Test hook. Never called from the trading path."""
    global _graph_cache
    with _graph_lock:
        _graph_cache = None
