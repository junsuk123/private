"""Turns the context hierarchy into the tensors the temporal hetero GNN consumes.

One snapshot is ``[T, N, F]`` node features over a fixed node roster, plus the ``[R, N, N]``
adjacency and ontology prior-bias tensors. The roster is the static ontology graph
(markets, venues, macro factors, regimes, strategy families, risk conditions, session
phases) **extended per cycle** with the sectors and stocks actually under consideration,
wired in through the same typed relations:

* ``Stock  --BELONGS_TO-->  Sector``      structural
* ``Stock  --TRADED_ON-->   Venue``       structural
* ``Sector --BELONGS_TO-->  Market``      structural
* ``MacroFactor --INFLUENCES--> Sector``  prior-weighted, from the sector's declared
  global counterpart
* ``TemporalEntity --INFLUENCES--> Market`` for the phase that is currently active

Feature vectors are **type-specific by position**: slot 3 means something different on a
``Stock`` than on a ``MacroFactor``, which is exactly why the model has one encoder per
node type. The layout is declared once in :data:`FEATURE_SLOTS` so training and serving
cannot drift apart.

Missing values are zeros *with the mask bit still set*, except when a node has no data at
all — then the node is masked out entirely and receives no prediction. Zero-filling an
absent feature is safe here only because every feature is centred: a zero means "no
signal", not "signal of magnitude zero in some absolute scale".
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Mapping, Sequence

import numpy as np

from app.context.domestic_context import DomesticContext
from app.context.global_context import GlobalContext
from app.context.sector_context import SectorContext
from app.context.temporal_context import TemporalSnapshot
from app.ontology.market_graph import (
    NODE_TYPES,
    RELATION_TYPES,
    MarketGraph,
    OntologyEdge,
    OntologyNode,
    default_market_graph,
)

__all__ = [
    "FEATURE_DIM",
    "FEATURE_SLOTS",
    "GraphSnapshot",
    "GraphSnapshotBuilder",
    "StockNodeObservation",
]

#: Feature vector layout per node type. Position is part of the checkpoint contract:
#: appending to a type's list is safe, reordering invalidates every trained model.
FEATURE_SLOTS: dict[str, tuple[str, ...]] = {
    "Market": (
        "direction",
        "breadth",
        "liquidity",
        "volatility",
        "flow",
        "leadership",
        "venue_divergence",
        "global_agreement",
        "confidence",
    ),
    "MacroFactor": ("score", "raw_score", "momentum", "freshness", "confidence"),
    "Sector": (
        "return",
        "breadth",
        "volume_z",
        "volatility",
        "relative_strength",
        "foreign_flow",
        "leader_strength",
        "leader_concentration",
        "global_alignment",
        "confidence",
    ),
    "Stock": (
        "return",
        "vwap_distance_bps",
        "ema_gap_bps",
        "momentum",
        "realized_volatility",
        "volume_intensity",
        "trade_intensity",
        "spread_bps",
        "depth",
        "orderbook_imbalance",
        "trade_imbalance",
        "relative_strength",
        "breakout_state",
        "data_age_seconds",
    ),
    "TemporalEntity": (
        "active",
        "session_progress",
        "minutes_from_open",
        "minutes_to_close",
        "day_of_week",
        "holiday_adjacent",
        "month_end",
        "quarter_end",
        "expiry",
    ),
    "MarketRegime": ("prior_probability",),
    "Strategy": ("ontology_suitability", "ontology_unsuitability"),
    "RiskCondition": ("active", "severity"),
    "Venue": ("active", "divergence"),
    "Industry": ("return", "breadth"),
    "TradeIntent": ("pending",),
}

#: Width of every node's feature vector. The widest type sets it; narrower types pad.
FEATURE_DIM = max(len(names) for names in FEATURE_SLOTS.values())

#: Sector name -> global indicator group, used to wire the cross-market INFLUENCES edges.
#: Lower-cased substring match, so "반도체/semiconductor" resolves without an exact table.
_SECTOR_GLOBAL_GROUP: tuple[tuple[str, str], tuple[str, str], ...] = (
    ("semiconductor", "semiconductor"),
    ("반도체", "semiconductor"),
    ("chip", "semiconductor"),
    ("it", "semiconductor"),
    ("tech", "equity"),
    ("bank", "rates"),
    ("금융", "rates"),
    ("insur", "rates"),
    ("energy", "commodity"),
    ("chem", "commodity"),
    ("화학", "commodity"),
    ("steel", "commodity"),
    ("철강", "commodity"),
    ("auto", "equity"),
    ("자동차", "equity"),
    ("ship", "equity"),
    ("bio", "equity"),
    ("제약", "equity"),
    ("health", "equity"),
)

#: Prior strength for the generated macro -> sector influence edges. Deliberately below
#: the hand-declared macro -> market priors in the config: a group-level mapping inferred
#: from a sector name is weaker evidence than a relation a person wrote down.
_GENERATED_INFLUENCE_PRIOR = 0.45
_GENERATED_PEER_PRIOR = 0.35
_GENERATED_TEMPORAL_PRIOR = 0.40

_PHASE_NODE = {
    "OPENING": "OPENING_PHASE",
    "OPEN_TRANSITION": "OPENING_PHASE",
    "MORNING_TREND": "OPENING_PHASE",
    "MIDDAY": "MIDDAY_PHASE",
    "AFTERNOON": "MIDDAY_PHASE",
    "CLOSING": "CLOSING_PHASE",
}

_MARKET_NODE = {"KR": "KR_MARKET", "US": "US_MARKET"}


@dataclass(frozen=True)
class StockNodeObservation:
    """One candidate's micro state for the graph."""

    ticker: str
    sector: str | None = None
    venue: str | None = None
    market_group: str = "KR"
    session_return: float | None = None
    vwap_distance_bps: float | None = None
    ema_gap_bps: float | None = None
    momentum: float | None = None
    realized_volatility: float | None = None
    volume_intensity: float | None = None
    trade_intensity: float | None = None
    spread_bps: float | None = None
    depth: float | None = None
    orderbook_imbalance: float | None = None
    trade_imbalance: float | None = None
    relative_strength: float | None = None
    breakout_state: float | None = None
    data_age_seconds: float | None = None
    peers: Sequence[str] = ()

    def feature_values(self) -> dict[str, float | None]:
        return {
            "return": self.session_return,
            "vwap_distance_bps": _scaled(self.vwap_distance_bps, 50.0),
            "ema_gap_bps": _scaled(self.ema_gap_bps, 50.0),
            "momentum": self.momentum,
            "realized_volatility": _scaled(self.realized_volatility, 0.01),
            "volume_intensity": self.volume_intensity,
            "trade_intensity": self.trade_intensity,
            "spread_bps": _scaled(self.spread_bps, 25.0),
            "depth": self.depth,
            "orderbook_imbalance": self.orderbook_imbalance,
            "trade_imbalance": self.trade_imbalance,
            "relative_strength": self.relative_strength,
            "breakout_state": self.breakout_state,
            "data_age_seconds": _scaled(self.data_age_seconds, 60.0),
        }


@dataclass(frozen=True)
class GraphSnapshot:
    """Fixed-shape tensors plus the roster needed to read the model's output."""

    node_ids: tuple[str, ...]
    node_types: tuple[str, ...]
    features: np.ndarray          # [T, N, F]
    adjacency: np.ndarray         # [R, N, N]
    prior_bias: np.ndarray        # [R, N, N]
    node_type_index: np.ndarray   # [N]
    node_mask: np.ndarray         # [N]
    captured_at: datetime
    graph: MarketGraph
    index_by_node: Mapping[str, int] = field(default_factory=dict)
    reason_codes: tuple[str, ...] = ()

    def index_of(self, node_id: str) -> int | None:
        return self.index_by_node.get(str(node_id))

    def stock_indices(self) -> dict[str, int]:
        return {
            node_id: position
            for node_id, position, node_type in zip(
                self.node_ids, range(len(self.node_ids)), self.node_types
            )
            if node_type == "Stock"
        }

    @property
    def active_node_count(self) -> int:
        return int(self.node_mask.sum())


def _scaled(value: float | None, scale: float) -> float | None:
    if value is None:
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    if not np.isfinite(number) or scale <= 0.0:
        return None
    return float(np.tanh(number / scale))


def _finite(value: Any) -> float | None:
    if value is None or isinstance(value, bool):
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if np.isfinite(number) else None


def _global_group_for_sector(sector: str) -> str | None:
    lowered = str(sector or "").strip().lower()
    if not lowered:
        return None
    for token, group in _SECTOR_GLOBAL_GROUP:
        if token in lowered:
            return group
    return None


class GraphSnapshotBuilder:
    """Assembles one :class:`GraphSnapshot` per cycle.

    ``max_nodes`` fixes the tensor width so a trained checkpoint stays loadable as the
    candidate set changes. Candidates beyond the budget are dropped in the order given and
    reported in ``reason_codes`` — a silently truncated roster would look identical to a
    quiet market.
    """

    #: Macro factor node -> the global indicator group it reads its features from.
    _MACRO_GROUP_ATTRIBUTE = "global_group"

    def __init__(
        self,
        *,
        graph: MarketGraph | None = None,
        max_nodes: int = 96,
        time_steps: int = 8,
    ) -> None:
        self._base_graph = graph or default_market_graph()
        self.max_nodes = max(len(self._base_graph.node_ids) + 1, int(max_nodes))
        self.time_steps = max(1, int(time_steps))

    def build(
        self,
        *,
        captured_at: datetime,
        temporal: TemporalSnapshot | None = None,
        global_context: GlobalContext | None = None,
        domestic_context: DomesticContext | None = None,
        sector_contexts: Sequence[SectorContext] = (),
        stocks: Sequence[StockNodeObservation] = (),
        history: Sequence[Mapping[str, Mapping[str, float]]] = (),
        active_risk_conditions: Mapping[str, float] | None = None,
        regime_prior: Mapping[str, float] | None = None,
    ) -> GraphSnapshot:
        """Build the snapshot.

        ``history`` supplies earlier time steps as ``{node_id: {slot: value}}`` mappings,
        oldest first. When it is shorter than ``time_steps`` the earliest available step is
        repeated backwards — the same convention the TCN uses, and for the same reason: a
        zero-filled history asserts a flat market that was never observed.
        """
        moment = (
            captured_at
            if captured_at.tzinfo
            else captured_at.replace(tzinfo=timezone.utc)
        ).astimezone(timezone.utc)
        reasons: list[str] = []

        nodes, edges, dropped = self._roster(
            sector_contexts, stocks, temporal, domestic_context
        )
        if dropped:
            reasons.append(f"GRAPH_NODE_BUDGET_EXCEEDED:{dropped}")

        graph = MarketGraph(
            nodes,
            edges,
            relations={
                name: spec
                for name in RELATION_TYPES
                if (spec := self._base_graph.relation_spec(name)) is not None
            },
            prior_bias_scale=self._base_graph.prior_bias_scale,
            source_path=self._base_graph.source_path,
        )
        node_ids = graph.node_ids
        index_by_node = {node_id: position for position, node_id in enumerate(node_ids)}
        size = self.max_nodes

        current_values = self._feature_values(
            graph,
            temporal=temporal,
            global_context=global_context,
            domestic_context=domestic_context,
            sector_contexts=sector_contexts,
            stocks=stocks,
            active_risk_conditions=active_risk_conditions or {},
            regime_prior=regime_prior or {},
        )

        features = np.zeros((self.time_steps, size, FEATURE_DIM), dtype=np.float32)
        steps: list[Mapping[str, Mapping[str, float]]] = list(history)[
            -(self.time_steps - 1) :
        ]
        steps.append(current_values)
        while len(steps) < self.time_steps:
            steps.insert(0, steps[0])
        for step_index, step in enumerate(steps):
            for node_id, values in step.items():
                position = index_by_node.get(node_id)
                if position is None:
                    continue
                node = graph.node(node_id)
                if node is None:
                    continue
                slots = FEATURE_SLOTS.get(node.node_type, ())
                for slot_index, slot in enumerate(slots):
                    value = _finite(values.get(slot))
                    if value is not None:
                        features[step_index, position, slot_index] = value

        node_mask = np.zeros(size, dtype=np.float32)
        node_type_index = np.zeros(size, dtype=np.int64)
        for node_id, position in index_by_node.items():
            node = graph.node(node_id)
            if node is None:
                continue
            node_mask[position] = 1.0
            node_type_index[position] = NODE_TYPES.index(node.node_type)

        adjacency = _pad_relation_tensor(graph.adjacency(), size, fill=0.0)
        prior_bias = _pad_relation_tensor(graph.prior_bias(), size, fill=-np.inf)

        return GraphSnapshot(
            node_ids=tuple(node_ids),
            node_types=tuple(
                graph.node(node_id).node_type for node_id in node_ids  # type: ignore[union-attr]
            ),
            features=features,
            adjacency=adjacency,
            prior_bias=prior_bias,
            node_type_index=node_type_index,
            node_mask=node_mask,
            captured_at=moment,
            graph=graph,
            index_by_node=index_by_node,
            reason_codes=tuple(reasons),
        )

    # ------------------------------------------------------------------ #
    # roster
    # ------------------------------------------------------------------ #
    def _roster(
        self,
        sector_contexts: Sequence[SectorContext],
        stocks: Sequence[StockNodeObservation],
        temporal: TemporalSnapshot | None,
        domestic_context: DomesticContext | None,
    ) -> tuple[list[OntologyNode], list[OntologyEdge], int]:
        nodes: list[OntologyNode] = [
            node
            for node_id in self._base_graph.node_ids
            if (node := self._base_graph.node(node_id)) is not None
        ]
        edges: list[OntologyEdge] = list(self._base_graph.edges())
        known = {node.node_id for node in nodes}
        budget = self.max_nodes - len(nodes)
        dropped = 0

        sector_nodes: dict[str, str] = {}
        for context in sector_contexts:
            node_id = _sector_node_id(context.sector)
            if node_id in known:
                continue
            if budget <= 0:
                dropped += 1
                continue
            budget -= 1
            known.add(node_id)
            sector_nodes[context.sector] = node_id
            nodes.append(
                OntologyNode(
                    node_id=node_id,
                    node_type="Sector",
                    label=context.sector,
                    attributes={"market_group": context.market_group},
                )
            )
            market_node = _MARKET_NODE.get(context.market_group.upper())
            if market_node:
                edges.append(
                    OntologyEdge(
                        source_id=node_id,
                        target_id=market_node,
                        relation="BELONGS_TO",
                        prior_strength=1.0,
                        learnable=False,
                    )
                )
            group = _global_group_for_sector(context.sector)
            macro_node = self._macro_node_for_group(group) if group else None
            if macro_node:
                edges.append(
                    OntologyEdge(
                        source_id=macro_node,
                        target_id=node_id,
                        relation="INFLUENCES",
                        prior_strength=_GENERATED_INFLUENCE_PRIOR,
                        learnable=True,
                        lag_min=0,
                        lag_max=960,
                        attributes={"generated": True, "global_group": group},
                    )
                )

        stock_nodes: dict[str, str] = {}
        for observation in stocks:
            node_id = _stock_node_id(observation.ticker)
            if node_id in known:
                continue
            if budget <= 0:
                dropped += 1
                continue
            budget -= 1
            known.add(node_id)
            stock_nodes[observation.ticker] = node_id
            nodes.append(
                OntologyNode(
                    node_id=node_id,
                    node_type="Stock",
                    label=observation.ticker,
                    attributes={
                        "sector": observation.sector or "",
                        "market_group": observation.market_group,
                    },
                )
            )
            sector_node = (
                sector_nodes.get(observation.sector)
                if observation.sector
                else None
            ) or (_sector_node_id(observation.sector) if observation.sector else None)
            if sector_node and sector_node in known:
                edges.append(
                    OntologyEdge(
                        source_id=node_id,
                        target_id=sector_node,
                        relation="BELONGS_TO",
                        prior_strength=1.0,
                        learnable=False,
                    )
                )
                # The sector's state has to be able to reach the stock, and BELONGS_TO is
                # structural and one-way. INFLUENCES carries the numeric relationship.
                edges.append(
                    OntologyEdge(
                        source_id=sector_node,
                        target_id=node_id,
                        relation="INFLUENCES",
                        prior_strength=_GENERATED_INFLUENCE_PRIOR,
                        learnable=True,
                        attributes={"generated": True},
                    )
                )
            venue = str(observation.venue or "").upper()
            if venue in known:
                edges.append(
                    OntologyEdge(
                        source_id=node_id,
                        target_id=venue,
                        relation="TRADED_ON",
                        prior_strength=1.0,
                        learnable=False,
                    )
                )
            market_node = _MARKET_NODE.get(str(observation.market_group).upper())
            if market_node and market_node in known:
                # TRADED_ON/BELONGS_TO correctly describe stock -> venue -> market,
                # but they are structural and directional.  They cannot carry the
                # market state back to the stock.  A learnable market -> stock edge is
                # the causal path through which US factors affect a US candidate and
                # KR factors affect a KR candidate in the shared heterogeneous GNN.
                edges.append(
                    OntologyEdge(
                        source_id=market_node,
                        target_id=node_id,
                        relation="INFLUENCES",
                        prior_strength=_GENERATED_INFLUENCE_PRIOR,
                        learnable=True,
                        attributes={
                            "generated": True,
                            "reason": "target_market_context",
                        },
                    )
                )

        for observation in stocks:
            source = stock_nodes.get(observation.ticker)
            if source is None:
                continue
            for peer in observation.peers:
                target = stock_nodes.get(peer) or _stock_node_id(peer)
                if target not in known or target == source:
                    continue
                edges.append(
                    OntologyEdge(
                        source_id=source,
                        target_id=target,
                        relation="CORRELATED_WITH",
                        prior_strength=_GENERATED_PEER_PRIOR,
                        learnable=True,
                        attributes={"generated": True},
                    )
                )

        if temporal is not None:
            phase_node = _PHASE_NODE.get(temporal.session_phase.value)
            market_node = _MARKET_NODE.get(temporal.market_group.upper())
            if phase_node and market_node:
                edges.append(
                    OntologyEdge(
                        source_id=phase_node,
                        target_id=market_node,
                        relation="INFLUENCES",
                        prior_strength=_GENERATED_TEMPORAL_PRIOR,
                        learnable=True,
                        attributes={"generated": True},
                    )
                )
            if temporal.expiry_context.value != "NONE" and market_node:
                edges.append(
                    OntologyEdge(
                        source_id="EXPIRY_WINDOW",
                        target_id=market_node,
                        relation="INFLUENCES",
                        prior_strength=_GENERATED_TEMPORAL_PRIOR,
                        learnable=True,
                        attributes={"generated": True},
                    )
                )
        if domestic_context is not None and domestic_context.venue_divergence:
            edges.append(
                OntologyEdge(
                    source_id="NXT",
                    target_id="KR_MARKET",
                    relation="INCREASES_RISK",
                    prior_strength=min(1.0, float(domestic_context.venue_divergence)),
                    learnable=True,
                    attributes={"generated": True, "reason": "venue_divergence"},
                )
            )

        # Duplicate edge ids can arise when a generated edge repeats a configured one;
        # the configured edge wins, because a hand-written prior outranks an inferred one.
        unique: dict[str, OntologyEdge] = {}
        for edge in edges:
            unique.setdefault(edge.edge_id, edge)
        return nodes, list(unique.values()), dropped

    def _macro_node_for_group(self, group: str) -> str | None:
        for node in self._base_graph.nodes_of_type("MacroFactor"):
            if node.attributes.get(self._MACRO_GROUP_ATTRIBUTE) == group:
                return node.node_id
        return None

    # ------------------------------------------------------------------ #
    # features
    # ------------------------------------------------------------------ #
    def _feature_values(
        self,
        graph: MarketGraph,
        *,
        temporal: TemporalSnapshot | None,
        global_context: GlobalContext | None,
        domestic_context: DomesticContext | None,
        sector_contexts: Sequence[SectorContext],
        stocks: Sequence[StockNodeObservation],
        active_risk_conditions: Mapping[str, float],
        regime_prior: Mapping[str, float],
    ) -> dict[str, dict[str, float]]:
        values: dict[str, dict[str, float]] = {}

        if domestic_context is not None:
            values["KR_MARKET"] = _drop_none(
                {
                    "direction": domestic_context.direction,
                    "breadth": domestic_context.breadth,
                    "liquidity": domestic_context.liquidity,
                    "volatility": _scaled(domestic_context.volatility, 0.01),
                    "flow": domestic_context.flow,
                    "leadership": domestic_context.leadership,
                    "venue_divergence": domestic_context.venue_divergence,
                    "global_agreement": domestic_context.global_agreement,
                    "confidence": domestic_context.confidence,
                }
            )
        if global_context is not None:
            values["US_MARKET"] = _drop_none(
                {
                    "direction": global_context.direction,
                    "volatility": global_context.volatility,
                    "confidence": global_context.confidence,
                    "global_agreement": global_context.global_alignment,
                }
            )
            for node in graph.nodes_of_type("MacroFactor"):
                group_name = node.attributes.get(self._MACRO_GROUP_ATTRIBUTE)
                score = global_context.groups.get(str(group_name)) if group_name else None
                if score is None:
                    continue
                values[node.node_id] = _drop_none(
                    {
                        "score": score.score,
                        "raw_score": score.raw_score,
                        "momentum": score.momentum,
                        "freshness": score.freshness,
                        "confidence": global_context.confidence,
                    }
                )

        for context in sector_contexts:
            node_id = _sector_node_id(context.sector)
            values[node_id] = _drop_none(
                {
                    "return": _scaled(context.sector_return, 0.01),
                    "breadth": context.breadth,
                    "volume_z": _scaled(context.volume_z, 1.0),
                    "volatility": _scaled(context.volatility, 0.01),
                    "relative_strength": _scaled(context.relative_strength, 0.01),
                    "foreign_flow": context.foreign_flow,
                    "leader_strength": _scaled(context.leader_strength, 0.02),
                    "leader_concentration": context.leader_concentration,
                    "global_alignment": context.global_alignment,
                    "confidence": context.confidence,
                }
            )

        for observation in stocks:
            values[_stock_node_id(observation.ticker)] = _drop_none(
                observation.feature_values()
            )

        if temporal is not None:
            active_phase = _PHASE_NODE.get(temporal.session_phase.value)
            for node in graph.nodes_of_type("TemporalEntity"):
                is_active = node.node_id == active_phase or (
                    node.node_id == "EXPIRY_WINDOW"
                    and temporal.expiry_context.value != "NONE"
                )
                values[node.node_id] = _drop_none(
                    {
                        "active": 1.0 if is_active else 0.0,
                        "session_progress": temporal.session_progress,
                        "minutes_from_open": _scaled(temporal.minutes_from_open, 120.0),
                        "minutes_to_close": _scaled(temporal.minutes_to_close, 120.0),
                        # Encoded on a circle so Monday and Friday are not five units
                        # apart on a line the model would read as an ordering.
                        "day_of_week": float(
                            np.sin(2.0 * np.pi * temporal.day_of_week / 7.0)
                        ),
                        "holiday_adjacent": 1.0 if temporal.holiday_adjacent else 0.0,
                        "month_end": 1.0 if temporal.month_end else 0.0,
                        "quarter_end": 1.0 if temporal.quarter_end else 0.0,
                        "expiry": 0.0
                        if temporal.expiry_context.value == "NONE"
                        else 1.0,
                    }
                )

        for node in graph.nodes_of_type("MarketRegime"):
            probability = _finite(regime_prior.get(node.node_id))
            values[node.node_id] = {"prior_probability": probability or 0.0}

        for node in graph.nodes_of_type("RiskCondition"):
            severity = _finite(active_risk_conditions.get(node.node_id))
            values[node.node_id] = {
                "active": 1.0 if severity else 0.0,
                "severity": severity or 0.0,
            }

        for node in graph.nodes_of_type("Strategy"):
            suitable = sum(
                edge.effective_weight
                for edge in graph.edges(relation="SUITABLE_FOR", target=node.node_id)
            )
            unsuitable = sum(
                edge.effective_weight
                for edge in graph.edges(relation="UNSUITABLE_FOR", target=node.node_id)
            )
            total = suitable + unsuitable
            values[node.node_id] = {
                "ontology_suitability": suitable / total if total else 0.0,
                "ontology_unsuitability": unsuitable / total if total else 0.0,
            }

        for node in graph.nodes_of_type("Venue"):
            divergence = (
                domestic_context.venue_divergence
                if domestic_context is not None and node.node_id in {"KRX", "NXT"}
                else None
            )
            values[node.node_id] = _drop_none(
                {"active": 1.0, "divergence": divergence}
            )

        return values


def _drop_none(values: Mapping[str, float | None]) -> dict[str, float]:
    return {
        name: float(number)
        for name, raw in values.items()
        if (number := _finite(raw)) is not None
    }


def _pad_relation_tensor(tensor: np.ndarray, size: int, *, fill: float) -> np.ndarray:
    relations, height, width = tensor.shape
    if height == size and width == size:
        return tensor
    padded = np.full((relations, size, size), fill, dtype=np.float32)
    padded[:, :height, :width] = tensor
    return padded


def _sector_node_id(sector: str) -> str:
    slug = "".join(
        character if character.isalnum() else "_"
        for character in str(sector or "").upper()
    ).strip("_")
    return f"SECTOR::{slug or 'UNKNOWN'}"


def _stock_node_id(ticker: str) -> str:
    return f"STOCK::{str(ticker or '').strip().upper()}"
