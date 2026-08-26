from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
import json
import os
from pathlib import Path
from typing import Any

import numpy as np

from app.cost import TradingCostEngine
from app.features.strategy_graph_context import (
    STRATEGY_GRAPH_CONTEXT_DIM,
    STRATEGY_GRAPH_CONTEXT_FIELDS,
    STRATEGY_GRAPH_CONTEXT_SCHEMA,
    as_context_mapping,
    build_strategy_graph_context,
    context_index,
)
from app.models.strategy_utility import (
    FixedShapeStrategyUtilityModel,
    StrategyUtilityModelConfig,
)
from app.models.strategy_utility.openvino_runtime import OpenVinoStrategyUtilityRuntime
from app.models.strategy_utility.strategy_graph import (
    RELATION_NAMES,
    STRATEGY_NODE_COUNT,
    diagonal_strategy_mask,
    strategy_node_features,
    strategy_relation_adjacency,
    strategy_ids_for_market,
)
from app.ontology.operational_gate import (
    ClosedWorldOntologyGate,
    OperationalFact,
    OperationalOntologySnapshot,
    StrategyGateRule,
)
from app.routing.shadow_comparison import (
    ShadowComparison,
    ShadowComparisonRecorder,
    ShadowDecision,
)
from app.routing.gnn_realtime_trust import (
    GnnRealtimeTrust,
    GnnRealtimeTrustEvaluator,
    default_gnn_realtime_trust_evaluator,
)
from app.routing.strategy_router import StrategyRouter
from app.strategy.catalog import is_short_strategy
from app.strategy.catalog import STRATEGY_IDS
from app.trading.contracts import StrategyUtilityEvidence


@dataclass(frozen=True)
class SlowIntelligenceSnapshot:
    snapshot_id: str
    symbol: str
    as_of: datetime
    valid_until: datetime
    feature_snapshot_id: str
    features: tuple[float, ...]
    data_fresh: bool
    tradable: bool
    allowed_strategy_ids: tuple[str, ...]
    feature_schema_name: str = "unspecified"
    #: Price the cost estimate is anchored on, in the instrument's own currency.
    #:
    #: Carried explicitly because the context vector no longer contains one. v4
    #: put the raw close in slot 0 and the cost engine read
    #: ``features[0] * 100_000`` — so removing the price level (an instrument
    #: identity leak) would silently have turned the reference price into
    #: whatever landed in slot 0. Zero means "not supplied", and the legacy
    #: derivation below still applies for pre-v5 snapshots.
    reference_price: float = 0.0


@dataclass(frozen=True)
class ShadowIntelligenceResult:
    ontology_snapshot_id: str
    cpu_evidence: tuple[StrategyUtilityEvidence, ...]
    npu_evidence: tuple[StrategyUtilityEvidence, ...]
    comparison: ShadowComparison


def slow_snapshot_from_live_feature_frame(
    frame: Any,
    *,
    allowed_strategy_ids: tuple[str, ...] = STRATEGY_IDS,
    valid_for_seconds: float = 5.0,
) -> SlowIntelligenceSnapshot:
    """Adapt the production live feature frame to the trained GNN contract.

    US daytime quotes can arrive through the REST fast-poll path rather than
    the event-driven websocket runtime.  Building the graph context here lets
    both ingestion paths feed the same ontology+GNN pipeline without inventing a
    second feature schema.

    The fields come from ``graph:``-prefixed entries the frame builder computes
    from completed minute bars, and they are read STRICTLY. The previous version
    assembled the vector here out of ``values.get(name, default)`` against the
    live model's feature dictionary; when that schema dropped columns, five slots
    silently became 0.0 at serving time while training kept supplying real values
    for the same positions. A raise (caught per symbol by the caller and recorded
    as a shadow error) is the correct response to an unbuildable context.
    """
    frame.validate()
    symbol = str(frame.symbol or "").strip().upper()
    as_of = frame.decision_time
    if as_of.tzinfo is None:
        as_of = as_of.replace(tzinfo=timezone.utc)
    price = max(0.0, float(frame.mark_price or 0.0))
    values = (
        frame.as_context_dict()
        if hasattr(frame, "as_context_dict")
        else frame.as_feature_dict()
    )
    context = build_strategy_graph_context(
        {
            name: values[f"graph:{name}"]
            for name in STRATEGY_GRAPH_CONTEXT_FIELDS
            if f"graph:{name}" in values
        }
    )
    tick_record_ids = tuple(
        getattr(frame.provenance, "tick_record_ids", ()) or ()
    )
    record_id = str(
        getattr(frame.provenance, "orderbook_record_id", "")
        or (tick_record_ids[-1] if tick_record_ids else "")
        or int(as_of.timestamp() * 1000)
    )
    return SlowIntelligenceSnapshot(
        snapshot_id=f"live-strategy:{symbol}:{record_id}",
        symbol=symbol,
        as_of=as_of,
        valid_until=as_of + timedelta(seconds=max(1.0, valid_for_seconds)),
        feature_snapshot_id=(
            f"live:{frame.feature_schema_hash}:{record_id}"
        ),
        features=context,
        data_fresh=True,
        tradable=(
            price > 0
            and context[context_index("spread_bps_scaled")] >= 0.0
        ),
        allowed_strategy_ids=allowed_strategy_ids,
        feature_schema_name=STRATEGY_GRAPH_CONTEXT_SCHEMA,
        reference_price=price,
    )


class ShadowIntelligenceService:
    """Periodic, order-free ontology + utility inference orchestration."""

    def __init__(
        self,
        *,
        feature_dim: int = 12,
        minimum_interval_seconds: float = 1.0,
        enable_npu_comparison: bool = False,
        comparison_path: str | Path = "logs/refactor-shadow-comparison.jsonl",
        trust_evaluator: GnnRealtimeTrustEvaluator | None = None,
    ) -> None:
        self.context_feature_dim = feature_dim
        # Graph mode is keyed off the width the strategy-graph contract declares,
        # not a hardcoded set of historical widths. The v3/v4 widths (27/28) stay
        # recognised so an operator running an old checkpoint still gets graph
        # mode and the schema check below, rather than silently degrading to the
        # single-node path.
        self.graph_mode = feature_dim in {27, 28, STRATEGY_GRAPH_CONTEXT_DIM}
        config = StrategyUtilityModelConfig(
            batch_size=1,
            time_steps=1,
            max_nodes=STRATEGY_NODE_COUNT if self.graph_mode else 1,
            feature_dim=(
                feature_dim + STRATEGY_NODE_COUNT
                if self.graph_mode
                else feature_dim
            ),
            relation_count=len(RELATION_NAMES) if self.graph_mode else 1,
            strategy_count=len(STRATEGY_IDS),
            hidden_dim=16,
            seed=17,
        )
        checkpoint_path = Path(
            os.getenv(
                "REFACTOR_GNN_CHECKPOINT",
                "data/models/strategy_utility/rgcn_shadow.npz",
            )
        )
        self.checkpoint_path = checkpoint_path
        checkpoint_model = None
        self.checkpoint_error: str | None = None
        if checkpoint_path.exists():
            try:
                checkpoint_model = FixedShapeStrategyUtilityModel.load_checkpoint(checkpoint_path)
            except Exception as exc:  # noqa: BLE001 - an unloadable model fails closed.
                # A HEAD-SHAPE mismatch is a schema change, not corruption, and the two
                # send an operator in different directions: "corrupt" means look for a
                # damaged file, while "schema" means retrain against the current
                # contract. The borrow channels widened the head from 8 to 11, so every
                # pre-existing checkpoint lands here — reporting that as corruption
                # would send everyone hunting a file that is perfectly intact.
                message = str(exc)
                self.checkpoint_error = (
                    "GNN_HEAD_SCHEMA_MISMATCH"
                    if "strategy_heads" in message or "no_trade_head" in message
                    else "GNN_CHECKPOINT_CORRUPT"
                )
        self.checkpoint_loaded = (
            checkpoint_model is not None and checkpoint_model.config == config
        )
        self.checkpoint_contract_reasons: tuple[str, ...] = ()
        if checkpoint_model is not None and checkpoint_model.config != config:
            reasons: list[str] = []
            if checkpoint_model.config.strategy_count != len(STRATEGY_IDS):
                reasons.append("GNN_STRATEGY_CATALOG_MISMATCH")
            if checkpoint_model.config.feature_dim != config.feature_dim:
                reasons.append("GNN_FEATURE_SCHEMA_MISMATCH")
            if (
                checkpoint_model.config.max_nodes != config.max_nodes
                or checkpoint_model.config.relation_count != config.relation_count
            ):
                reasons.append("GNN_GRAPH_SCHEMA_MISMATCH")
            if not reasons:
                reasons.append("GNN_CHECKPOINT_SHAPE_MISMATCH")
            self.checkpoint_contract_reasons = tuple(reasons)
        self.model_input_schema = "unspecified"
        self.model_strategy_ids: tuple[str, ...] = ()
        #: Strategies whose upside (MFE) head was actually taught. Empty means no
        #: strategy may contribute a positive net-edge forecast — fail closed, so
        #: a checkpoint that never reported its supervision cannot be read as
        #: having proved any.
        self.upside_supervised_strategy_ids: tuple[str, ...] = ()
        self.upside_authorized_strategy_markets: dict[str, tuple[str, ...]] = {}
        self.live_authorized = False
        self.live_authorized_markets: tuple[str, ...] = ()
        self.authorization_scope = "none"
        self.checkpoint_hash: str | None = None
        if self.checkpoint_loaded:
            try:
                metadata = json.loads(
                    checkpoint_path.with_suffix(".json").read_text(encoding="utf-8")
                )
                self.model_input_schema = str(
                    metadata.get("input_feature_schema")
                    or (
                        "counterfactual_quantiles_v1"
                        if metadata.get("method")
                        == "causal_feature_encoder_plus_ridge_calibrated_heads"
                        else "unspecified"
                    )
                )
                self.model_strategy_ids = tuple(
                    str(item) for item in metadata.get("strategy_ids", ())
                )
                self.live_authorized = bool(metadata.get("live_authorized"))
                declared_markets = metadata.get("live_authorized_markets")
                if isinstance(declared_markets, (list, tuple)):
                    self.live_authorized_markets = tuple(
                        market
                        for market in (str(item).upper() for item in declared_markets)
                        if market in {"KRX", "US"}
                    )
                    self.live_authorized = bool(self.live_authorized_markets)
                elif self.live_authorized:
                    # Legacy cards carried only the aggregate flag. Newly trained
                    # cards always carry explicit market authority.
                    self.live_authorized_markets = ("KRX", "US")
                self.authorization_scope = str(
                    metadata.get("authorization_scope") or "none"
                )
                self.checkpoint_hash = (
                    str(metadata.get("checkpoint_hash"))
                    if metadata.get("checkpoint_hash")
                    else None
                )
                self.upside_supervised_strategy_ids = (
                    _upside_supervised_strategy_ids(metadata)
                )
                self.upside_authorized_strategy_markets = (
                    _upside_authorized_strategy_markets(metadata)
                )
            except (OSError, ValueError, json.JSONDecodeError):
                self.model_input_schema = "unknown"
                self.checkpoint_error = "GNN_CHECKPOINT_METADATA_INVALID"
        self.model = (
            checkpoint_model
            if self.checkpoint_loaded
            else FixedShapeStrategyUtilityModel(config)
        )
        self.cpu = OpenVinoStrategyUtilityRuntime(self.model, requested_device="CPU")
        self.npu = (
            OpenVinoStrategyUtilityRuntime(self.model, requested_device="NPU")
            if enable_npu_comparison
            else None
        )
        self.minimum_interval = timedelta(seconds=max(0, minimum_interval_seconds))
        self.last_run: dict[str, datetime] = {}
        self.recorder = ShadowComparisonRecorder(comparison_path)
        # The web dashboard and live shadow route consume the same trust
        # evidence.  Accepting the process-wide evaluator prevents both paths
        # from independently rescanning the large comparison journal whenever
        # their caches expire.  Standalone/offline callers keep an isolated
        # evaluator by default.
        if trust_evaluator is not None:
            self.trust_evaluator = trust_evaluator
        elif Path(comparison_path) == Path("logs/refactor-shadow-comparison.jsonl"):
            self.trust_evaluator = default_gnn_realtime_trust_evaluator()
        else:
            # Tests and offline experiments using a private comparison journal
            # must remain isolated from live process state.
            self.trust_evaluator = GnnRealtimeTrustEvaluator(
                comparison_path=comparison_path,
                database_path=os.getenv(
                    "REALTIME_MARKET_DATA_DB",
                    "data/store/realtime_market_data.sqlite3",
                ),
                checkpoint_metadata_path=checkpoint_path.with_suffix(".json"),
                stale_while_refresh=True,
            )
        self.cost_engine = TradingCostEngine()
        self.gate = ClosedWorldOntologyGate()
        self.router = StrategyRouter(
            minimum_net_edge_bps=max(
                0.0,
                float(
                    os.getenv(
                        "GNN_ROUTER_MIN_NET_EDGE_BPS",
                        str(
                            float(
                                os.getenv(
                                    "REALTIME_MIN_BUY_NET_RETURN_US",
                                    "0.0005",
                                )
                            )
                            * 10_000.0
                        ),
                    )
                ),
            )
        )

    def _checkpoint_live_authorized_for(self, symbol: str) -> bool:
        market = "KRX" if str(symbol).isdigit() and len(str(symbol)) == 6 else "US"
        return market in self.live_authorized_markets

    def evaluate(
        self,
        snapshot: SlowIntelligenceSnapshot,
        *,
        legacy_action: str = "NO_TRADE",
    ) -> ShadowIntelligenceResult | None:
        previous = self.last_run.get(snapshot.symbol)
        if previous is not None and snapshot.as_of - previous < self.minimum_interval:
            return None
        if len(snapshot.features) != self.context_feature_dim:
            raise ValueError("slow intelligence feature dimension mismatch")
        self.last_run[snapshot.symbol] = snapshot.as_of
        ontology = self._ontology(snapshot)
        model_block_reasons = self._model_block_reasons(snapshot)
        if model_block_reasons:
            decisions = [
                ShadowDecision("legacy", legacy_action, None, None, ("LEGACY_OBSERVED",)),
                ShadowDecision(
                    "ontology",
                    "GATE_ONLY" if ontology.allowed_strategy_ids else "NO_TRADE",
                    None,
                    None,
                    (
                        ("ONTOLOGY_GATE_ONLY",)
                        if ontology.allowed_strategy_ids
                        else ("NO_ONTOLOGY_ADMISSIBLE_STRATEGY",)
                    ),
                ),
                ShadowDecision("cpu_gnn", "NO_TRADE", None, None, model_block_reasons),
            ]
            comparison = self.recorder.compare(
                correlation_id=snapshot.snapshot_id,
                symbol=snapshot.symbol,
                as_of=snapshot.as_of,
                decisions=tuple(decisions),
            )
            return ShadowIntelligenceResult(
                ontology_snapshot_id=ontology.snapshot_id,
                cpu_evidence=(),
                npu_evidence=(),
                comparison=comparison,
            )
        inputs = self._inputs(snapshot, ontology.allowed_strategy_ids)
        cpu_output = self.cpu.infer(*inputs)
        cpu_evidence = self._evidence(snapshot, ontology, cpu_output, "openvino-cpu")
        cpu_route = self.router.route(
            as_of=snapshot.as_of,
            symbol=snapshot.symbol,
            ontology=ontology,
            evidence=cpu_evidence,
        )
        realtime_trust = self.trust_evaluator.evaluate(snapshot.as_of)
        npu_evidence: tuple[StrategyUtilityEvidence, ...] = ()
        checkpoint_live_authorized = self._checkpoint_live_authorized_for(
            snapshot.symbol
        )
        decisions = [
            ShadowDecision("legacy", legacy_action, None, None, ("LEGACY_OBSERVED",)),
            ShadowDecision(
                "ontology",
                "GATE_ONLY" if ontology.allowed_strategy_ids else "NO_TRADE",
                None,
                None,
                (
                    ("ONTOLOGY_GATE_ONLY",)
                    if ontology.allowed_strategy_ids
                    else ("NO_ONTOLOGY_ADMISSIBLE_STRATEGY",)
                ),
            ),
            _shadow_route(
                "cpu_gnn",
                cpu_route,
                realtime_trust=realtime_trust,
                evidence=cpu_evidence,
                checkpoint_hash=self.checkpoint_hash,
                checkpoint_live_authorized=checkpoint_live_authorized,
            ),
        ]
        if self.npu is not None:
            npu_output = self.npu.infer(*inputs)
            npu_evidence = self._evidence(snapshot, ontology, npu_output, "openvino-npu")
            npu_route = self.router.route(
                as_of=snapshot.as_of,
                symbol=snapshot.symbol,
                ontology=ontology,
                evidence=npu_evidence,
            )
            decisions.append(
                _shadow_route(
                    "npu_gnn",
                    npu_route,
                    realtime_trust=realtime_trust,
                    evidence=npu_evidence,
                    checkpoint_hash=self.checkpoint_hash,
                    checkpoint_live_authorized=checkpoint_live_authorized,
                )
            )
        comparison = self.recorder.compare(
            correlation_id=snapshot.snapshot_id,
            symbol=snapshot.symbol,
            as_of=snapshot.as_of,
            decisions=tuple(decisions),
            validation_candidates=_validation_candidates(
                cpu_evidence,
                checkpoint_hash=self.checkpoint_hash,
                realtime_trust=realtime_trust,
                checkpoint_live_authorized=checkpoint_live_authorized,
            ),
        )
        return ShadowIntelligenceResult(
            ontology_snapshot_id=ontology.snapshot_id,
            cpu_evidence=cpu_evidence,
            npu_evidence=npu_evidence,
            comparison=comparison,
        )

    def _model_block_reasons(
        self,
        snapshot: SlowIntelligenceSnapshot,
    ) -> tuple[str, ...]:
        """Reject checkpoint outputs whose provenance cannot support this live frame."""
        if self.checkpoint_error:
            return (self.checkpoint_error,)
        if not self.checkpoint_path.exists():
            return ("GNN_CHECKPOINT_MISSING",)
        if not self.checkpoint_loaded:
            return self.checkpoint_contract_reasons or ("GNN_CHECKPOINT_SHAPE_MISMATCH",)
        reasons: list[str] = []
        if self.model_strategy_ids != STRATEGY_IDS:
            reasons.append("GNN_STRATEGY_CATALOG_MISMATCH")
        if self.model_input_schema != snapshot.feature_schema_name:
            reasons.append("GNN_FEATURE_SCHEMA_MISMATCH")
        # Offline promotion controls order authority, not shadow inference.
        # Blocking inference here creates a permanent deadlock: an unpromoted
        # checkpoint emits no validation forecasts, so it can never accumulate
        # the forward outcomes used by the realtime trust gate.  Structural
        # contract failures still block above; a quality-gate failure continues
        # through the validation-only path and is kept non-executable below.
        return tuple(reasons)

    def _ontology(self, snapshot: SlowIntelligenceSnapshot):
        facts = {
            "data_fresh": _fact(snapshot, "data_fresh", snapshot.data_fresh),
            "tradable": _fact(snapshot, "tradable", snapshot.tradable),
        }
        compatibility, expressible = _compatibility_with_provenance(snapshot.features)
        market_strategy_ids = set(strategy_ids_for_market(snapshot.symbol))
        for strategy_id in STRATEGY_IDS:
            compatibility_score = compatibility.get(strategy_id, 0.0)
            unavailable = strategy_id not in expressible
            facts[f"allow:{strategy_id}"] = _fact(
                snapshot,
                f"allow:{strategy_id}",
                strategy_id in snapshot.allowed_strategy_ids
                and strategy_id in market_strategy_ids,
            )
            facts[f"compat:{strategy_id}"] = _fact(
                snapshot,
                f"compat:{strategy_id}",
                compatibility_score,
            )
            facts[f"compat_unavailable:{strategy_id}"] = _fact(
                snapshot,
                f"compat_unavailable:{strategy_id}",
                unavailable,
            )
            facts[f"compatible:{strategy_id}"] = _fact(
                snapshot,
                f"compatible:{strategy_id}",
                compatibility_score > 0.0,
            )
        operational = OperationalOntologySnapshot(
            snapshot_id=f"ontology:{snapshot.snapshot_id}",
            symbol=snapshot.symbol,
            as_of=snapshot.as_of,
            valid_until=snapshot.valid_until,
            facts=facts,
        )
        rules = tuple(
            StrategyGateRule(
                strategy_id,
                required_true=(
                    "data_fresh",
                    "tradable",
                    f"allow:{strategy_id}",
                    f"compatible:{strategy_id}",
                ),
                compatibility_weights={f"compat:{strategy_id}": 1.0},
            )
            for strategy_id in STRATEGY_IDS
        )
        return self.gate.evaluate(operational, rules)

    def _inputs(self, snapshot: SlowIntelligenceSnapshot, allowed: tuple[str, ...]):
        if self.graph_mode:
            allowed = tuple(
                strategy_id
                for strategy_id in allowed
                if strategy_id in set(strategy_ids_for_market(snapshot.symbol))
            )
            x = strategy_node_features(snapshot.features).reshape(
                1,
                1,
                STRATEGY_NODE_COUNT,
                -1,
            )
            adjacency = strategy_relation_adjacency(
                allowed,
                market=snapshot.symbol,
            ).reshape(
                1,
                1,
                len(RELATION_NAMES),
                STRATEGY_NODE_COUNT,
                STRATEGY_NODE_COUNT,
            )
            node_mask = np.asarray(
                [[[
                    1.0 if strategy_id in allowed else 0.0
                    for strategy_id in STRATEGY_IDS
                ]]],
                dtype=np.float32,
            )
            strategy_mask = diagonal_strategy_mask(allowed).reshape(
                1,
                STRATEGY_NODE_COUNT,
                len(STRATEGY_IDS),
            )
            return x, adjacency, node_mask, strategy_mask
        x = np.asarray(snapshot.features, dtype=np.float32).reshape(1, 1, 1, -1)
        adjacency = np.ones((1, 1, 1, 1, 1), dtype=np.float32)
        node_mask = np.ones((1, 1, 1), dtype=np.float32)
        strategy_mask = np.asarray(
            [[[1.0 if strategy in allowed else 0.0 for strategy in STRATEGY_IDS]]],
            dtype=np.float32,
        )
        return x, adjacency, node_mask, strategy_mask

    def _evidence(self, snapshot, ontology, output, version):
        values = []
        is_krx = snapshot.symbol.isdigit() and len(snapshot.symbol) == 6
        reference_price = max(1e-9, _reference_price(snapshot))
        baseline_cost = self.cost_engine.estimate(
            symbol=snapshot.symbol,
            market="KR" if is_krx else "US",
            venue="KRX" if is_krx else "NASD",
            instrument_type="domestic_stock" if is_krx else "overseas_stock",
            entry_price=reference_price,
            expected_exit_price=reference_price,
            quantity=1,
        ).total_cost_rate * 10_000.0
        for index, strategy_id in enumerate(STRATEGY_IDS):
            node_index = index if self.graph_mode else 0
            allowed = strategy_id in ontology.allowed_strategy_ids
            gross = float(output.gross_return_bps[0, node_index, index])
            cost = max(
                baseline_cost,
                float(output.cost_bps[0, node_index, index]),
            )
            probability = float(output.probability_success[0, node_index, index])
            mfe = float(output.mfe_bps[0, node_index, index])
            # Drop the upside term when nothing taught it.
            #
            # The decoder builds the expectation as
            # ``probability * mfe - (1 - probability) * mae``, and MFE is trained
            # only on realized PROFITABLE fills. On the 2026-08-03 checkpoint that
            # is 0-21 rows per strategy against 5-112 for MAE, so for most
            # strategies the only positive term in the forecast is an untrained
            # head. Live measurement: forecasts this model called positive
            # realized -39 to -163bps and were right 13-43% of the time, while its
            # negative forecasts were right ~97%. The downside half is evidence;
            # the upside half was noise wearing the same units.
            #
            # Removing ``probability * mfe`` leaves ``-(1 - probability) * mae``
            # minus costs — an estimate built only from rows that exist. It is
            # necessarily non-positive, so StrategyRouter rejects it as
            # NON_POSITIVE_NET_EDGE and it stays in the shadow log as a
            # calibration sample instead of a fabricated opportunity.
            #
            # It comes off GROSS, not off net: the contract in
            # ``StrategyUtilityEvidence`` requires net == gross - cost, and a gross
            # forecast carrying an unsupported upside is exactly as fabricated as
            # the net one. Both numbers have to shed it together.
            market_key = "KRX" if is_krx else "US"
            if market_key not in self.upside_authorized_strategy_markets.get(
                strategy_id, ()
            ):
                gross -= probability * mfe
            values.append(
                StrategyUtilityEvidence(
                    evidence_id=f"{version}:{snapshot.snapshot_id}:{strategy_id}",
                    as_of=snapshot.as_of,
                    symbol=snapshot.symbol,
                    strategy_id=strategy_id,
                    ontology_allowed=allowed,
                    hard_block_reasons=ontology.blocked_strategy_reasons.get(strategy_id, ()),
                    compatibility_score=ontology.compatibility_scores[strategy_id],
                    probability_success=probability,
                    expected_gross_return_bps=gross,
                    expected_cost_bps=cost,
                    expected_net_return_bps=gross - cost,
                    expected_adverse_excursion_bps=float(output.mae_bps[0, node_index, index]),
                    expected_favorable_excursion_bps=mfe,
                    fill_probability=float(output.fill_probability[0, node_index, index]),
                    expected_holding_seconds=float(output.holding_seconds[0, node_index, index]),
                    aleatoric_uncertainty=float(output.aleatoric_uncertainty[0, node_index, index]),
                    epistemic_uncertainty_or_proxy=0,
                    utility=float(output.utility[0, node_index, index]),
                    model_version=version,
                    feature_snapshot_id=snapshot.feature_snapshot_id,
                    ontology_snapshot_id=ontology.snapshot_id,
                    explanation_paths=ontology.explanation_paths[strategy_id],
                )
            )
        return tuple(values)


def _fact(snapshot, name: str, value: bool | float | int | str) -> OperationalFact:
    return OperationalFact(
        name=name,
        value=value,
        observed_at=snapshot.as_of,
        valid_from=snapshot.as_of,
        valid_until=snapshot.valid_until,
        source="slow-intelligence-snapshot",
        confidence=1,
    )


# Why a strategy's compatibility is zero. A zero that means "this snapshot
# cannot express the relation" is a DIFFERENT finding from a zero that means "the
# relation evaluated to nothing", and collapsing the two is how eight of sixteen
# strategies became permanently unreachable without anything reporting it: the
# gate requires ``compatible:{id}``, ``compatibility.get(id, 0.0)`` returned the
# default for any id missing from the map, and a missing key looked exactly like
# a computed zero. These reasons are what the dashboard needs to separate
# "measured no edge" from "never evaluated".
COMPATIBILITY_UNAVAILABLE_REASONS: dict[str, str] = {
    # Cross-sectional: a residual needs peer returns, and one symbol's tick
    # window cannot produce them. The electing layer computes them and passes
    # them in ElectionContext.residual_return_*; this snapshot has no slot.
    "residual_relative_strength": "CONTEXT_UNAVAILABLE:PEER_RESIDUALS",
    "residual_relative_weakness": "CONTEXT_UNAVAILABLE:PEER_RESIDUALS",
    "cross_sectional_relative_strength": "CONTEXT_UNAVAILABLE:PEER_RANKING",
    # Point-in-time news: an event relation needs the event and its age.
    "event_momentum": "CONTEXT_UNAVAILABLE:EVENT_FACTS",
}


def compatibility_coverage() -> dict[str, str]:
    """Per-strategy status of the relation map: computed, or why it cannot be.

    Exists so the omission this map suffered cannot recur silently — a strategy
    added to the catalogue with neither a relation nor a documented reason shows
    up here as ``UNDECLARED`` and fails the contract test.

    It asks :func:`_named_relation_scores` which ids have a relation, NOT
    :func:`_strategy_compatibility`. The latter applies the closed-world fill
    before returning, so every catalogue id was present in its result and the
    ``UNDECLARED`` branch below was unreachable — the guard reported all 23 ids as
    ``COMPUTED`` while five of them had no relation at all and were therefore
    permanently blocked by the ontology gate. The guard built to catch exactly that
    omission was defeated by the line that makes the omission invisible.
    """
    computed = set(_named_relation_scores(tuple([0.0] * STRATEGY_GRAPH_CONTEXT_DIM)))
    coverage: dict[str, str] = {}
    for strategy_id in STRATEGY_IDS:
        if strategy_id in COMPATIBILITY_UNAVAILABLE_REASONS:
            coverage[strategy_id] = COMPATIBILITY_UNAVAILABLE_REASONS[strategy_id]
        elif strategy_id in computed:
            coverage[strategy_id] = "COMPUTED"
        else:
            coverage[strategy_id] = "UNDECLARED"
    return coverage


def _compatibility_with_provenance(
    features: tuple[float, ...],
) -> tuple[dict[str, float], frozenset[str]]:
    """Scores, plus which ids the contract could actually express.

    The pair is the point. ``_strategy_compatibility`` alone cannot answer "was this a
    measured zero or an absent relation", and the gate needs that distinction: a relation
    this snapshot cannot compute is not evidence against the strategy.
    """
    scores = (
        _named_relation_scores(features)
        if len(features) == STRATEGY_GRAPH_CONTEXT_DIM
        else _legacy_relation_scores(features)
    )
    expressible = frozenset(scores)
    for strategy_id in STRATEGY_IDS:
        scores.setdefault(strategy_id, 0.0)
    return scores, expressible


def _strategy_compatibility(features: tuple[float, ...]) -> dict[str, float]:
    """Domain priors as soft ontology relations, closed over the whole catalogue.

    Every id in ``STRATEGY_IDS`` appears in the result: the ones whose facts this
    snapshot contract cannot supply are an explicit 0.0 carrying a reason in
    ``COMPATIBILITY_UNAVAILABLE_REASONS``, so "unreachable" is a reported state
    rather than a missing dictionary key.

    The fill lives HERE and not inside the relation functions, because
    :func:`compatibility_coverage` has to be able to tell a computed relation from a
    filled default. While the fill ran inside them, it made every id look computed
    and the coverage guard could never fail.
    """
    scores = (
        _named_relation_scores(features)
        if len(features) == STRATEGY_GRAPH_CONTEXT_DIM
        else _legacy_relation_scores(features)
    )
    # Closed world, stated explicitly: every remaining catalogue id is present with
    # a 0.0 whose reason is declared, so a strategy is never silently excluded by
    # absence from the relation map.
    for strategy_id in STRATEGY_IDS:
        scores.setdefault(strategy_id, 0.0)
    return scores


def _named_relation_scores(features: tuple[float, ...]) -> dict[str, float]:
    """Relations the ALIGNED contract can express, and only those.

    Reads by NAME. The version before the contract indexed ``features[4]`` and
    ``features[6]`` directly and went on scoring after those slots came to hold
    different quantities in training and in serving.

    Returns only the ids with a real relation -- no closed-world padding -- so the
    caller can distinguish "this relation evaluated to zero" from "this contract
    cannot express this relation at all".
    """
    named = as_context_mapping(features)

    def value(name: str, default: float = 0.0) -> float:
        raw = float(named.get(name, default))
        return raw if np.isfinite(raw) else default

    def unit(raw: float) -> float:
        return max(0.0, min(1.0, raw))

    # Execution quality is UNKNOWN, not perfect, when the minute carried no book
    # sample. Without the availability factor a missing spread reads as 0.0 and
    # ``1 - 0/10`` scores it as the best possible book -- so the priors would peak
    # exactly where the least is known, on roughly nine of ten KRX minutes. A
    # relation whose facts are absent is worth 0.0 here, which is the same
    # closed-world convention as COMPATIBILITY_UNAVAILABLE_REASONS.
    microstructure_known = unit(value("microstructure_available"))
    spread_quality = microstructure_known * unit(
        1.0 - abs(value("spread_bps_scaled")) / 10.0
    )
    order_imbalance = value("orderbook_imbalance")
    # v4 used ``aggressor_imbalance_5s`` here, which the historical path cannot
    # produce; the scaled bar return is the directional-pressure measure both
    # sides actually share.
    momentum = value("return_1m_scaled")
    trend_strength = unit(0.5 + 0.5 * momentum)
    breakout_pressure = unit(
        0.5
        + 0.25 * order_imbalance
        + 0.25 * momentum
    )
    vwap_deviation = abs(value("distance_from_vwap"))
    mean_reversion_context = unit(vwap_deviation * 100.0) * spread_quality
    volatility = unit(abs(value("realized_volatility_30m")) / 2.0)
    rvgi_available = unit(value("rvgi_available"))
    rvgi_cross = unit(value("rvgi_bullish_cross"))
    box_available = unit(value("box_available"))
    box_position = unit(value("box_position"))
    rvgi_box = (
        rvgi_available
        * box_available
        * unit(0.45 * rvgi_cross + 0.35 * box_position + 0.20 * breakout_pressure)
    )
    # --- Trend structure, v6 ------------------------------------------------ #
    # These relations exist because the v5 contract could not express them at all,
    # and the closed-world gate turns "cannot express" into a permanent veto: every
    # trend, volatility-regime and range-regime arm in the catalogue was unreachable
    # on every pass, which left the elector choosing among mean-reversion arms only.
    # Like the relations above, each states the COARSE thesis and leaves exact
    # thresholds to the owned algorithm.
    trend_known = unit(value("trend_available"))
    #: ADX 25 is the conventional "trending, not ranging" line; adx_scaled is ADX/100.
    trend_conviction = trend_known * unit(value("adx_scaled") / 0.25)
    directional_bias = unit(0.5 + 0.5 * value("supertrend_direction"))
    dmi_bias = unit(0.5 + 0.5 * value("dmi_spread_scaled") / 0.25)
    ema_bias = unit(0.5 + 0.5 * value("ema_separation_pct"))
    keltner_known = unit(value("keltner_available"))
    oscillator_known = unit(value("oscillator_available"))
    #: CHOP above ~61 is the conventional "range, not trend" reading.
    range_regime = oscillator_known * unit(value("choppiness_scaled") / 0.61)

    # --- Session structure, v6 ---------------------------------------------- #
    session_known = unit(value("session_structure_available"))
    opening_position = value("opening_range_position")
    #: 0 below 0.9 of the range, 1.0 at or past 1.1 — crisp about the boundary the
    #: thesis is actually about, rather than scoring mid-range drift as half a break.
    above_opening_range = unit((opening_position - 0.9) * 5.0)
    below_opening_range = unit((0.1 - opening_position) * 5.0)
    #: The intraday-momentum effect is a statement about the CLOSE, so how late in
    #: the session it is belongs in the relation.
    session_maturity = unit(value("minutes_since_session_open") / 330.0)
    first_half_hour = value("first_half_hour_return_pct")

    # ``spread_quality`` is zero whenever the minute carried no book sample, which is
    # ~89.5% of KRX bars. Multiplying a relation by it therefore does not express "the
    # book is poor" — it expresses "we did not look", and on nine bars in ten it zeroes
    # the relation outright. That is the right factor for a thesis the book is PART of,
    # and the wrong one for a thesis measured from bars over hours: a three-hour trend
    # continuation does not stop being a trend because one minute went unsampled, and
    # spread is an execution concern the guard already re-checks at submit time with a
    # live book rather than a stale bar column.
    #
    # So the factor is applied by whether the THESIS reads the book, not uniformly.
    book_thesis = spread_quality
    computed = {
        "intraday_momentum": trend_strength * spread_quality,
        "breakout_volume": breakout_pressure * spread_quality,
        "vwap_mean_reversion": mean_reversion_context,
        # Same first-class ontology/model mask as every other strategy. The
        # relation expresses only the coarse thesis: a liquid VWAP displacement
        # with positive completed-minute pressure. Exact EMA/MACD/RSI thresholds
        # remain the owned algorithm's point-in-time responsibility.
        "bar_confirmed_vwap_recovery": (
            mean_reversion_context * unit(0.5 + 0.5 * momentum)
        ),
        "liquidity_shock_reversal": volatility * unit(abs(order_imbalance)),
        "rvgi_box_breakout": rvgi_box,
        # Low IN the box is the whole relation, so it is 1 - box_position and not
        # box_position: rvgi_box_breakout above reads the SAME feature the other way
        # up for the opposite thesis. Gated on box_available for the closed-world
        # reason stated above — an absent box must score 0.0, not "at the floor".
        "range_support_reversion": (
            box_available * unit(1.0 - box_position) * spread_quality
        ),
        # Restored: this relation existed on the legacy path and was dropped when the
        # aligned contract replaced it, which left the strategy scoring a permanent
        # 0.0 -- ontology-blocked on every pass -- with nothing reporting it.
        #
        # Faithful port, not a new invention. ``vwap_premium`` in the legacy form is
        # ``(price - vwap) / vwap``, which IS ``distance_from_vwap`` here, and the
        # aggressor term takes the same substitution the contract already documents
        # for intraday_momentum and breakout_volume: the historical path cannot
        # produce ``aggressor_imbalance_5s``, and the scaled bar return is the
        # directional-pressure measure both sides share. Minutes-to-close stays the
        # owned algorithm's point-in-time responsibility, as it was before.
        # A continuation thesis needs the trend to be real (ADX), pointed (supertrend)
        # and confirmed by directional movement (DMI) — three readings of the same
        # structure, so they multiply rather than average.
        "supertrend_dmi_continuation": (
            trend_conviction * directional_bias * dmi_bias
        ),
        # The bar-trend arm reads separation rather than ADX: fast EMA above slow,
        # and price holding above the fast one.
        "bar_trend_continuation": (
            trend_known
            * ema_bias
            * unit(0.5 + 0.5 * value("ema_fast_distance_pct"))
        ),
        # A channel breakout is only interesting where the channel is being pushed
        # AND the move has directional conviction; without ADX this scores every
        # drift through the band as a breakout.
        "keltner_volatility_breakout": (
            keltner_known
            * unit(value("keltner_position"))
            * trend_conviction
        ),
        # The opposite regime to the three above, and deliberately expressed from the
        # same numbers the other way up: a high choppiness index with price at the
        # lower Bollinger extreme is the range-reversion thesis.
        "choppiness_range_reversion": (
            range_regime * unit(1.0 - value("bb_percent_b"))
        ),
        # Exhaustion, not continuation: the book leans one way while the last
        # completed bar moved the other. Their product is negative exactly when they
        # disagree, which is the whole relation.
        # This one IS a book thesis — microprice against flow — so an unsampled book
        # really does mean it cannot be evaluated.
        "ofi_microprice_exhaustion_reversal": (
            microstructure_known
            * unit(abs(order_imbalance))
            * unit(0.5 - 2.0 * momentum * order_imbalance)
        ),
        # Anchored VWAP reversion is displacement measured against the symbol's own
        # volatility rather than an absolute band — a 30bp gap is a long way for a
        # quiet name and noise for a violent one, which is what makes it adaptive.
        "adaptive_anchored_vwap_reversion": (
            unit(vwap_deviation * 100.0 / max(value("atr_pct"), 0.05))
            * book_thesis
        ),
        # A range break is the range being cleared, with the completed-minute
        # pressure to suggest it holds. The width term keeps a one-tick "range" from
        # scoring as a decisive break.
        "opening_range_breakout": (
            session_known
            * above_opening_range
            * breakout_pressure
            * unit(value("opening_range_width_pct") / 0.2)
        ),
        "opening_range_breakdown": (
            session_known
            * below_opening_range
            * unit(1.0 - breakout_pressure)
            * unit(value("opening_range_width_pct") / 0.2)
        ),
        # Gap context is about a gap existing and being large enough to organise the
        # session around; which side to take is the owned algorithm's submode.
        "gap_context": (
            unit(value("session_gap_available"))
            * unit(abs(value("session_gap_pct")) / 1.0)
            * spread_quality
        ),
        # The intraday-momentum effect: the first half hour's sign carries into the
        # close. Both arms read the SAME field with opposite sign, which is what makes
        # them a matched long/short pair rather than two unrelated theses.
        "market_intraday_momentum": (
            session_known * session_maturity * unit(first_half_hour)
        ),
        "market_intraday_momentum_short": (
            session_known * session_maturity * unit(-first_half_hour)
        ),
        "overnight_gap_carry": (
            unit(0.5 + 0.5 * momentum)
            * unit(0.5 + 50.0 * value("distance_from_vwap"))
            * volatility
            * spread_quality
        ),
    }
    return computed


def _reference_price(snapshot) -> float:
    """Price the cost estimate is anchored on.

    Prefers the explicit field. Pre-v5 snapshots do not carry one, and for those
    the price really is in slot 0 as ``close / 100_000`` — so the old derivation
    is kept for exactly those widths rather than applied blindly to a vector
    whose slot 0 is now ``microstructure_available``.
    """
    explicit = float(getattr(snapshot, "reference_price", 0.0) or 0.0)
    if explicit > 0 and np.isfinite(explicit):
        return explicit
    features = snapshot.features
    if len(features) == STRATEGY_GRAPH_CONTEXT_DIM:
        # v5 carries no price. A missing reference is reported as zero so the
        # caller's clamp makes the cost estimate obviously wrong rather than
        # quietly plausible.
        return 0.0
    return float(features[0]) * 100_000.0 if features else 0.0


def _legacy_relation_scores(features: tuple[float, ...]) -> dict[str, float]:
    """Priors for the pre-v5 schemas, read positionally as they always were.

    Kept verbatim so an operator still running a v3/v4 or 12-field checkpoint
    gets the behaviour that checkpoint was calibrated against. New work belongs
    in the named path above; this exists only so the contract fix does not
    quietly re-score deployed artifacts.

    Like the named path, returns only the ids it can score; the closed-world fill
    belongs to :func:`_strategy_compatibility`.
    """

    def value(index: int, default: float = 0.0) -> float:
        try:
            raw = float(features[index])
            return raw if np.isfinite(raw) else default
        except (IndexError, TypeError, ValueError):
            return default

    def unit(raw: float) -> float:
        return max(0.0, min(1.0, raw))

    spread_quality = unit(1.0 - abs(value(2)) / 10.0)
    order_imbalance = value(4)
    aggressor = value(6)
    trend_strength = unit(0.5 + 0.5 * aggressor)
    breakout_pressure = unit(0.5 + 0.25 * order_imbalance + 0.25 * aggressor)
    price = value(0)
    vwap = value(7)
    vwap_deviation = abs(price - vwap) / max(abs(vwap), 1e-9)
    mean_reversion_context = unit(vwap_deviation * 100.0) * spread_quality
    volatility = unit(abs(value(8)) / 2.0)
    rvgi_box = (
        unit(value(12))
        * unit(value(18))
        * unit(0.45 * unit(value(17)) + 0.35 * unit(value(23)) + 0.20 * breakout_pressure)
    )
    # Signed, unlike ``vwap_deviation``: a carry needs the close to be ABOVE the
    # session VWAP, and an absolute displacement cannot tell that from below it.
    vwap_premium = (price - vwap) / max(abs(vwap), 1e-9)
    closing_drive = unit(0.5 + 0.5 * aggressor) * unit(0.5 + 50.0 * vwap_premium)
    computed = {
        "intraday_momentum": trend_strength * spread_quality,
        "breakout_volume": breakout_pressure * spread_quality,
        "vwap_mean_reversion": mean_reversion_context,
        "liquidity_shock_reversal": volatility * unit(abs(order_imbalance)),
        "rvgi_box_breakout": rvgi_box,
        # Buyers in control at the close, on a name whose volatility can carry a
        # gap worth the round trip, in a book tight enough to get out of.
        "overnight_gap_carry": closing_drive * volatility * spread_quality,
    }
    return computed


def _shadow_route(
    path,
    route,
    *,
    realtime_trust: GnnRealtimeTrust | None = None,
    evidence: tuple[StrategyUtilityEvidence, ...] = (),
    checkpoint_hash: str | None = None,
    checkpoint_live_authorized: bool = True,
):
    selected = route.selected
    strategy_trusted = bool(
        realtime_trust is not None
        and checkpoint_live_authorized
        and realtime_trust.passed
        and selected is not None
        and _strategy_market_trusted(
            realtime_trust, selected.strategy_id, selected.symbol
        )
    )
    reason_codes = tuple(route.reason_codes)
    if not checkpoint_live_authorized:
        reason_codes = (*reason_codes, "GNN_CHECKPOINT_NOT_LIVE_AUTHORIZED")
    if realtime_trust is not None:
        reason_codes = (
            *reason_codes,
            (
                "GNN_REALTIME_TRUST_PASSED"
                if strategy_trusted
                else (
                    "GNN_REALTIME_MODEL_TRUST_PASSED"
                    if realtime_trust.passed
                    else "GNN_REALTIME_TRUST_NOT_READY"
                )
            ),
            *realtime_trust.reason_codes,
        )
    validation_candidate = (
        selected
        if (
            selected is not None
            and selected.ontology_allowed
            and not selected.hard_block_reasons
            and selected.compatibility_score > 0.0
            and selected.expected_net_return_bps > 0.0
        )
        else None
    )
    if validation_candidate is None:
        reason_codes = (
            *reason_codes,
            "GNN_NO_ONTOLOGY_ADMISSIBLE_VALIDATION_CANDIDATE",
        )
    return ShadowDecision(
        path=path,
        # A non-promoted checkpoint may be measured, never executed.  Keeping
        # the selected strategy and metrics below preserves observability while
        # the explicit NO_TRADE prevents a later consumer from treating the
        # shadow winner as an order permission.
        action=route.action if checkpoint_live_authorized else "NO_TRADE",
        strategy_id=selected.strategy_id if selected else None,
        utility=(
            route.weighted_utility
            if selected and route.weighted_utility is not None
            else selected.utility if selected else None
        ),
        reason_codes=tuple(dict.fromkeys(reason_codes)),
        probability_success=(
            validation_candidate.probability_success
            if validation_candidate
            else None
        ),
        expected_net_return_bps=(
            validation_candidate.expected_net_return_bps
            if validation_candidate
            else None
        ),
        expected_cost_bps=(
            validation_candidate.expected_cost_bps
            if validation_candidate
            else None
        ),
        total_uncertainty=(
            validation_candidate.aleatoric_uncertainty
            + validation_candidate.epistemic_uncertainty_or_proxy
            if validation_candidate
            else None
        ),
        ontology_compatibility=(
            validation_candidate.compatibility_score
            if validation_candidate
            else None
        ),
        realtime_trust_score=(
            realtime_trust.score if realtime_trust is not None else None
        ),
        realtime_trust_samples=(
            realtime_trust.sample_count if realtime_trust is not None else None
        ),
        validation_strategy_id=(
            validation_candidate.strategy_id if validation_candidate else None
        ),
        checkpoint_hash=checkpoint_hash,
    )


def _upside_supervised_strategy_ids(metadata: dict) -> tuple[str, ...]:
    """Which strategies' upside heads carry evidence, per the checkpoint report.

    Prefers the explicit ``upside_supervised_strategy_ids`` written by training.
    Checkpoints predating that field still carry the same fact in
    ``label_outcomes[*].positive_net`` — the realized profitable fills are exactly
    the rows that trained the MFE channel — so derive it rather than forcing a
    retrain to regain a safety property.

    Fails closed: no recognizable evidence means no strategy is authorized to
    forecast an upside. That is the honest reading of a checkpoint that never said.
    """
    declared = metadata.get("upside_supervised_strategy_ids")
    if isinstance(declared, (list, tuple)):
        return tuple(str(item) for item in declared)
    minimum = int(
        metadata.get("minimum_upside_supervision_rows")
        or os.getenv("GNN_MIN_UPSIDE_SUPERVISION_ROWS", "20")
    )
    supervision = metadata.get("strategy_supervision")
    if isinstance(supervision, dict):
        return tuple(
            str(strategy_id)
            for strategy_id, row in supervision.items()
            if isinstance(row, dict)
            and int(row.get("upside_rows") or 0) >= minimum
        )
    outcomes = metadata.get("label_outcomes")
    if isinstance(outcomes, dict):
        return tuple(
            str(strategy_id)
            for strategy_id, row in outcomes.items()
            if isinstance(row, dict)
            and int(row.get("positive_net") or 0) >= minimum
        )
    return ()


def _upside_authorized_strategy_markets(
    metadata: dict,
) -> dict[str, tuple[str, ...]]:
    """Markets where a strategy's upside head is both taught and profitable.

    A globally trained head may see KRX and US examples, but their round-trip
    costs are radically different.  A strategy with positive KRX outcomes must
    not lend that permission to US forecasts, and US losses must not suppress a
    KRX edge.  Old checkpoints without per-market reports retain the previous
    all-market behaviour for strategies that explicitly declared supervision.
    """
    by_market = metadata.get("label_outcomes_by_market")
    if not isinstance(by_market, dict) or not by_market:
        return {
            strategy_id: ("KRX", "US")
            for strategy_id in _upside_supervised_strategy_ids(metadata)
        }
    minimum_fills = max(
        10,
        int(os.getenv("GNN_MIN_MARKET_UPSIDE_FILLED_ROWS", "20")),
    )
    minimum_mean_net_bps = max(
        0.0, float(os.getenv("GNN_MIN_MARKET_MEAN_NET_BPS", "5.0"))
    )
    authorized: dict[str, list[str]] = {}
    for market, outcomes in by_market.items():
        market_key = str(market or "").upper()
        if market_key not in {"KRX", "US"} or not isinstance(outcomes, dict):
            continue
        for strategy_id, row in outcomes.items():
            if not isinstance(row, dict):
                continue
            filled = int(row.get("filled") or 0)
            mean_net = row.get("mean_net_return_bps_when_filled")
            try:
                positive_expectancy = float(mean_net) >= minimum_mean_net_bps
            except (TypeError, ValueError):
                positive_expectancy = False
            if filled >= minimum_fills and positive_expectancy:
                authorized.setdefault(str(strategy_id), []).append(market_key)
    return {
        strategy_id: tuple(sorted(set(markets)))
        for strategy_id, markets in authorized.items()
    }


def _strategy_market_trusted(
    trust: GnnRealtimeTrust,
    strategy_id: str,
    symbol: str,
) -> bool:
    market = "KRX" if str(symbol).isdigit() and len(str(symbol)) == 6 else "US"
    configured = trust.trusted_strategy_markets.get(str(strategy_id), ())
    return market in configured


def _validation_candidates(
    evidence: tuple[StrategyUtilityEvidence, ...],
    *,
    checkpoint_hash: str | None,
    realtime_trust: GnnRealtimeTrust | None = None,
    checkpoint_live_authorized: bool = True,
) -> tuple[ShadowDecision, ...]:
    """Persist every admissible strategy forecast for validation and joint ranking.

    These rows still grant no order permission.  Per-strategy trust is carried so
    the downstream election can compare all valid ``(symbol, strategy)`` pairs
    without mistaking a model-wide pass for strategy-specific authorization.
    """
    return tuple(
        ShadowDecision(
            path="cpu_gnn_validation",
            action="VALIDATE_ONLY",
            strategy_id=item.strategy_id,
            utility=item.utility,
            reason_codes=tuple(
                dict.fromkeys(
                    (
                        "ORDER_PERMISSION_NOT_GRANTED",
                        *(
                            ()
                            if checkpoint_live_authorized
                            else ("GNN_CHECKPOINT_NOT_LIVE_AUTHORIZED",)
                        ),
                        (
                            "GNN_REALTIME_TRUST_PASSED"
                            if (
                                checkpoint_live_authorized
                                and realtime_trust is not None
                                and realtime_trust.passed
                                and _strategy_market_trusted(
                                    realtime_trust,
                                    item.strategy_id,
                                    item.symbol,
                                )
                            )
                            else (
                                "GNN_REALTIME_MODEL_TRUST_PASSED"
                                if realtime_trust is not None
                                and realtime_trust.passed
                                else "GNN_REALTIME_TRUST_NOT_READY"
                            )
                        ),
                        *(realtime_trust.reason_codes if realtime_trust else ()),
                    )
                )
            ),
            probability_success=item.probability_success,
            expected_net_return_bps=item.expected_net_return_bps,
            expected_cost_bps=item.expected_cost_bps,
            total_uncertainty=(
                item.aleatoric_uncertainty
                + item.epistemic_uncertainty_or_proxy
            ),
            ontology_compatibility=item.compatibility_score,
            validation_strategy_id=item.strategy_id,
            checkpoint_hash=checkpoint_hash,
            position_direction=(
                "SHORT" if is_short_strategy(item.strategy_id) else "LONG"
            ),
        )
        for item in evidence
        if (
            item.ontology_allowed
            and not item.hard_block_reasons
            and item.compatibility_score > 0.0
        )
    )
