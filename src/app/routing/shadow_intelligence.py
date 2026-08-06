from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
import json
import math
import os
from pathlib import Path
from typing import Any

import numpy as np

from app.cost import TradingCostEngine
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
from app.routing.gnn_realtime_trust import GnnRealtimeTrust, GnnRealtimeTrustEvaluator
from app.routing.strategy_router import StrategyRouter
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
    the event-driven websocket runtime.  Building the 28-field graph context
    here lets both ingestion paths feed the same ontology+GNN pipeline without
    inventing a second feature schema.
    """
    frame.validate()
    values = frame.as_feature_dict()
    symbol = str(frame.symbol or "").strip().upper()
    as_of = frame.decision_time
    if as_of.tzinfo is None:
        as_of = as_of.replace(tzinfo=timezone.utc)
    price = max(0.0, float(frame.mark_price or 0.0))
    price_scale = 100_000.0
    scaled_price = price / price_scale
    distance_from_vwap = float(values.get("distance_from_vwap", 0.0) or 0.0)
    vwap = (
        price / (1.0 + distance_from_vwap)
        if price > 0 and math.isfinite(distance_from_vwap)
        and abs(1.0 + distance_from_vwap) > 1e-9
        else price
    )
    aggressor = max(
        -1.0,
        min(1.0, float(values.get("aggressor_imbalance_5s", 0.0) or 0.0)),
    )
    volume_5s = max(
        0.0,
        math.expm1(max(0.0, float(values.get("volume_5s_log", 0.0) or 0.0))),
    )
    signed_flow = aggressor * volume_5s

    def finite(name: str, default: float = 0.0) -> float:
        raw = float(values.get(name, default) or 0.0)
        return raw if math.isfinite(raw) else default

    rvgi_box = (
        finite("rvgi_available"),
        finite("rvgi"),
        finite("rvgi_signal"),
        finite("rvgi_diff"),
        finite("rvgi_slope"),
        finite("rvgi_bullish_cross"),
        finite("box_available"),
        finite("box_high") / max(price, 1e-12),
        finite("box_low") / max(price, 1e-12),
        finite("box_mid") / max(price, 1e-12),
        finite("box_width_pct"),
        finite("box_position"),
        finite("breakout_distance_bps") / 100.0,
        finite("box_previous_close") / max(price, 1e-12),
        1.0 if finite("box_context_timestamp_epoch") > 0 else 0.0,
    )
    context = (
        scaled_price,
        scaled_price,
        finite("spread_bps") / 100.0,
        scaled_price,
        finite("orderbook_imbalance"),
        signed_flow / 10_000.0,
        aggressor,
        vwap / price_scale,
        finite("realized_volatility_3m") * 100.0,
        1.0,
        0.0,
        1.0,
        *rvgi_box,
        1.0 if symbol.isdigit() and len(symbol) == 6 else 0.0,
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
        features=tuple(float(item) for item in context),
        data_fresh=True,
        tradable=price > 0 and finite("spread_bps", -1.0) >= 0.0,
        allowed_strategy_ids=allowed_strategy_ids,
        feature_schema_name="realtime_strategy_graph_v4_market",
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
    ) -> None:
        self.context_feature_dim = feature_dim
        self.graph_mode = feature_dim in {27, 28}
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
        self.live_authorized = False
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
        if not self.live_authorized:
            reasons.append("GNN_NOT_LIVE_AUTHORIZED")
        return tuple(reasons)

    def _ontology(self, snapshot: SlowIntelligenceSnapshot):
        facts = {
            "data_fresh": _fact(snapshot, "data_fresh", snapshot.data_fresh),
            "tradable": _fact(snapshot, "tradable", snapshot.tradable),
        }
        compatibility = _strategy_compatibility(snapshot.features)
        for strategy_id in STRATEGY_IDS:
            compatibility_score = compatibility.get(strategy_id, 0.0)
            facts[f"allow:{strategy_id}"] = _fact(
                snapshot,
                f"allow:{strategy_id}",
                strategy_id in snapshot.allowed_strategy_ids,
            )
            facts[f"compat:{strategy_id}"] = _fact(
                snapshot,
                f"compat:{strategy_id}",
                compatibility_score,
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
            x = strategy_node_features(snapshot.features).reshape(
                1,
                1,
                STRATEGY_NODE_COUNT,
                -1,
            )
            adjacency = strategy_relation_adjacency(allowed).reshape(
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
        reference_price = max(
            1e-9,
            float(snapshot.features[0]) * 100_000.0,
        )
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
            if strategy_id not in self.upside_supervised_strategy_ids:
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
    # Session structure: opening range and the first-half-hour return are
    # measured from session boundaries, not from the sub-second window.
    "opening_range_breakout": "CONTEXT_UNAVAILABLE:OPENING_RANGE",
    "opening_range_breakdown": "CONTEXT_UNAVAILABLE:OPENING_RANGE",
    "market_intraday_momentum": "CONTEXT_UNAVAILABLE:FIRST_HALF_HOUR",
    "market_intraday_momentum_short": "CONTEXT_UNAVAILABLE:FIRST_HALF_HOUR",
    "gap_context": "CONTEXT_UNAVAILABLE:SESSION_OPEN_GAP",
    # Point-in-time news: an event relation needs the event and its age.
    "event_momentum": "CONTEXT_UNAVAILABLE:EVENT_FACTS",
    # A meaningful anchor (session open, volatility spike, news time) is what
    # separates this from plain VWAP reversion; the snapshot carries only the
    # rolling VWAP proxy.
    "adaptive_anchored_vwap_reversion": "CONTEXT_UNAVAILABLE:VWAP_ANCHOR",
    # The depth fields in this contract are neutral placeholders (the causal
    # minute-bar proxy has no L2), so microprice OFI cannot be formed. Inventing
    # one from bar range would be a fabricated relation, not a weak one.
    "ofi_microprice_exhaustion_reversal": "CONTEXT_UNAVAILABLE:L2_MICROPRICE",
}


def compatibility_coverage() -> dict[str, str]:
    """Per-strategy status of the relation map: computed, or why it cannot be.

    Exists so the omission this map suffered cannot recur silently — a strategy
    added to the catalogue with neither a relation nor a documented reason shows
    up here as ``UNDECLARED`` and fails the contract test.
    """
    computed = set(_strategy_compatibility(tuple([0.0] * 28)))
    coverage: dict[str, str] = {}
    for strategy_id in STRATEGY_IDS:
        if strategy_id in COMPATIBILITY_UNAVAILABLE_REASONS:
            coverage[strategy_id] = COMPATIBILITY_UNAVAILABLE_REASONS[strategy_id]
        elif strategy_id in computed:
            coverage[strategy_id] = "COMPUTED"
        else:
            coverage[strategy_id] = "UNDECLARED"
    return coverage


def _strategy_compatibility(features: tuple[float, ...]) -> dict[str, float]:
    """Domain priors encoded as soft ontology relations, never BUY signals.

    The GNN learns outcome utility; these values only describe how strongly the
    current facts instantiate each strategy's domain relationship. Every id in
    ``STRATEGY_IDS`` appears in the result: the ones whose facts this snapshot
    contract cannot supply are an explicit 0.0 carrying a reason in
    ``COMPATIBILITY_UNAVAILABLE_REASONS``, so "unreachable" is a reported state
    rather than a missing dictionary key.
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
    breakout_pressure = unit(
        0.5
        + 0.25 * order_imbalance
        + 0.25 * aggressor
    )
    price = value(0)
    vwap = value(7)
    vwap_deviation = abs(price - vwap) / max(abs(vwap), 1e-9)
    mean_reversion_context = unit(vwap_deviation * 100.0) * spread_quality
    volatility = unit(abs(value(8)) / 2.0)
    rvgi_available = unit(value(12))
    rvgi_cross = unit(value(17))
    box_available = unit(value(18))
    box_position = unit(value(23))
    rvgi_box = (
        rvgi_available
        * box_available
        * unit(0.45 * rvgi_cross + 0.35 * box_position + 0.20 * breakout_pressure)
    )
    computed = {
        "intraday_momentum": trend_strength * spread_quality,
        "breakout_volume": breakout_pressure * spread_quality,
        "vwap_mean_reversion": mean_reversion_context,
        "liquidity_shock_reversal": volatility * unit(abs(order_imbalance)),
        "rvgi_box_breakout": rvgi_box,
    }
    # Closed world, stated explicitly: every remaining catalogue id is present
    # with a 0.0 whose reason is declared, so a strategy is never silently
    # excluded by absence from this dictionary.
    for strategy_id in STRATEGY_IDS:
        computed.setdefault(strategy_id, 0.0)
    return computed


def _shadow_route(
    path,
    route,
    *,
    realtime_trust: GnnRealtimeTrust | None = None,
    evidence: tuple[StrategyUtilityEvidence, ...] = (),
    checkpoint_hash: str | None = None,
):
    selected = route.selected
    strategy_trusted = bool(
        realtime_trust is not None
        and realtime_trust.passed
        and selected is not None
        and selected.strategy_id in realtime_trust.trusted_strategy_ids
    )
    reason_codes = tuple(route.reason_codes)
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
        action=route.action,
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


def _validation_candidates(
    evidence: tuple[StrategyUtilityEvidence, ...],
    *,
    checkpoint_hash: str | None,
) -> tuple[ShadowDecision, ...]:
    """Persist forecasts for forward validation without granting order rights."""
    return tuple(
        ShadowDecision(
            path="cpu_gnn_validation",
            action="VALIDATE_ONLY",
            strategy_id=item.strategy_id,
            utility=item.utility,
            reason_codes=("ORDER_PERMISSION_NOT_GRANTED",),
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
        )
        for item in evidence
        if (
            item.ontology_allowed
            and not item.hard_block_reasons
            and item.compatibility_score > 0.0
        )
    )
