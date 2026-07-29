from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta
import json
import os
from pathlib import Path

import numpy as np

from app.models.strategy_utility import (
    FixedShapeStrategyUtilityModel,
    StrategyUtilityModelConfig,
)
from app.models.strategy_utility.openvino_runtime import OpenVinoStrategyUtilityRuntime
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
from app.routing.strategy_router import StrategyRouter
from app.strategy.experts import ALL_EXPERT_TYPES
from app.trading.contracts import StrategyUtilityEvidence


STRATEGY_IDS = tuple(expert.strategy_id for expert in (kind() for kind in ALL_EXPERT_TYPES))


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
        config = StrategyUtilityModelConfig(
            batch_size=1,
            time_steps=1,
            max_nodes=1,
            feature_dim=feature_dim,
            relation_count=1,
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
        checkpoint_model = (
            FixedShapeStrategyUtilityModel.load_checkpoint(checkpoint_path)
            if checkpoint_path.exists()
            else None
        )
        self.checkpoint_loaded = (
            checkpoint_model is not None and checkpoint_model.config == config
        )
        self.model_input_schema = "unspecified"
        self.live_authorized = False
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
                self.live_authorized = bool(metadata.get("live_authorized"))
            except (OSError, ValueError, json.JSONDecodeError):
                self.model_input_schema = "unknown"
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
        self.gate = ClosedWorldOntologyGate()
        self.router = StrategyRouter()

    def evaluate(
        self,
        snapshot: SlowIntelligenceSnapshot,
        *,
        legacy_action: str = "NO_TRADE",
    ) -> ShadowIntelligenceResult | None:
        previous = self.last_run.get(snapshot.symbol)
        if previous is not None and snapshot.as_of - previous < self.minimum_interval:
            return None
        if len(snapshot.features) != self.model.config.feature_dim:
            raise ValueError("slow intelligence feature dimension mismatch")
        self.last_run[snapshot.symbol] = snapshot.as_of
        ontology = self._ontology(snapshot)
        model_block_reasons = self._model_block_reasons(snapshot)
        if model_block_reasons:
            decisions = [
                ShadowDecision("legacy", legacy_action, None, None, ("LEGACY_OBSERVED",)),
                ShadowDecision(
                    "ontology",
                    "ADMISSIBLE" if ontology.allowed_strategy_ids else "NO_TRADE",
                    ontology.allowed_strategy_ids[0] if ontology.allowed_strategy_ids else None,
                    None,
                    (),
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
        npu_evidence: tuple[StrategyUtilityEvidence, ...] = ()
        decisions = [
            ShadowDecision("legacy", legacy_action, None, None, ("LEGACY_OBSERVED",)),
            ShadowDecision(
                "ontology",
                "ADMISSIBLE" if ontology.allowed_strategy_ids else "NO_TRADE",
                ontology.allowed_strategy_ids[0] if ontology.allowed_strategy_ids else None,
                None,
                (),
            ),
            _shadow_route("cpu_gnn", cpu_route),
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
            decisions.append(_shadow_route("npu_gnn", npu_route))
        comparison = self.recorder.compare(
            correlation_id=snapshot.snapshot_id,
            symbol=snapshot.symbol,
            as_of=snapshot.as_of,
            decisions=tuple(decisions),
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
        if not self.checkpoint_loaded:
            return ()
        reasons: list[str] = []
        if self.model_input_schema != snapshot.feature_schema_name:
            reasons.append(
                "MODEL_INPUT_SCHEMA_MISMATCH:"
                f"{self.model_input_schema}!={snapshot.feature_schema_name}"
            )
        if not self.live_authorized:
            reasons.append("UTILITY_MODEL_NOT_LIVE_AUTHORIZED")
        return tuple(reasons)

    def _ontology(self, snapshot: SlowIntelligenceSnapshot):
        facts = {
            "data_fresh": _fact(snapshot, "data_fresh", snapshot.data_fresh),
            "tradable": _fact(snapshot, "tradable", snapshot.tradable),
        }
        for strategy_id in STRATEGY_IDS:
            facts[f"allow:{strategy_id}"] = _fact(
                snapshot,
                f"allow:{strategy_id}",
                strategy_id in snapshot.allowed_strategy_ids,
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
                required_true=("data_fresh", "tradable", f"allow:{strategy_id}"),
            )
            for strategy_id in STRATEGY_IDS
        )
        return self.gate.evaluate(operational, rules)

    def _inputs(self, snapshot: SlowIntelligenceSnapshot, allowed: tuple[str, ...]):
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
        for index, strategy_id in enumerate(STRATEGY_IDS):
            allowed = strategy_id in ontology.allowed_strategy_ids
            gross = float(output.gross_return_bps[0, 0, index])
            cost = float(output.cost_bps[0, 0, index])
            values.append(
                StrategyUtilityEvidence(
                    evidence_id=f"{version}:{snapshot.snapshot_id}:{strategy_id}",
                    as_of=snapshot.as_of,
                    symbol=snapshot.symbol,
                    strategy_id=strategy_id,
                    ontology_allowed=allowed,
                    hard_block_reasons=ontology.blocked_strategy_reasons.get(strategy_id, ()),
                    compatibility_score=ontology.compatibility_scores[strategy_id],
                    probability_success=float(output.probability_success[0, 0, index]),
                    expected_gross_return_bps=gross,
                    expected_cost_bps=cost,
                    expected_net_return_bps=gross - cost,
                    expected_adverse_excursion_bps=float(output.mae_bps[0, 0, index]),
                    expected_favorable_excursion_bps=float(output.mfe_bps[0, 0, index]),
                    fill_probability=float(output.fill_probability[0, 0, index]),
                    expected_holding_seconds=float(output.holding_seconds[0, 0, index]),
                    aleatoric_uncertainty=float(output.aleatoric_uncertainty[0, 0, index]),
                    epistemic_uncertainty_or_proxy=0,
                    utility=float(output.utility[0, 0, index]),
                    model_version=version,
                    feature_snapshot_id=snapshot.feature_snapshot_id,
                    ontology_snapshot_id=ontology.snapshot_id,
                    explanation_paths=ontology.explanation_paths[strategy_id],
                )
            )
        return tuple(values)


def _fact(snapshot, name: str, value: bool) -> OperationalFact:
    return OperationalFact(
        name=name,
        value=value,
        observed_at=snapshot.as_of,
        valid_from=snapshot.as_of,
        valid_until=snapshot.valid_until,
        source="slow-intelligence-snapshot",
        confidence=1,
    )


def _shadow_route(path, route):
    return ShadowDecision(
        path=path,
        action=route.action,
        strategy_id=route.selected.strategy_id if route.selected else None,
        utility=route.selected.utility if route.selected else None,
        reason_codes=route.reason_codes,
    )
