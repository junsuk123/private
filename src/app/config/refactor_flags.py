from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Mapping


@dataclass(frozen=True)
class RefactorFeatureFlags:
    legacy_vote_path: bool = True
    websocket_market_data: bool = False
    local_chart_engine: bool = False
    ontology_router: bool = False
    gnn_shadow: bool = False
    gnn_rerank: bool = False
    npu_inference: bool = False
    strategy_owned_execution: bool = False
    live_enabled: bool = False

    @classmethod
    def from_env(cls, env: Mapping[str, str] | None = None) -> "RefactorFeatureFlags":
        values = os.environ if env is None else env
        flags = cls(
            legacy_vote_path=_bool(values, "REFACTOR_LEGACY_VOTE_PATH", True),
            websocket_market_data=_bool(values, "REFACTOR_WEBSOCKET_MARKET_DATA", False),
            local_chart_engine=_bool(values, "REFACTOR_LOCAL_CHART_ENGINE", False),
            ontology_router=_bool(values, "REFACTOR_ONTOLOGY_ROUTER", False),
            gnn_shadow=_bool(values, "REFACTOR_GNN_SHADOW", False),
            gnn_rerank=_bool(values, "REFACTOR_GNN_RERANK", False),
            npu_inference=_bool(values, "REFACTOR_NPU_INFERENCE", False),
            strategy_owned_execution=_bool(values, "REFACTOR_STRATEGY_OWNED_EXECUTION", False),
            live_enabled=_bool(values, "REFACTOR_LIVE_ENABLED", False),
        )
        flags.validate()
        return flags

    def validate(self) -> None:
        if self.live_enabled and not self.strategy_owned_execution:
            raise ValueError("refactor live mode requires strategy-owned execution")
        if self.gnn_rerank and not self.ontology_router:
            raise ValueError("GNN reranking requires the ontology router")
        if self.npu_inference and not (self.gnn_shadow or self.gnn_rerank):
            raise ValueError("NPU inference requires a GNN shadow or rerank mode")
        if not self.legacy_vote_path and not self.ontology_router:
            raise ValueError("at least one routing path must remain enabled")


def _bool(values: Mapping[str, str], key: str, default: bool) -> bool:
    raw = values.get(key)
    if raw is None:
        return default
    normalized = str(raw).strip().lower()
    if normalized in {"1", "true", "yes", "on"}:
        return True
    if normalized in {"0", "false", "no", "off"}:
        return False
    raise ValueError(f"{key} must be a boolean")
