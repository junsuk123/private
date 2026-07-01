from __future__ import annotations

from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path
from typing import Any, Mapping

import yaml


DEFAULT_REGISTRY_PATH = Path("config/theory_registry.yaml")


@dataclass(frozen=True)
class TheoryMetadata:
    theory_id: str
    family: str
    style: str
    default_action_bias: str
    horizon_bucket: str
    expected_holding_minutes: int
    required_evidence: tuple[str, ...] = ()
    invalid_when: tuple[str, ...] = ()
    compatible_with: tuple[str, ...] = ()
    conflicts_with: tuple[str, ...] = ()
    evidence_cluster: str = "unknown_cluster"
    validation_status: str = "unvalidated"
    default_weight: float = 0.1

    @classmethod
    def from_mapping(cls, theory_id: str, data: Mapping[str, Any]) -> "TheoryMetadata":
        validation = data.get("validation") or {}
        return cls(
            theory_id=theory_id,
            family=str(data["family"]),
            style=str(data["style"]),
            default_action_bias=str(data.get("default_action_bias", "WATCH")).upper(),
            horizon_bucket=str(data.get("horizon_bucket", "short_intraday")),
            expected_holding_minutes=max(0, int(data.get("expected_holding_minutes", 0))),
            required_evidence=tuple(str(item) for item in data.get("required_evidence", ()) or ()),
            invalid_when=tuple(str(item) for item in data.get("invalid_when", ()) or ()),
            compatible_with=tuple(str(item) for item in data.get("compatible_with", ()) or ()),
            conflicts_with=tuple(str(item) for item in data.get("conflicts_with", ()) or ()),
            evidence_cluster=str(data.get("evidence_cluster", "unknown_cluster")),
            validation_status=str(validation.get("status", data.get("validation_status", "unvalidated"))),
            default_weight=float(validation.get("default_weight", data.get("default_weight", 0.1))),
        )

    def as_dict(self) -> dict[str, Any]:
        return {
            "theory_id": self.theory_id,
            "family": self.family,
            "style": self.style,
            "default_action_bias": self.default_action_bias,
            "horizon_bucket": self.horizon_bucket,
            "expected_holding_minutes": self.expected_holding_minutes,
            "required_evidence": self.required_evidence,
            "invalid_when": self.invalid_when,
            "compatible_with": self.compatible_with,
            "conflicts_with": self.conflicts_with,
            "evidence_cluster": self.evidence_cluster,
            "validation": {
                "status": self.validation_status,
                "default_weight": self.default_weight,
            },
        }


@dataclass(frozen=True)
class OntologyVotingConfig:
    mode: str = "theory_aware_multi_action"
    decision_margin: float = 0.10
    enable_conflict_resolver: bool = True
    enable_evidence_clustering: bool = True
    enable_position_aware_action: bool = True
    default_unvalidated_theory_weight: float = 0.1
    default_partially_validated_theory_weight: float = 0.5
    default_validated_theory_weight: float = 1.0
    max_same_cluster_contribution: float = 1.0
    high_conflict_hold_threshold: float = 0.4


@dataclass(frozen=True)
class NpuVotingConfig:
    enabled: bool = True
    device_preference: str = "NPU"
    fallback_device: str = "CPU"
    batch_buckets: tuple[int, ...] = (128, 256, 512, 1024, 2048, 4096)
    compile_cache_enabled: bool = True
    profile_enabled: bool = True
    numerical_tolerance: float = 0.0001
    min_batch_for_npu: int = 128
    use_async_inference: bool = True
    modules: dict[str, dict[str, bool]] = field(default_factory=dict)


class TheoryRegistry:
    def __init__(
        self,
        theories: Mapping[str, TheoryMetadata],
        *,
        ontology_voting: OntologyVotingConfig | None = None,
        npu: NpuVotingConfig | None = None,
    ) -> None:
        self.theories = dict(theories)
        self.ontology_voting = ontology_voting or OntologyVotingConfig()
        self.npu = npu or NpuVotingConfig()

    @classmethod
    def load(cls, path: str | Path = DEFAULT_REGISTRY_PATH) -> "TheoryRegistry":
        config_path = Path(path)
        loaded = yaml.safe_load(config_path.read_text(encoding="utf-8")) if config_path.exists() else {}
        loaded = loaded or {}
        theories = {
            theory_id: TheoryMetadata.from_mapping(theory_id, data or {})
            for theory_id, data in (loaded.get("theories") or {}).items()
        }
        voting = _ontology_voting_config(loaded.get("ontology_voting") or {})
        npu = _npu_config(loaded.get("npu") or {})
        registry = cls(theories, ontology_voting=voting, npu=npu)
        registry.validate()
        return registry

    def validate(self) -> None:
        required = ("family", "style", "default_action_bias", "horizon_bucket")
        for theory in self.theories.values():
            data = theory.as_dict()
            missing = [name for name in required if not data.get(name)]
            if missing:
                raise ValueError(f"{theory.theory_id} missing required fields: {missing}")

    def get(self, theory_id: str) -> TheoryMetadata | None:
        return self.theories.get(theory_id)

    def require(self, theory_id: str) -> TheoryMetadata:
        metadata = self.get(theory_id)
        if metadata is None:
            raise KeyError(f"Unknown theory_id: {theory_id}")
        return metadata

    def weight_for(self, theory_id: str) -> float:
        metadata = self.get(theory_id)
        if metadata is None:
            return self.ontology_voting.default_unvalidated_theory_weight
        if metadata.default_weight > 0:
            return metadata.default_weight
        status = metadata.validation_status.lower()
        if status == "validated":
            return self.ontology_voting.default_validated_theory_weight
        if status == "partially_validated":
            return self.ontology_voting.default_partially_validated_theory_weight
        return self.ontology_voting.default_unvalidated_theory_weight


@lru_cache(maxsize=1)
def get_theory_registry() -> TheoryRegistry:
    return TheoryRegistry.load()


def reset_theory_registry_cache() -> None:
    get_theory_registry.cache_clear()


def _ontology_voting_config(data: Mapping[str, Any]) -> OntologyVotingConfig:
    defaults = OntologyVotingConfig()
    return OntologyVotingConfig(
        mode=str(data.get("mode", defaults.mode)),
        decision_margin=float(data.get("decision_margin", defaults.decision_margin)),
        enable_conflict_resolver=bool(data.get("enable_conflict_resolver", defaults.enable_conflict_resolver)),
        enable_evidence_clustering=bool(data.get("enable_evidence_clustering", defaults.enable_evidence_clustering)),
        enable_position_aware_action=bool(data.get("enable_position_aware_action", defaults.enable_position_aware_action)),
        default_unvalidated_theory_weight=float(data.get("default_unvalidated_theory_weight", defaults.default_unvalidated_theory_weight)),
        default_partially_validated_theory_weight=float(data.get("default_partially_validated_theory_weight", defaults.default_partially_validated_theory_weight)),
        default_validated_theory_weight=float(data.get("default_validated_theory_weight", defaults.default_validated_theory_weight)),
        max_same_cluster_contribution=float(data.get("max_same_cluster_contribution", defaults.max_same_cluster_contribution)),
        high_conflict_hold_threshold=float(data.get("high_conflict_hold_threshold", defaults.high_conflict_hold_threshold)),
    )


def _npu_config(data: Mapping[str, Any]) -> NpuVotingConfig:
    defaults = NpuVotingConfig()
    return NpuVotingConfig(
        enabled=bool(data.get("enabled", defaults.enabled)),
        device_preference=str(data.get("device_preference", defaults.device_preference)),
        fallback_device=str(data.get("fallback_device", defaults.fallback_device)),
        batch_buckets=tuple(int(item) for item in data.get("batch_buckets", defaults.batch_buckets)),
        compile_cache_enabled=bool(data.get("compile_cache_enabled", defaults.compile_cache_enabled)),
        profile_enabled=bool(data.get("profile_enabled", defaults.profile_enabled)),
        numerical_tolerance=float(data.get("numerical_tolerance", defaults.numerical_tolerance)),
        min_batch_for_npu=int(data.get("min_batch_for_npu", defaults.min_batch_for_npu)),
        use_async_inference=bool(data.get("use_async_inference", defaults.use_async_inference)),
        modules=dict(data.get("modules", defaults.modules)),
    )
