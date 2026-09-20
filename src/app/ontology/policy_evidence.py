"""Closed-world evidence projection for ontology-derived numerical policies.

Projection is dependency-light and safe to use in a decision cycle. RDF and
SHACL are imported only by explicit audit/export calls, never on a market tick.
No inference here authorizes an order, invents an observation or assigns a
missing value a neutral zero.
"""
from __future__ import annotations

import hashlib
import json
import math
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from types import MappingProxyType
from typing import Any, Iterable, Mapping
from urllib.parse import quote

POLICY_NAMESPACE = "https://obaits.local/ontology/policy#"
STRATEGY_RELATION_IRIS = MappingProxyType({
    "same_methodology_family": POLICY_NAMESPACE + "sameMethodologyFamily",
    "confirming_methodology": POLICY_NAMESPACE + "confirmsStrategy",
    "contrasting_methodology": POLICY_NAMESPACE + "contrastsStrategy",
})
SCHEMA_PATH = Path(__file__).with_name("policy_ontology.ttl")
SHAPES_PATH = Path(__file__).with_name("policy_shapes.ttl")

_SOURCES = frozenset({
    "kis", "kis_realtime", "live_market_snapshot", "orderbook_snapshot",
    "account_snapshot", "live_signal_predictor", "context_runtime.domestic",
    "context_runtime.regime", "context_runtime.global", "context_runtime.temporal",
    "strategy_performance_store", "indicator_contract",
    "validated_temporal_rgcn",
})
_LOCAL_ALIASES = {
    "direction": "trend_strength", "breadth": "market_breadth",
    "volatility": "realized_volatility", "liquidity": "liquidity_score",
    "flow": "flow_imbalance", "venue_divergence": "venue_divergence",
    "leadership": "leadership", "global_agreement": "global_agreement",
}
_BOUNDS = {
    "realized_volatility": (0.0, 1.0), "downside_volatility": (0.0, 1.0),
    "spread_rate": (0.0, 1.0), "liquidity_score": (0.0, 1.0),
    "market_breadth": (-1.0, 1.0), "trend_strength": (-1.0, 1.0),
    "drawdown_rate": (0.0, 1.0), "regime_confidence": (0.0, 1.0),
    "data_quality_score": (0.0, 1.0), "flow_imbalance": (-1.0, 1.0),
    "model_uncertainty": (0.0, 1.0), "venue_divergence": (0.0, 1.0),
    "global_agreement": (-1.0, 1.0), "leadership": (-1.0, 1.0),
    "probability_success": (0.0, 1.0), "change_point_probability": (0.0, 1.0),
}


def _market(value: Any) -> str:
    text = str(getattr(value, "value", value) or "").strip().upper()
    if text in {"KR", "KRX", "NXT", "KOSPI", "KOSDAQ", "KONEX"}:
        return "KR"
    if text in {"US", "NASDAQ", "NASD", "NYSE", "AMEX", "NYSEAMERICAN"}:
        return "US"
    return text


def _datetime(value: Any) -> datetime | None:
    if isinstance(value, str):
        try:
            value = datetime.fromisoformat(value.replace("Z", "+00:00"))
        except ValueError:
            return None
    if not isinstance(value, datetime) or value.tzinfo is None or value.utcoffset() is None:
        return None
    return value.astimezone(timezone.utc)


def _finite(value: Any) -> float | None:
    if value is None or isinstance(value, bool):
        return None
    try:
        number = float(value)
    except (TypeError, ValueError, OverflowError):
        return None
    return number if math.isfinite(number) else None


def _get(value: Any, key: str, default: Any = None) -> Any:
    return value.get(key, default) if isinstance(value, Mapping) else getattr(value, key, default)


@dataclass(frozen=True)
class PolicyObservation:
    metric: str
    value: float
    market: str
    observed_at: datetime
    source: str
    max_age_seconds: float
    unit: str = "ratio"
    origin_market: str | None = None
    applicable_markets: tuple[str, ...] = ()
    derived_from: tuple[str, ...] = ()
    horizon_seconds: float | None = None
    evidence_id: str = ""

    def as_dict(self) -> dict[str, Any]:
        result = asdict(self)
        result["observed_at"] = self.observed_at.isoformat() if isinstance(self.observed_at, datetime) else str(self.observed_at)
        return result


@dataclass(frozen=True)
class ObservationVerdict:
    observation: PolicyObservation
    accepted: bool
    reason_codes: tuple[str, ...]
    age_seconds: float | None


@dataclass(frozen=True)
class EvidenceProjection:
    market: str
    as_of: datetime
    usable: bool
    reason_codes: tuple[str, ...]
    observations: tuple[PolicyObservation, ...]
    rejected: tuple[ObservationVerdict, ...] = ()
    regime: str = "UNKNOWN"
    context_id: str = ""
    required_metrics: tuple[str, ...] = ()

    @property
    def values(self) -> Mapping[str, float]:
        return MappingProxyType({item.metric: float(item.value) for item in self.observations})

    def as_dict(self) -> dict[str, Any]:
        return {
            "market": self.market, "as_of": self.as_of.isoformat(),
            "usable": self.usable, "reason_codes": list(self.reason_codes),
            "values": dict(self.values), "regime": self.regime,
            "context_id": self.context_id, "required_metrics": list(self.required_metrics),
            "observations": [item.as_dict() for item in self.observations],
            "rejected": [{"observation": item.observation.as_dict(),
                          "reason_codes": list(item.reason_codes), "age_seconds": item.age_seconds}
                         for item in self.rejected],
        }


def project_market_evidence(
    market: str, *, as_of: datetime, observations: Iterable[PolicyObservation],
    required_metrics: Iterable[str] = (), regime: str = "UNKNOWN", context_id: str = "",
    max_observations: int = 256, additional_sources: Iterable[str] = (),
) -> EvidenceProjection:
    """Validate market, source, exact timestamp, horizon, bounds and finite values.

    Different timestamps for a metric select the newest valid reading. Conflicting
    values at that same timestamp invalidate that metric rather than relying on
    input order. Cross-market evidence can only populate ``global_*`` metrics.
    ``usable`` means complete evidence for the caller's required metrics, never
    order permission. Additional sources must be explicitly admitted by an adapter.
    """
    target, moment = _market(market), _datetime(as_of)
    if moment is None:
        raise ValueError("as_of must be a timezone-aware datetime")
    required = tuple(dict.fromkeys(str(name) for name in required_metrics))
    reasons: list[str] = []
    rejected: list[ObservationVerdict] = []
    buckets: dict[str, list[PolicyObservation]] = {}
    sources = _SOURCES | frozenset(str(source).lower() for source in additional_sources)
    if target not in {"KR", "US"}:
        reasons.append("ONTO_POLICY_MARKET_UNKNOWN")
    budget = max(1, min(4096, int(max_observations)))
    for index, observation in enumerate(observations):
        if index >= budget:
            reasons.append("ONTO_POLICY_EVIDENCE_BUDGET_EXCEEDED")
            break
        codes: list[str] = []
        observed = _datetime(observation.observed_at)
        age = (moment - observed).total_seconds() if observed is not None else None
        origin = _market(observation.origin_market or observation.market)
        observed_market = _market(observation.market)
        applicable = {_market(item) for item in observation.applicable_markets}
        is_global = observation.metric.startswith("global_")
        if observed_market != target or (origin != target and not is_global):
            codes.append("ONTO_POLICY_MARKET_MISMATCH")
        if origin != target and (not is_global or target not in applicable):
            codes.append("ONTO_POLICY_CROSS_MARKET_UNVERIFIED")
        source = str(observation.source or "").strip().lower().split(":", 1)[0]
        if source not in sources:
            codes.append("ONTO_POLICY_SOURCE_UNVERIFIED")
        if (source.startswith("context_runtime.") or source == "validated_temporal_rgcn") and not observation.derived_from:
            codes.append("ONTO_POLICY_PROVENANCE_MISSING")
        if not str(observation.metric or "").strip() or not str(observation.unit or "").strip():
            codes.append("ONTO_POLICY_METRIC_INVALID")
        expected_unit = "ratio" if observation.metric in _BOUNDS else {"expected_net_return_bps": "bps", "expected_adverse_excursion_bps": "bps", "expected_downside_net_bps": "bps", "minutes_to_close": "minutes", "session_progress": "ratio"}.get(observation.metric)
        if expected_unit is not None and observation.unit != expected_unit:
            codes.append("ONTO_POLICY_UNIT_MISMATCH")
        if observed is None:
            codes.append("ONTO_POLICY_TIMESTAMP_INVALID")
        elif age is not None and age < 0.0:
            codes.append("ONTO_POLICY_FROM_FUTURE")
        maximum_age = _finite(observation.max_age_seconds)
        if maximum_age is None or maximum_age <= 0.0:
            codes.append("ONTO_POLICY_FRESHNESS_INVALID")
        elif age is not None and age > maximum_age:
            codes.append("ONTO_POLICY_STALE")
        value = _finite(observation.value)
        if value is None:
            codes.append("ONTO_POLICY_VALUE_INVALID")
        elif observation.metric in _BOUNDS:
            lower, upper = _BOUNDS[observation.metric]
            if value < lower or value > upper:
                codes.append("ONTO_POLICY_VALUE_OUT_OF_RANGE")
        if observation.horizon_seconds is not None:
            horizon = _finite(observation.horizon_seconds)
            if horizon is None or horizon <= 0.0:
                codes.append("ONTO_POLICY_HORIZON_INVALID")
        # A volatility rate without its sampling horizon cannot size a time-based
        # policy: a minute standard deviation and an annual one are not comparable.
        elif observation.metric in {"realized_volatility", "downside_volatility"}:
            codes.append("ONTO_POLICY_HORIZON_MISSING")
        if codes:
            rejected.append(ObservationVerdict(observation, False, tuple(dict.fromkeys(codes)), age))
        else:
            buckets.setdefault(observation.metric, []).append(observation)
    accepted: list[PolicyObservation] = []
    for metric, candidates in sorted(buckets.items()):
        newest = max(_datetime(item.observed_at) for item in candidates)
        readings = [item for item in candidates if _datetime(item.observed_at) == newest]
        signatures = {(float(item.value), item.unit, item.horizon_seconds) for item in readings}
        if len(signatures) != 1:
            for item in readings:
                rejected.append(ObservationVerdict(item, False, ("ONTO_POLICY_CONFLICTING_EVIDENCE",), (moment - newest).total_seconds()))
            continue
        accepted.append(sorted(readings, key=lambda item: (item.source, item.evidence_id))[0])
    available = {item.metric for item in accepted}
    missing = [name for name in required if name not in available]
    reasons.extend("ONTO_POLICY_REQUIRED_MISSING:" + name for name in missing)
    if not accepted:
        reasons.append("ONTO_POLICY_NO_VALID_EVIDENCE")
    fatal = bool(reasons)
    reasons.extend(code for verdict in rejected for code in verdict.reason_codes)
    return EvidenceProjection(
        market=target, as_of=moment, usable=not fatal,
        reason_codes=tuple(dict.fromkeys(reasons)), observations=tuple(accepted),
        rejected=tuple(rejected), regime=str(regime or "UNKNOWN"),
        context_id=str(context_id or ""), required_metrics=required,
    )


def project_context_evidence(
    market: str, *, as_of: datetime, domestic: Any = None, global_context: Any = None,
    regime: Any = None, temporal: Any = None, context_id: str = "",
    max_age_seconds: float = 120.0, required_metrics: Iterable[str] = (),
    volatility_horizon_seconds: float | None = None,
) -> EvidenceProjection:
    """Project existing context objects/mappings while preserving their timestamps.

    Regime feature snapshots have no market identity themselves. They are usable
    only with an explicit cycle/context identifier supplied by the market-partitioned
    caller. Global derived factors additionally need accepted indicator relations;
    their age is anchored to the oldest contributing observation, not cycle time.
    """
    target = _market(market)
    records: list[PolicyObservation] = []
    local_context = str(_get(domestic, "context_id", "") or context_id)
    local_identity_matches = domestic is None or _market(_get(domestic, "market", "")) == target
    if domestic is not None:
        local_market = _market(_get(domestic, "market", ""))
        observed_at = _get(domestic, "captured_at")
        for field, metric in _LOCAL_ALIASES.items():
            value = _get(domestic, field)
            if value is not None:
                records.append(PolicyObservation(
                    metric, value, local_market, observed_at,
                    "context_runtime.domestic", max_age_seconds,
                    derived_from=(local_context,) if local_context else (),
                    horizon_seconds=volatility_horizon_seconds if metric == "realized_volatility" else None,
                ))
    elif regime is not None and context_id:
        snapshot = _get(regime, "feature_snapshot", {}) or {}
        for field, metric in _LOCAL_ALIASES.items():
            value = _get(snapshot, field)
            if value is not None:
                records.append(PolicyObservation(
                    metric, value, target, _get(regime, "evaluated_at"),
                    "context_runtime.regime", max_age_seconds, derived_from=(context_id,),
                    horizon_seconds=volatility_horizon_seconds if metric == "realized_volatility" else None,
                ))
    if regime is not None and (local_context or context_id) and local_identity_matches:
        confidence = _get(regime, "confidence")
        if confidence is not None:
            records.append(PolicyObservation(
                "regime_confidence", confidence, target, _get(regime, "evaluated_at"),
                "context_runtime.regime", max_age_seconds,
                derived_from=(local_context or context_id,),
            ))
    if global_context is not None:
        from app.ontology.indicator_graph import _CONTRACTS

        relations = _get(global_context, "indicator_relations", ()) or ()
        valid = []
        allowances = []
        moment = _datetime(as_of)
        context_at = _datetime(_get(global_context, "captured_at"))
        context_current = context_at is not None and moment is not None and 0 <= (moment - context_at).total_seconds() <= max_age_seconds
        declared_market = _get(global_context, "market")
        if declared_market is not None and _market(declared_market) != target:
            context_current = False
        for item in relations:
            contract = _CONTRACTS.get(str(item.get("indicator") or "").upper())
            observed = _datetime(item.get("observed_at"))
            source = str(item.get("source") or "").lower().split(":", 1)[0]
            if not context_current or item.get("usable") is not True or contract is None or observed is None:
                continue
            if _market(item.get("target_market")) != target or target not in contract.targets or source not in contract.providers:
                continue
            if not 0 <= (moment - observed).total_seconds() <= contract.max_age_seconds:
                continue
            valid.append(item)
            allowances.append(contract.max_age_seconds)
        identifier = str(_get(global_context, "context_id", "") or "")
        if valid and identifier:
            oldest = min(_datetime(item["observed_at"]) for item in valid)
            for field in ("direction", "momentum", "risk_sentiment", "volatility", "rates_pressure", "fx_pressure", "global_alignment"):
                value = _get(global_context, field)
                if value is not None:
                    records.append(PolicyObservation(
                        "global_" + field.removeprefix("global_"), value, target, oldest,
                        "context_runtime.global", min(allowances), origin_market="GLOBAL",
                        applicable_markets=(target,),
                        derived_from=(identifier, *(str(item["source"]) + ":" + str(item["indicator"]) for item in valid)),
                    ))
    if temporal is not None:
        for metric in ("minutes_to_close", "session_progress"):
            value = _get(temporal, metric)
            if value is not None:
                records.append(PolicyObservation(
                    metric, value, _market(_get(temporal, "market_group", "")),
                    _get(temporal, "as_of"), "context_runtime.temporal", max_age_seconds,
                    unit="minutes" if metric == "minutes_to_close" else "ratio",
                    derived_from=(context_id,) if context_id else (),
                ))
    routing = _get(regime, "routing_regime")
    if routing is None and isinstance(regime, Mapping):
        routing = _get(regime.get("routing", {}), "regime")
    return project_market_evidence(
        target, as_of=as_of, observations=records, required_metrics=required_metrics,
        regime=routing or _get(regime, "dominant", "UNKNOWN"), context_id=local_context or context_id,
    )


def materialize_evidence_graph(
    projection: EvidenceProjection, *, threshold_values: Mapping[str, float] | None = None,
    performance_assessments: Iterable[Mapping[str, Any]] = (),
    instrument_symbol: str | None = None,
) -> Any:
    """Create an RDF audit graph explicitly outside the realtime receive path.

    Only accepted evidence is linked through ``usesEvidence``. Rejected records
    remain visible through ``rejectedEvidence`` and cannot support a policy.
    This serialization does not execute OWL rules or grant live authorization.
    """
    from rdflib import Graph, Literal, Namespace, RDF, XSD

    pol = Namespace(POLICY_NAMESPACE)
    graph = Graph().parse(SCHEMA_PATH, format="turtle")
    graph.bind("pol", pol)
    identity = {"market": projection.market, "as_of": projection.as_of.isoformat(),
                "context_id": projection.context_id, "symbol": instrument_symbol,
                "regime": projection.regime,
                "observations": [item.as_dict() for item in projection.observations],
                "rejected": [item.observation.as_dict() for item in projection.rejected]}
    digest = hashlib.sha256(json.dumps(identity, sort_keys=True, default=str).encode()).hexdigest()[:24]
    root = pol["projection_" + digest]
    graph.add((root, RDF.type, pol.EvidenceProjection))
    graph.add((root, pol.appliesToMarket, pol[projection.market]))
    graph.add((root, pol.asOf, Literal(projection.as_of, datatype=XSD.dateTime)))
    graph.add((root, pol.usable, Literal(projection.usable)))
    graph.add((root, pol.contextId, Literal(projection.context_id)))
    instrument = None
    if instrument_symbol:
        instrument = pol["instrument_" + quote(projection.market + "_" + str(instrument_symbol), safe="_")]
        graph.add((instrument, RDF.type, pol.Instrument))
        graph.add((instrument, pol.tradedInMarket, pol[projection.market]))
        graph.add((instrument, pol.symbol, Literal(str(instrument_symbol))))
        graph.add((root, pol.projectionForInstrument, instrument))
    for reason in projection.reason_codes:
        graph.add((root, pol.reasonCode, Literal(reason)))
    regime = pol["regime_" + quote(projection.market + "_" + projection.regime, safe="_")]
    graph.add((regime, RDF.type, pol.Regime))
    graph.add((regime, pol.appliesToMarket, pol[projection.market]))
    graph.add((regime, pol.regimeLabel, Literal(projection.regime)))
    graph.add((root, pol.observedRegime, regime))
    verdicts = [ObservationVerdict(item, True, (), (projection.as_of - _datetime(item.observed_at)).total_seconds()) for item in projection.observations]
    verdicts.extend(projection.rejected)
    for index, verdict in enumerate(verdicts):
        record = verdict.observation
        node = pol[f"observation_{digest}_{index}"]
        graph.add((node, RDF.type, pol.AcceptedObservation if verdict.accepted else pol.RejectedObservation))
        graph.add((node, RDF.type, pol.Observation))
        graph.add((root, pol.usesEvidence if verdict.accepted else pol.rejectedEvidence, node))
        graph.add((node, pol.appliesToMarket, pol[_market(record.market)]))
        graph.add((node, pol.originMarket, pol[_market(record.origin_market or record.market)]))
        if instrument is not None and record.metric in {"realized_volatility", "downside_volatility", "spread_rate", "quote_age_seconds", "expected_net_return_bps", "expected_downside_net_bps"}:
            graph.add((node, pol.observedInstrument, instrument))
        metric = pol["metric_" + quote(record.metric, safe="_")]
        graph.add((metric, RDF.type, pol.Metric))
        graph.add((metric, pol.metricName, Literal(record.metric)))
        graph.add((node, pol.observedMetric, metric))
        source = pol["source_" + quote(record.source, safe="_")]
        graph.add((source, RDF.type, pol.DataSource))
        graph.add((source, pol.sourceName, Literal(record.source)))
        graph.add((node, pol.providedBy, source))
        number = _finite(record.value)
        if number is not None:
            graph.add((node, pol.numericValue, Literal(number, datatype=XSD.double)))
        graph.add((node, pol.unit, Literal(record.unit)))
        observed = _datetime(record.observed_at)
        if observed is not None:
            graph.add((node, pol.observedAt, Literal(observed, datatype=XSD.dateTime)))
        maximum = _finite(record.max_age_seconds)
        if maximum is not None:
            graph.add((node, pol.maximumAgeSeconds, Literal(maximum, datatype=XSD.double)))
        if verdict.age_seconds is not None:
            graph.add((node, pol.ageSeconds, Literal(verdict.age_seconds, datatype=XSD.double)))
        if record.horizon_seconds is not None and _finite(record.horizon_seconds) is not None:
            graph.add((node, pol.horizonSeconds, Literal(float(record.horizon_seconds), datatype=XSD.double)))
        for reference in record.derived_from:
            graph.add((node, pol.evidenceReference, Literal(reference)))
        if record.evidence_id:
            graph.add((node, pol.evidenceReference, Literal(record.evidence_id)))
        for reason in verdict.reason_codes:
            graph.add((node, pol.reasonCode, Literal(reason)))
    if threshold_values is not None:
        threshold_digest = hashlib.sha256((digest + json.dumps(dict(threshold_values), sort_keys=True, default=str)).encode()).hexdigest()[:24]
        policy = pol["policy_" + threshold_digest]
        assessment = pol["risk_" + digest]
        graph.add((policy, RDF.type, pol.ThresholdPolicy))
        graph.add((policy, pol.appliesToMarket, pol[projection.market]))
        graph.add((policy, pol.derivedFromProjection, root))
        graph.add((assessment, RDF.type, pol.RiskAssessment))
        graph.add((assessment, pol.appliesToMarket, pol[projection.market]))
        graph.add((assessment, pol.assessedFrom, root))
        graph.add((assessment, pol.generatesPolicy, policy))
        if instrument is not None:
            graph.add((policy, pol.policyForInstrument, instrument))
            graph.add((assessment, pol.assessesInstrument, instrument))
        graph.add((policy, pol.policyMode, Literal("ADAPTIVE" if projection.usable else "PROTECTIVE_FALLBACK")))
        for name, value in sorted(threshold_values.items()):
            number = _finite(value)
            if number is None:
                raise ValueError("threshold values must be finite numbers")
            threshold = pol["threshold_" + threshold_digest + "_" + quote(str(name), safe="_")]
            graph.add((threshold, RDF.type, pol.Threshold))
            graph.add((threshold, pol.metricName, Literal(str(name))))
            graph.add((threshold, pol.numericValue, Literal(number, datatype=XSD.double)))
            graph.add((policy, pol.hasThreshold, threshold))
    for index, assessment in enumerate(performance_assessments):
        for source in ("live", "shadow"):
            if isinstance(assessment.get(source), Mapping):
                item = {**assessment, **assessment[source]}
                item["observed_at"] = assessment[source].get("last_observed_at") or assessment.get("observed_at")
                _attach_performance(graph, pol, root, digest, str(index) + "_" + source, projection, item)
        if "live" not in assessment and "shadow" not in assessment:
            _attach_performance(graph, pol, root, digest, str(index), projection, assessment)
    return graph


def _attach_performance(graph: Any, pol: Any, root: Any, digest: str, index: str, projection: EvidenceProjection, assessment: Mapping[str, Any]) -> None:
    from rdflib import Literal, RDF, XSD

    if _market(assessment.get("market")) != projection.market:
        raise ValueError("strategy performance market must match projection")
    strategy_id = str(assessment.get("strategy_id") or "")
    if not strategy_id:
        raise ValueError("strategy performance requires strategy_id")
    strategy = pol["strategy_" + quote(projection.market + "_" + strategy_id, safe="_")]
    node = pol[f"performance_{digest}_{index}"]
    graph.add((strategy, RDF.type, pol.Strategy))
    graph.add((strategy, pol.appliesToMarket, pol[projection.market]))
    graph.add((strategy, pol.strategyId, Literal(strategy_id)))
    graph.add((node, RDF.type, pol.StrategyPerformanceObservation))
    graph.add((node, pol.appliesToMarket, pol[projection.market]))
    graph.add((node, pol.observedStrategy, strategy))
    graph.add((strategy, pol.hasPerformance, node))
    graph.add((root, pol.considersPerformance, node))
    regime = pol["regime_" + quote(projection.market + "_" + str(assessment.get("regime") or "UNKNOWN"), safe="_")]
    graph.add((regime, RDF.type, pol.Regime))
    graph.add((regime, pol.appliesToMarket, pol[projection.market]))
    graph.add((regime, pol.regimeLabel, Literal(str(assessment.get("regime") or "UNKNOWN"))))
    graph.add((node, pol.observedRegime, regime))
    for key, predicate in (("realized_net_bps", "realizedNetBps"), ("expected_net_bps", "expectedNetBps"),
                           ("effective_sample_count", "effectiveSampleCount"), ("sample_count", "sampleCount"),
                           ("lower_net_bps", "lowerNetBps"), ("upper_net_bps", "upperNetBps")):
        number = _finite(assessment.get(key))
        if number is not None:
            graph.add((node, pol[predicate], Literal(number, datatype=XSD.double)))
    for key, predicate in (("performance_state", "performanceState"), ("evidence_source", "evidenceSource")):
        if assessment.get(key) is not None:
            graph.add((node, pol[predicate], Literal(str(assessment[key]))))
    observed = _datetime(assessment.get("observed_at"))
    if observed is not None:
        graph.add((node, pol.observedAt, Literal(observed, datatype=XSD.dateTime)))
    for reference in assessment.get("evidence_refs", ()):
        graph.add((node, pol.evidenceReference, Literal(str(reference))))


def validate_evidence_graph(graph: Any) -> tuple[bool, str]:
    """Run SHACL only for audit/tests, not synchronous order or quote processing."""
    from pyshacl import validate
    from rdflib import Graph

    conforms, _, report = validate(graph, shacl_graph=Graph().parse(SHAPES_PATH, format="turtle"), inference="rdfs")
    return bool(conforms), str(report)
