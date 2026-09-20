from __future__ import annotations

from dataclasses import replace
from datetime import datetime, timedelta, timezone
from types import SimpleNamespace

import pytest
from rdflib import Graph, Literal, Namespace, OWL, RDF, RDFS, XSD

from app.ontology.policy_evidence import (
    POLICY_NAMESPACE, SCHEMA_PATH, EvidenceProjection, PolicyObservation,
    materialize_evidence_graph, project_context_evidence, project_market_evidence,
    validate_evidence_graph,
)

NOW = datetime(2026, 9, 21, 1, 30, tzinfo=timezone.utc)
POL = Namespace(POLICY_NAMESPACE)


def observation(metric="spread_rate", value=0.001, **kwargs):
    values = dict(metric=metric, value=value, market="KR", observed_at=NOW,
                  source="kis_realtime", max_age_seconds=30.0)
    values.update(kwargs)
    return PolicyObservation(**values)


def projection(*observations, **kwargs):
    return project_market_evidence("KR", as_of=NOW, observations=observations,
                                   context_id="kr-cycle-17", **kwargs)


def test_formal_schema_has_all_five_components_and_typed_properties():
    graph = Graph().parse(SCHEMA_PATH, format="turtle")
    for name in ("Market", "Session", "Instrument", "Observation", "Regime", "Strategy", "Position", "RiskAssessment", "ThresholdPolicy"):
        assert (POL[name], RDF.type, OWL.Class) in graph
    assert (POL.KR, RDF.type, OWL.NamedIndividual) in graph
    assert (POL.US, RDF.type, POL.Market) in graph
    for property_type in (OWL.ObjectProperty, OWL.DatatypeProperty):
        properties = list(graph.subjects(RDF.type, property_type))
        assert properties
        for predicate in properties:
            assert list(graph.objects(predicate, RDFS.domain))
            assert list(graph.objects(predicate, RDFS.range))
    assert (POL.KR, OWL.differentFrom, POL.US) in graph
    assert (POL.AcceptedObservation, OWL.disjointWith, POL.RejectedObservation) in graph


def test_projection_preserves_real_zero_and_signed_evidence():
    result = projection(observation("market_breadth", -0.6), observation("spread_rate", 0.0))
    assert result.usable
    assert result.values == {"market_breadth": -0.6, "spread_rate": 0.0}
    with pytest.raises(TypeError):
        result.values["market_breadth"] = 0.0


@pytest.mark.parametrize(("changes", "reason"), [
    ({"value": float("nan")}, "ONTO_POLICY_VALUE_INVALID"),
    ({"value": float("inf")}, "ONTO_POLICY_VALUE_INVALID"),
    ({"value": True}, "ONTO_POLICY_VALUE_INVALID"),
    ({"value": -0.1}, "ONTO_POLICY_VALUE_OUT_OF_RANGE"),
    ({"observed_at": NOW + timedelta(microseconds=1)}, "ONTO_POLICY_FROM_FUTURE"),
    ({"observed_at": NOW - timedelta(seconds=31)}, "ONTO_POLICY_STALE"),
    ({"observed_at": NOW.replace(tzinfo=None)}, "ONTO_POLICY_TIMESTAMP_INVALID"),
    ({"market": "US"}, "ONTO_POLICY_MARKET_MISMATCH"),
    ({"source": "untrusted_blog"}, "ONTO_POLICY_SOURCE_UNVERIFIED"),
    ({"source": "context_runtime.domestic"}, "ONTO_POLICY_PROVENANCE_MISSING"),
    ({"max_age_seconds": float("nan")}, "ONTO_POLICY_FRESHNESS_INVALID"),
    ({"max_age_seconds": -1}, "ONTO_POLICY_FRESHNESS_INVALID"),
    ({"unit": "bps"}, "ONTO_POLICY_UNIT_MISMATCH"),
])
def test_invalid_evidence_cannot_become_a_policy_input(changes, reason):
    result = projection(observation(**changes), required_metrics=("spread_rate",))
    assert not result.usable
    assert "spread_rate" not in result.values
    assert reason in result.reason_codes
    assert "ONTO_POLICY_REQUIRED_MISSING:spread_rate" in result.reason_codes


def test_cross_market_asserted_applicability_never_substitutes_local_volatility():
    result = projection(observation("realized_volatility", 0.01, origin_market="US",
                                    applicable_markets=("KR",), horizon_seconds=60))
    assert not result.usable
    assert "ONTO_POLICY_MARKET_MISMATCH" in result.reason_codes
    global_result = projection(observation("global_volatility", 0.3, origin_market="US",
                                           applicable_markets=("KR",)))
    assert global_result.usable


def test_volatility_horizon_is_mandatory_and_preserved():
    result = projection(observation("realized_volatility", 0.002))
    assert not result.usable
    assert "ONTO_POLICY_HORIZON_MISSING" in result.reason_codes
    result = projection(observation("realized_volatility", 0.002, horizon_seconds=60))
    assert result.usable
    assert result.observations[0].horizon_seconds == 60


def test_latest_valid_evidence_selected_and_equal_time_conflict_rejected():
    older = observation(value=0.001, observed_at=NOW - timedelta(seconds=10))
    newest = observation(value=0.002)
    assert projection(newest, older).values["spread_rate"] == 0.002
    conflict = projection(newest, observation(value=0.003))
    assert not conflict.usable
    assert "ONTO_POLICY_CONFLICTING_EVIDENCE" in conflict.reason_codes
    assert projection(newest, newest).values == {"spread_rate": 0.002}


def test_sparse_context_accepts_only_present_metrics_and_budget_failure_is_closed():
    result = projection(observation("trend_strength", 0.2))
    assert result.usable
    assert "realized_volatility" not in result.values
    limited = projection(observation(), observation("liquidity_score", 0.8), max_observations=1)
    assert not limited.usable
    assert "ONTO_POLICY_EVIDENCE_BUDGET_EXCEEDED" in limited.reason_codes


def test_domestic_context_market_and_original_capture_time_cannot_be_relabelled():
    domestic = SimpleNamespace(market="US", context_id="us-context", captured_at=NOW,
                               direction=0.4, volatility=0.005, liquidity=0.8)
    result = project_context_evidence("KR", as_of=NOW, domestic=domestic,
                                      volatility_horizon_seconds=60)
    assert not result.usable
    domestic.market = "KR"
    domestic.captured_at = NOW - timedelta(minutes=10)
    result = project_context_evidence("KR", as_of=NOW, domestic=domestic,
                                      volatility_horizon_seconds=60)
    assert not result.usable
    assert "ONTO_POLICY_STALE" in result.reason_codes


def test_regime_confidence_cannot_rescue_explicitly_wrong_market_context():
    domestic = SimpleNamespace(market="US", context_id="us-context", captured_at=NOW, direction=.4)
    regime = SimpleNamespace(evaluated_at=NOW, confidence=.9, routing_regime="TREND_LOW_VOL")
    result = project_context_evidence("KR", as_of=NOW, domestic=domestic, regime=regime, context_id="kr-cycle")
    assert not result.usable
    assert "regime_confidence" not in result.values


def test_regime_snapshot_needs_explicit_market_cycle_provenance():
    regime = SimpleNamespace(feature_snapshot={"direction": -0.4, "volatility": 0.01,
                              "liquidity": None}, evaluated_at=NOW, confidence=0.8,
                              routing_regime="RISK_OFF")
    assert not project_context_evidence("US", as_of=NOW, regime=regime).usable
    result = project_context_evidence("US", as_of=NOW, regime=regime, context_id="US-cycle",
                                      volatility_horizon_seconds=60)
    assert result.usable
    assert result.regime == "RISK_OFF"
    assert result.values["trend_strength"] == -0.4
    assert result.observations[0].derived_from == ("US-cycle",)
    assert "liquidity_score" not in result.values


def test_global_context_requires_indicator_links_and_retains_underlying_time():
    context = {"context_id": "global-1", "captured_at": NOW, "direction": 0.4,
               "indicator_relations": [{"usable": True, "target_market": "KR",
                   "observed_at": (NOW - timedelta(days=4)).isoformat(),
                   "source": "fred", "indicator": "SP500"}]}
    result = project_context_evidence("KR", as_of=NOW, global_context=context)
    assert not result.usable
    assert "global_direction" not in result.values
    context["indicator_relations"][0]["observed_at"] = (NOW - timedelta(hours=20)).isoformat()
    daily = project_context_evidence("KR", as_of=NOW, global_context=context)
    assert daily.usable
    assert daily.observations[0].observed_at == NOW - timedelta(hours=20)
    context["indicator_relations"][0]["observed_at"] = NOW.isoformat()
    result = project_context_evidence("KR", as_of=NOW, global_context=context)
    assert result.usable
    assert result.values == {"global_direction": 0.4}
    context["indicator_relations"][0]["target_market"] = "US"
    assert not project_context_evidence("KR", as_of=NOW, global_context=context).usable


def test_rdf_export_and_shacl_validate_auditable_policy_and_rejected_records():
    result = projection(observation(), observation("liquidity_score", float("nan")))
    graph = materialize_evidence_graph(result, threshold_values={"stop_loss_rate": 0.009})
    conforms, report = validate_evidence_graph(graph)
    assert conforms, report
    assert len(list(graph.subjects(RDF.type, POL.AcceptedObservation))) == 1
    assert len(list(graph.subjects(RDF.type, POL.RejectedObservation))) == 1
    policy = next(graph.subjects(RDF.type, POL.ThresholdPolicy))
    assert (policy, POL.policyMode, Literal("ADAPTIVE")) in graph


@pytest.mark.parametrize("corruption", ["market", "future", "stale", "contradiction", "missing_source", "adaptive_incomplete"])
def test_shacl_rejects_contradictory_or_incomplete_audit_graph(corruption):
    graph = materialize_evidence_graph(projection(observation()), threshold_values={"stop_loss_rate": 0.01})
    node = next(graph.subjects(RDF.type, POL.AcceptedObservation))
    root = next(graph.subjects(RDF.type, POL.EvidenceProjection))
    if corruption == "market":
        graph.set((node, POL.appliesToMarket, POL.US))
    elif corruption == "future":
        graph.set((node, POL.observedAt, Literal(NOW + timedelta(seconds=1), datatype=XSD.dateTime)))
    elif corruption == "stale":
        graph.set((node, POL.ageSeconds, Literal(31.0, datatype=XSD.double)))
    elif corruption == "contradiction":
        graph.add((node, RDF.type, POL.RejectedObservation))
    elif corruption == "missing_source":
        graph.remove((node, POL.providedBy, None))
    elif corruption == "adaptive_incomplete":
        graph.set((root, POL.usable, Literal(False)))
    assert not validate_evidence_graph(graph)[0]


def test_strategy_performance_preserves_live_shadow_and_realized_predicted_separation():
    assessment = {
        "strategy_id": "us_momentum", "market": "KR", "regime": "TREND_LOW_VOL",
        "performance_state": "SHADOW_ONLY", "observed_at": NOW.isoformat(),
        "live": {"evidence_source": "LIVE", "sample_count": 2, "realized_net_bps": -25,
                 "expected_net_bps": 15},
        "shadow": {"evidence_source": "SHADOW", "sample_count": 0,
                   "realized_net_bps": None, "expected_net_bps": None},
    }
    graph = materialize_evidence_graph(projection(observation()), performance_assessments=(assessment,))
    assert validate_evidence_graph(graph)[0]
    nodes = list(graph.subjects(RDF.type, POL.StrategyPerformanceObservation))
    assert len(nodes) == 2
    live = next(node for node in nodes if (node, POL.evidenceSource, Literal("LIVE")) in graph)
    shadow = next(node for node in nodes if (node, POL.evidenceSource, Literal("SHADOW")) in graph)
    assert float(graph.value(live, POL.realizedNetBps)) == -25.0
    assert float(graph.value(live, POL.expectedNetBps)) == 15.0
    assert graph.value(shadow, POL.realizedNetBps) is None
    with pytest.raises(ValueError, match="market"):
        materialize_evidence_graph(projection(observation()), performance_assessments=({**assessment, "market": "US"},))


def test_materializer_does_not_serialize_nonfinite_thresholds():
    with pytest.raises(ValueError, match="finite"):
        materialize_evidence_graph(projection(observation()), threshold_values={"stop_loss_rate": float("nan")})


def test_same_cycle_different_instruments_and_policy_versions_have_distinct_rdf_identity():
    data = projection(observation())
    first = materialize_evidence_graph(data, instrument_symbol="005930", threshold_values={"stop": .01})
    other = materialize_evidence_graph(data, instrument_symbol="000660", threshold_values={"stop": .01})
    revised = materialize_evidence_graph(data, instrument_symbol="005930", threshold_values={"stop": .005})
    assert set(first.subjects(RDF.type, POL.EvidenceProjection)) != set(other.subjects(RDF.type, POL.EvidenceProjection))
    assert set(first.subjects(RDF.type, POL.ThresholdPolicy)) != set(revised.subjects(RDF.type, POL.ThresholdPolicy))
    assert set(first.subjects(RDF.type, POL.EvidenceProjection)) == set(revised.subjects(RDF.type, POL.EvidenceProjection))


def test_naive_decision_time_rejected_instead_of_guessing_timezone():
    with pytest.raises(ValueError, match="timezone-aware"):
        project_market_evidence("KR", as_of=NOW.replace(tzinfo=None), observations=())
