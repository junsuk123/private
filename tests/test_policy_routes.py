from __future__ import annotations

from datetime import datetime, timezone

from fastapi import FastAPI
from fastapi.testclient import TestClient
from rdflib import Graph, Namespace, RDF

from app.ontology.policy_evidence import POLICY_NAMESPACE, PolicyObservation, materialize_evidence_graph, project_market_evidence
from app.ontology.policy_routes import create_policy_router


class Runtime:
    def __init__(self):
        self.snapshot_calls = 0
        self.audit_calls = []

    def resolve(self, **kwargs):
        raise AssertionError("read-only routes cannot resolve or obtain market data")

    def snapshot(self):
        self.snapshot_calls += 1
        return {"policy_count": 1, "policies": {"KR:005930": {"policy": {"symbol": "005930"}}}}

    def evidence_graph(self, symbol, market):
        self.audit_calls.append((symbol, market))
        if symbol != "005930":
            raise KeyError(symbol)
        now = datetime(2026, 9, 21, tzinfo=timezone.utc)
        projection = project_market_evidence(market, as_of=now,
            observations=(PolicyObservation("spread_rate", .001, market, now, "kis_realtime", 20),))
        return materialize_evidence_graph(projection)


def client(runtime):
    app = FastAPI()
    app.include_router(create_policy_router(lambda: runtime))
    return TestClient(app)


def test_snapshot_only_reads_cache_and_schema_does_not_touch_runtime():
    runtime = Runtime()
    http = client(runtime)
    assert http.get("/api/ontology/policy").json()["policy_count"] == 1
    schema = http.get("/api/ontology/policy/schema")
    assert schema.status_code == 200
    assert schema.headers["content-type"].startswith("text/turtle")
    graph = Graph().parse(data=schema.text, format="turtle")
    assert len(graph) > 100
    assert runtime.snapshot_calls == 1 and runtime.audit_calls == []


def test_explicit_rdf_audit_works_without_resolve_and_validates_paths():
    runtime = Runtime()
    http = client(runtime)
    response = http.get("/api/ontology/policy/KR/005930")
    assert response.status_code == 200
    pol = Namespace(POLICY_NAMESPACE)
    graph = Graph().parse(data=response.text, format="turtle")
    assert list(graph.subjects(RDF.type, pol.EvidenceProjection))
    assert runtime.audit_calls == [("005930", "KR")]
    assert http.get("/api/ontology/policy/UK/ABC").status_code == 422
    assert http.get("/api/ontology/policy/KR/%3Cscript%3E").status_code == 422
    assert http.get("/api/ontology/policy/KR/999999").status_code == 404
    assert http.post("/api/ontology/policy/KR/005930").status_code == 405


def test_absent_runtime_never_starts_work():
    http = client(None)
    assert http.get("/api/ontology/policy").json() == {"available": False, "policy_count": 0, "policies": {}}
    assert http.get("/api/ontology/policy/schema").status_code == 200
    assert http.get("/api/ontology/policy/KR/005930").status_code == 503
