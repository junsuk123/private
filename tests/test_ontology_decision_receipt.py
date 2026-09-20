from dataclasses import replace
from datetime import timedelta

import pytest

from app.ontology.decision_receipt import decision_receipt_to_rdf, validate_plan_authority
from app.trading.trade_plan import EntryRule, ExitRules, TradePlan
from test_ontology_risk_authority import NOW, _case


def _approved_plan(tmp_path):
    manager, policy, intent, account, market = _case(tmp_path)
    result = manager.validate(intent, account, market, ontology_policy=policy, now=NOW)
    assert result.approved
    receipt = result.metadata["ontology_risk_authority"]
    return TradePlan(
        plan_id="actual-authorized-plan", created_at=NOW, expires_at=policy.expires_at,
        symbol="000660", market="KRX", direction="LONG", strategy_id="example",
        quantity=result.final_order.quantity, max_notional=receipt["actual_notional"],
        entry_rule=EntryRule("entry", min_price=990, max_price=receipt["authorized_price"]),
        exit_rules=ExitRules(.1, .01), cancel_rule="cancel", expected_net_edge_bps=900,
        cost_snapshot=result.metadata["cost_breakdown"],
        risk_snapshot={"ontology_authority": receipt, "ontology_risk_policy": policy.as_dict()},
        weekday_time_context={}, source_ids=("actual-quote",), reference_price=1000,
        order_contract={"position_direction": "LONG", "position_effect": "OPEN", "execution_product": "CASH"},
    )


def test_receipt_allows_only_original_bounds_and_broker_clipping(tmp_path):
    plan = _approved_plan(tmp_path)
    assert validate_plan_authority(plan, NOW, symbol="000660", market="KOSPI", price=1000) == ()
    clipped = plan.with_broker_clip(20, reason="native_cash")
    assert validate_plan_authority(clipped, NOW, price=995) == ()
    assert validate_plan_authority(plan.with_entry_fill(1000, 10), NOW) == ()
    assert validate_plan_authority(replace(plan, expires_at=NOW + timedelta(seconds=1)), NOW) == ()


@pytest.mark.parametrize("mutation,reason", [
    (lambda p: replace(p, risk_snapshot={}), "ONTOLOGY_AUTHORITY_RECEIPT_MISSING"),
    (lambda p: replace(p, quantity=p.quantity + 1), "ONTOLOGY_AUTHORITY_QUANTITY_EXCEEDED"),
    (lambda p: replace(p, max_notional=p.max_notional + 1), "ONTOLOGY_AUTHORITY_PRICE_OR_NOTIONAL_EXCEEDED"),
    (lambda p: replace(p, reference_price=1001), "ONTOLOGY_AUTHORITY_PRICE_OR_NOTIONAL_EXCEEDED"),
    (lambda p: replace(p, entry_rule=replace(p.entry_rule, max_price=1001)), "ONTOLOGY_AUTHORITY_PRICE_OR_NOTIONAL_EXCEEDED"),
    (lambda p: replace(p, direction="SHORT"), "ONTOLOGY_AUTHORITY_CONTRACT_MISMATCH"),
    (lambda p: replace(p, order_contract={}), "ONTOLOGY_AUTHORITY_CONTRACT_MISMATCH"),
    (lambda p: replace(p, symbol="005930"), "ONTOLOGY_AUTHORITY_SCOPE_MISMATCH"),
    (lambda p: replace(p, market="US"), "ONTOLOGY_AUTHORITY_SCOPE_MISMATCH"),
    (lambda p: replace(p, expires_at=p.expires_at + timedelta(seconds=1)), "ONTOLOGY_AUTHORITY_EXPIRED_OR_FUTURE"),
])
def test_plan_cannot_expand_or_relabel_an_existing_grant(tmp_path, mutation, reason):
    assert reason in validate_plan_authority(mutation(_approved_plan(tmp_path)), NOW)


@pytest.mark.parametrize("field,value,reason", [
    ("approved", "true", "ONTOLOGY_AUTHORITY_RECEIPT_INVALID"),
    ("authority_id", "legacy-risk-manager", "ONTOLOGY_AUTHORITY_RECEIPT_INVALID"),
    ("phase", "exposure_reduction", "ONTOLOGY_AUTHORITY_RECEIPT_INVALID"),
    ("policy_id", "other-policy", "ONTOLOGY_AUTHORITY_POLICY_MISMATCH"),
    ("evaluated_at", (NOW + timedelta(seconds=1)).isoformat(), "ONTOLOGY_AUTHORITY_EXPIRED_OR_FUTURE"),
    ("evaluated_at", NOW.replace(tzinfo=None).isoformat(), "ONTOLOGY_AUTHORITY_EXPIRED_OR_FUTURE"),
    ("expires_at", NOW.isoformat(), "ONTOLOGY_AUTHORITY_EXPIRED_OR_FUTURE"),
    ("quantity", float("nan"), "ONTOLOGY_AUTHORITY_QUANTITY_EXCEEDED"),
    ("authorized_price", float("inf"), "ONTOLOGY_AUTHORITY_PRICE_OR_NOTIONAL_EXCEEDED"),
    ("all_in_cost_rate", float("nan"), "ONTOLOGY_AUTHORITY_RECEIPT_INVALID"),
])
def test_malformed_receipt_fails_closed_without_market_reassessment(tmp_path, field, value, reason):
    plan = _approved_plan(tmp_path)
    snapshot = dict(plan.risk_snapshot)
    snapshot["ontology_authority"] = {**snapshot["ontology_authority"], field: value}
    assert reason in validate_plan_authority(replace(plan, risk_snapshot=snapshot), NOW)


def test_expiry_price_and_fully_filled_grants_are_not_reusable(tmp_path):
    plan = _approved_plan(tmp_path)
    assert "ONTOLOGY_AUTHORITY_EXPIRED_OR_FUTURE" in validate_plan_authority(plan, plan.expires_at)
    assert "ONTOLOGY_AUTHORITY_PRICE_OUTSIDE_APPROVAL" in validate_plan_authority(plan, NOW, price=1001)
    assert "ONTOLOGY_AUTHORITY_PRICE_OUTSIDE_APPROVAL" in validate_plan_authority(plan, NOW, price=989)
    assert "ONTOLOGY_AUTHORITY_QUANTITY_EXCEEDED" in validate_plan_authority(plan.with_entry_fill(1000, plan.quantity), NOW)


@pytest.mark.parametrize("value", [float("nan"), float("inf"), "unknown", None, 0, -1])
def test_frozen_execution_quote_limit_must_remain_finite_and_positive(tmp_path, value):
    plan = _approved_plan(tmp_path)
    snapshot = dict(plan.risk_snapshot)
    snapshot["ontology_risk_policy"] = {**snapshot["ontology_risk_policy"], "max_quote_age_seconds": value}
    assert "ONTOLOGY_AUTHORITY_POLICY_INVALID" in validate_plan_authority(replace(plan, risk_snapshot=snapshot), NOW)


def test_actual_approved_and_rejected_decisions_export_named_ontology_instances(tmp_path):
    from rdflib import Namespace, OWL, RDF, Literal, XSD
    from app.ontology.policy_evidence import POLICY_NAMESPACE
    pol = Namespace(POLICY_NAMESPACE)
    plan = _approved_plan(tmp_path)
    receipt = plan.risk_snapshot["ontology_authority"]
    graph = decision_receipt_to_rdf(receipt)
    node = next(graph.subjects(RDF.type, pol.ApprovedRiskDecision))
    assert (node, RDF.type, OWL.NamedIndividual) in graph
    assert (node, pol.decidedBy, pol.OntologyRiskAuthority) in graph
    policy = graph.value(node, pol.usesThresholdPolicy)
    assert (policy, pol.policyId, Literal(receipt["policy_id"])) in graph
    assert (node, pol.approvedQuantity, Literal(plan.quantity, datatype=XSD.integer)) in graph
    manager, _, intent, account, market = _case(tmp_path)
    rejected = manager.validate(intent, account, market, now=NOW).metadata["ontology_risk_authority"]
    graph = decision_receipt_to_rdf(rejected)
    node = next(graph.subjects(RDF.type, pol.RejectedRiskDecision))
    assert (node, pol.reasonCode, Literal("ONTOLOGY_POLICY_UNAVAILABLE")) in graph
    assert (node, pol.approved, Literal(False, datatype=XSD.boolean)) in graph
