"""Validate a frozen ontology decision's bounds without judging the trade again."""
from __future__ import annotations

from collections.abc import Mapping
from datetime import datetime
import math

from app.ontology.risk_authority import AUTHORITY_ID


def _time(value):
    try:
        parsed = value if isinstance(value, datetime) else datetime.fromisoformat(str(value))
        return parsed if parsed.tzinfo is not None and parsed.utcoffset() is not None else None
    except (ValueError, TypeError, OverflowError):
        return None


def _number(value):
    if isinstance(value, bool):
        return None
    try:
        parsed = float(value)
        return parsed if math.isfinite(parsed) else None
    except (ValueError, TypeError, OverflowError):
        return None


def _market(value):
    text = str(value or "").strip().upper()
    if text in {"KR", "KRX", "KOSPI", "KOSDAQ", "KONEX", "NXT"}:
        return "KR"
    if text in {"US", "US-LISTED", "NASDAQ", "NASD", "NYSE", "AMEX", "OVERSEAS"}:
        return "US"
    return None


def validate_plan_authority(plan, now, *, symbol=None, market=None, price=None) -> tuple[str, ...]:
    """Accept only a current, scoped cash-long grant that contains this order.

    No account/model/graph calls or discretionary risk criteria are used here.
    Broker clipping may reduce quantity; it can never extend the original grant.
    """
    snapshot = getattr(plan, "risk_snapshot", None)
    receipt = snapshot.get("ontology_authority") if isinstance(snapshot, Mapping) else None
    policy = snapshot.get("ontology_risk_policy") if isinstance(snapshot, Mapping) else None
    if not isinstance(receipt, Mapping):
        return ("ONTOLOGY_AUTHORITY_RECEIPT_MISSING",)
    reasons: list[str] = []

    def require(okay, code):
        if not okay and code not in reasons:
            reasons.append(code)

    require(receipt.get("authority_id") == AUTHORITY_ID and receipt.get("approved") is True
            and receipt.get("phase") == "entry_assessment" and bool(receipt.get("evaluation_id")),
            "ONTOLOGY_AUTHORITY_RECEIPT_INVALID")
    plan_symbol = str(getattr(plan, "symbol", "")).strip().upper()
    plan_market = _market(getattr(plan, "market", None))
    require(bool(plan_symbol) and receipt.get("symbol") == plan_symbol
            and (symbol is None or str(symbol).strip().upper() == plan_symbol)
            and plan_market is not None and receipt.get("market") == plan_market
            and (market is None or _market(market) == plan_market), "ONTOLOGY_AUTHORITY_SCOPE_MISMATCH")
    require(isinstance(policy, Mapping) and bool(receipt.get("policy_id"))
            and receipt.get("policy_id") == policy.get("policy_id")
            and str(policy.get("symbol", "")).upper() == plan_symbol
            and _market(policy.get("market")) == plan_market, "ONTOLOGY_AUTHORITY_POLICY_MISMATCH")
    quote_limit = _number(policy.get("max_quote_age_seconds")) if isinstance(policy, Mapping) else None
    require(isinstance(policy, Mapping) and policy.get("valid_for_entry") is True
            and quote_limit is not None and quote_limit > 0
            and all(not isinstance(value, float) or math.isfinite(value) for value in policy.values()),
            "ONTOLOGY_AUTHORITY_POLICY_INVALID")
    contract = getattr(plan, "order_contract", None)
    require(str(getattr(plan, "direction", "")).upper() == "LONG"
            and receipt.get("side") == "BUY" and receipt.get("position_direction") == "LONG"
            and receipt.get("position_effect") == "OPEN" and receipt.get("execution_product") == "CASH"
            and isinstance(contract, Mapping) and contract.get("position_direction") == "LONG"
            and contract.get("position_effect") == "OPEN" and contract.get("execution_product") == "CASH",
            "ONTOLOGY_AUTHORITY_CONTRACT_MISMATCH")
    moment = _time(now)
    evaluated = _time(receipt.get("evaluated_at"))
    expires = _time(receipt.get("expires_at"))
    created = _time(getattr(plan, "created_at", None))
    plan_expires = _time(getattr(plan, "expires_at", None))
    policy_as_of = _time(policy.get("as_of")) if isinstance(policy, Mapping) else None
    policy_expires = _time(policy.get("expires_at")) if isinstance(policy, Mapping) else None
    require(all((moment, evaluated, expires, created, plan_expires, policy_as_of, policy_expires))
            and policy_as_of <= evaluated <= moment < expires <= policy_expires
            and created <= moment < plan_expires <= expires,
            "ONTOLOGY_AUTHORITY_EXPIRED_OR_FUTURE")
    quantity = _number(receipt.get("quantity"))
    actual_quantity = _number(receipt.get("actual_quantity"))
    planned = _number(getattr(plan, "quantity", None))
    filled = _number(getattr(plan, "filled_quantity", None))
    remaining = _number(getattr(plan, "remaining_quantity", None))
    require(all(value is not None and value.is_integer() for value in (quantity, actual_quantity, planned, filled, remaining))
            and 0 < planned <= quantity == actual_quantity and 0 <= filled <= planned
            and remaining == planned - filled and remaining > 0,
            "ONTOLOGY_AUTHORITY_QUANTITY_EXCEEDED")
    authorized = _number(receipt.get("authorized_price"))
    notional = _number(receipt.get("actual_notional"))
    plan_notional = _number(getattr(plan, "max_notional", None))
    reference = _number(getattr(plan, "reference_price", None))
    rule = getattr(plan, "entry_rule", None)
    maximum_price = _number(getattr(rule, "max_price", None))
    require(authorized is not None and authorized > 0 and notional is not None and notional > 0
            and quantity is not None and abs(notional - quantity * authorized) <= max(1e-8, notional * 1e-12)
            and plan_notional is not None and 0 < plan_notional <= notional + 1e-8
            and reference is not None and 0 < reference <= authorized
            and maximum_price is not None and 0 < maximum_price <= authorized + 1e-10,
            "ONTOLOGY_AUTHORITY_PRICE_OR_NOTIONAL_EXCEEDED")
    require(all(_number(receipt.get(name)) is not None and _number(receipt.get(name)) >= 0
                for name in ("all_in_cost_rate", "required_gross_return")), "ONTOLOGY_AUTHORITY_RECEIPT_INVALID")
    if price is not None:
        actual = _number(price)
        require(actual is not None and authorized is not None and 0 < actual <= authorized
                and hasattr(rule, "price_permitted") and rule.price_permitted(actual),
                "ONTOLOGY_AUTHORITY_PRICE_OUTSIDE_APPROVAL")
    return tuple(reasons)


def decision_receipt_to_rdf(receipt: Mapping):
    """Materialize an audit graph on request, outside execution's numeric path.

    Rejected decisions remain exportable, including those without a policy or an
    executable quantity. The receipt is an observation of a past decision, not
    an approval obtained by RDF inference.
    """
    from urllib.parse import quote

    from rdflib import Graph, Literal, Namespace, OWL, RDF, XSD
    from app.ontology.policy_evidence import POLICY_NAMESPACE

    evaluation_id = str(receipt.get("evaluation_id") or "")
    market = _market(receipt.get("market"))
    symbol = str(receipt.get("symbol") or "").strip().upper()
    if not evaluation_id or market is None or not symbol or not isinstance(receipt.get("approved"), bool):
        raise ValueError("Decision receipt requires identity, market, symbol and an explicit verdict")
    pol = Namespace(POLICY_NAMESPACE)
    graph = Graph()
    graph.bind("pol", pol)
    node = pol["decision_" + quote(evaluation_id, safe="_")]
    instrument = pol["instrument_" + quote(market + "_" + symbol, safe="_")]
    approved = receipt["approved"]
    for kind in (OWL.NamedIndividual, pol.RiskDecision, pol.ApprovedRiskDecision if approved else pol.RejectedRiskDecision):
        graph.add((node, RDF.type, kind))
    graph.add((pol.OntologyRiskAuthority, RDF.type, OWL.NamedIndividual))
    graph.add((pol.OntologyRiskAuthority, RDF.type, pol.RiskAuthority))
    graph.add((instrument, RDF.type, pol.Instrument))
    graph.add((instrument, pol.symbol, Literal(symbol)))
    graph.add((instrument, pol.appliesToMarket, pol[market]))
    graph.add((node, pol.decidedBy, pol.OntologyRiskAuthority))
    graph.add((node, pol.decisionForInstrument, instrument))
    graph.add((node, pol.appliesToMarket, pol[market]))
    graph.add((node, pol.approved, Literal(approved, datatype=XSD.boolean)))
    policy_id = str(receipt.get("policy_id") or "")
    if policy_id:
        policy = pol["policy_" + quote(policy_id, safe="_")]
        graph.add((policy, RDF.type, pol.ThresholdPolicy))
        graph.add((policy, pol.policyId, Literal(policy_id)))
        graph.add((node, pol.usesThresholdPolicy, policy))
    for field, predicate in (("evaluated_at", pol.evaluatedAt), ("expires_at", pol.validUntil)):
        stamp = _time(receipt.get(field))
        if stamp is not None:
            graph.add((node, predicate, Literal(stamp, datatype=XSD.dateTime)))
    quantity = _number(receipt.get("actual_quantity"))
    if quantity is not None and quantity >= 0 and quantity.is_integer():
        graph.add((node, pol.approvedQuantity, Literal(int(quantity), datatype=XSD.integer)))
    price = _number(receipt.get("authorized_price"))
    if price is not None and price >= 0:
        graph.add((node, pol.authorizedPrice, Literal(price, datatype=XSD.double)))
    graph.add((node, pol.decisionPhase, Literal(str(receipt.get("phase") or ""))))
    for reason in receipt.get("reason_codes", ()):
        graph.add((node, pol.reasonCode, Literal(str(reason))))
    return graph
