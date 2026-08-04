"""Adapters between the custom ``KnowledgeGraph`` and the RDF layer (T03/T13).

The migration is *additive*: the existing string-triple ``KnowledgeGraph``
remains the primary in-memory store. This module projects it into an
:class:`~app.graph.rdf_graph.RdfTradingGraph` so OWL RL materialization and
SHACL validation can run on standards-based RDF, and projects RDF back into the
UI node/edge shape the GUI already consumes.

Design notes
------------
* Predicate strings are mapped to ``tr:`` properties. The four polarity
  relations (supports/contradicts/increases/decreases) become object
  properties pointing at canonical signal/risk individuals so OWL RL
  classification (``trading_rules.ttl``) fires.
* Unknown predicates degrade gracefully to ``tr:`` datatype properties with a
  string literal — always valid RDF, never a crash.
* ``evidence_id`` becomes an explicit ``ev:`` ``tr:EvidenceItem`` individual
  linked via ``tr:hasEvidence`` (open provenance model, no reification).
"""

from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path
from typing import Iterable, Mapping

from rdflib import Literal, URIRef
from rdflib.namespace import RDF, RDFS, XSD

from app.graph.knowledge_graph import KnowledgeGraph, Triple
from app.graph.rdf_graph import (
    EV,
    RES,
    TR,
    RdfTradingGraph,
    evidence_iri,
    resource_iri,
    slug,
)

# The four semantic-evidence relations (sub-properties of tr:hasSemanticEvidence).
_EVIDENCE_PREDICATES = {
    "supportsSignal",
    "contradictsSignal",
    "increasesRiskOf",
    "decreasesRiskOf",
}

# Canonical signal/risk individuals defined in trading_rules.ttl. Mapping an
# emitted object string here lets the OWL RL rules classify the asset.
# value = (individual local name in tr:, rdf:type class local name)
_CANONICAL_OBJECTS: dict[str, tuple[str, str]] = {
    "BuyCandidate": ("Signal_BuyCandidate", "PositiveSignal"),
    "SellCandidate": ("Signal_SellCandidate", "PositiveSignal"),
    "HoldWithTrailingStop": ("Signal_HoldWithTrailingStop", "PositiveSignal"),
    "WaitOrTakeProfit": ("Signal_WaitOrTakeProfit", "PositiveSignal"),
    "NetProfitability": ("Signal_NetProfitability", "PositiveSignal"),
    "TradeForbidden": ("Risk_TradeForbidden", "RiskFactor"),
    "VolatilityRisk": ("Risk_Volatility", "RiskFactor"),
    "ThinLiquidityPriceImpactRisk": ("Risk_ThinLiquidity", "RiskFactor"),
    "ConcentratedPositionRisk": ("Risk_ConcentratedPosition", "RiskFactor"),
}

# Object (relationship) predicates whose object is another entity, not a literal.
# value = tr: property local name.
_OBJECT_PREDICATES: dict[str, str] = {
    "belongsToSector": "belongsToSector",
    "hasTicker": "hasTicker",
    "hasRecentNews": "hasNewsEvent",
    "hasRecentDisclosure": "hasDisclosureEvent",
    "affectedByMacroFactor": "hasMacroIndicator",
    "hasTechnicalIndicator": "hasTechnicalIndicator",
    "generatesSemanticFeature": "hasSemanticFeature",
    "hasExposureTo": "hasExposureTo",
    "isIncludedInPortfolio": "isIncludedInPortfolio",
    "generatesOrderIntent": "generatesOrderIntent",
    "isApprovedByRiskManager": "isApprovedByRiskManager",
    "hasRiskManagerDecision": "hasRiskManagerDecision",
    "isRejectedByRiskRule": "isRejectedByRiskRule",
    "isExecutedAs": "isExecutedAs",
    "hasBrokerQuote": "hasBrokerQuote",
    "hasMarketSnapshot": "hasMarketSnapshot",
}

# rdf:type class for the object individual of each evidence predicate.
_EVIDENCE_OBJECT_CLASS = {
    "supportsSignal": "StrategySignal",
    "contradictsSignal": "ContradictorySignal",
    "increasesRiskOf": "RiskFactor",
    "decreasesRiskOf": "RiskFactor",
}

_DOMESTIC_MARKETS = {"KRX", "KOSPI", "KOSDAQ", "KONEX"}


def _property_iri(predicate: str) -> URIRef:
    """Map a predicate string to a tr: property IRI (rename where needed)."""
    renamed = _OBJECT_PREDICATES.get(predicate, predicate)
    return TR[renamed if renamed[0].islower() else predicate]


def _object_individual(predicate: str, object_: str) -> tuple[URIRef, str]:
    """Return (IRI, type_class_local) for the object individual of an evidence
    or relationship predicate."""
    canonical = _CANONICAL_OBJECTS.get(object_)
    if canonical is not None:
        local, cls = canonical
        return TR[local], cls
    cls = _EVIDENCE_OBJECT_CLASS.get(predicate, "TradingEntity")
    return resource_iri(object_), cls


def add_triple_to_rdf(rdf: RdfTradingGraph, triple: Triple) -> None:
    """Convert a single custom ``Triple`` into RDF assertions."""
    subject = resource_iri(triple.subject)
    predicate = triple.predicate
    obj = triple.object

    if predicate in _EVIDENCE_PREDICATES:
        obj_iri, obj_cls = _object_individual(predicate, obj)
        rdf.add(subject, TR[predicate], obj_iri)
        rdf.add(obj_iri, RDF.type, TR[obj_cls])
    elif predicate == "isListedOn":
        # object is a market/exchange code -> data property
        rdf.add(subject, TR.hasExchange, Literal(str(obj)))
    elif predicate in _OBJECT_PREDICATES:
        prop = TR[_OBJECT_PREDICATES[predicate]]
        rdf.add(subject, prop, resource_iri(obj))
    else:
        # Unknown / metric predicate -> datatype property with string literal.
        rdf.add(subject, TR[slug(predicate) if not predicate[:1].isalpha() else predicate],
                Literal(str(obj)))

    # Provenance: attach an explicit EvidenceItem node.
    if triple.evidence_id:
        ev_node = evidence_iri(triple.evidence_id)
        rdf.add(subject, TR.hasEvidence, ev_node)
        rdf.add(ev_node, RDF.type, TR.EvidenceItem)
        source_name = str(triple.evidence_id).split(":", 1)[0]
        rdf.add(ev_node, TR.hasSourceName, Literal(source_name))
        rdf.add(ev_node, TR.hasAnalysisCycleId, Literal(rdf.cycle_id))


def knowledge_graph_to_rdf(
    graph: KnowledgeGraph,
    *,
    cycle_id: str | None = None,
    markets: Mapping[str, object] | None = None,
    live_tickers: Iterable[str] | None = None,
) -> RdfTradingGraph:
    """Project an entire ``KnowledgeGraph`` into an ``RdfTradingGraph``.

    ``markets`` (ticker -> MarketSnapshot-like) enables asset typing
    (Domestic/Foreign stock), symbol/market data properties, broker-quote and
    source-metadata assertions. ``live_tickers`` marks candidates as
    ``tr:LiveTradingCandidate`` so live-only SHACL shapes apply.
    """
    rdf = RdfTradingGraph(cycle_id=cycle_id)
    for triple in graph.triples():
        add_triple_to_rdf(rdf, triple)

    if markets:
        live_set = {str(t) for t in (live_tickers or ())}
        for ticker, market in markets.items():
            _assert_market_entity(rdf, ticker, market, live=str(ticker) in live_set)
    return rdf


def _assert_market_entity(rdf: RdfTradingGraph, ticker: str, market: object, *, live: bool) -> None:
    """Type a ticker as a (Domestic/Foreign) stock and attach snapshot facts."""
    subject = resource_iri(ticker)
    market_code = str(getattr(market, "market", "") or "").upper()
    if market_code in _DOMESTIC_MARKETS:
        rdf.add(subject, RDF.type, TR.DomesticStock)
    elif market_code:
        rdf.add(subject, RDF.type, TR.ForeignStock)
    else:
        rdf.add(subject, RDF.type, TR.Stock)
    rdf.add(subject, TR.hasSymbol, Literal(str(ticker)))
    company_name = str(getattr(market, "company_name", "") or "").strip()
    if company_name and company_name != str(ticker):
        rdf.add(subject, TR.hasName, Literal(company_name))
    if market_code:
        rdf.add(subject, TR.hasMarket, Literal(market_code))

    last_price = getattr(market, "last_price", None)
    source = getattr(market, "source", None)
    if source is not None:
        rdf.add(subject, TR.hasSourceTrustLevel, Literal(str(getattr(source, "source_type", "unknown"))))
        is_synthetic = bool(getattr(source, "is_synthetic", False))
        rdf.add(subject, TR.hasIsSynthetic, Literal(is_synthetic))
        # A delayed / non-realtime broker snapshot is treated as stale for live.
        is_stale = bool(getattr(source, "is_delayed", False)) or not bool(
            getattr(source, "is_realtime", False)
        )
        rdf.add(subject, TR.hasIsStale, Literal(is_stale))
        retrieved = getattr(source, "retrieved_at", None)
        if retrieved is not None:
            rdf.add(subject, TR.hasAsOfTime, Literal(retrieved.isoformat(), datatype=XSD.dateTime))

    if last_price is not None and float(last_price) > 0:
        quote = resource_iri(f"quote:{ticker}")
        rdf.add(subject, TR.hasBrokerQuote, quote)
        rdf.add(quote, RDF.type, TR.BrokerQuote)
        rdf.add(quote, TR.hasLastPrice, Literal(str(round(float(last_price), 6)), datatype=XSD.decimal))

    if live:
        rdf.add(subject, RDF.type, TR.LiveTradingCandidate)
        # Confidence is a required live field; a placeholder ensures the node is
        # evaluated. Real confidence is set by the policy scorer integration.
        rdf.add(subject, TR.hasConfidence, Literal("0.0", datatype=XSD.decimal))


# NPU score schema (mirrors app.graph.npu_classifier.SCORE_SCHEMA).
_NPU_SCORE_PROPERTIES = (
    ("support_score", "hasSupportScore"),
    ("risk_score", "hasRiskScore"),
    ("momentum_score", "hasQualityScore"),
    ("value_score", "hasExpectedReturn"),
    ("liquidity_score", "hasQualityScore"),
    ("confidence_score", "hasConfidence"),
)


def attach_scoring_provenance(
    rdf: RdfTradingGraph,
    *,
    backend: str,
    model_kind: str,
    npu_scores: Mapping[str, tuple] | None = None,
) -> None:
    """Represent candidate-scorer output as RDF evidence + semantic features (T08).

    The NPU / CPU-fallback / heuristic score is preserved as data properties on
    per-ticker ``tr:EvidenceItem`` individuals, tagged with the source backend.
    OWL never consumes these as a trade authorization; they are evidence only.
    """
    source_type = str(backend or "unknown")
    for ticker, scores in (npu_scores or {}).items():
        subject = resource_iri(ticker)
        ev_node = evidence_iri(f"npu:{ticker}")
        rdf.add(subject, TR.derivedFromEvidence, ev_node)
        rdf.add(ev_node, RDF.type, TR.EvidenceItem)
        rdf.add(ev_node, TR.hasSourceName, Literal("candidate_scorer"))
        rdf.add(ev_node, TR.hasSourceType, Literal(source_type))
        rdf.add(ev_node, RDFS.label, Literal(f"{source_type}:{model_kind}"))
        rdf.add(ev_node, TR.hasAnalysisCycleId, Literal(rdf.cycle_id))
        for idx, (_name, prop) in enumerate(_NPU_SCORE_PROPERTIES):
            if idx < len(scores):
                rdf.add(ev_node, TR[prop], Literal(str(round(float(scores[idx]), 6)), datatype=XSD.decimal))


def attach_account_snapshot(rdf: RdfTradingGraph, account: object, *, node_name: str = "account") -> None:
    """Assert an ``tr:AccountSnapshot`` individual (T05 AccountSnapshotShape)."""
    node = resource_iri(f"account:{node_name}")
    rdf.add(node, RDF.type, TR.AccountSnapshot)
    total = getattr(account, "equity", None)
    if total is not None:
        rdf.add(node, TR.hasTotalAssetValue, Literal(str(round(float(total), 6)), datatype=XSD.decimal))
    cash_by_currency = getattr(account, "cash_by_currency", None) or {}
    krw = cash_by_currency.get("KRW", getattr(account, "cash", None))
    if krw is not None:
        rdf.add(node, TR.hasAvailableKRW, Literal(str(round(float(krw), 6)), datatype=XSD.decimal))
    if "USD" in cash_by_currency:
        rdf.add(node, TR.hasAvailableUSD, Literal(str(round(float(cash_by_currency["USD"]), 6)), datatype=XSD.decimal))
    return node


# ---------------------------------------------------------------------------
# RDF -> UI node/edge projection (compatible with web.py graph payload shape)
# ---------------------------------------------------------------------------
_UI_SKIP_PREDICATES = {RDF.type}


@lru_cache(maxsize=1)
def _instrument_name_map() -> dict[str, str]:
    """Load code -> display-name overrides (e.g. KRX Korean stock names)."""
    mapping: dict[str, str] = {}
    try:
        path = Path("config/instrument_names.json")
        if path.exists():
            data = json.loads(path.read_text(encoding="utf-8"))
            if isinstance(data, dict):
                for key, value in data.items():
                    key = str(key).strip()
                    value = str(value).strip()
                    if key and not key.startswith("_") and value:
                        mapping[key] = value
    except Exception:
        pass
    return mapping


def _display_label(nid: str, attrs: Mapping[str, str]) -> str:
    """Prefer a configured human-readable name over the bare exchange code."""
    symbol = str(attrs.get("hasSymbol") or _local(nid))
    base = symbol.split(".", 1)[0]
    names = _instrument_name_map()
    if base in names:
        return names[base]
    if symbol in names:
        return names[symbol]
    name_attr = attrs.get("hasName")
    if name_attr and str(name_attr) != symbol:
        return str(name_attr)
    return symbol


def _local(term: object) -> str:
    text = str(term)
    for sep in ("#", "/"):
        if sep in text:
            text = text.rsplit(sep, 1)[-1]
    return text


def rdf_to_ui_payload(
    rdf: RdfTradingGraph,
    *,
    inferred_triples: Iterable[tuple] | None = None,
) -> dict:
    """Project RDF into a ``{"nodes": [...], "links": [...]}`` UI payload.

    Blank nodes (OWL restriction classes produced by materialization) are
    dropped. When ``inferred_triples`` is given, links present only in that set
    are flagged ``inferred: true`` so the GUI can separate asserted vs inferred.
    """
    inferred_set = {t for t in (inferred_triples or ())}
    merged = rdf.merged_graph()

    node_types: dict[str, list[str]] = {}
    node_attrs: dict[str, dict] = {}
    links: list[dict] = []

    for s, p, o in merged:
        if isinstance(s, URIRef) is False:
            continue
        sid = str(s)
        node_attrs.setdefault(sid, {})
        if p == RDF.type and isinstance(o, URIRef):
            node_types.setdefault(sid, []).append(_local(o))
            continue
        if p in _UI_SKIP_PREDICATES:
            continue
        if isinstance(o, URIRef):
            node_attrs.setdefault(str(o), {})
            links.append(
                {
                    "source": sid,
                    "target": str(o),
                    "predicate": _local(p),
                    "inferred": (s, p, o) in inferred_set,
                }
            )
        else:  # literal -> node attribute
            node_attrs[sid][_local(p)] = str(o)

    nodes = []
    for nid, attrs in node_attrs.items():
        types = node_types.get(nid, [])
        nodes.append(
            {
                "id": nid,
                "label": _display_label(nid, attrs),
                "kind": _kind_from_types(types),
                "rdf_types": types,
                "inferred": bool(types) and nid not in _asserted_ids(merged, inferred_set),
                "attributes": attrs,
            }
        )
    return {"nodes": nodes, "links": links}


def _asserted_ids(merged, inferred_set) -> set[str]:
    # A node is "asserted" if it participates in at least one non-inferred triple.
    asserted: set[str] = set()
    for triple in merged:
        if triple in inferred_set:
            continue
        s, _, o = triple
        if isinstance(s, URIRef):
            asserted.add(str(s))
        if isinstance(o, URIRef):
            asserted.add(str(o))
    return asserted


def _kind_from_types(types: list[str]) -> str:
    priority = [
        ("TradeForbiddenAsset", "risk"),
        ("RiskFactor", "risk"),
        ("SyntheticDataAsset", "risk"),
        ("StaleDataAsset", "risk"),
        ("BuyCandidate", "candidate"),
        ("SellCandidate", "candidate"),
        ("CandidateAsset", "candidate"),
        ("DomesticStock", "ticker"),
        ("ForeignStock", "ticker"),
        ("Stock", "ticker"),
        ("EvidenceItem", "evidence"),
        ("PositiveSignal", "signal"),
        ("StrategySignal", "signal"),
        ("NewsEvent", "event"),
        ("DisclosureEvent", "event"),
    ]
    for local, kind in priority:
        if local in types:
            return kind
    return "resource"


# ---------------------------------------------------------------------------
# Macro / micro reasoning result projection (hierarchical refactor).
# Represents MacroReasoningResult / MicroReasoningResult as RDF evidence nodes
# whose datatype properties satisfy the SHACL shapes in trading_shapes.ttl.
# ADVISORY ONLY — no FinalOrder is ever asserted here.
# ---------------------------------------------------------------------------
def attach_macro_result_rdf(rdf: RdfTradingGraph, macro_result, *, node_name: str = "macro:latest") -> URIRef:
    node = resource_iri(node_name)
    rdf.add(node, RDF.type, TR.MacroReasoningResult)
    rdf.add(node, TR.hasMarketRegimeName, Literal(macro_result.market_regime.value))
    rdf.add(node, TR.hasMacroRiskLevelName, Literal(macro_result.macro_risk_level.value))
    rdf.add(node, TR.hasMacroConfidence, Literal(str(round(float(macro_result.macro_confidence), 4)), datatype=XSD.decimal))
    for code in macro_result.reason_codes:
        rdf.add(node, TR.hasMacroReasonCode, Literal(str(code)))
    for sym in macro_result.candidate_symbols:
        sym_node = resource_iri(str(sym))
        rdf.add(sym_node, RDF.type, TR.MacroCandidateSymbol)
        rdf.add(node, TR.selectsCandidateSymbol, sym_node)
    for strat in macro_result.allowed_micro_strategies:
        s = resource_iri(f"strategy:allow:{strat}")
        rdf.add(s, RDF.type, TR.AllowedMicroStrategy)
        rdf.add(node, TR.allowsMicroStrategy, s)
    for strat in macro_result.blocked_micro_strategies:
        s = resource_iri(f"strategy:block:{strat}")
        rdf.add(s, RDF.type, TR.BlockedMicroStrategy)
        rdf.add(node, TR.blocksMicroStrategy, s)
    rdf.add(node, TR.hasAnalysisCycleId, Literal(rdf.cycle_id))
    return node


def attach_micro_result_rdf(rdf: RdfTradingGraph, micro_result, *, node_name: str | None = None) -> URIRef:
    node = resource_iri(node_name or f"micro:{micro_result.symbol}")
    rdf.add(node, RDF.type, TR.MicroReasoningResult)
    rdf.add(node, TR.hasSymbol, Literal(str(micro_result.symbol)))
    rdf.add(node, TR.hasMicroRegimeName, Literal(micro_result.micro_regime.value))
    rdf.add(node, TR.hasMicroConfidence, Literal(str(round(float(micro_result.confidence), 4)), datatype=XSD.decimal))
    rdf.add(node, TR.hasExecutionQualityName, Literal(micro_result.execution_quality.value))
    rdf.add(node, TR.hasEntrySignalName, Literal(micro_result.entry_signal.value))
    rdf.add(node, TR.hasExitSignalName, Literal(micro_result.exit_signal.value))
    rdf.add(node, TR.hasMicroRegimeName, Literal(micro_result.micro_regime.value))
    _dec = lambda v: Literal(str(round(float(v), 4)), datatype=XSD.decimal)
    if micro_result.expected_entry_price is not None:
        rdf.add(node, TR.hasExpectedEntryPrice, _dec(micro_result.expected_entry_price))
    if micro_result.expected_exit_price is not None:
        rdf.add(node, TR.hasExpectedExitPrice, _dec(micro_result.expected_exit_price))
    if micro_result.expected_gross_return_bps is not None:
        rdf.add(node, TR.hasExpectedGrossReturnBps, _dec(micro_result.expected_gross_return_bps))
    if micro_result.expected_net_return_bps is not None:
        rdf.add(node, TR.hasExpectedNetReturnBps, _dec(micro_result.expected_net_return_bps))
    if micro_result.downside_risk_bps is not None:
        rdf.add(node, TR.hasDownsideRiskBps, _dec(micro_result.downside_risk_bps))
    for code in micro_result.reason_codes:
        rdf.add(node, TR.hasMicroReasonCode, Literal(str(code)))
    rdf.add(node, TR.hasAnalysisCycleId, Literal(rdf.cycle_id))
    return node
