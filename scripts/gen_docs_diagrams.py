#!/usr/bin/env python3
"""Generate the top-level README diagrams into docs/diagrams/.

Pure-Python SVG authoring (no matplotlib/graphviz/rasterizer dependency).
Style matches the existing docs/diagrams/system_overview.svg. Re-run after an
architecture change:

    python scripts/gen_docs_diagrams.py

Outputs:
    docs/diagrams/ontology_gnn_layers.svg
    docs/diagrams/data_to_decision_flow.svg
"""

from __future__ import annotations

from pathlib import Path

OUT = Path(__file__).resolve().parents[1] / "docs" / "diagrams"

FONT = "Segoe UI, Noto Sans KR, Arial, sans-serif"

STYLE = """
    .bg{fill:#f8fafc}
    .title{fill:#0f172a;font-size:30px;font-weight:800}
    .subtitle{fill:#475569;font-size:15px}
    .band{fill:#f1f5f9;stroke:#cbd5e1}
    .lane{fill:#94a3b8;font-size:13px;font-weight:800;letter-spacing:.06em}
    .h{fill:#0f172a;font-size:17px;font-weight:800}
    .t{fill:#334155;font-size:13px}
    .m{fill:#64748b;font-size:12px;font-style:italic}
    .note{fill:#7f1d1d;font-size:12.5px;font-weight:700}
    .flow{stroke:#475569;stroke-width:2.4;fill:none;marker-end:url(#arrow)}
    .dash{stroke:#94a3b8;stroke-width:2;fill:none;stroke-dasharray:7 6;marker-end:url(#arrowlite)}
    .badge{fill:#fff;font-size:12.5px;font-weight:800;text-anchor:middle}
    .foot{fill:#475569;font-size:13px}
"""

PALETTE = {
    "green": ("#ecfdf5", "#10b981"),
    "blue": ("#eff6ff", "#3b82f6"),
    "cyan": ("#ecfeff", "#06b6d4"),
    "amber": ("#fefce8", "#eab308"),
    "violet": ("#f5f3ff", "#8b5cf6"),
    "red": ("#fef2f2", "#ef4444"),
    "teal": ("#f0fdfa", "#14b8a6"),
    "orange": ("#fff7ed", "#f97316"),
    "slate": ("#f1f5f9", "#94a3b8"),
}


def esc(text: str) -> str:
    return text.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")


def header(width: int, height: int, title: str, subtitle: str) -> str:
    return (
        f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {width} {height}" '
        f'font-family="{FONT}">\n'
        "  <defs>\n"
        '    <marker id="arrow" markerWidth="10" markerHeight="10" refX="8" refY="3" '
        'orient="auto" markerUnits="strokeWidth">'
        '<path d="M0,0 L9,3 L0,6 Z" fill="#475569"/></marker>\n'
        '    <marker id="arrowlite" markerWidth="10" markerHeight="10" refX="8" refY="3" '
        'orient="auto" markerUnits="strokeWidth">'
        '<path d="M0,0 L9,3 L0,6 Z" fill="#94a3b8"/></marker>\n'
        f"    <style>{STYLE}    </style>\n"
        "  </defs>\n\n"
        f'  <rect class="bg" x="0" y="0" width="{width}" height="{height}"/>\n'
        f'  <text class="title" x="52" y="52">{esc(title)}</text>\n'
        f'  <text class="subtitle" x="52" y="79">{esc(subtitle)}</text>\n\n'
    )


def band(x: int, y: int, w: int, h: int, label: str) -> str:
    return (
        f'  <rect class="band" x="{x}" y="{y}" width="{w}" height="{h}" rx="18"/>\n'
        f'  <text class="lane" x="{x + 24}" y="{y + 26}">{esc(label)}</text>\n'
    )


def card_height(lines: list[str], module: str | None) -> int:
    """Height that fits the title, every body line, and the italic module caption."""
    return 55 + 19 * len(lines) + (20 if module else 6)


def card(
    x: int,
    y: int,
    w: int,
    title: str,
    lines: list[str],
    color: str = "blue",
    badge: str | None = None,
    module: str | None = None,
    h: int | None = None,
) -> str:
    fill, stroke = PALETTE[color]
    box_h = h if h is not None else card_height(lines, module)
    out = (
        f'  <rect x="{x}" y="{y}" width="{w}" height="{box_h}" rx="12" fill="{fill}" '
        f'stroke="{stroke}" stroke-width="1.8"/>\n'
    )
    text_x = x + 20
    if badge:
        out += (
            f'  <circle cx="{x + 26}" cy="{y + 26}" r="13" fill="{stroke}"/>'
            f'<text class="badge" x="{x + 26}" y="{y + 31}">{esc(badge)}</text>\n'
        )
        text_x = x + 48
    out += f'  <text class="h" x="{text_x}" y="{y + 32}">{esc(title)}</text>\n'
    cursor = y + 55
    for line in lines:
        out += f'  <text class="t" x="{x + 20}" y="{cursor}">{esc(line)}</text>\n'
        cursor += 19
    if module:
        out += f'  <text class="m" x="{x + 20}" y="{cursor + 8}">{esc(module)}</text>\n'
    return out


def arrow(x1: int, y1: int, x2: int, y2: int, dashed: bool = False) -> str:
    cls = "dash" if dashed else "flow"
    return f'  <path class="{cls}" d="M{x1},{y1} L{x2},{y2}"/>\n'


def curve(x1: int, y1: int, x2: int, y2: int, bow: int = 60, dashed: bool = True) -> str:
    cls = "dash" if dashed else "flow"
    cx = (x1 + x2) // 2
    return f'  <path class="{cls}" d="M{x1},{y1} C{cx + bow},{y1} {cx - bow},{y2} {x2},{y2}"/>\n'


def label(x: int, y: int, text: str, cls: str = "m", anchor: str = "start") -> str:
    return (
        f'  <text class="{cls}" x="{x}" y="{y}" text-anchor="{anchor}">{esc(text)}</text>\n'
    )




# --------------------------------------------------------------------------------------
# Diagram 1: ontology and GNN layers
# --------------------------------------------------------------------------------------

ONTOLOGY_LAYERS = [
    (
        "L0 · Evidence and facts",
        [
            "KIS realtime ticks / orderbooks / session, account and cash snapshot,",
            "broker quotes, news / RSS / DART events, macro and sector snapshots.",
            "Every assertion carries source, timestamp, quality, synthetic and stale flags.",
        ],
        "green",
        "app.data.source_policy · app.features.feature_provenance",
    ),
    (
        "L1 · Working triple store",
        [
            "In-memory (subject, predicate, object, evidence_id) graph, plus a compact",
            "integer-id fact table with validity windows for hot-path lookups.",
        ],
        "blue",
        "graph.knowledge_graph.KnowledgeGraph · graph.fact_table.FactTable",
    ),
    (
        "L2 · RDF assertion graph",
        [
            "Projection into rdflib with stable IRIs (tr: schema, res: instances,",
            "ev: evidence) inside one named graph per analysis cycle.",
        ],
        "blue",
        "graph.rdf_graph · graph.rdf_adapter",
    ),
    (
        "L3 · RDFS / OWL 2 RL closure   (OPEN WORLD)",
        [
            "trading_core.ttl class and property hierarchy plus trading_rules.ttl",
            "owl:hasValue restrictions materialise semantic labels such as",
            "BuyCandidate, TradeForbiddenAsset, SyntheticDataAsset.",
        ],
        "cyan",
        "graph.owl_reasoner · graph.semantic_materializer  —  never permits or blocks a trade",
    ),
    (
        "L4 · SHACL validation   (CLOSED WORLD)",
        [
            "trading_shapes.ttl checks required fields, positive prices, stale and",
            "synthetic data, order structure, final-order preconditions.",
        ],
        "amber",
        "graph.shacl_validator  —  mode=live blocks, mode=paper warns",
    ),
    (
        "L5 · Operational strategy gate   (TIME-VALID, FAIL-CLOSED)",
        [
            "OperationalFact(value, observed_at, valid_from, valid_until, source, confidence)",
            "→ allowed strategy IDs, hard-block reasons, soft compatibility scores,",
            "source-linked explanation paths. A missing required fact is a hard block.",
        ],
        "red",
        "app.ontology.operational_gate.ClosedWorldOntologyGate",
    ),
    (
        "L6 · Macro → micro reasoning",
        [
            "MacroMarketReasoner: regime, risk level, sector ranking, candidate symbols,",
            "allowed / blocked strategies. Bounded-parallel MicroSymbolReasoner per candidate:",
            "entry, exit, execution quality, expected net return. Arbiter ranks SELL before BUY.",
        ],
        "violet",
        "graph.macro_reasoner · micro_reasoner · ontology_coordinator · global_trade_arbiter",
    ),
]

GNN_LAYERS = [
    (
        "G0 · Fixed-shape tensor projection",
        [
            "X [B,T,N,F] · A [B,T,R,N,N] · node_mask [B,T,N] · strategy_mask [B,N,S]",
            "Topology is built on CPU and stays outside the model graph.",
            "Shadow checkpoint B1 T1 N1 F12 R1 S7 · benchmark B1 T4 N16 F12 R4 S7",
        ],
        "violet",
        "app.npu.tensor_schemas · routing.shadow_intelligence",
    ),
    (
        "G1 · Relational message passing (R-GCN)",
        [
            "H = ReLU( X · W_self  +  Σ_r  A_r · X · W_r ),  then H ×= node_mask",
            "Dense MatMul / Add / ReLU only. No scatter, gather-by-index or sparse ops.",
        ],
        "violet",
        "models.strategy_utility.rgcn.FixedShapeStrategyUtilityModel",
    ),
    (
        "G2 · Causal temporal pooling",
        [
            "Z = Σ_t  w_t · H_t  over observations at or before the decision snapshot.",
            "No future padding, no look-ahead, fixed T.",
        ],
        "violet",
        None,
    ),
    (
        "G3 · Multi-task heads per stock × strategy",
        [
            "p_success · gross_bps · cost_bps · MAE · MFE · p_fill · holding_s ·",
            "aleatoric uncertainty, plus a separate NoTrade head.",
        ],
        "violet",
        "8 head channels × 7 strategies",
    ),
    (
        "G4 · Cost-adjusted utility",
        [
            "net = gross_bps − cost_bps",
            "U = p_success·net − (1−p_success)·MAE − uncertainty + 0.1·p_fill·MFE",
            "U = −∞ wherever the ontology mask or the node mask is 0.",
        ],
        "orange",
        "rgcn.output_from_raw",
    ),
    (
        "G5 · Strategy router",
        [
            "argmax U over admissible strategies, otherwise a first-class NO_TRADE.",
            "Emits StrategyUtilityEvidence with the ontology snapshot id and explanations.",
        ],
        "red",
        "routing.strategy_router.StrategyRouter",
    ),
    (
        "Runtime and promotion status",
        [
            "OpenVINO CPU is the verified runtime. NPU compiles without fallback but",
            "measured slower (p50 1.14 ms vs 0.33 ms) and outside the utility tolerance.",
            "Shadow inference only: rgcn_shadow.npz is scope-limited, never order authority.",
        ],
        "slate",
        "models.strategy_utility.openvino_runtime · routing.shadow_comparison",
    ),
]


def _stack(x: int, width: int, top: int, layers: list, gap: int = 22) -> tuple[str, list[int], int]:
    """Render a vertical stack of cards, returning svg, each card's y, and the bottom."""
    svg = ""
    positions: list[int] = []
    cursor = top
    for index, (title, lines, color, module) in enumerate(layers):
        height = card_height(lines, module)
        positions.append(cursor)
        svg += card(x, cursor, width, title, lines, color=color, module=module)
        if index:
            svg += arrow(x + width // 2, cursor - gap, x + width // 2, cursor - 4)
        cursor += height + gap
    return svg, positions, cursor - gap


def ontology_gnn_layers() -> str:
    width = 1640
    left_x, left_w = 38, 846
    right_x, right_w = 950, 652
    band_y = 110
    top = band_y + 46

    left_svg, left_y, left_bottom = _stack(left_x + 24, left_w - 48, top, ONTOLOGY_LAYERS)
    right_svg, right_y, right_bottom = _stack(right_x + 24, right_w - 48, top, GNN_LAYERS)
    band_h = max(left_bottom, right_bottom) + 18 - band_y

    auth_y = band_y + band_h + 24
    height = auth_y + 128 + 52

    svg = header(
        width,
        height,
        "Ontology and GNN layers",
        "How symbolic knowledge and the fixed-shape temporal R-GCN are stacked, and where the deterministic authority boundary sits.",
    )
    svg += band(left_x, band_y, left_w, band_h, "ONTOLOGY STACK — CPU, EXPLAINABLE, ADVISORY")
    svg += band(right_x, band_y, right_w, band_h, "STRATEGY-UTILITY GNN — FIXED SHAPE, SHADOW ONLY")
    svg += left_svg + right_svg

    # The operational gate is the only ontology output the model is allowed to consume.
    gate_top = left_y[5]
    gate_height = card_height(ONTOLOGY_LAYERS[5][1], ONTOLOGY_LAYERS[5][3])
    svg += curve(left_x + left_w - 24, gate_top + 24, right_x + 24, right_y[0] + 40, bow=70)
    svg += curve(
        left_x + left_w - 24,
        gate_top + gate_height - 24,
        right_x + 24,
        right_y[4] + 40,
        bow=70,
    )
    svg += label(
        52,
        102,
        "Dashed: the L5 operational gate is the only ontology output the model consumes — "
        "allowed strategy IDs become strategy_mask, and a hard block forces utility = −∞.",
    )

    svg += band(38, auth_y, 1564, 128, "DETERMINISTIC AUTHORITY")
    svg += label(1578, auth_y + 26, "the only path to a real order", cls="lane", anchor="end")
    gates = [
        ("TradingCostEngine + ProfitabilityGate", "cost.profitability_gate"),
        ("DynamicExitPolicy + PositionSizer", "trading.dynamic_exit_policy · risk.position_sizing"),
        ("PrincipalProtection + RiskManager", "risk.principal_protection · risk.manager"),
        ("LiveExecutionCoordinator → KIS limit order", "execution.live_execution_coordinator"),
    ]
    gate_w, gate_gap = 364, 24
    for index, (title, module) in enumerate(gates):
        gx = 62 + index * (gate_w + gate_gap)
        gy = auth_y + 44
        svg += (
            f'  <rect x="{gx}" y="{gy}" width="{gate_w}" height="62" rx="12" fill="#fee2e2" '
            'stroke="#dc2626" stroke-width="1.8"/>\n'
            f'  <text class="h" x="{gx + 18}" y="{gy + 26}" font-size="14">{esc(title)}</text>\n'
            f'  <text class="m" x="{gx + 18}" y="{gy + 48}">{esc(module)}</text>\n'
        )
        if index:
            svg += arrow(gx - gate_gap, gy + 31, gx - 4, gy + 31)

    svg += arrow(left_x + left_w // 2, left_bottom, left_x + left_w // 2, auth_y + 38)
    svg += arrow(right_x + right_w // 2, right_bottom, right_x + right_w // 2, auth_y + 38)

    svg += label(
        52,
        height - 22,
        "Safety invariant: no ontology, GNN, NPU, ML or LLM output can create a FinalOrder. "
        "They classify, rank, explain and mask; the deterministic gates decide and submit.",
        cls="note",
    )
    svg += "</svg>\n"
    return svg


# --------------------------------------------------------------------------------------
# Diagram 2: data to decision flow
# --------------------------------------------------------------------------------------

SOURCES = [
    (
        "1",
        "KIS realtime WebSocket",
        ["Domestic trade ticks", "Orderbook levels", "Session / market phase"],
        "data.kis_realtime",
    ),
    (
        "2",
        "KIS REST",
        ["Account, cash, holdings", "Order status and fills", "Overseas quote polling"],
        "execution.kis_real · kis_overseas",
    ),
    (
        "3",
        "Broker quotes and FX",
        ["Refreshed bid / ask", "Currency cash split", "Tick size and venue rules"],
        "execution.exchange_resolver",
    ),
    (
        "4",
        "News, RSS, disclosure",
        ["Article and DART events", "Keyword / OpenVINO / LLM", "Polarity and novelty"],
        "data.event_pipeline",
    ),
    (
        "5",
        "Market and macro",
        ["Index and sector snapshots", "ECOS / FRED macro series", "Listed universe catalog"],
        "research.service · public_collectors",
    ),
]

STORAGE = [
    (
        "Source trust and freshness",
        [
            "Source-type inference, quality score, live eligibility.",
            "Synthetic / sample / hash data rejected for live decisions.",
            "Quote and orderbook age limits enforce staleness blocks.",
        ],
        "data.source_policy · config/live_trading_safety.json",
    ),
    (
        "Event bus and market state",
        [
            "Bounded queue keeps DB work off the WebSocket callback.",
            "In-memory per-symbol state with sequence / gap detection.",
            "Reconnect marks state uncertain instead of silently resuming.",
        ],
        "data.event_pipeline · data.event_runtime (flagged)",
    ),
    (
        "Realtime and research stores",
        [
            "Minute bars, ticks, orderbooks in realtime_market_data.sqlite3.",
            "Normalised research records, raw archive, account snapshots.",
            "Retention pruning, deduplication, recursive secret redaction.",
        ],
        "data.realtime_store · storage.local_store · audit.logger",
    ),
    (
        "Session and market phase",
        [
            "Exchange calendar classification for KRX / NXT / US.",
            "REST snapshot fallback when the market is fully closed.",
            "Blank fields and zero ticks resolve to a phase, not an error.",
        ],
        "data.market_session · data.rest_snapshot_fallback",
    ),
]

FEATURES = [
    (
        "Indicators",
        [
            "SMA, EMA, MACD, RSI, Bollinger,",
            "ATR, volume spike, Donchian,",
            "rolling z-score. NaN-safe, pure.",
        ],
        "features.indicator_engine · technical.indicators",
    ),
    (
        "Microstructure",
        [
            "mid, spread_bps, microprice,",
            "orderbook imbalance, OFI,",
            "trade imbalance, VWAP, RV.",
        ],
        "data.event_pipeline · features.short_horizon_features",
    ),
    (
        "Live feature frame",
        [
            "Schema live_short_horizon_v2 with",
            "provenance and freshness flags.",
            "Triple-barrier short-horizon labels.",
        ],
        "features.live_feature_frame · models.labeling",
    ),
    (
        "Semantic and flow features",
        [
            "Event polarity and materiality,",
            "foreign / institution / retail flow,",
            "informed-imbalance derivations.",
        ],
        "features.semantic_feature_engine · graph.builders",
    ),
]

REASONING = [
    (
        "Technical prediction",
        [
            "Rule-based regime, methodology",
            "providers, mandatory VWAP and",
            "volume confirmation, conservative",
            "expected exit price and downside.",
        ],
        "app.technical.* — advisory",
    ),
    (
        "Macro–micro ontology",
        [
            "Market regime and risk, sector",
            "ranking, candidate symbols,",
            "strategy permissions, per-symbol",
            "entry / exit and SELL-first ranking.",
        ],
        "app.graph.* — advisory",
    ),
    (
        "Semantic labels and scoring",
        [
            "RDF / OWL RL inferred classes,",
            "SHACL validation report, weighted",
            "support / contradiction / risk",
            "scoring with reasoning paths.",
        ],
        "graph.reasoner.SemanticPolicyScorer — advisory",
    ),
    (
        "Learned models",
        [
            "Live short-horizon predictor",
            "(CPU or OpenVINO, auxiliary only)",
            "and the strategy-utility R-GCN",
            "running in shadow observation.",
        ],
        "models.* · routing.shadow_intelligence — advisory",
    ),
]

DECISIONS = [
    (
        "Exit evaluated first",
        [
            "Every cycle scores SELL / REDUCE for held",
            "symbols before any BUY is considered.",
            "An open SELL at the same effective price",
            "is kept instead of re-submitted.",
        ],
        "trading.shared_decision_engine.evaluate_exit_for_holding",
    ),
    (
        "DynamicExitPolicy",
        [
            "One resolver for take-profit, profit lock,",
            "trailing giveback, soft / hard / emergency",
            "stop. Loss exits need strong deterioration",
            "evidence and are blocked inside the noise band.",
        ],
        "trading.dynamic_exit_policy · config/dynamic_exit_policy.yaml",
    ),
    (
        "ProfitabilityGate",
        [
            "The single BUY authority. Expected NET return",
            "after fees, tax, spread, slippage and impact",
            "must clear a volatility- and liquidity-aware",
            "minimum edge, or the BUY is rejected.",
        ],
        "cost.profitability_gate · cost.trading_cost_engine",
    ),
    (
        "Sizing and risk",
        [
            "Risk budget, cash, liquidity and exposure caps",
            "size the order; principal protection and the",
            "RiskManager perform the final validation that",
            "turns an intent into a FinalOrder.",
        ],
        "risk.position_sizing · risk.principal_protection · risk.manager",
    ),
]

EXECUTION = [
    (
        "Order pricing",
        [
            "Limit price from book and tick rules,",
            "no-chase cap, emergency sell offsets.",
        ],
        "execution.order_pricing_policy",
    ),
    (
        "Guarded submission",
        [
            "Live flags, KIS health, arming, idempotency",
            "reservation, limit orders only.",
        ],
        "execution.live_execution_coordinator",
    ),
    (
        "Journals and reconciliation",
        [
            "Order and causal journals, status polling,",
            "broker-authoritative position rebuild.",
        ],
        "execution.live_order_journal · causal_journal",
    ),
    (
        "Surfaces and feedback",
        [
            "/account, /display, /display/ontology,",
            "audit logs, retraining artifacts.",
        ],
        "web_account_routes · account_dashboard",
    ),
]

MAPPING = [
    (
        "Orderbook and spread",
        [
            "→ BUY gate (SPREAD_TOO_WIDE, SPREAD_CONSUMES_ALPHA)",
            "→ execution limit price and no-chase cap",
            "→ execution-quality rejection of alpha-eating fills",
        ],
    ),
    (
        "Tick freshness and session",
        [
            "→ NoTrade and MODEL_FEATURE_UNAVAILABLE reason codes",
            "→ ontology fact validity windows (stale = hard block)",
            "→ REST snapshot fallback when the market is closed",
        ],
    ),
    (
        "Cash, holdings, realised PnL",
        [
            "→ position sizing and INSUFFICIENT_CASH_FOR_ONE_SHARE",
            "→ daily loss budget and BUY stop thresholds",
            "→ concentration and drawdown reduction exits",
        ],
    ),
    (
        "Volatility and liquidity",
        [
            "→ required minimum net return for a BUY",
            "→ take-profit, trailing and soft-stop levels",
            "→ macro BLOCK_BUY under a high-volatility regime",
        ],
    ),
    (
        "News and disclosure events",
        [
            "→ macro risk level and market-wide BUY block",
            "→ event-momentum admissibility and TTL expiry",
            "→ semantic risk evidence in the reasoning paths",
        ],
    ),
    (
        "Model and ontology output",
        [
            "→ expected exit price (never inflated above the honest estimate)",
            "→ strategy admissibility mask and compatibility score",
            "→ bounded exit-deterioration penalty on profitable positions",
        ],
    ),
]


def data_to_decision_flow() -> str:
    width = 1640
    inner_x, inner_w = 62, 1516
    center = width // 2
    band_gap = 34

    def row(
        y: int,
        title: str,
        entries: list,
        color: str,
        count: int,
        gap: int,
        badged: bool = False,
        chained: bool = False,
    ) -> tuple[str, int]:
        """Render one band of equal-height cards; return svg and the band height."""
        box_w = (inner_w - gap * (count - 1)) // count
        box_h = max(
            card_height(entry[2] if badged else entry[1], entry[-1]) for entry in entries
        )
        band_h = 40 + box_h + 18
        svg_out = band(38, y, 1564, band_h, title)
        for index, entry in enumerate(entries):
            column = index % count
            bx = inner_x + column * (box_w + gap)
            by = y + 40
            if badged:
                svg_out += card(
                    bx, by, box_w, entry[1], entry[2],
                    color=color, badge=entry[0], module=entry[3], h=box_h,
                )
            else:
                svg_out += card(
                    bx, by, box_w, entry[0], entry[1],
                    color=color, module=entry[2] if len(entry) > 2 else None, h=box_h,
                )
            if chained and column:
                svg_out += arrow(bx - gap, by + box_h // 2, bx - 4, by + box_h // 2)
        return svg_out, band_h

    body = ""
    cursor = 110
    bands = [
        ("1 · SOURCES", SOURCES, "green", 5, 28, True, False),
        ("2 · NORMALISE, TRUST, STORE", STORAGE, "blue", 4, 30, False, False),
        ("3 · FEATURES — POINT IN TIME, NO LOOK-AHEAD", FEATURES, "amber", 4, 30, False, False),
        ("4 · REASONING — ADVISORY EVIDENCE ONLY", REASONING, "violet", 4, 30, False, False),
        (
            "5 · DECISION AUTHORITY — DETERMINISTIC, SELL BEFORE BUY",
            DECISIONS, "red", 4, 30, False, True,
        ),
        ("6 · EXECUTION AND FEEDBACK", EXECUTION, "teal", 4, 30, False, False),
    ]
    for index, (title, entries, color, count, gap, badged, chained) in enumerate(bands):
        chunk, band_h = row(cursor, title, entries, color, count, gap, badged, chained)
        body += chunk
        if index:
            body += arrow(center, cursor - band_gap, center, cursor - 4)
        cursor += band_h + band_gap

    # Final band: the data-to-decision mapping, two rows of three cards.
    map_gap = 33
    map_w = (inner_w - map_gap * 2) // 3
    map_h = max(card_height(lines, None) for _, lines in MAPPING)
    map_band_h = 40 + map_h * 2 + 16 + 18
    body += band(38, cursor, 1564, map_band_h, "7 · WHICH DATA IS ALLOWED TO DECIDE WHAT")
    body += arrow(center, cursor - band_gap, center, cursor - 4)
    for index, (title, lines) in enumerate(MAPPING):
        column, map_row = index % 3, index // 3
        body += card(
            inner_x + column * (map_w + map_gap),
            cursor + 40 + map_row * (map_h + 16),
            map_w,
            title,
            lines,
            color="slate",
            h=map_h,
        )
    height = cursor + map_band_h + 52

    svg = header(
        width,
        height,
        "Data → process → decision map",
        "Which input reaches which processing stage, and which decision it is actually allowed to influence.",
    )
    svg += body
    svg += label(
        52,
        height - 22,
        "Advisory layers may narrow the action set and supply an honest expected edge. "
        "Only the cost, exit, sizing and risk gates can approve, and only LiveExecutionCoordinator can submit.",
        cls="note",
    )
    svg += "</svg>\n"
    return svg


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    for name, builder in (
        ("ontology_gnn_layers.svg", ontology_gnn_layers),
        ("data_to_decision_flow.svg", data_to_decision_flow),
    ):
        (OUT / name).write_text(builder(), encoding="utf-8")
        print(f"wrote docs/diagrams/{name}")


if __name__ == "__main__":
    main()
