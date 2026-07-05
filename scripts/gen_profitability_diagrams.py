#!/usr/bin/env python3
"""Generate the profitability-refactor SVG diagrams into docs/diagrams/.

Pure-Python SVG authoring (no matplotlib/graphviz dependency, no rasterizer needed —
GitHub renders SVG in markdown). Re-run to regenerate:

    python scripts/gen_profitability_diagrams.py
"""

from __future__ import annotations

from pathlib import Path

OUT = Path(__file__).resolve().parents[1] / "docs" / "diagrams"

# Theme (light background, readable in both GitHub themes).
BG = "#ffffff"
INK = "#1b1f24"
MUTED = "#57606a"
BLUE = "#1f6feb"
GREEN = "#1a7f37"
RED = "#cf222e"
AMBER = "#9a6700"
LINE = "#8b949e"
FONT = "font-family='Segoe UI, Helvetica, Arial, sans-serif'"


def _hdr(w: int, h: int, title: str) -> str:
    return (
        f"<svg xmlns='http://www.w3.org/2000/svg' width='{w}' height='{h}' "
        f"viewBox='0 0 {w} {h}'>\n"
        f"<rect width='{w}' height='{h}' fill='{BG}'/>\n"
        f"<text x='{w//2}' y='34' {FONT} font-size='20' font-weight='700' "
        f"fill='{INK}' text-anchor='middle'>{title}</text>\n"
        "<defs><marker id='arrow' markerWidth='10' markerHeight='10' refX='8' refY='3' "
        "orient='auto' markerUnits='strokeWidth'>"
        f"<path d='M0,0 L8,3 L0,6 z' fill='{LINE}'/></marker></defs>\n"
    )


def box(x, y, w, h, label, sub="", fill=BLUE, text="#ffffff", rx=10):
    out = f"<rect x='{x}' y='{y}' width='{w}' height='{h}' rx='{rx}' fill='{fill}'/>\n"
    if sub:
        out += (
            f"<text x='{x+w//2}' y='{y+h//2-4}' {FONT} font-size='14' font-weight='700' "
            f"fill='{text}' text-anchor='middle'>{label}</text>\n"
        )
        out += (
            f"<text x='{x+w//2}' y='{y+h//2+14}' {FONT} font-size='11' "
            f"fill='{text}' text-anchor='middle'>{sub}</text>\n"
        )
    else:
        out += (
            f"<text x='{x+w//2}' y='{y+h//2+5}' {FONT} font-size='14' font-weight='700' "
            f"fill='{text}' text-anchor='middle'>{label}</text>\n"
        )
    return out


def note(x, y, s, size=12, fill=MUTED, anchor="start", weight="400"):
    return (
        f"<text x='{x}' y='{y}' {FONT} font-size='{size}' font-weight='{weight}' "
        f"fill='{fill}' text-anchor='{anchor}'>{s}</text>\n"
    )


def arrow(x1, y1, x2, y2, color=LINE):
    return (
        f"<line x1='{x1}' y1='{y1}' x2='{x2}' y2='{y2}' stroke='{color}' "
        f"stroke-width='2' marker-end='url(#arrow)'/>\n"
    )


def _esc(s: str) -> str:
    return s.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")


def decision_flow() -> str:
    s = _hdr(900, 340, "BUY decision flow (profitability-first)")
    xs = [20, 190, 360, 530, 700]
    y = 120
    w, h = 150, 66
    steps = [
        ("Signal + Model", "predicted edge", BLUE),
        ("ProfitabilityGate", "net &#8805; dynamic floor", GREEN),
        ("PositionSizer", "fractional Kelly", BLUE),
        ("RiskManager", "same gate (backstop)", BLUE),
        ("ExecutionQuality", "spread / slippage / fill", BLUE),
    ]
    for i, (label, sub, fill) in enumerate(steps):
        s += box(xs[i], y, w, h, label, sub, fill=fill)
        if i < len(steps) - 1:
            s += arrow(xs[i] + w, y + h // 2, xs[i + 1], y + h // 2)
    # order box
    s += box(360, 250, 150, 54, "KIS order", "(guarded / armed)", fill=INK)
    s += arrow(775, y + h, 460, 250)
    # reject lane
    s += note(20, 230, "Any stage may REJECT → no order (reasons surfaced to GUI):", 13, RED, weight="700")
    s += note(
        20,
        252,
        _esc("MISSING_EXPECTED_EXIT_PRICE, BELOW_TARGET_NET_RETURN_AFTER_COST, BELOW_BREAK_EVEN_WITH_MARGIN,"),
        11,
        MUTED,
    )
    s += note(
        20,
        270,
        _esc("COST_BURDEN_HIGH, SPREAD_TOO_WIDE, SPREAD_CONSUMES_ALPHA, LIQUIDITY_TOO_LOW, SLIPPAGE_RISK_HIGH"),
        11,
        MUTED,
    )
    s += note(
        20,
        310,
        "Expected exit price is a REAL predicted edge (no fabricated 100bps floor).",
        12,
        AMBER,
        weight="700",
    )
    s += "</svg>\n"
    return s


def before_after() -> str:
    s = _hdr(900, 340, "Before → After: net-profitability gate")
    # before
    s += note(30, 70, "BEFORE", 15, RED, weight="700")
    s += box(30, 84, 180, 56, "Signal", "fabricated 100bps", fill=RED)
    s += arrow(210, 112, 260, 112)
    s += box(260, 84, 180, 56, "RiskManager", "garbage-in exit price", fill=MUTED)
    s += note(30, 172, "Replay of live journal:", 13, INK, weight="700")
    s += note(30, 194, _esc("gross PnL +1,342  /  NET PnL -1,701"), 13, RED, weight="700")
    s += note(30, 214, _esc("net expectancy -0.00117  /  101 of 181 net-negative"), 12, MUTED)
    # after
    s += note(470, 70, "AFTER", 15, GREEN, weight="700")
    s += box(470, 84, 170, 56, "Real edge", "model / fallback est.", fill=BLUE)
    s += arrow(640, 112, 690, 112)
    s += box(690, 84, 180, 56, "ProfitabilityGate", "net &#8805; required min", fill=GREEN)
    s += note(470, 172, "A BUY is allowed only when", 13, INK, weight="700")
    s += note(470, 194, _esc("net_expected_return ≥ required_min_net_return"), 12, GREEN, weight="700")
    s += note(470, 214, _esc("AND exit ≥ break-even+buffer AND spread/liquidity/cost OK"), 11, MUTED)
    s += note(470, 234, _esc("gross-positive / net-negative trades are rejected"), 12, MUTED)
    # divider
    s += f"<line x1='450' y1='60' x2='450' y2='250' stroke='{LINE}' stroke-dasharray='4 4'/>\n"
    s += note(450, 300, "One authoritative decision object across candidate, risk, engine, and GUI.", 12, MUTED, anchor="middle")
    s += "</svg>\n"
    return s


def exit_policy() -> str:
    s = _hdr(900, 320, "DynamicExitPolicy — unified thresholds + loss-exit governance")
    s += box(30, 80, 200, 60, "Resolved exit policy", "logged once", fill=GREEN)
    s += note(40, 165, "Dynamic (cost/vol/liquidity/spread aware):", 12, INK, weight="700")
    s += note(40, 185, "take-profit, profit-lock, trailing giveback, soft-stop", 11, MUTED)
    s += note(40, 205, "Capital circuit-breakers: hard-stop, emergency-stop", 11, MUTED)
    s += note(40, 240, "Env overrides honored (backward parity); logged.", 11, AMBER)
    # loss exit decision
    s += box(470, 70, 200, 54, "Loss-exit decision", "", fill=INK)
    s += box(470, 150, 190, 48, "BLOCK", "noise band / no evidence", fill=RED)
    s += box(680, 150, 190, 48, "ALLOW", "strong deterioration", fill=GREEN)
    s += arrow(540, 124, 560, 150, RED)
    s += arrow(600, 124, 760, 150, GREEN)
    s += note(680, 218, "hard/emergency stop, ontology SELL/REDUCE", 11, MUTED)
    s += note(680, 234, "dominance, strong negative forecast,", 11, MUTED)
    s += note(680, 250, "liquidity/regime, daily-loss budget", 11, MUTED)
    s += note(470, 285, "Losers no longer always-held or always-dumped — exit on evidence, hold through noise.", 12, INK, weight="700")
    s += "</svg>\n"
    return s


def architecture() -> str:
    s = _hdr(900, 340, "Profitability architecture — single sources of truth")
    comps = [
        (30, 70, "ProfitabilityGate", "profitability_policy.yaml", GREEN),
        (250, 70, "DynamicExitPolicy", "dynamic_exit_policy.yaml", GREEN),
        (470, 70, "PositionSizer", "position_sizing_policy.yaml", BLUE),
        (690, 70, "ExecutionQuality", "+ slippage store", BLUE),
    ]
    for x, y, label, sub, fill in comps:
        s += box(x, y, 190, 60, label, sub, fill=fill)
    s += box(250, 190, 400, 56, "SharedLiveDecisionEngine + RiskManager", "consume all four; log resolved values", fill=INK)
    for x, _, _, _, _ in comps:
        s += arrow(x + 95, 130, 450, 190)
    s += box(250, 285, 400, 44, "GUI: net PnL, break-even, expectancy, rejection reasons", "", fill=BLUE)
    s += arrow(450, 246, 450, 285)
    s += "</svg>\n"
    return s


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    diagrams = {
        "profitability_decision_flow.svg": decision_flow(),
        "profitability_before_after.svg": before_after(),
        "profitability_dynamic_exit.svg": exit_policy(),
        "profitability_architecture.svg": architecture(),
    }
    for name, svg in diagrams.items():
        (OUT / name).write_text(svg, encoding="utf-8")
        print(f"wrote docs/diagrams/{name}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
