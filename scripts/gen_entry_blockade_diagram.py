"""Render docs/diagrams/entry_blockade_chain.svg.

The diagram exists because "why has nothing traded" was, in production, answered
by a single reason code from whichever layer happened to fail last. For 11,614
consecutive cycles that read NO_POSITIVE_NET_GNN_EDGE, which named the GNN while
the actual constraint was that no scanned market was in its regular session.

Rendering the chain makes two things visible that prose does not:
* the order — the FIRST unmet link is the answer, later links were never reached;
* the fact that several links can be independently broken at once, so fixing the
  one that happens to be reported does not necessarily unblock trading.

Pure stdlib string building, no plotting dependency, deterministic output.
"""

from __future__ import annotations

from pathlib import Path
from xml.sax.saxutils import escape

OUTPUT = Path("docs/diagrams/entry_blockade_chain.svg")

BG = "#0d1420"
PANEL = "#131c2b"
LINE = "#22304a"
TEXT = "#e6edf6"
MUTED = "#8195ad"
GREEN = "#42d392"
RED = "#ff6678"
AMBER = "#f0b90b"

# (stage, korean label, verdict, detail) — the state measured on the live server
# at 2026-07-30T23:4x UTC.
STAGES = [
    ("engine_running", "엔진 실행", "pass", "11,614 사이클 정상 동작"),
    ("live_armed", "라이브 무장", "pass", "live_armed=true, 일일 손실 예산 잔여"),
    (
        "market_session",
        "시장 세션",
        "fail",
        "US=after / KRX=pre — 스캔 대상 시장이 정규장 아님",
    ),
    ("buy_candidates", "매수 후보", "pass", "6종목 (RIVN·SOFI·NIO·F·PFE·BAC)"),
    (
        "micro_buy_intents",
        "마이크로 전략",
        "fail",
        "0/6 실행가능 — LOW_LIQUIDITY_TECHNICAL_BLOCK 외",
    ),
    (
        "strategy_election",
        "전략 선택",
        "fail",
        "GNN entry_authorized=false (전 전략) → 순환 교착",
    ),
    ("position", "포지션", "fail", "SCANNING — 진입 없음"),
]

ROOT_CAUSES = [
    (
        "① 시장 세션",
        "시간외 호가로 평가",
        "F: 분당 2주 · 스프레드 33bp · liquidity_score 2e-05.\n"
        "\"완전 마감 아님\"을 \"거래 가능\"으로 읽고 있었음.",
        "allows_new_entry() — 정규장에서만 신규 진입",
    ),
    (
        "② GNN 권한 교착",
        "CALIBRATED_AWAITING_POSITIVE_EDGE",
        "entry_authorized 에는 실현 체결 5건 필요, 보유 1건.\n"
        "체결하려면 권한이 필요하고 권한은 체결을 요구함.",
        "보수적 bandit — require_live_gnn 을 거부권이 아닌 감점으로",
    ),
    (
        "③ 진단 오표시",
        "NO_POSITIVE_NET_GNN_EDGE",
        "마지막으로 실패한 계층의 코드 하나만 노출되어\n실제 원인(세션)이 가려짐.",
        "ENTRY BLOCKADE 체인 — 계층별 판정을 모두 노출",
    ),
]


def _text(x, y, content, *, size=12, fill=TEXT, weight="400", anchor="start", family=None):
    font = family or "'Segoe UI','Malgun Gothic',sans-serif"
    return (
        f'<text x="{x}" y="{y}" font-family="{font}" font-size="{size}" '
        f'fill="{fill}" font-weight="{weight}" text-anchor="{anchor}">{escape(content)}</text>'
    )


def build() -> str:
    width, height = 1180, 760
    parts: list[str] = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" '
        f'viewBox="0 0 {width} {height}" role="img" '
        f'aria-label="신규 진입 차단 체인 진단">',
        f'<rect width="{width}" height="{height}" fill="{BG}"/>',
        _text(40, 48, "왜 신규 진입이 없는가 — ENTRY BLOCKADE 체인", size=22, weight="700"),
        _text(
            40,
            74,
            "체인은 순서대로 평가되며, 처음 막힌 단계가 실제 원인입니다. 그 뒤 단계는 도달하지 않았을 뿐 통과한 것이 아닙니다.",
            size=12,
            fill=MUTED,
        ),
        _text(40, 96, "측정 시점: 2026-07-30T23:4x UTC · 누적 11,614 사이클 · 체결 0건", size=11, fill=MUTED),
    ]

    # --- Chain -------------------------------------------------------------
    top = 128
    row_h = 52
    first_fail = next(i for i, s in enumerate(STAGES) if s[2] == "fail")
    for index, (_stage, label, verdict, detail) in enumerate(STAGES):
        y = top + index * row_h
        unreachable = index > first_fail
        if verdict == "pass":
            accent, mark = GREEN, "✓"
        else:
            accent, mark = RED, "✕"
        if unreachable:
            accent, mark = LINE, "·"
        opacity = "0.45" if unreachable else "1"
        parts.append(f'<g opacity="{opacity}">')
        parts.append(
            f'<rect x="40" y="{y}" width="640" height="42" rx="8" fill="{PANEL}" '
            f'stroke="{LINE}"/>'
        )
        parts.append(f'<rect x="40" y="{y}" width="4" height="42" rx="2" fill="{accent}"/>')
        parts.append(_text(62, y + 27, mark, size=14, fill=accent, weight="700"))
        parts.append(_text(86, y + 19, label, size=13, weight="700"))
        detail_text = "앞 단계에서 막혀 평가되지 않음" if unreachable else detail
        parts.append(_text(86, y + 34, detail_text, size=10.5, fill=MUTED))
        parts.append("</g>")
        if index < len(STAGES) - 1:
            parts.append(
                f'<line x1="52" y1="{y + 42}" x2="52" y2="{y + row_h}" '
                f'stroke="{LINE}" stroke-width="1.5"/>'
            )

    # Marker for the reported-vs-actual gap.
    fail_y = top + first_fail * row_h
    parts.append(
        f'<path d="M 690 {fail_y + 21} L 716 {fail_y + 21}" stroke="{RED}" '
        f'stroke-width="2" marker-end="url(#arrow-red)"/>'
    )
    parts.append(_text(724, fail_y + 17, "실제 원인", size=12, fill=RED, weight="700"))
    parts.append(_text(724, fail_y + 33, "(체인이 없으면 보이지 않음)", size=10, fill=MUTED))

    reported_y = top + 5 * row_h
    parts.append(
        f'<path d="M 690 {reported_y + 21} L 716 {reported_y + 21}" stroke="{AMBER}" '
        f'stroke-width="2" marker-end="url(#arrow-amber)"/>'
    )
    parts.append(
        _text(724, reported_y + 17, "기존에 표시되던 사유", size=12, fill=AMBER, weight="700")
    )
    parts.append(
        _text(724, reported_y + 33, "NO_POSITIVE_NET_GNN_EDGE", size=10, fill=MUTED)
    )
    # The three failures are independent; the chain can only reveal them one at a
    # time, so "dimmed" must not be read as "fine".
    parts.append(
        _text(
            724,
            reported_y + 52,
            "이 단계도 독립적으로 막혀 있습니다.",
            size=10,
            fill=MUTED,
        )
    )
    parts.append(
        _text(
            724,
            reported_y + 66,
            "앞 단계를 고쳐야 비로소 드러납니다.",
            size=10,
            fill=MUTED,
        )
    )

    # --- Root causes -------------------------------------------------------
    base = top + len(STAGES) * row_h + 30
    parts.append(_text(40, base, "독립적으로 깨져 있던 세 가지 원인과 조치", size=15, weight="700"))
    card_w, gap = 358, 20
    for index, (title, symptom, body, fix) in enumerate(ROOT_CAUSES):
        x = 40 + index * (card_w + gap)
        y = base + 16
        parts.append(
            f'<rect x="{x}" y="{y}" width="{card_w}" height="152" rx="10" '
            f'fill="{PANEL}" stroke="{LINE}"/>'
        )
        parts.append(_text(x + 16, y + 26, title, size=13, weight="700"))
        parts.append(_text(x + 16, y + 45, symptom, size=10, fill=AMBER, family="Consolas,monospace"))
        for line_index, line in enumerate(body.split("\n")):
            parts.append(_text(x + 16, y + 68 + line_index * 15, line, size=10, fill=MUTED))
        parts.append(
            f'<line x1="{x + 16}" y1="{y + 108}" x2="{x + card_w - 16}" y2="{y + 108}" '
            f'stroke="{LINE}"/>'
        )
        parts.append(_text(x + 16, y + 127, "조치", size=9.5, fill=MUTED, weight="700"))
        fix_lines = _wrap(fix, 46)
        for line_index, line in enumerate(fix_lines[:2]):
            parts.append(_text(x + 16, y + 141 + line_index * 13, line, size=10, fill=GREEN))

    parts.append(
        f'<defs>'
        f'<marker id="arrow-red" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="6" '
        f'markerHeight="6" orient="auto"><path d="M 0 0 L 10 5 L 0 10 z" fill="{RED}"/></marker>'
        f'<marker id="arrow-amber" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="6" '
        f'markerHeight="6" orient="auto"><path d="M 0 0 L 10 5 L 0 10 z" fill="{AMBER}"/></marker>'
        f'</defs>'
    )
    parts.append("</svg>")
    return "\n".join(parts)


def _wrap(text: str, width: int) -> list[str]:
    words = text.split()
    lines: list[str] = []
    current = ""
    for word in words:
        candidate = f"{current} {word}".strip()
        if len(candidate) > width and current:
            lines.append(current)
            current = word
        else:
            current = candidate
    if current:
        lines.append(current)
    return lines


def main() -> None:
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT.write_text(build(), encoding="utf-8")
    print(f"wrote {OUTPUT} ({OUTPUT.stat().st_size} bytes)")


if __name__ == "__main__":
    main()
