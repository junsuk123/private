"""4단계 ``MarketPhase`` 를 쓰는 기존 호출자를 위한 backward-compatible wrapper.

**이 모듈은 더 이상 세션 판정 로직을 갖지 않는다.** 모든 판정은
:mod:`app.data.market_capabilities` 의 :class:`~app.data.market_capabilities.MarketSessionService`
가 수행하고, 여기서는 그 결과를 PRE / REGULAR / AFTER / CLOSED 4단계로 축약해 돌려준다.

왜 축약이 필요한가
------------------
PRE/REGULAR/AFTER/CLOSED 는 KRX 경매, KRX 시간외 종가, 시간외 단일가, NXT 3세션,
미국 주간거래를 구분할 수 없다. 그래서 새 코드는 ``SessionId`` 와 ``MarketCapability`` 를
직접 쓰고, 이 wrapper 는 아직 옮기지 않은 호출자(대시보드 배지, 로그 문구 등)를 위해서만
남아 있다. **신규 코드에서 쓰지 말 것.**

변경된 동작 (의도적)
--------------------
미국 애프터마켓 종료가 ET 20:00 → ET 17:00(EST) / 18:00(EDT) 로 좁혀졌다.
KIS 공식 문서가 애프터마켓 주문 가능 시간을 "06:00~07:00 KST (Summer Time 05:00~07:00)"
로 명시하기 때문이다. 기존 값은 2~3시간 과대였고, 그 구간의 주문은 브로커에서 거부됐다.
근거: ``docs/kis_market_session_capability_matrix.md`` §5.1.
"""

from __future__ import annotations

from datetime import datetime, timezone
from enum import Enum

from app.data.market_capabilities import (
    MarketGroup,
    SessionId,
    Venue,
    default_service,
    normalize_market_group,
)

_KRX_GROUP_NAMES = {"KRX", "KR", "KOSPI", "KOSDAQ", "KONEX"}
_US_GROUP_NAMES = {"US", "USA", "NASDAQ", "NASD", "NAS", "NYSE", "NYS", "AMEX", "AMS", "OVERSEAS"}


class MarketPhase(str, Enum):
    """Intraday session phase for a market group (legacy, 4단계)."""

    PRE = "pre"          # 프리마켓 / 장전
    REGULAR = "regular"  # 정규장
    AFTER = "after"      # 애프터마켓 / 장후
    CLOSED = "closed"    # 완전 마감


#: 세분화된 SessionId → legacy phase.
_PHASE_BY_SESSION: dict[SessionId, MarketPhase] = {
    SessionId.KRX_PREOPEN: MarketPhase.PRE,
    SessionId.KRX_OPENING_AUCTION: MarketPhase.PRE,
    SessionId.KRX_REGULAR: MarketPhase.REGULAR,
    # 종가 단일가는 정규장 매매의 마지막 국면이다. legacy 호출자는 15:20-15:30 을
    # REGULAR 로 보고 있었고 그 의미는 유지한다.
    SessionId.KRX_CLOSING_AUCTION: MarketPhase.REGULAR,
    SessionId.KRX_AFTER_CLOSE: MarketPhase.AFTER,
    SessionId.KRX_AFTER_SINGLE_PRICE: MarketPhase.AFTER,
    SessionId.NXT_PRE: MarketPhase.PRE,
    SessionId.NXT_REGULAR: MarketPhase.REGULAR,
    SessionId.NXT_POST: MarketPhase.AFTER,
    SessionId.US_PREMARKET: MarketPhase.PRE,
    SessionId.US_REGULAR: MarketPhase.REGULAR,
    SessionId.US_AFTERMARKET: MarketPhase.AFTER,
    # 미국 주간거래는 legacy 4단계에 대응하는 국면이 없다. 한국 낮 시간에 열리는
    # 별도 세션이므로 PRE(장전거래) 로 축약한다 — KIS 문서도 "주간거래(장전거래)" 라
    # 표기한다. 정확한 판정이 필요한 호출자는 SessionId 를 직접 쓸 것.
    SessionId.US_DAYTIME: MarketPhase.PRE,
}

#: legacy KRX 호출자가 기대하던 phase 사이의 빈 구간 (15:30-15:40 은 KRX 세션이
#: 없지만 장후로 취급된다). (start_minute, end_minute, phase) — Asia/Seoul 분 단위.
_KRX_LEGACY_GAP_WINDOWS = ((15 * 60 + 30, 15 * 60 + 40, MarketPhase.AFTER),)


def _normalize_group(group: str) -> str:
    """Legacy 그룹 정규화. ``"KRX"`` / ``"US"`` / 원문 대문자를 반환한다."""
    name = str(group or "").upper().strip()
    if name in _KRX_GROUP_NAMES:
        return "KRX"
    if name in _US_GROUP_NAMES:
        return "US"
    return name


def _as_utc(now_utc: datetime | None) -> datetime:
    current = now_utc or datetime.now(timezone.utc)
    if current.tzinfo is None:
        current = current.replace(tzinfo=timezone.utc)
    return current


def _phase_from_sessions(
    group: MarketGroup,
    now_utc: datetime,
    *,
    venues: frozenset[Venue] | None = None,
) -> MarketPhase:
    service = default_service()
    active = [
        item
        for item in service.active_capabilities(group, now_utc)
        if venues is None or item.venue in venues
    ]
    if not active:
        return MarketPhase.CLOSED
    phases = {_PHASE_BY_SESSION.get(item.session, MarketPhase.CLOSED) for item in active}
    # 여러 세션이 겹치면 "가장 활발한" 국면을 보고한다.
    for candidate in (MarketPhase.REGULAR, MarketPhase.PRE, MarketPhase.AFTER):
        if candidate in phases:
            return candidate
    return MarketPhase.CLOSED


def _krx_legacy_gap_phase(now_utc: datetime) -> MarketPhase:
    from zoneinfo import ZoneInfo

    local = now_utc.astimezone(ZoneInfo("Asia/Seoul"))
    minute = local.hour * 60 + local.minute
    for start, end, phase in _KRX_LEGACY_GAP_WINDOWS:
        if start <= minute < end:
            return phase
    return MarketPhase.CLOSED


def market_phase(group: str, now_utc: datetime | None = None) -> MarketPhase:
    """Classify the current intraday phase for a market group ("KRX" or "US").

    국내는 **KRX venue 만** 본다 (NXT 를 포함한 스트리밍 판정은 :func:`streaming_phase`).
    """
    normalized = _normalize_group(group)
    current = _as_utc(now_utc)
    if normalized == "KRX":
        phase = _phase_from_sessions(
            MarketGroup.KR, current, venues=frozenset({Venue.KRX})
        )
        if phase is MarketPhase.CLOSED and default_service().is_trading_day(
            MarketGroup.KR, current
        ):
            return _krx_legacy_gap_phase(current)
        return phase
    if normalized == "US":
        # 미국 주간거래는 야간 세션과 성격이 달라 legacy 판정에서는 제외한다.
        # (주간거래를 인식해야 하는 호출자는 SessionId.US_DAYTIME 을 직접 조회할 것.)
        return _phase_from_sessions(
            MarketGroup.US, current, venues=frozenset({Venue.NASDAQ, Venue.NYSE, Venue.AMEX})
        )
    return MarketPhase.CLOSED


def is_market_fully_closed(group: str, now_utc: datetime | None = None) -> bool:
    """True when the group has no session at all (not pre, regular, or after)."""
    return market_phase(group, now_utc) is MarketPhase.CLOSED


def streaming_phase(
    group: str,
    now_utc: datetime | None = None,
    *,
    include_nxt: bool = True,
) -> MarketPhase:
    """Phase for DATA STREAMING, which is wider than the order-gating phase.

    NXT (넥스트레이드) 는 08:00-20:00 KST 로 운영되므로 통합(KRX+NXT) 실시간 피드는
    KRX 09:00-15:30 밖에서도 체결을 전달한다. 주문 게이팅은 이 값을 쓰지 않는다 —
    19:00 에 KRX 로 주문을 보내면 거부되고, NXT 주문 route 는 별도 capability 다.
    """
    normalized = _normalize_group(group)
    current = _as_utc(now_utc)
    if normalized != "KRX":
        return market_phase(group, current)
    if not include_nxt:
        return market_phase("KRX", current)
    phase = _phase_from_sessions(
        MarketGroup.KR, current, venues=frozenset({Venue.KRX, Venue.NXT})
    )
    if phase is MarketPhase.CLOSED and default_service().is_trading_day(
        MarketGroup.KR, current
    ):
        return _krx_legacy_gap_phase(current)
    return phase


def market_has_live_session(group: str, now_utc: datetime | None = None) -> bool:
    """True when the group is in any tradeable/quotable session (pre/regular/after)."""
    return not is_market_fully_closed(group, now_utc)


def allows_new_entry(group: str, now_utc: datetime | None = None) -> bool:
    """May a NEW position be opened in this group right now?

    canonical service 에 위임한다. "닫혀 있지 않다"는 "매수해도 된다"가 아니다:

        US after-hours, 2026-07-30 23:4x UTC — F quoting 2 shares/minute at a
        33bps spread, liquidity_score 2e-05.

    이 데이터에서 모든 후보가 ``hold`` / ``LOW_LIQUIDITY_TECHNICAL_BLOCK`` 으로 귀결되어
    election 이 매수 의도를 0건 받았는데, 운영자에게는 GNN 실패로 보였다. 세션 제약을
    명시적으로 표현하면 종목별 유동성 실패로 위장되지 않는다.

    청산은 이 함수로 게이팅하지 않는다 — 포지션은 항상 닫을 수 있어야 하고, 청산 경로는
    얇은 호가에 대한 자체 가격 가드를 갖는다. 세션별 청산 가능 여부는
    ``MarketSessionService.exit_allowed`` 를 쓸 것.

    세션별 제어는 ``TRADING_ALLOW_ENTRY_<SESSION>`` 이고,
    ``TRADING_ALLOW_EXTENDED_HOURS_ENTRY`` 는 backward-compatible alias 로 유지된다.
    """
    market = normalize_market_group(group)
    if market is None:
        return False
    return default_service().new_entry_allowed(market, _as_utc(now_utc))


def new_entry_session_report(
    groups: tuple[str, ...] = ("KRX", "US"),
    now_utc: datetime | None = None,
) -> dict:
    """Per-group phase plus new-entry permission, for the dashboard.

    "왜 아무것도 거래되지 않았는가" 를 운영자가 타임스탬프 대조 없이 답할 수 있어야 한다.
    세분화된 세션 정보가 필요하면 ``MarketSessionService.session_report`` 를 쓸 것.
    """
    current = _as_utc(now_utc)
    service = default_service()
    per_group: dict[str, dict] = {}
    for group in groups:
        normalized = _normalize_group(group)
        market = normalize_market_group(group)
        entry = {
            "phase": market_phase(group, current).value,
            "streaming_phase": streaming_phase(group, current).value,
            "allows_new_entry": allows_new_entry(group, current),
            "fully_closed": is_market_fully_closed(group, current),
        }
        if market is not None:
            active = service.active_capabilities(market, current)
            entry.update(
                {
                    "session": service.primary_capability(market, current).session.value,
                    "active_sessions": [item.session.value for item in active],
                    "data_available": any(item.data_available for item in active),
                    "trade_available": any(item.trade_available for item in active),
                    "exit_allowed": any(item.exit_allowed for item in active),
                    "new_entry_block_reasons": list(
                        service.new_entry_block_reasons(market, current)
                    ),
                }
            )
        per_group[normalized] = entry
    return {
        "as_of": current.isoformat(),
        "calendar_version": service.calendar.version,
        "extended_hours_entry_enabled": any(
            entry["allows_new_entry"] and entry["phase"] != MarketPhase.REGULAR.value
            for entry in per_group.values()
        ),
        "any_group_allows_new_entry": any(
            item["allows_new_entry"] for item in per_group.values()
        ),
        "groups": per_group,
    }


__all__ = [
    "MarketPhase",
    "allows_new_entry",
    "is_market_fully_closed",
    "market_has_live_session",
    "market_phase",
    "new_entry_session_report",
    "streaming_phase",
]
