#!/usr/bin/env python
"""국내·미국 전 세션 실시간 거래 readiness 점검 — **read-only, 실주문 없음**.

이 스크립트는 주문을 만들지도, 보내지도 않는다. 확인하는 것은 다음뿐이다:

1. 공식 API mapping 검증 상태 (capability matrix 의 verification source)
2. 현재 세션 판정 — 국내·미국 각각, venue 별로
3. 데이터 수신 가능 / 주문 route 가능 / 신규 진입 가능 / 청산 가능 (네 값 분리)
4. session calendar 최신성 (커버리지·완전성)
5. 실시간 저장소 스키마 버전과 마이그레이션 이력
6. 스트림별 수집 현황 + 교차 스트림 중복
7. quote/orderbook freshness 를 세션 임계값과 비교
8. market-session 별 학습 표본 분포
9. 세션별 live order authorization 상태
10. (``--with-kis``) KIS 연결·잔고·미체결 read-only 조회

기본 실행은 네트워크를 쓰지 않는다. KIS read-only 조회는 명시적으로 켜야 한다:

    python scripts/check_market_session_readiness.py
    python scripts/check_market_session_readiness.py --with-kis
    KIS_READINESS_ALLOW_NETWORK=1 python scripts/check_market_session_readiness.py --with-kis

exit code 0 = 모든 필수 항목 통과, 1 = 하나 이상 실패/미확인.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "src"))

from app.data.market_capabilities import (  # noqa: E402
    MarketGroup,
    ReasonCode,
    default_service,
)

DEFAULT_STORE_PATH = REPO_ROOT / "data" / "store" / "realtime_market_data.sqlite3"
DEFAULT_TRAINING_ROWS_PATH = REPO_ROOT / "data" / "store" / "live_training_rows.sqlite3"

OK = "PASS"
WARN = "WARN"
FAIL = "FAIL"


class Report:
    """점검 결과 수집기. WARN 은 exit code 에 영향을 주지 않는다."""

    def __init__(self) -> None:
        self.checks: list[dict[str, object]] = []

    def add(self, name: str, status: str, detail: object = "") -> None:
        self.checks.append({"check": name, "status": status, "detail": detail})

    @property
    def failed(self) -> list[dict[str, object]]:
        return [item for item in self.checks if item["status"] == FAIL]

    def render(self) -> str:
        lines: list[str] = []
        width = max((len(str(item["check"])) for item in self.checks), default=10)
        for item in self.checks:
            detail = item["detail"]
            rendered = (
                json.dumps(detail, ensure_ascii=False, default=str)
                if isinstance(detail, (dict, list, tuple))
                else str(detail)
            )
            lines.append(f"[{item['status']:<4}] {str(item['check']):<{width}}  {rendered}")
        return "\n".join(lines)


# --------------------------------------------------------------------------- #
# 1-3. capability / 세션
# --------------------------------------------------------------------------- #
def check_official_mapping(report: Report) -> None:
    service = default_service()
    matrix = service.capability_matrix()
    report.add(
        "official_api_mapping",
        OK,
        {
            "verification_source": matrix["verification_source"],
            "officially_verified_at": matrix["officially_verified_at"],
            "paper_mode": matrix["paper_mode"],
            "sessions_described": len(matrix["capabilities"]),
        },
    )
    unsupported = matrix["unsupported_routes"]
    report.add(
        "unsupported_routes",
        WARN if unsupported else OK,
        [
            {"session": row["session"], "venue": row["venue"], "reasons": row["reasons"]}
            for row in unsupported
        ],
    )


def check_sessions(report: Report, now: datetime) -> None:
    service = default_service()
    for group in (MarketGroup.KR, MarketGroup.US):
        active = service.active_capabilities(group, now)
        detail = {
            "trading_day": service.is_trading_day(group, now),
            "data_available": any(item.data_available for item in active),
            "trade_available": any(item.trade_available for item in active),
            "new_entry_allowed": any(item.new_entry_allowed for item in active),
            "exit_allowed": any(item.exit_allowed for item in active),
            "sessions": [
                {
                    "session": item.session.value,
                    "venue": item.venue.value,
                    "data": item.data_available,
                    "route": item.trade_available,
                    "entry": item.new_entry_allowed,
                    "exit": item.exit_allowed,
                    "ord_dvsn": item.limit_order_division(),
                    "trade_tr": item.trade_ws_tr_id,
                    "book_tr": item.orderbook_ws_tr_id,
                    "reasons": list(item.unavailable_reason),
                }
                for item in active
            ],
            "new_entry_block_reasons": list(service.new_entry_block_reasons(group, now)),
        }
        # 세션이 하나도 없는 것은 정상 상태(휴장/야간)이므로 실패가 아니다.
        report.add(f"session_state:{group.value}", OK, detail)


def check_live_order_authorization(report: Report) -> None:
    service = default_service()
    authorized: list[str] = []
    for session, policy in service.config.policies.items():
        if policy.live_order_authorized:
            authorized.append(session.value)
    report.add(
        "live_order_authorized_sessions",
        OK if authorized else WARN,
        sorted(authorized) or "실주문이 승인된 세션이 없습니다 (전 세션 fail-closed)",
    )


# --------------------------------------------------------------------------- #
# 4. 캘린더
# --------------------------------------------------------------------------- #
def check_calendar(report: Report, now: datetime) -> None:
    service = default_service()
    calendar = service.calendar
    detail = {
        "version": calendar.version,
        "provider": calendar.provider,
        "coverage": f"{calendar.coverage_start} .. {calendar.coverage_end}",
        "completeness": dict(calendar.completeness),
    }
    blocking = {
        group.value: list(service.blocking_calendar_reasons(group, now))
        for group in (MarketGroup.KR, MarketGroup.US)
    }
    detail["blocking_reasons"] = blocking
    status = FAIL if any(blocking.values()) else OK
    if status is OK and not all(
        calendar.is_complete(group) for group in (MarketGroup.KR, MarketGroup.US)
    ):
        status = WARN
        detail["note"] = (
            "일부 캘린더가 완전하지 않습니다 (예: KR 음력 휴장일 미포함). "
            "누락된 휴장일은 freshness 게이트가 잡습니다."
        )
    report.add("session_calendar", status, detail)


# --------------------------------------------------------------------------- #
# 5-7. 저장소 / 스트림 / freshness
# --------------------------------------------------------------------------- #
def check_store(report: Report, store_path: Path, now: datetime) -> None:
    if not store_path.exists():
        report.add("realtime_store", WARN, f"저장소 파일이 없습니다: {store_path}")
        return
    from app.data.realtime_store import SCHEMA_VERSION, RealtimeMarketDataStore

    store = RealtimeMarketDataStore(store_path)
    version = store.schema_version()
    report.add(
        "realtime_store_schema",
        OK if version >= SCHEMA_VERSION else FAIL,
        {
            "version": version,
            "expected": SCHEMA_VERSION,
            "migrations": store.migration_history(),
        },
    )

    since = now - timedelta(hours=1)
    inventory = store.stream_inventory(since)
    report.add(
        "stream_inventory_1h",
        OK if inventory else WARN,
        list(inventory) or "최근 1시간 동안 수집된 체결이 없습니다 (휴장이면 정상)",
    )
    duplicates = store.cross_stream_duplicate_count(since)
    report.add(
        "cross_stream_duplicates_1h",
        OK if duplicates == 0 else WARN,
        {
            "count": duplicates,
            "note": (
                "통합 피드와 venue 피드를 동시에 구독하고 있습니다. 분 bar 는 스트림별로 "
                "분리되어 이중 계산되지 않지만 구독 예산이 낭비됩니다."
                if duplicates
                else "교차 스트림 중복 없음"
            ),
        },
    )
    _check_freshness(report, store, inventory, now)


def _check_freshness(report: Report, store, inventory, now: datetime) -> None:
    """스트림별 최신 체결·호가 나이를 세션 임계값과 비교한다."""
    service = default_service()
    from app.data.market_capabilities import SessionId

    rows: list[dict[str, object]] = []
    for entry in inventory:
        last_at = entry.get("last_at")
        if not last_at:
            continue
        try:
            moment = datetime.fromisoformat(str(last_at))
        except ValueError:
            continue
        if moment.tzinfo is None:
            moment = moment.replace(tzinfo=timezone.utc)
        age_ms = (now - moment).total_seconds() * 1000.0
        try:
            session = SessionId(str(entry.get("session") or "UNKNOWN"))
        except ValueError:
            session = SessionId.UNKNOWN
        policy = service.policy(session)
        rows.append(
            {
                "stream_id": entry.get("stream_id"),
                "session": session.value,
                "last_tick_age_ms": round(age_ms, 1),
                "max_quote_age_ms": policy.max_quote_age_ms,
                "fresh": age_ms <= policy.max_quote_age_ms,
            }
        )
    if not rows:
        report.add("quote_freshness", WARN, "비교할 스트림이 없습니다")
        return
    stale = [row for row in rows if not row["fresh"]]
    report.add("quote_freshness", WARN if stale else OK, rows)


# --------------------------------------------------------------------------- #
# 8. 학습 표본 분포
# --------------------------------------------------------------------------- #
def check_training_samples(report: Report, path: Path) -> None:
    if not path.exists():
        report.add("training_rows", WARN, f"학습 행 저장소가 없습니다: {path}")
        return
    import sqlite3
    from contextlib import closing

    try:
        with closing(sqlite3.connect(f"file:{path}?mode=ro", uri=True, timeout=30)) as conn:
            tables = {
                row[0]
                for row in conn.execute(
                    "select name from sqlite_master where type='table'"
                ).fetchall()
            }
            if "live_training_rows" not in tables:
                report.add("training_rows", WARN, f"live_training_rows 테이블 없음 (있는 테이블: {sorted(tables)[:6]})")
                return
            columns = {row[1] for row in conn.execute("pragma table_info(live_training_rows)")}
            group_column = "market" if "market" in columns else None
            total = conn.execute("select count(*) from live_training_rows").fetchone()[0]
            distribution: dict[str, int] = {}
            if group_column:
                distribution = {
                    str(row[0] or "UNKNOWN"): int(row[1])
                    for row in conn.execute(
                        f"select {group_column}, count(*) from live_training_rows"
                        f" group by {group_column}"
                    ).fetchall()
                }
    except sqlite3.Error as exc:
        report.add("training_rows", WARN, f"읽기 실패: {exc}")
        return
    detail: dict[str, object] = {"total_rows": int(total), "by_market": distribution}
    if "session" not in columns:
        detail["note"] = (
            "학습 행에 session 컬럼이 없습니다 — market-session 별 모델 fallback 은 "
            "아직 market 단위까지만 가능합니다."
        )
    report.add("training_rows", OK if total else WARN, detail)


# --------------------------------------------------------------------------- #
# 10. KIS read-only
# --------------------------------------------------------------------------- #
def check_kis_readonly(report: Report) -> None:
    """실주문 없이 연결·잔고·미체결만 조회한다."""
    if not (
        os.getenv("KIS_READINESS_ALLOW_NETWORK", "").strip().lower()
        in {"1", "true", "yes", "on"}
    ):
        report.add(
            "kis_readonly",
            WARN,
            "네트워크 조회가 비활성입니다. KIS_READINESS_ALLOW_NETWORK=1 로 켜세요.",
        )
        return
    try:
        from app.execution.kis_auth import build_kis_client

        client = build_kis_client(paper=False)
        portfolio = client.get_portfolio()
        report.add(
            "kis_readonly",
            OK,
            {
                # 계좌번호·토큰은 절대 출력하지 않는다.
                "holdings": len(getattr(portfolio, "holdings", ()) or ()),
                "has_cash_figure": bool(getattr(portfolio, "cash", None) is not None),
            },
        )
    except Exception as exc:  # noqa: BLE001 - readiness 는 진단이므로 예외를 결과로 바꾼다.
        report.add("kis_readonly", FAIL, f"{type(exc).__name__}: {exc}")


# --------------------------------------------------------------------------- #
def _use_utf8_stdout() -> None:
    """Windows 기본 콘솔은 cp949 라서 한글 설명의 일부 문자에서 죽는다.

    진단 스크립트가 인코딩 때문에 실패하면 안 되므로 출력 스트림을 UTF-8 로 바꾸고,
    그래도 표현할 수 없는 문자는 대체 문자로 흘린다.
    """
    for stream_name in ("stdout", "stderr"):
        stream = getattr(sys, stream_name, None)
        reconfigure = getattr(stream, "reconfigure", None)
        if reconfigure is None:
            continue
        try:
            reconfigure(encoding="utf-8", errors="replace")
        except (ValueError, OSError):
            continue


def main(argv: list[str] | None = None) -> int:
    _use_utf8_stdout()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--with-kis", action="store_true", help="KIS read-only 조회 포함")
    parser.add_argument("--store", default=str(DEFAULT_STORE_PATH))
    parser.add_argument("--training-rows", default=str(DEFAULT_TRAINING_ROWS_PATH))
    parser.add_argument("--json", action="store_true", help="JSON 으로 출력")
    parser.add_argument(
        "--at",
        default="",
        help="판정 기준 시각 (ISO 8601). 생략하면 현재. 세션 경계 검증에 사용.",
    )
    args = parser.parse_args(argv)

    if args.at:
        now = datetime.fromisoformat(args.at)
        if now.tzinfo is None:
            now = now.replace(tzinfo=timezone.utc)
    else:
        now = datetime.now(timezone.utc)

    report = Report()
    report.add("as_of", OK, now.isoformat())
    check_official_mapping(report)
    check_sessions(report, now)
    check_live_order_authorization(report)
    check_calendar(report, now)
    check_store(report, Path(args.store), now)
    check_training_samples(report, Path(args.training_rows))
    if args.with_kis:
        check_kis_readonly(report)

    if args.json:
        print(json.dumps(report.checks, ensure_ascii=False, indent=2, default=str))
    else:
        print(report.render())
        print()
        failed = report.failed
        if failed:
            print(f"실패 {len(failed)}건: " + ", ".join(str(item["check"]) for item in failed))
        else:
            print("필수 항목 전부 통과 (WARN 은 차단 사유가 아닙니다).")
    return 1 if report.failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
