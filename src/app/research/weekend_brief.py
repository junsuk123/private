"""Weekend macro/event research turned into a FALSIFIABLE Monday-open prior.

Why a prior and not a report
----------------------------
Both venues are shut from the KRX Friday close until the KRX Monday open, so no
price forms and no strategy can act. That window is still informative — macro
prints, weekend news and the US Friday session all reprice risk — but a weekend
"analysis" that is never checked against what actually happened is unfalsifiable,
and this repository has already paid for unmeasurable features: six strategies sat
in the catalogue for the life of the training set producing zero labels because
nothing ever scored them.

So the output is a single committed claim — direction, magnitude, confidence —
written BEFORE the Monday open and scored AFTER it. It earns trust the same way the
GNN does, or it does not earn it.

What actually carries information here
--------------------------------------
Measured on this repository's own stores, not assumed:

* ``events`` (13,408 rows): rich and usable — sectors, tickers, event_labels
  (earnings/guidance/macro/...), classification_confidence.
* ``macro_metrics`` (FRED): thin (12 rows total) but the daily series — VIX, the
  10y yield, the broad dollar index — are real numbers whose weekend CHANGE is
  economically meaningful. Depth is the limiting factor, and the brief reports its
  own input coverage rather than hiding it.
* ``news_sentiment``: DELIBERATELY UNUSED. 155,959 of 158,338 stored scores are
  exactly +1.0 (98.5%). A feed that says "positive" to almost everything cannot
  discriminate, and averaging it would produce a confident-looking number carrying
  no information.

The primary driver is the US Friday session move. Overnight/closed-market returns
predicting the next session's opening gap is the best-established linkage available
here (Lin/Engle/Ito 1994; Lee 2012 for KRX specifically, where the night session
leads regular-session price discovery), and it is computable from data already held.
"""

from __future__ import annotations

import json
import math
import sqlite3
from contextlib import closing
from dataclasses import dataclass, field
from datetime import datetime, time, timedelta, timezone
from pathlib import Path
from statistics import fmean
from typing import Any, Iterable, Mapping, Sequence

KST = timezone(timedelta(hours=9))

# KRX regular session boundaries in Korea time.
KRX_OPEN = time(9, 0)
KRX_CLOSE = time(15, 30)

DEFAULT_STORE_PATH = "data/store/weekend_brief.sqlite3"

UP = "UP"
DOWN = "DOWN"
NEUTRAL = "NEUTRAL"


@dataclass(frozen=True)
class WeekendWindow:
    """Friday KRX close -> Monday KRX open, the interval with no tradable price."""

    start: datetime
    end: datetime

    @property
    def key(self) -> str:
        """Stable identity: the Monday date the prior is about."""
        return self.end.astimezone(KST).strftime("%Y-%m-%d")

    def contains(self, moment: datetime) -> bool:
        return self.start <= moment < self.end


def weekend_window(now: datetime) -> WeekendWindow | None:
    """The weekend window ``now`` belongs to, or ``None`` on a normal trading day.

    Covers Friday after the close, all of Saturday and Sunday, and Monday before
    the open — the whole span during which a Monday prior is still un-resolved.
    """
    local = now.astimezone(KST)
    weekday = local.weekday()  # Mon=0 .. Sun=6
    local_time = local.timetz().replace(tzinfo=None)

    if weekday == 4 and local_time >= KRX_CLOSE:  # Friday after close
        friday = local
    elif weekday in (5, 6):  # Saturday / Sunday
        friday = local - timedelta(days=weekday - 4)
    elif weekday == 0 and local_time < KRX_OPEN:  # Monday pre-open
        friday = local - timedelta(days=3)
    else:
        return None

    start = datetime.combine(friday.date(), KRX_CLOSE, tzinfo=KST)
    monday = friday + timedelta(days=(7 - friday.weekday()) % 7 or 3)
    # From Friday, the next Monday is +3 days.
    monday = friday + timedelta(days=3)
    end = datetime.combine(monday.date(), KRX_OPEN, tzinfo=KST)
    return WeekendWindow(start=start, end=end)


@dataclass(frozen=True)
class WeekendSignals:
    """Everything measurable about the closed window, with its own coverage."""

    window_key: str
    us_session_move_bps: float | None
    vix_change: float | None
    treasury_10y_change: float | None
    dollar_index_change: float | None
    event_count: int
    negative_event_count: int
    macro_event_count: int
    top_sectors: tuple[str, ...]
    inputs_available: int
    inputs_expected: int

    @property
    def coverage(self) -> float:
        if self.inputs_expected <= 0:
            return 0.0
        return round(self.inputs_available / self.inputs_expected, 3)

    def as_dict(self) -> dict[str, Any]:
        return {
            "window_key": self.window_key,
            "us_session_move_bps": self.us_session_move_bps,
            "vix_change": self.vix_change,
            "treasury_10y_change": self.treasury_10y_change,
            "dollar_index_change": self.dollar_index_change,
            "event_count": self.event_count,
            "negative_event_count": self.negative_event_count,
            "macro_event_count": self.macro_event_count,
            "top_sectors": list(self.top_sectors),
            "coverage": self.coverage,
        }


@dataclass(frozen=True)
class MondayOpenPrior:
    """One committed, checkable claim about the KRX Monday open gap."""

    window_key: str
    direction: str
    magnitude_bps: float
    confidence: float
    reason_codes: tuple[str, ...]
    signals: WeekendSignals
    computed_at: datetime
    reasons: tuple[str, ...] = field(default_factory=tuple)

    def as_dict(self) -> dict[str, Any]:
        return {
            "window_key": self.window_key,
            "direction": self.direction,
            "magnitude_bps": round(self.magnitude_bps, 2),
            "confidence": round(self.confidence, 4),
            "reason_codes": list(self.reason_codes),
            "reasons": list(self.reasons),
            "computed_at": self.computed_at.isoformat(),
            "signals": self.signals.as_dict(),
        }


# --------------------------------------------------------------------------- #
# Signal collection                                                            #
# --------------------------------------------------------------------------- #
def collect_weekend_signals(
    window: WeekendWindow,
    *,
    research_db: str | Path = "data/store/research.sqlite3",
    us_session_move_bps: float | None = None,
) -> WeekendSignals:
    """Aggregate the closed window from the research store.

    ``us_session_move_bps`` is injected rather than derived here so the caller owns
    the market-data dependency and this stays unit-testable.
    """
    events: list[Mapping[str, Any]] = []
    macro: dict[str, list[tuple[datetime, float]]] = {}
    path = Path(research_db)
    if path.exists():
        try:
            with closing(sqlite3.connect(path)) as conn:
                events = _load_events(conn, window)
                macro = _load_macro_series(conn, window)
        except sqlite3.Error:
            events, macro = [], {}

    # Prefer an LLM re-classification where one exists. The stored keyword verdict is
    # saturated (98.5% positive), so counting it would understate weekend stress; the
    # override table is written by the weekend enrichment pass.
    overrides: dict[str, str] = {}
    try:
        from app.research.weekend_enrichment import EventReclassificationStore

        overrides = EventReclassificationStore().sentiment_overrides(
            window.start.astimezone(timezone.utc).isoformat(),
            window.end.astimezone(timezone.utc).isoformat(),
        )
    except Exception:  # noqa: BLE001 - enrichment is optional, never required
        overrides = {}

    def _sentiment_of(row: Mapping[str, Any]) -> str:
        event_id = str(row.get("event_id") or "")
        if event_id and event_id in overrides:
            return overrides[event_id].upper()
        return str(row.get("sentiment", "")).upper()

    negative = sum(1 for row in events if _sentiment_of(row) == "NEGATIVE")
    macro_events = sum(
        1
        for row in events
        if "macro" in {str(x).lower() for x in (row.get("event_labels") or ())}
        or str(row.get("event_type", "")).upper() == "MACRO"
    )
    sector_counts: dict[str, int] = {}
    for row in events:
        for sector in row.get("sectors") or ():
            key = str(sector).strip()
            if key:
                sector_counts[key] = sector_counts.get(key, 0) + 1
    top_sectors = tuple(
        name for name, _ in sorted(sector_counts.items(), key=lambda kv: -kv[1])[:5]
    )

    vix = _series_change(macro.get("us_vix_close"))
    tsy = _series_change(macro.get("us_treasury_10y_yield"))
    usd = _series_change(macro.get("us_broad_dollar_index"))

    expected = 4  # us move + three macro deltas
    available = sum(
        1 for value in (us_session_move_bps, vix, tsy, usd) if value is not None
    )
    return WeekendSignals(
        window_key=window.key,
        us_session_move_bps=us_session_move_bps,
        vix_change=vix,
        treasury_10y_change=tsy,
        dollar_index_change=usd,
        event_count=len(events),
        negative_event_count=negative,
        macro_event_count=macro_events,
        top_sectors=top_sectors,
        inputs_available=available,
        inputs_expected=expected,
    )


def _load_events(conn: sqlite3.Connection, window: WeekendWindow) -> list[dict[str, Any]]:
    rows = conn.execute(
        "select payload from records where kind='events'"
        " and observed_at >= ? and observed_at < ?",
        (window.start.astimezone(timezone.utc).isoformat(), window.end.astimezone(timezone.utc).isoformat()),
    ).fetchall()
    out: list[dict[str, Any]] = []
    for (payload,) in rows:
        try:
            data = json.loads(payload)
        except (TypeError, ValueError):
            continue
        if isinstance(data, dict):
            out.append(data)
    return out


def _load_macro_series(
    conn: sqlite3.Connection, window: WeekendWindow, *, lookback_days: int = 21
) -> dict[str, list[tuple[datetime, float]]]:
    """Recent observations per FRED series, up to the window end.

    A lookback is required because the weekend itself contains no prints: the
    "weekend change" is the move across the last observations available before the
    Monday open.
    """
    since = (window.end - timedelta(days=lookback_days)).astimezone(timezone.utc)
    rows = conn.execute(
        "select payload, observed_at from records where kind='macro_metrics'"
        " and observed_at >= ? and observed_at < ? order by observed_at",
        (since.isoformat(), window.end.astimezone(timezone.utc).isoformat()),
    ).fetchall()
    series: dict[str, list[tuple[datetime, float]]] = {}
    for payload, observed_at in rows:
        try:
            data = json.loads(payload)
        except (TypeError, ValueError):
            continue
        name = str(data.get("name") or "")
        value = data.get("value")
        if value is None:
            value = data.get("metric_value")
        try:
            numeric = float(value)
        except (TypeError, ValueError):
            continue
        try:
            moment = datetime.fromisoformat(str(observed_at))
        except (TypeError, ValueError):
            continue
        if name:
            series.setdefault(name, []).append((moment, numeric))
    return series


# US proxies for the move KRX has not yet priced, best first, as
# ``(exchange_code, symbol)``.
#
# EWY is the iShares MSCI South Korea ETF: it trades in New York AFTER the KRX close,
# so its Friday move IS the market repricing Korean equities inside the exact window
# this prior is about. SPY/QQQ are broad risk proxies and only stand in when EWY is
# unavailable — they answer "was it risk-on" rather than "what happened to Korea".
US_PROXY_PRIORITY: tuple[tuple[str, str], ...] = (
    ("AMS", "EWY"),
    ("AMS", "SPY"),
    ("NAS", "QQQ"),
)


def us_session_move_bps(
    window: WeekendWindow | None = None,
    *,
    client: Any | None = None,
    proxies: Sequence[tuple[str, str]] = US_PROXY_PRIORITY,
) -> tuple[float | None, str | None]:
    """The US session move in bps, plus which proxy supplied it.

    Taken from the broker's overseas quote, whose ``rate`` field already carries the
    session's percent change against the previous close — precisely the move KRX has
    not traded on, since the US Friday session runs entirely after KRX shuts.

    The stored ``typed_market_snapshots`` were tried first and rejected on evidence:
    across the whole weekend window every proxy held exactly ONE distinct price, and
    the values were not real (SPY recorded at 275.94 against an actual 747.03). A
    frozen, wrong series would have produced a confident 0.0bps "flat" reading.

    Returns ``(None, None)`` when no proxy answers. A missing primary signal must be
    reported as missing — 0.0 would be a claim that nothing happened.
    """
    del window  # the quote is point-in-time; the window only frames interpretation
    if client is None:
        try:
            from app.execution.kis_real import KisDevelopersApiClient

            client = KisDevelopersApiClient(paper=False, enabled=True)
        except Exception:  # noqa: BLE001 - no broker, no primary signal
            return None, None

    for exchange, symbol in proxies:
        try:
            response = client._get(  # noqa: SLF001 - read-only quotation endpoint
                "/uapi/overseas-price/v1/quotations/price",
                tr_id="HHDFS00000300",
                params={"AUTH": "", "EXCD": exchange, "SYMB": symbol},
            )
        except Exception:  # noqa: BLE001 - try the next proxy
            continue
        output = (response or {}).get("output") or {}
        raw_rate = output.get("rate")
        if raw_rate in (None, ""):
            continue
        try:
            percent = float(raw_rate)
        except (TypeError, ValueError):
            continue
        # A quote that reports an exactly-zero change is far more likely to be an
        # unrefreshed field than a market that closed perfectly flat.
        if percent == 0.0:
            continue
        return percent * 100.0, symbol
    return None, None


def _series_change(points: Sequence[tuple[datetime, float]] | None) -> float | None:
    """Change between the two most recent observations; ``None`` if unavailable.

    Two observations is the minimum that can express a change at all. Returning 0.0
    for a single observation would assert "no move" from no evidence.
    """
    if not points or len(points) < 2:
        return None
    ordered = sorted(points, key=lambda item: item[0])
    return ordered[-1][1] - ordered[-2][1]


# --------------------------------------------------------------------------- #
# Prior                                                                        #
# --------------------------------------------------------------------------- #
def build_monday_prior(
    signals: WeekendSignals,
    *,
    computed_at: datetime,
    us_beta: float = 0.6,
) -> MondayOpenPrior:
    """Turn weekend signals into a direction/magnitude/confidence claim.

    The US session move is the primary term — closed-market returns predicting the
    next session's opening gap is the best-supported linkage available here, and KRX
    opens after the US Friday close. ``us_beta`` is the pass-through: KRX does not
    reproduce the whole US move, and 0.6 is a deliberately conservative default that
    the scoring history is meant to correct.

    Macro deltas act as MODIFIERS, never as the primary term:
      * VIX up   -> risk-off, drag
      * 10y up   -> discount-rate pressure, drag
      * dollar up-> foreign outflow pressure on KRX, drag

    Confidence scales with input coverage. A prior built from one of four inputs
    must not look as certain as one built from four.
    """
    reasons: list[str] = []
    codes: list[str] = []
    magnitude = 0.0

    if signals.us_session_move_bps is not None:
        contribution = signals.us_session_move_bps * us_beta
        magnitude += contribution
        codes.append("US_SESSION_SPILLOVER")
        reasons.append(
            f"US session moved {signals.us_session_move_bps:.0f}bps; "
            f"pass-through {us_beta:.2f} -> {contribution:.0f}bps"
        )
    else:
        codes.append("US_SESSION_MOVE_UNAVAILABLE")

    # Modifiers, in bps, deliberately small relative to the primary term.
    if signals.vix_change is not None and abs(signals.vix_change) > 0.01:
        drag = -signals.vix_change * 4.0
        magnitude += drag
        codes.append("VIX_RISK_REPRICING")
        reasons.append(f"VIX change {signals.vix_change:+.2f} -> {drag:+.0f}bps")
    if signals.treasury_10y_change is not None and abs(signals.treasury_10y_change) > 0.005:
        drag = -signals.treasury_10y_change * 20.0
        magnitude += drag
        codes.append("RATES_REPRICING")
        reasons.append(
            f"10y change {signals.treasury_10y_change:+.3f} -> {drag:+.0f}bps"
        )
    if signals.dollar_index_change is not None and abs(signals.dollar_index_change) > 0.05:
        drag = -signals.dollar_index_change * 3.0
        magnitude += drag
        codes.append("DOLLAR_PRESSURE")
        reasons.append(
            f"dollar index change {signals.dollar_index_change:+.2f} -> {drag:+.0f}bps"
        )

    # Weekend negative-event concentration. The stored sentiment score is saturated
    # (98.5% positive) so only the RARE negative carries information; it is used as a
    # dampener on conviction, never as a directional signal of its own.
    negative_share = (
        signals.negative_event_count / signals.event_count
        if signals.event_count > 0
        else 0.0
    )
    if negative_share > 0.05:
        codes.append("ELEVATED_NEGATIVE_EVENT_SHARE")
        reasons.append(f"negative events {negative_share:.1%} of weekend flow")

    if magnitude > 5.0:
        direction = UP
    elif magnitude < -5.0:
        direction = DOWN
    else:
        direction = NEUTRAL
        codes.append("BELOW_DIRECTIONAL_THRESHOLD")

    # Confidence: coverage-limited, dampened by negative-event concentration, and
    # capped well below certainty. This prior has no track record yet.
    confidence = 0.5 * signals.coverage
    confidence *= max(0.5, 1.0 - negative_share)
    if direction == NEUTRAL:
        confidence *= 0.5
    confidence = max(0.0, min(0.75, confidence))

    return MondayOpenPrior(
        window_key=signals.window_key,
        direction=direction,
        magnitude_bps=magnitude,
        confidence=confidence,
        reason_codes=tuple(dict.fromkeys(codes)),
        signals=signals,
        computed_at=computed_at,
        reasons=tuple(reasons),
    )


def score_prior(prior: MondayOpenPrior, realized_gap_bps: float) -> dict[str, Any]:
    """Compare a committed prior with the realized Monday open gap."""
    if realized_gap_bps > 5.0:
        realized_direction = UP
    elif realized_gap_bps < -5.0:
        realized_direction = DOWN
    else:
        realized_direction = NEUTRAL
    return {
        "window_key": prior.window_key,
        "predicted_direction": prior.direction,
        "realized_direction": realized_direction,
        "direction_correct": prior.direction == realized_direction,
        "predicted_bps": round(prior.magnitude_bps, 2),
        "realized_bps": round(realized_gap_bps, 2),
        "absolute_error_bps": round(abs(prior.magnitude_bps - realized_gap_bps), 2),
        "confidence": round(prior.confidence, 4),
    }


# --------------------------------------------------------------------------- #
# Storage                                                                      #
# --------------------------------------------------------------------------- #
class WeekendBriefStore:
    """Priors and their scores, keyed by the Monday they describe."""

    def __init__(self, path: str | Path = DEFAULT_STORE_PATH) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with closing(sqlite3.connect(self.path)) as conn:
            conn.execute(
                """
                create table if not exists monday_prior (
                    window_key text primary key,
                    computed_at text not null,
                    direction text not null,
                    magnitude_bps real not null,
                    confidence real not null,
                    payload text not null,
                    realized_gap_bps real,
                    scored_at text,
                    score_payload text
                )
                """
            )
            conn.commit()

    def save_prior(self, prior: MondayOpenPrior) -> None:
        """Upsert, but NEVER overwrite a prior that has already been scored.

        Rewriting a scored prior would let a later run quietly improve its own track
        record — the record has to be of what was actually claimed beforehand.
        """
        with closing(sqlite3.connect(self.path)) as conn:
            existing = conn.execute(
                "select scored_at from monday_prior where window_key=?",
                (prior.window_key,),
            ).fetchone()
            if existing and existing[0]:
                return
            conn.execute(
                """
                insert into monday_prior (
                    window_key, computed_at, direction, magnitude_bps, confidence, payload
                ) values (?, ?, ?, ?, ?, ?)
                on conflict(window_key) do update set
                    computed_at = excluded.computed_at,
                    direction = excluded.direction,
                    magnitude_bps = excluded.magnitude_bps,
                    confidence = excluded.confidence,
                    payload = excluded.payload
                """,
                (
                    prior.window_key,
                    prior.computed_at.isoformat(),
                    prior.direction,
                    prior.magnitude_bps,
                    prior.confidence,
                    json.dumps(prior.as_dict(), ensure_ascii=False),
                ),
            )
            conn.commit()

    def latest_prior(self) -> dict[str, Any] | None:
        with closing(sqlite3.connect(self.path)) as conn:
            row = conn.execute(
                "select payload, realized_gap_bps, scored_at, score_payload"
                " from monday_prior order by window_key desc limit 1"
            ).fetchone()
        if not row:
            return None
        payload = json.loads(row[0])
        payload["realized_gap_bps"] = row[1]
        payload["scored_at"] = row[2]
        payload["score"] = json.loads(row[3]) if row[3] else None
        return payload

    def record_score(self, window_key: str, realized_gap_bps: float) -> dict[str, Any] | None:
        with closing(sqlite3.connect(self.path)) as conn:
            row = conn.execute(
                "select payload, scored_at from monday_prior where window_key=?",
                (window_key,),
            ).fetchone()
            if not row or row[1]:
                return None  # unknown window, or already scored
            payload = json.loads(row[0])
            prior = MondayOpenPrior(
                window_key=payload["window_key"],
                direction=payload["direction"],
                magnitude_bps=float(payload["magnitude_bps"]),
                confidence=float(payload["confidence"]),
                reason_codes=tuple(payload.get("reason_codes") or ()),
                signals=WeekendSignals(
                    window_key=payload["window_key"],
                    us_session_move_bps=None,
                    vix_change=None,
                    treasury_10y_change=None,
                    dollar_index_change=None,
                    event_count=0,
                    negative_event_count=0,
                    macro_event_count=0,
                    top_sectors=(),
                    inputs_available=0,
                    inputs_expected=4,
                ),
                computed_at=datetime.fromisoformat(payload["computed_at"]),
            )
            score = score_prior(prior, realized_gap_bps)
            conn.execute(
                "update monday_prior set realized_gap_bps=?, scored_at=?, score_payload=?"
                " where window_key=?",
                (
                    realized_gap_bps,
                    datetime.now(timezone.utc).isoformat(),
                    json.dumps(score, ensure_ascii=False),
                    window_key,
                ),
            )
            conn.commit()
        return score

    def track_record(self) -> dict[str, Any]:
        """Accuracy so far. Trust must be earned, exactly like the GNN's."""
        with closing(sqlite3.connect(self.path)) as conn:
            rows = conn.execute(
                "select score_payload from monday_prior where score_payload is not null"
            ).fetchall()
        scores = [json.loads(row[0]) for row in rows if row[0]]
        if not scores:
            return {"scored": 0, "direction_accuracy": None, "mean_absolute_error_bps": None}
        correct = sum(1 for s in scores if s.get("direction_correct"))
        return {
            "scored": len(scores),
            "direction_accuracy": round(correct / len(scores), 4),
            "mean_absolute_error_bps": round(
                fmean(float(s.get("absolute_error_bps") or 0.0) for s in scores), 2
            ),
        }
