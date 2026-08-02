"""The three inputs the weekend prior was missing, and why each was missing.

1. US session move — the primary term was absent. ``typed_market_snapshots`` was
   tried and rejected on evidence: across the weekend window every proxy held
   exactly ONE distinct price and the values were wrong (SPY recorded at 275.94
   against an actual 747.03). The broker quote's ``rate`` field carries the session
   change directly.
2. FRED history — ``collect_latest`` fetched a batch and returned only its newest
   row, so the macro store held 12 observations across 6 series. Two points is the
   minimum that can express a change and several series had one.
3. Event sentiment — the keyword classifier scored 155,959 of 158,338 events at
   exactly +1.0 (98.5%). A re-classification pass measured 55% positive on the same
   material, which is a feed that can actually discriminate.
"""

from __future__ import annotations

from datetime import datetime, timezone

import pytest

from app.research.weekend_brief import (
    US_PROXY_PRIORITY,
    KST,
    us_session_move_bps,
    weekend_window,
)
from app.research.weekend_enrichment import (
    EventReclassificationStore,
    ReclassifyResult,
    backfill_macro_history,
    reclassify_events,
)

NOW_SAT = datetime(2026, 8, 1, 12, 0, tzinfo=KST)


# --------------------------------------------------------------------------- #
# 1. US session move                                                           #
# --------------------------------------------------------------------------- #
class _QuoteClient:
    """Stands in for the KIS client; records which proxies were asked."""

    def __init__(self, rates: dict[str, str | None]) -> None:
        self.rates = rates
        self.asked: list[str] = []

    def _get(self, path, *, tr_id, params):  # noqa: D401 - mirrors the real signature
        symbol = params["SYMB"]
        self.asked.append(symbol)
        rate = self.rates.get(symbol, "__missing__")
        if rate == "__missing__":
            raise RuntimeError("quote unavailable")
        return {"output": {"rate": rate} if rate is not None else {}}


def test_ewy_is_preferred_because_it_prices_korea() -> None:
    """EWY trades in New York after KRX closes, so its move IS the repricing."""
    assert US_PROXY_PRIORITY[0][1] == "EWY"
    client = _QuoteClient({"EWY": "-2.55", "SPY": "+0.72"})
    move, proxy = us_session_move_bps(client=client)
    assert proxy == "EWY"
    assert move == pytest.approx(-255.0)
    assert client.asked == ["EWY"], "a working primary must not fall through"


def test_percent_is_converted_to_bps() -> None:
    move, _ = us_session_move_bps(client=_QuoteClient({"EWY": "+1.25"}))
    assert move == pytest.approx(125.0)


def test_falls_through_to_broad_proxies() -> None:
    client = _QuoteClient({"SPY": "+0.72"})  # EWY raises
    move, proxy = us_session_move_bps(client=client)
    assert proxy == "SPY"
    assert move == pytest.approx(72.0)


def test_no_proxy_answers_reports_missing_not_flat() -> None:
    """0.0 would assert 'the US did not move', which is a claim, not an absence."""
    move, proxy = us_session_move_bps(client=_QuoteClient({}))
    assert move is None and proxy is None


def test_exactly_zero_rate_is_treated_as_unrefreshed() -> None:
    """A quote reporting a perfectly flat session is far more likely to be stale."""
    client = _QuoteClient({"EWY": "0.00", "SPY": "+0.40"})
    move, proxy = us_session_move_bps(client=client)
    assert proxy == "SPY"
    assert move == pytest.approx(40.0)


def test_unparseable_rate_does_not_crash() -> None:
    move, proxy = us_session_move_bps(client=_QuoteClient({"EWY": "n/a", "SPY": "+0.5"}))
    assert proxy == "SPY"


# --------------------------------------------------------------------------- #
# 2. FRED history                                                              #
# --------------------------------------------------------------------------- #
class _Collector:
    def __init__(self, per_series: int = 100, fail: set[str] | None = None) -> None:
        self.per_series = per_series
        self.fail = fail or set()
        self.calls: list[str] = []

    def collect_history(self, series_id, name, *, limit=120):
        self.calls.append(series_id)
        if series_id in self.fail:
            raise RuntimeError("fred unavailable")
        return tuple(object() for _ in range(min(self.per_series, limit)))


class _Store:
    def __init__(self, fail: bool = False) -> None:
        self.saved = 0
        self.batches = 0
        self.fail = fail

    def save_macro_metrics(self, records):
        if self.fail:
            raise RuntimeError("database is locked")
        self.batches += 1
        self.saved += len(records)
        return len(records)


def test_backfill_persists_whole_series_not_one_point() -> None:
    """The defect: one observation per series per cycle."""
    store = _Store()
    result = backfill_macro_history(
        series=[("VIXCLS", "us_vix_close"), ("DGS10", "us_treasury_10y_yield")],
        store=store,
        collector=_Collector(per_series=100),
    )
    assert result.series_attempted == 2
    assert result.records_written == 200
    assert store.batches == 2, "records must be saved in batches, not one at a time"


def test_one_failing_series_does_not_abort_the_sweep() -> None:
    store = _Store()
    collector = _Collector(per_series=50, fail={"CPIAUCSL"})
    result = backfill_macro_history(
        series=[("CPIAUCSL", "cpi"), ("VIXCLS", "vix"), ("DGS10", "t10")],
        store=store,
        collector=collector,
    )
    assert len(collector.calls) == 3
    assert result.records_written == 100
    assert len(result.failures) == 1
    assert "CPIAUCSL" in result.failures[0]


def test_a_locked_database_is_reported_not_swallowed() -> None:
    """Observed for real: the running server holds the same SQLite file."""
    result = backfill_macro_history(
        series=[("VIXCLS", "vix")], store=_Store(fail=True), collector=_Collector()
    )
    assert result.records_written == 0
    assert result.failures and "save" in result.failures[0]


# --------------------------------------------------------------------------- #
# 3. Event re-classification                                                   #
# --------------------------------------------------------------------------- #
class _Verdict:
    def __init__(self, sentiment: str) -> None:
        self.sentiment = sentiment
        self.confidence = 0.8
        self.model = "qwen2.5:1.5b-instruct"


class _Classifier:
    def __init__(self, verdicts: list[str], fail_on: int | None = None) -> None:
        self.verdicts = verdicts
        self.fail_on = fail_on
        self.calls = 0

    def classify(self, title, body, known_tickers):
        index = self.calls
        self.calls += 1
        if self.fail_on is not None and index == self.fail_on:
            raise RuntimeError("model timeout")
        return _Verdict(self.verdicts[index % len(self.verdicts)])


def _events(n: int, sentiment: str = "POSITIVE"):
    return [
        {
            "event_id": f"e{i}",
            "title": f"headline {i}",
            "summary": "body",
            "sentiment": sentiment,
            "_observed_at": "2026-08-01T01:00:00+00:00",
        }
        for i in range(n)
    ]


def test_disagreement_with_the_keyword_verdict_is_counted(tmp_path) -> None:
    store = EventReclassificationStore(tmp_path / "e.sqlite3")
    result = reclassify_events(
        _events(4, sentiment="POSITIVE"),
        classifier=_Classifier(["NEGATIVE", "POSITIVE", "NEUTRAL", "NEGATIVE"]),
        store=store,
    )
    assert result.reclassified == 4
    assert result.changed_sentiment == 3  # everything except the POSITIVE one
    assert store.distribution() == {"NEGATIVE": 2, "POSITIVE": 1, "NEUTRAL": 1}


def test_one_model_failure_does_not_end_the_batch(tmp_path) -> None:
    result = reclassify_events(
        _events(5),
        classifier=_Classifier(["NEGATIVE"], fail_on=2),
        store=EventReclassificationStore(tmp_path / "e.sqlite3"),
    )
    assert result.considered == 5
    assert result.reclassified == 4
    assert result.failures == 1


def test_missing_classifier_is_reported_not_silent() -> None:
    """considered=0/failures=0 alone is indistinguishable from 'nothing to do'."""
    result = reclassify_events([], classifier=None, store=None)
    if result.classifier_available:
        pytest.skip("an LLM classifier is configured in this environment")
    assert result.considered == 0
    assert result.classifier_available is False


def test_reclassification_never_touches_the_original_record(tmp_path) -> None:
    """Corrections live in their own table so both verdicts stay inspectable."""
    store = EventReclassificationStore(tmp_path / "e.sqlite3")
    events = _events(1, sentiment="POSITIVE")
    reclassify_events(events, classifier=_Classifier(["NEGATIVE"]), store=store)
    assert events[0]["sentiment"] == "POSITIVE", "the source event must be unchanged"
    assert store.distribution() == {"NEGATIVE": 1}


def test_overrides_are_scoped_to_the_window(tmp_path) -> None:
    store = EventReclassificationStore(tmp_path / "e.sqlite3")
    reclassify_events(_events(2), classifier=_Classifier(["NEGATIVE"]), store=store)
    inside = store.sentiment_overrides(
        "2026-08-01T00:00:00+00:00", "2026-08-01T02:00:00+00:00"
    )
    outside = store.sentiment_overrides(
        "2026-07-01T00:00:00+00:00", "2026-07-02T00:00:00+00:00"
    )
    assert len(inside) == 2
    assert outside == {}


def test_result_dict_exposes_availability() -> None:
    assert ReclassifyResult().as_dict()["classifier_available"] is True
