"""Candidate discovery must not propose instruments the account cannot order.

The defect this guards
----------------------
``RiskManager`` carried ``derivatives_allowed`` / ``leverage_etf_allowed`` from the
start, but nothing classified a candidate, so the flags asserted a policy that was
never applied to anything. Meanwhile discovery ranks KRX by turnover — and leveraged
and inverse ETPs sit at the top of a turnover ranking most sessions, because they are
the highest-beta way to express an index view.

Measured on the live KIS volume ranking, 2026-08-11 10:40 KST: 12 of the top 30 names
were leveraged ETPs or leveraged ETNs, and the persisted session universe held six of
them (252670 KODEX 200선물인버스2X, 114800 KODEX 인버스, 233740 KODEX 코스닥150레버리지,
252710 TIGER 200선물인버스2X, 251340 KODEX 코스닥150선물인버스, 122630 KODEX 레버리지).
None is orderable without the 기본예탁금 and 사전 의무교육 that leveraged ETPs require,
so each one occupied a session-locked slot that could never produce a trade.
"""

from __future__ import annotations

import pytest

from app.data.instrument_eligibility import (
    CATEGORY_ETF,
    CATEGORY_ETN,
    CATEGORY_EQUITY,
    CATEGORY_LEVERAGED_ETP,
    classify,
    filter_tradable,
)
from app.trading.domestic_universe import UniverseDecision, resolve_universe

# Real rows from the live KIS volume ranking, so the fixtures cannot drift into
# names KRX does not actually issue.
LEVERAGED = (
    ("252670", "KODEX 200선물인버스2X"),
    ("114800", "KODEX 인버스"),
    ("233740", "KODEX 코스닥150레버리지"),
    ("252710", "TIGER 200선물인버스2X"),
    ("251340", "KODEX 코스닥150선물인버스"),
    ("462330", "KODEX 2차전지산업레버리지"),
    ("122630", "KODEX 레버리지"),
    # KRX issues alphanumeric codes for newer ETF listings; the shape check must not
    # refuse these for the wrong reason, nor let them through for lack of one.
    ("0193T0", "KODEX SK하이닉스단일종목레버리지"),
    ("0197X0", "SOL SK하이닉스선물단일종목인버스2X"),
    ("0195S0", "TIGER SK하이닉스단일종목레버리지"),
)

PERMITTED = (
    ("005930", "삼성전자", CATEGORY_EQUITY),
    ("006340", "대원전선", CATEGORY_EQUITY),
    ("229200", "KODEX 코스닥150", CATEGORY_ETF),
    ("360750", "TIGER 미국S&P500", CATEGORY_ETF),
    ("228790", "TIGER 화장품", CATEGORY_ETF),
    # Contains a digit next to a letter but is not a multiple; the bounded pattern
    # must not read "TOP2플러스" as a 2x.
    ("0167A0", "SOL AI반도체TOP2플러스", CATEGORY_ETF),
)


@pytest.mark.parametrize("symbol,name", LEVERAGED)
def test_leveraged_and_inverse_etps_are_refused(symbol: str, name: str) -> None:
    verdict = classify(symbol, name, market="KR")
    assert verdict.category == CATEGORY_LEVERAGED_ETP, name
    assert verdict.tradable is False


@pytest.mark.parametrize("symbol,name,category", PERMITTED)
def test_ordinary_listings_are_permitted(symbol: str, name: str, category: str) -> None:
    verdict = classify(symbol, name, market="KR")
    assert verdict.category == category, name
    assert verdict.tradable is True


@pytest.mark.parametrize(
    "symbol,name",
    (
        ("Q530107", "삼성 인버스 2X 코스닥150 선물 ETN"),
        ("Q520057", "미래에셋 인버스 2X 코스닥150 선물 ETN"),
    ),
)
def test_etn_codes_are_refused(symbol: str, name: str) -> None:
    assert classify(symbol, name, market="KR").category == CATEGORY_ETN


def test_exchange_traded_derivative_codes_are_refused() -> None:
    """Futures and options are not in the equity code space at all."""
    for code in ("101S3000", "201S3300", "301S9000"):
        verdict = classify(code, market="KR")
        assert verdict.tradable is False, code


@pytest.mark.parametrize(
    "ticker,expect_tradable",
    (
        ("005930.KS", True),
        ("122630.KS", False),
        ("069540.KQ", True),
        # A US class suffix is part of the ticker and must survive the strip.
        ("BRK.B", True),
    ),
)
def test_board_suffixed_tickers_classify_on_the_listing_code(
    ticker: str, expect_tradable: bool
) -> None:
    """``005930.KS`` reaches the risk gate; refusing it as an unsupported shape would
    have blocked every ordinary KR order the moment the gate started binding."""
    names = {
        "005930.KS": "삼성전자",
        "122630.KS": "KODEX 레버리지",
        "069540.KQ": "라이브플렉스",
        "BRK.B": "Berkshire Hathaway",
    }
    assert classify(ticker, names[ticker]).tradable is expect_tradable


def test_filtering_preserves_the_symbol_form_it_was_given() -> None:
    """Discovery must get its own identifiers back, not a rewritten code."""
    kept, _ = filter_tradable(
        ["005930.KS", "122630.KS"],
        {"005930.KS": "삼성전자", "122630.KS": "KODEX 레버리지"},
        market="KR",
    )
    assert kept == ("005930.KS",)


def test_unnamed_listing_is_permitted_but_reported() -> None:
    """Refusing every unnamed candidate turns a ranking hiccup into an empty universe."""
    verdict = classify("005930", None, market="KR")
    assert verdict.tradable is True
    assert "INSTRUMENT_NAME_UNRESOLVED" in verdict.reason_codes


def test_flags_re_admit_without_a_code_change() -> None:
    """Completing the 기본예탁금/교육 is a config change, not a code change."""
    symbols = [code for code, _ in LEVERAGED]
    names = dict(LEVERAGED)

    blocked, excluded = filter_tradable(symbols, names, market="KR")
    assert blocked == ()
    assert len(excluded) == len(symbols)

    allowed, still_excluded = filter_tradable(
        symbols, names, market="KR", leverage_etf_allowed=True
    )
    assert set(allowed) == set(symbols)
    assert still_excluded == ()


def test_plain_etfs_require_their_own_account_permission() -> None:
    symbols = ["229200", "360750"]
    names = {"229200": "KODEX 코스닥150", "360750": "TIGER 미국S&P500"}

    blocked, excluded = filter_tradable(symbols, names, market="KR")
    assert blocked == ()
    assert {item.category for item in excluded} == {CATEGORY_ETF}
    assert all("INSTRUMENT_ETF_NOT_PERMITTED" in item.reason_codes for item in excluded)

    allowed, still_excluded = filter_tradable(
        symbols, names, market="KR", etf_allowed=True
    )
    assert tuple(allowed) == tuple(symbols)
    assert still_excluded == ()


# --- Universe integration ----------------------------------------------------- #


def test_session_locked_universe_purges_untradable_incumbents() -> None:
    """The load-bearing half: session locking would otherwise hold them all day.

    Filtering only the fresh ranking leaves an incumbent that got in before the
    filter existed sitting in its slot until the next session — which is exactly the
    state the 2026-08-11 universe was found in.
    """
    state = {
        "session_date": "2026-08-11",
        "symbols": ["252670", "005930", "122630", "006340"],
        "names": {
            "252670": "KODEX 200선물인버스2X",
            "005930": "삼성전자",
            "122630": "KODEX 레버리지",
            "006340": "대원전선",
        },
    }
    decision = resolve_universe(
        ("003010", "067290"),
        state=state,
        size=4,
        names={"003010": "혜인", "067290": "JW신약"},
        now=_at("2026-08-11"),
    )

    assert decision.source == "session_locked"
    assert "252670" not in decision.symbols
    assert "122630" not in decision.symbols
    assert {"005930", "006340"} <= set(decision.symbols)
    assert set(decision.dropped) == {"252670", "122630"}
    # The vacated slots were never going to produce a candidate, so refilling them is
    # recovering nominal capacity rather than the churn session locking prevents.
    assert len(decision.symbols) == 4
    assert {"003010", "067290"} <= set(decision.symbols)


def test_backfill_converges_after_one_refresh() -> None:
    """Once the refill is persisted the next refresh must take the plain locked path."""
    state = {
        "session_date": "2026-08-11",
        "symbols": ["005930", "006340", "003010", "067290"],
        "names": {"005930": "삼성전자", "006340": "대원전선"},
    }
    decision = resolve_universe(
        ("079650", "095910"),
        state=state,
        size=4,
        now=_at("2026-08-11"),
    )
    assert decision.source == "session_locked"
    assert decision.added == ()
    assert decision.dropped == ()
    assert decision.symbols == ("005930", "006340", "003010", "067290")


def test_reselection_never_admits_a_leveraged_etp() -> None:
    decision = resolve_universe(
        ("252670", "005930", "122630", "006340"),
        state={"session_date": "2026-08-10", "symbols": []},
        size=10,
        names=dict(LEVERAGED) | {"005930": "삼성전자", "006340": "대원전선"},
        now=_at("2026-08-11"),
    )
    assert decision.source == "reselected"
    assert decision.symbols == ("005930", "006340")
    assert {row["symbol"] for row in decision.excluded} == {"252670", "122630"}


def test_names_are_persisted_so_a_restart_can_still_classify() -> None:
    """Without persisted names the session-locked path has nothing to classify by."""
    decision = UniverseDecision(
        symbols=("005930",),
        session_date="2026-08-11",
        source="reselected",
        names={"005930": "삼성전자", "122630": "KODEX 레버리지"},
    )
    # Only the chosen symbols' names are kept, so the file cannot grow into a rolling
    # cache of everything ever ranked.
    kept = {s: decision.names[s] for s in decision.symbols if s in decision.names}
    assert kept == {"005930": "삼성전자"}


def _at(day: str):
    from datetime import datetime, timezone

    return datetime.fromisoformat(day).replace(tzinfo=timezone.utc)
