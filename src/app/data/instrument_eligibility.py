"""Which instruments this account is actually permitted to trade.

The gap this fills
------------------
``RiskManager`` has carried ``derivatives_allowed`` / ``leverage_etf_allowed``
since the beginning, but its own comment records why they could not bind:

    Still a global assertion (as the original combined check was), because there
    is no instrument classification on the intent to make it per-order.

So the flags asserted "the account may not trade these" while nothing ever asked
what a candidate *was*. Discovery ranks KRX names by turnover, and the most-traded
names on KRX are routinely leveraged and inverse ETPs — they are the highest-beta
way to express an index view, so they sit at the top of a turnover ranking almost
every session. Measured on the 2026-08-11 session universe: 9 of 24 held names were
ETFs and 6 of those were leveraged or inverse (KODEX 200선물인버스2X, KODEX 인버스,
KODEX 코스닥150레버리지, TIGER 200선물인버스2X, KODEX 코스닥150선물인버스,
KODEX 레버리지). None of them is orderable on an account without the 기본예탁금 and
사전 의무교육 that leveraged ETPs and exchange-traded derivatives require.

Why the name and not the code
-----------------------------
There is no arithmetic on a 6-digit KRX code that separates KODEX 레버리지 (122630)
from KODEX 200 (069500) — they are adjacent allocations in the same space. What does
separate them is the listed name: KRX requires a leveraged or inverse ETP to carry
its multiple in 종목명, which is why ``레버리지`` / ``인버스`` / ``2X`` are reliable
and a code prefix is not.

The name is already in hand. Every KIS ranking row carries ``hts_kor_isnm`` next to
the code, and discovery was reading the code and discarding the row.

What this does NOT claim
------------------------
Classification is only as good as the name it was given. A symbol with no resolvable
name is reported ``UNKNOWN`` and, for the KRX 6-digit space, allowed — refusing every
unnamed candidate would empty the universe on any ranking hiccup. The fail-closed
half is the code shape: anything that is not a plain 6-digit KRX listing (futures and
option codes, ETN ``Q``-codes) is refused outright, because the execution layer has no
path to construct such an order regardless of what the account is permitted to do.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Iterable, Mapping, Sequence

# --- Categories -------------------------------------------------------------- #
CATEGORY_EQUITY = "EQUITY"
CATEGORY_ETF = "ETF"
CATEGORY_LEVERAGED_ETP = "LEVERAGED_ETP"
CATEGORY_ETN = "ETN"
CATEGORY_DERIVATIVE = "DERIVATIVE"
CATEGORY_UNKNOWN = "UNKNOWN"

# --- Reason codes ------------------------------------------------------------ #
INSTRUMENT_LEVERAGED_ETP_NOT_PERMITTED = "INSTRUMENT_LEVERAGED_ETP_NOT_PERMITTED"
INSTRUMENT_DERIVATIVE_NOT_PERMITTED = "INSTRUMENT_DERIVATIVE_NOT_PERMITTED"
INSTRUMENT_ETN_NOT_PERMITTED = "INSTRUMENT_ETN_NOT_PERMITTED"
INSTRUMENT_ETF_NOT_PERMITTED = "INSTRUMENT_ETF_NOT_PERMITTED"
INSTRUMENT_CODE_SHAPE_UNSUPPORTED = "INSTRUMENT_CODE_SHAPE_UNSUPPORTED"
INSTRUMENT_NAME_UNRESOLVED = "INSTRUMENT_NAME_UNRESOLVED"

# Tokens KRX requires in the listed name of a leveraged or inverse ETP. ``인버스``
# is in the same set as ``레버리지`` deliberately: the 기본예탁금 / 사전교육
# requirement is written against "1배를 초과하거나 음(-)의 배율", so a plain -1x
# inverse carries it exactly as a 2x does.
_LEVERAGE_NAME_TOKENS: tuple[str, ...] = (
    "레버리지",
    "인버스",
    "곱버스",
    "LEVERAGE",
    "LEVERAGED",
    "INVERSE",
    "ULTRASHORT",
    "ULTRAPRO",
)

# ``2X`` / ``-1X`` / ``3X`` and their spaced forms. Bounded on both sides so a name
# like "TIGER 2차전지" or an ISIN fragment cannot match.
_MULTIPLE_PATTERN = re.compile(r"(?<![A-Z0-9])[-+]?[1-9]\s*X(?![A-Z0-9])")

# Exchange-traded derivative code shapes: KRX futures/options are not in the equity
# code space at all (e.g. 101S3000, 201S3300, 301S...). Matched so they can be named
# in a reason code rather than falling into the generic "unsupported shape" bucket.
_DERIVATIVE_CODE_PATTERN = re.compile(r"^[1-4][0-9]{2}[A-Z][0-9]{3,4}$")

# ETN codes are 7 characters and, on KIS, commonly ``Q`` + 6 digits.
_ETN_CODE_PATTERN = re.compile(r"^(Q[0-9]{6}|5[0-9]{6})$")

# KRX listing codes are 6 characters starting with a digit. They are NOT all numeric:
# newer ETF listings carry a letter (0193T0 KODEX SK하이닉스단일종목레버리지,
# 0167A0 SOL AI반도체TOP2플러스 both appear in today's turnover ranking). Matching only
# ``[0-9]{6}`` would classify those as an unsupported code shape and refuse a plain
# ETF for the wrong reason — and, worse, would refuse the two leveraged ones for a
# reason that no longer holds if the shape check is ever widened.
#
# Note for anyone widening discovery: ``strategy_performance_store.market_for_symbol``
# routes "6-digit numeric -> KR, everything else -> US", so these codes are currently
# filtered out upstream by ``_extract_domestic_symbol`` and would be misrouted to US
# if that filter were relaxed without fixing the routing rule too.
_KR_EQUITY_CODE_PATTERN = re.compile(r"^[0-9][0-9A-Z]{5}$")
# A US ticker: letters, optionally with a class suffix. Anything else is not a plain
# listing we can route.
_US_TICKER_PATTERN = re.compile(r"^[A-Z]{1,5}(\.[A-Z])?$")


@dataclass(frozen=True)
class InstrumentVerdict:
    """What an instrument is, and whether this account may trade it."""

    symbol: str
    name: str
    market: str
    category: str
    tradable: bool
    reason_codes: tuple[str, ...] = ()

    def as_dict(self) -> dict[str, object]:
        return {
            "symbol": self.symbol,
            "name": self.name,
            "market": self.market,
            "category": self.category,
            "tradable": self.tradable,
            "reason_codes": list(self.reason_codes),
        }


def _normalize_name(name: str | None) -> str:
    """Upper-cased with whitespace removed, so ``인버스 2X`` == ``인버스2X``."""
    return re.sub(r"\s+", "", str(name or "")).upper()


#: KRX market suffixes the codebase attaches to a listing code in some paths
#: (``005930.KS``). Stripped before classification: the suffix says which board the
#: listing is on, never what the instrument is, and leaving it attached would fail
#: the code-shape check and refuse an ordinary equity as an unsupported product.
_KR_MARKET_SUFFIXES: tuple[str, ...] = (".KS", ".KQ", ".KRX", ".KN")


def _strip_market_suffix(code: str) -> str:
    for suffix in _KR_MARKET_SUFFIXES:
        if code.endswith(suffix):
            return code[: -len(suffix)]
    return code


def _market_for(symbol: str, market: str | None) -> str:
    if market:
        resolved = str(market).strip().upper()
        if resolved in {"KR", "KRX", "KOSPI", "KOSDAQ", "KONEX"}:
            return "KR"
        if resolved in {"US", "USA", "NASD", "NASDAQ", "NYSE", "AMEX"}:
            return "US"
    text = _strip_market_suffix(str(symbol or "").strip().upper())
    # 6 characters beginning with a digit is the KRX shape; a US ticker is letters.
    return "KR" if len(text) == 6 and text[:1].isdigit() else "US"


def classify(symbol: str, name: str | None = None, *, market: str | None = None) -> InstrumentVerdict:
    """Classify one instrument from its code and (when known) its listed name."""
    raw = str(symbol or "").strip().upper()
    resolved_market = _market_for(raw, market)
    # Only a KRX code carries a board suffix; a US ``BRK.B`` dot is part of the
    # ticker itself and must survive.
    code = _strip_market_suffix(raw) if resolved_market == "KR" else raw
    normalized = _normalize_name(name)

    # U.S. ticker shape and display name cannot distinguish an ETF from a common
    # share (e.g. SPY vs SPY Inc.).  Use the locally cached official Nasdaq Trader
    # ETF flag; this lookup is memory-cached and performs no network IO.
    if resolved_market == "US":
        try:
            from app.data.instrument_catalog import us_instrument

            catalog_record = us_instrument(code)
        except Exception:  # noqa: BLE001 - unavailable metadata stays explicit below.
            catalog_record = None
        if catalog_record is not None and catalog_record.is_etf:
            return InstrumentVerdict(
                code,
                catalog_record.security_name or str(name or ""),
                resolved_market,
                CATEGORY_ETF,
                True,
                (),
            )
        if catalog_record is not None and not normalized:
            name = catalog_record.security_name
            normalized = _normalize_name(name)

    # --- Code shape, checked first and fail-closed ---------------------------- #
    # This half does not depend on a name being available, and it is about what the
    # execution layer can construct rather than about what the account may hold.
    if _DERIVATIVE_CODE_PATTERN.match(code):
        return InstrumentVerdict(
            code, str(name or ""), resolved_market, CATEGORY_DERIVATIVE, False,
            (INSTRUMENT_DERIVATIVE_NOT_PERMITTED,),
        )
    if _ETN_CODE_PATTERN.match(code):
        return InstrumentVerdict(
            code, str(name or ""), resolved_market, CATEGORY_ETN, False,
            (INSTRUMENT_ETN_NOT_PERMITTED,),
        )
    shape_ok = (
        _KR_EQUITY_CODE_PATTERN.match(code)
        if resolved_market == "KR"
        else _US_TICKER_PATTERN.match(code)
    )
    if not shape_ok:
        return InstrumentVerdict(
            code, str(name or ""), resolved_market, CATEGORY_UNKNOWN, False,
            (INSTRUMENT_CODE_SHAPE_UNSUPPORTED,),
        )

    # --- Name, which is the only thing that reveals leverage ------------------ #
    if not normalized:
        # Allowed, but reported. See the module docstring: refusing every unnamed
        # candidate turns one ranking hiccup into an empty universe.
        return InstrumentVerdict(
            code, "", resolved_market, CATEGORY_UNKNOWN, True, (INSTRUMENT_NAME_UNRESOLVED,)
        )

    if "ETN" in normalized:
        return InstrumentVerdict(
            code, str(name), resolved_market, CATEGORY_ETN, False,
            (INSTRUMENT_ETN_NOT_PERMITTED,),
        )
    leveraged = any(token in normalized for token in _LEVERAGE_NAME_TOKENS) or bool(
        _MULTIPLE_PATTERN.search(normalized)
    )
    if leveraged:
        return InstrumentVerdict(
            code, str(name), resolved_market, CATEGORY_LEVERAGED_ETP, False,
            (INSTRUMENT_LEVERAGED_ETP_NOT_PERMITTED,),
        )

    category = CATEGORY_ETF if _looks_like_etf(normalized) else CATEGORY_EQUITY
    return InstrumentVerdict(code, str(name), resolved_market, category, True, ())


# Brand prefixes that mark a Korean ETF. Only used to LABEL an already-permitted
# instrument; nothing is refused on the strength of this list.
_ETF_BRANDS: tuple[str, ...] = (
    "KODEX", "TIGER", "KBSTAR", "ARIRANG", "HANARO", "KOSEF", "SOL", "ACE",
    "PLUS", "RISE", "TIMEFOLIO", "WOORI", "히어로즈", "마이티", "ETF",
)


def _looks_like_etf(normalized_name: str) -> bool:
    return any(brand in normalized_name for brand in _ETF_BRANDS)


def is_tradable(
    symbol: str,
    name: str | None = None,
    *,
    market: str | None = None,
    derivatives_allowed: bool = False,
    etf_allowed: bool = False,
    leverage_etf_allowed: bool = False,
) -> bool:
    """``True`` when the account's permissions cover this instrument.

    The two flags are the same ones :class:`~app.schemas.domain.RiskRules` already
    carries, so completing the 기본예탁금 / 사전교육 and flipping them re-admits the
    instruments without any code change here.
    """
    verdict = classify(symbol, name, market=market)
    if verdict.category == CATEGORY_ETF:
        return bool(etf_allowed)
    if verdict.tradable:
        return True
    if verdict.category == CATEGORY_LEVERAGED_ETP:
        return bool(leverage_etf_allowed)
    if verdict.category in {CATEGORY_DERIVATIVE, CATEGORY_ETN}:
        return bool(derivatives_allowed)
    # An unsupported code shape is not a permission question — the execution layer
    # cannot build the order at all — so no flag re-admits it.
    return False


def filter_tradable(
    symbols: Sequence[str],
    names: Mapping[str, str] | None = None,
    *,
    market: str | None = None,
    derivatives_allowed: bool = False,
    etf_allowed: bool = False,
    leverage_etf_allowed: bool = False,
) -> tuple[tuple[str, ...], tuple[InstrumentVerdict, ...]]:
    """Split ``symbols`` into the permitted ones and the verdicts that excluded the rest.

    Returns ``(kept, excluded_verdicts)``. The excluded verdicts are returned rather
    than logged so the caller can report *why* discovery shrank — a universe that
    silently halves is indistinguishable from a broken ranking feed.
    """
    lookup = {str(k).strip().upper(): v for k, v in (names or {}).items()}
    kept: list[str] = []
    excluded: list[InstrumentVerdict] = []
    for raw in symbols:
        symbol = str(raw or "").strip().upper()
        if not symbol:
            continue
        verdict = classify(symbol, lookup.get(symbol), market=market)
        permitted = verdict.tradable
        if verdict.category == CATEGORY_ETF:
            permitted = bool(etf_allowed)
            if not permitted:
                verdict = InstrumentVerdict(
                    verdict.symbol,
                    verdict.name,
                    verdict.market,
                    verdict.category,
                    False,
                    (INSTRUMENT_ETF_NOT_PERMITTED,),
                )
        if not permitted:
            if verdict.category == CATEGORY_LEVERAGED_ETP:
                permitted = bool(leverage_etf_allowed)
            elif verdict.category in {CATEGORY_DERIVATIVE, CATEGORY_ETN}:
                permitted = bool(derivatives_allowed)
        if permitted:
            kept.append(symbol)
        else:
            excluded.append(verdict)
    return tuple(dict.fromkeys(kept)), tuple(excluded)


def excluded_summary(verdicts: Iterable[InstrumentVerdict]) -> dict[str, int]:
    """Category -> count, for a one-line report of what discovery dropped."""
    counts: dict[str, int] = {}
    for verdict in verdicts:
        counts[verdict.category] = counts.get(verdict.category, 0) + 1
    return counts
