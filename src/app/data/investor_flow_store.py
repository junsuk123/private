"""Persisted daily investor flow (개인/외국인/기관 순매수) for KRX symbols.

Why this store exists
---------------------
``residual_relative_strength`` requires informed-flow evidence: residual strength
with no institutional or foreign buying behind it is usually a squeeze, so the
expert treats flow as mandatory rather than optional. The counterfactual labeler
had no such data, hardcoded ``investor_flow`` to 0.0, and the strategy therefore
never fired in training — it was uneval*uable*, not unprofitable.

Order-book imbalance is NOT a substitute. Imbalance describes resting quotes over
seconds; investor flow describes who actually accumulated over a day. Filling one
with the other would have produced a trained model scoring a feature that does not
mean what its name says, so this fetches the real thing instead.

Granularity
-----------
KIS reports investor flow per business day, not intraday. That matches how the
feature is used: a day's net institutional buying is slow context for an intraday
thesis, in the same way sector rank and gap reference are. Every minute bar within
one business day therefore shares that day's flow, and this is a deliberate
property, not an interpolation.
"""

from __future__ import annotations

import sqlite3
from contextlib import closing
from dataclasses import dataclass
from datetime import date, datetime
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

DEFAULT_STORE_PATH = "data/store/investor_flow.sqlite3"


@dataclass(frozen=True)
class InvestorFlowDay:
    symbol: str
    business_date: str  # YYYYMMDD, as KIS reports it
    close_price: float
    retail_net_buy_value: float
    foreign_net_buy_value: float
    institution_net_buy_value: float

    @property
    def informed_net_buy_value(self) -> float:
        """Foreign + institutional net buying — the informed-flow proxy.

        Retail is deliberately excluded rather than subtracted: this is used as a
        magnitude of informed demand, and mixing in a retail term would make a
        heavily retail-sold day look identical to an institution-bought one.
        """
        return self.foreign_net_buy_value + self.institution_net_buy_value

    def as_dict(self) -> dict[str, Any]:
        return {
            "symbol": self.symbol,
            "business_date": self.business_date,
            "close_price": self.close_price,
            "retail_net_buy_value": self.retail_net_buy_value,
            "foreign_net_buy_value": self.foreign_net_buy_value,
            "institution_net_buy_value": self.institution_net_buy_value,
            "informed_net_buy_value": self.informed_net_buy_value,
        }


class InvestorFlowStore:
    def __init__(self, path: str | Path = DEFAULT_STORE_PATH) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._ensure_schema()

    def _connect(self) -> sqlite3.Connection:
        return sqlite3.connect(self.path, timeout=15)

    def _ensure_schema(self) -> None:
        with closing(self._connect()) as conn:
            conn.execute(
                """
                create table if not exists investor_flow_daily (
                    symbol text not null,
                    business_date text not null,
                    close_price real,
                    retail_net_buy_value real,
                    foreign_net_buy_value real,
                    institution_net_buy_value real,
                    updated_at text not null,
                    primary key (symbol, business_date)
                )
                """
            )
            conn.execute(
                "create index if not exists idx_investor_flow_date"
                " on investor_flow_daily (business_date)"
            )
            conn.commit()

    def upsert_many(self, rows: Iterable[Mapping[str, Any]]) -> int:
        """Insert or refresh rows. Returns the number written.

        Upsert rather than insert-ignore: the newest business day is still being
        traded when it is first fetched, so its figures change during the session
        and a stale first read must not win forever.
        """
        payload = []
        stamp = datetime.now().astimezone().isoformat()
        for row in rows:
            symbol = str(row.get("symbol") or "").strip()
            business_date = str(row.get("business_date") or "").strip()
            if not symbol or len(business_date) != 8:
                continue
            payload.append(
                (
                    symbol,
                    business_date,
                    _float_or_none(row.get("close_price")),
                    _float_or_none(row.get("retail_net_buy_value")),
                    _float_or_none(row.get("foreign_net_buy_value")),
                    _float_or_none(row.get("institution_net_buy_value")),
                    stamp,
                )
            )
        if not payload:
            return 0
        with closing(self._connect()) as conn:
            conn.executemany(
                """
                insert into investor_flow_daily (
                    symbol, business_date, close_price, retail_net_buy_value,
                    foreign_net_buy_value, institution_net_buy_value, updated_at
                ) values (?, ?, ?, ?, ?, ?, ?)
                on conflict(symbol, business_date) do update set
                    close_price = excluded.close_price,
                    retail_net_buy_value = excluded.retail_net_buy_value,
                    foreign_net_buy_value = excluded.foreign_net_buy_value,
                    institution_net_buy_value = excluded.institution_net_buy_value,
                    updated_at = excluded.updated_at
                """,
                payload,
            )
            conn.commit()
        return len(payload)

    def history(self, symbol: str) -> tuple[InvestorFlowDay, ...]:
        with closing(self._connect()) as conn:
            rows = conn.execute(
                """
                select symbol, business_date, close_price, retail_net_buy_value,
                       foreign_net_buy_value, institution_net_buy_value
                from investor_flow_daily where symbol = ?
                order by business_date
                """,
                (str(symbol or "").strip(),),
            ).fetchall()
        return tuple(_row_to_day(row) for row in rows)

    def load_all(self) -> dict[str, dict[str, InvestorFlowDay]]:
        """Every symbol's history keyed by ``business_date``, for labeling."""
        with closing(self._connect()) as conn:
            rows = conn.execute(
                """
                select symbol, business_date, close_price, retail_net_buy_value,
                       foreign_net_buy_value, institution_net_buy_value
                from investor_flow_daily order by symbol, business_date
                """
            ).fetchall()
        result: dict[str, dict[str, InvestorFlowDay]] = {}
        for row in rows:
            day = _row_to_day(row)
            result.setdefault(day.symbol, {})[day.business_date] = day
        return result

    def coverage(self) -> dict[str, Any]:
        with closing(self._connect()) as conn:
            row = conn.execute(
                "select count(*), count(distinct symbol),"
                " min(business_date), max(business_date) from investor_flow_daily"
            ).fetchone()
        return {
            "rows": int(row[0] or 0),
            "symbols": int(row[1] or 0),
            "first_business_date": row[2],
            "last_business_date": row[3],
        }


def business_date_for(moment: datetime | date) -> str:
    """KIS-style ``YYYYMMDD`` for a timestamp, in Korean market local time.

    Bars are stored in UTC; a KRX session that starts at 00:00 UTC belongs to the
    Korean calendar day nine hours ahead, so converting is not optional.
    """
    if isinstance(moment, datetime):
        try:
            from zoneinfo import ZoneInfo

            local = moment.astimezone(ZoneInfo("Asia/Seoul"))
        except Exception:  # noqa: BLE001 - fall back to the raw date, never crash
            local = moment
        return local.strftime("%Y%m%d")
    return moment.strftime("%Y%m%d")


def _row_to_day(row: Sequence[Any]) -> InvestorFlowDay:
    return InvestorFlowDay(
        symbol=str(row[0]),
        business_date=str(row[1]),
        close_price=float(row[2] or 0.0),
        retail_net_buy_value=float(row[3] or 0.0),
        foreign_net_buy_value=float(row[4] or 0.0),
        institution_net_buy_value=float(row[5] or 0.0),
    )


def _float_or_none(value: Any) -> float | None:
    try:
        if value is None:
            return None
        result = float(value)
    except (TypeError, ValueError):
        return None
    return result if result == result else None
