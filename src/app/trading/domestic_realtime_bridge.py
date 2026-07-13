from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from app.trading.us_realtime_bridge import _kis_get


@dataclass(frozen=True)
class DomesticRankingSpec:
    path: str
    tr_id: str
    params: dict[str, str]
    symbol_keys: tuple[str, ...]


DOMESTIC_RANKING_SPECS: dict[str, DomesticRankingSpec] = {
    "volume_rank": DomesticRankingSpec(
        path="/uapi/domestic-stock/v1/quotations/volume-rank",
        tr_id="FHPST01710000",
        params={
            "FID_COND_MRKT_DIV_CODE": "J",
            "FID_COND_SCR_DIV_CODE": "20171",
            "FID_INPUT_ISCD": "0000",
            "FID_DIV_CLS_CODE": "0",
            "FID_BLNG_CLS_CODE": "0",
            "FID_TRGT_CLS_CODE": "111111111",
            "FID_TRGT_EXLS_CLS_CODE": "000000",
            "FID_INPUT_PRICE_1": "0",
            "FID_INPUT_PRICE_2": "0",
            "FID_VOL_CNT": "0",
            "FID_INPUT_DATE_1": "0",
        },
        symbol_keys=("mksc_shrn_iscd", "stck_shrn_iscd", "pdno"),
    ),
    "fluctuation": DomesticRankingSpec(
        path="/uapi/domestic-stock/v1/ranking/fluctuation",
        tr_id="FHPST01700000",
        params={
            "fid_cond_mrkt_div_code": "J",
            "fid_cond_scr_div_code": "20170",
            "fid_input_iscd": "0000",
            "fid_rank_sort_cls_code": "0",
            "fid_input_cnt_1": "0",
            "fid_prc_cls_code": "0",
            "fid_input_price_1": "",
            "fid_input_price_2": "",
            "fid_vol_cnt": "",
            "fid_trgt_cls_code": "0",
            "fid_trgt_exls_cls_code": "0",
            "fid_div_cls_code": "0",
            "fid_rsfl_rate1": "",
            "fid_rsfl_rate2": "",
        },
        symbol_keys=("stck_shrn_iscd", "mksc_shrn_iscd", "pdno"),
    ),
    "volume_power": DomesticRankingSpec(
        path="/uapi/domestic-stock/v1/ranking/volume-power",
        tr_id="FHPST01680000",
        params={
            "fid_cond_mrkt_div_code": "J",
            "fid_cond_scr_div_code": "20168",
            "fid_input_iscd": "0000",
            "fid_div_cls_code": "0",
            "fid_input_price_1": "",
            "fid_input_price_2": "",
            "fid_vol_cnt": "",
            "fid_trgt_exls_cls_code": "0",
            "fid_trgt_cls_code": "0",
        },
        symbol_keys=("stck_shrn_iscd", "mksc_shrn_iscd", "pdno"),
    ),
    "quote_balance": DomesticRankingSpec(
        path="/uapi/domestic-stock/v1/ranking/quote-balance",
        tr_id="FHPST01720000",
        params={
            "fid_cond_mrkt_div_code": "J",
            "fid_cond_scr_div_code": "20172",
            "fid_input_iscd": "0000",
            "fid_rank_sort_cls_code": "0",
            "fid_div_cls_code": "0",
            "fid_trgt_cls_code": "0",
            "fid_trgt_exls_cls_code": "0",
            "fid_input_price_1": "",
            "fid_input_price_2": "",
            "fid_vol_cnt": "",
        },
        symbol_keys=("mksc_shrn_iscd", "stck_shrn_iscd", "pdno"),
    ),
}


def fetch_domestic_ranking_symbols(
    *,
    sources: tuple[str, ...] = ("volume_rank", "fluctuation", "volume_power"),
    max_symbols: int = 30,
) -> dict[str, Any]:
    """Fetch KRX buy-discovery candidates from KIS domestic ranking APIs.

    KIS returns at most 30 rows per ranking endpoint. This combines the official
    volume, fluctuation, and volume-power rankings so the realtime engine has
    fresh domestic names beyond held symbols and the static collection list.
    """
    selected: list[str] = []
    errors: dict[str, str] = {}
    for source in sources:
        key = str(source or "").strip().lower()
        if not key:
            continue
        spec = DOMESTIC_RANKING_SPECS.get(key)
        if spec is None:
            errors[key] = "unknown domestic ranking source"
            continue
        try:
            data = _kis_get(spec.path, spec.tr_id, spec.params)
        except Exception as exc:  # noqa: BLE001 - discovery is best-effort.
            errors[key] = f"{exc.__class__.__name__}: {exc}"
            continue
        for row in _ranking_rows(data):
            symbol = _extract_domestic_symbol(row, spec.symbol_keys)
            if symbol:
                selected.append(symbol)
    unique = tuple(dict.fromkeys(selected))
    if max_symbols > 0:
        unique = unique[:max_symbols]
    return {"symbols": unique, "errors": errors, "ok": not errors}


def _ranking_rows(data: dict[str, Any]) -> tuple[dict[str, Any], ...]:
    rows: list[dict[str, Any]] = []
    for key in ("output", "output1", "output2"):
        value = data.get(key)
        if isinstance(value, list):
            rows.extend(row for row in value if isinstance(row, dict))
        elif isinstance(value, dict):
            rows.append(value)
    return tuple(rows)


def _extract_domestic_symbol(row: dict[str, Any], keys: tuple[str, ...]) -> str:
    for key in keys:
        raw = str(row.get(key) or "").upper().strip()
        if raw.isdigit() and len(raw) == 6:
            return raw
    return ""
