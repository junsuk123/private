from __future__ import annotations

import os
import time
from dataclasses import dataclass
from typing import Mapping

import numpy as np


REASON_INSUFFICIENT_LIQUIDITY = 1 << 0
REASON_HALT_STATUS = 1 << 1
REASON_MANAGEMENT_STATUS = 1 << 2
REASON_INFORMED_IMBALANCE = 1 << 3
REASON_INFORMED_DISTRIBUTION = 1 << 4
REASON_JOINT_BUYING = 1 << 5
REASON_JOINT_SELLING = 1 << 6
REASON_RETAIL_ABSORBED = 1 << 7
REASON_RETAIL_DEMAND_SELLING = 1 << 8
REASON_CROWDED_FLOW = 1 << 9
REASON_PRICE_CONFIRMATION = 1 << 10
REASON_PRICE_DIVERGENCE = 1 << 11
REASON_HIGH_VALUE_VOLUME = 1 << 12
REASON_FOREIGN_INSTITUTION_MOMENTUM = 1 << 13
REASON_BREAKOUT_PRIORITY = 1 << 14
REASON_BASELINE_LIQUIDITY = 1 << 15

REASON_NAMES: Mapping[int, str] = {
    REASON_INSUFFICIENT_LIQUIDITY: "insufficient_liquidity",
    REASON_HALT_STATUS: "halt_status",
    REASON_MANAGEMENT_STATUS: "management_stock_status",
    REASON_INFORMED_IMBALANCE: "informedorderflowimbalance",
    REASON_INFORMED_DISTRIBUTION: "informedorderflowdistribution",
    REASON_JOINT_BUYING: "foreigninstitutionjointbuying",
    REASON_JOINT_SELLING: "foreigninstitutionjointselling",
    REASON_RETAIL_ABSORBED: "retailsupplyabsorbedbyinformedflow",
    REASON_RETAIL_DEMAND_SELLING: "retaildemandmeetsinformedselling",
    REASON_CROWDED_FLOW: "crowdedsamedirectionflow",
    REASON_PRICE_CONFIRMATION: "orderflowpriceconfirmation",
    REASON_PRICE_DIVERGENCE: "orderflowpricedivergence",
    REASON_HIGH_VALUE_VOLUME: "high_trading_value_and_volume_surge",
    REASON_FOREIGN_INSTITUTION_MOMENTUM: "foreign_institution_buying_with_momentum",
    REASON_BREAKOUT_PRIORITY: "breakout_priority",
    REASON_BASELINE_LIQUIDITY: "baseline_liquidity_candidate",
}


@dataclass(frozen=True)
class VectorizedScreeningResult:
    selected_indices: np.ndarray
    rejected_indices: np.ndarray
    accepted_indices: np.ndarray
    scores: np.ndarray
    reason_masks: np.ndarray
    hard_reject_masks: np.ndarray
    profile: dict[str, float | int | str]


def screen_candidates_vectorized(
    *,
    trading_value: np.ndarray,
    liquidity_score: np.ndarray,
    volume_change_rate: np.ndarray,
    price_change_rate: np.ndarray,
    foreign_net_buy: np.ndarray,
    institution_net_buy: np.ndarray,
    retail_net_buy: np.ndarray,
    program_net_buy: np.ndarray,
    upper_limit_near: np.ndarray,
    new_52week_high: np.ndarray,
    halt_status: np.ndarray,
    management_stock_status: np.ndarray,
    domestic_market: np.ndarray,
    min_trading_value: float,
    min_liquidity_score: float,
    top_k: int,
) -> VectorizedScreeningResult:
    started_total = time.perf_counter()
    started_score = started_total
    trading_value = _f32(trading_value)
    liquidity_score = _f32(liquidity_score)
    volume_change_rate = _f32(volume_change_rate)
    price_change_rate = _f32(price_change_rate)
    foreign_net_buy = _f32(foreign_net_buy)
    institution_net_buy = _f32(institution_net_buy)
    retail_net_buy = _f32(retail_net_buy)
    program_net_buy = _f32(program_net_buy)
    upper_limit_near = np.asarray(upper_limit_near, dtype=bool)
    new_52week_high = np.asarray(new_52week_high, dtype=bool)
    halt_status = np.asarray(halt_status, dtype=bool)
    management_stock_status = np.asarray(management_stock_status, dtype=bool)
    domestic_market = np.asarray(domestic_market, dtype=bool)
    native = _try_native_screening(
        trading_value=trading_value,
        liquidity_score=liquidity_score,
        volume_change_rate=volume_change_rate,
        price_change_rate=price_change_rate,
        foreign_net_buy=foreign_net_buy,
        institution_net_buy=institution_net_buy,
        retail_net_buy=retail_net_buy,
        program_net_buy=program_net_buy,
        upper_limit_near=upper_limit_near,
        new_52week_high=new_52week_high,
        halt_status=halt_status,
        management_stock_status=management_stock_status,
        domestic_market=domestic_market,
        min_trading_value=min_trading_value,
        min_liquidity_score=min_liquidity_score,
        top_k=top_k,
        started_total=started_total,
    )
    if native is not None:
        return native

    count = int(trading_value.shape[0])
    hard_reject = np.zeros(count, dtype=np.uint32)
    hard_reject[trading_value < float(min_trading_value)] |= REASON_INSUFFICIENT_LIQUIDITY
    hard_reject[liquidity_score < float(min_liquidity_score)] |= REASON_INSUFFICIENT_LIQUIDITY
    hard_reject[halt_status] |= REASON_HALT_STATUS
    hard_reject[management_stock_status] |= REASON_MANAGEMENT_STATUS
    accepted_mask = hard_reject == 0

    reason_masks = np.zeros(count, dtype=np.uint32)
    scores = (
        np.minimum(1.0, liquidity_score) * np.float32(0.35)
        + np.maximum(0.0, volume_change_rate) * np.float32(0.18)
        + np.maximum(0.0, price_change_rate) * np.float32(3.0)
    ).astype(np.float32, copy=False)

    norm_value = np.maximum(np.float32(1.0), np.abs(trading_value))
    foreign = np.round(foreign_net_buy / norm_value, 6)
    institution = np.round(institution_net_buy / norm_value, 6)
    retail = np.round(retail_net_buy / norm_value, 6)
    program = np.round(program_net_buy / norm_value, 6)
    informed = np.round(0.55 * foreign + 0.45 * institution - 0.20 * retail + 0.15 * program, 6)
    absorption = np.round(-(retail * (0.55 * foreign + 0.45 * institution)), 6)
    signed_efficiency = np.round(price_change_rate * informed, 6)
    volume_pressure = np.round(np.maximum(0.0, volume_change_rate), 6)
    flow_raw = (
        16.0 * informed
        + 10.0 * absorption
        + 80.0 * signed_efficiency
        + 0.10 * np.minimum(2.5, volume_pressure)
    )
    flow_score = np.round(np.clip(flow_raw, -1.4, 1.4), 4).astype(np.float32)
    scores += np.where(domestic_market, flow_score * np.float32(0.25), np.float32(0.0))

    reason_masks[(domestic_market & (informed >= 0.018))] |= REASON_INFORMED_IMBALANCE
    reason_masks[(domestic_market & (informed <= -0.018))] |= REASON_INFORMED_DISTRIBUTION
    reason_masks[(domestic_market & (foreign > 0) & (institution > 0) & (price_change_rate > 0))] |= REASON_JOINT_BUYING
    reason_masks[(domestic_market & (foreign < 0) & (institution < 0))] |= REASON_JOINT_SELLING
    reason_masks[(domestic_market & (absorption >= 0.002) & (informed > 0))] |= REASON_RETAIL_ABSORBED
    reason_masks[(domestic_market & (absorption >= 0.002) & (informed < 0))] |= REASON_RETAIL_DEMAND_SELLING
    reason_masks[(domestic_market & (absorption <= -0.002))] |= REASON_CROWDED_FLOW
    reason_masks[(domestic_market & (signed_efficiency >= 0.0007) & (volume_pressure >= 1.0))] |= REASON_PRICE_CONFIRMATION
    reason_masks[(domestic_market & (signed_efficiency <= -0.0007))] |= REASON_PRICE_DIVERGENCE

    high_value_volume = accepted_mask & (trading_value >= float(min_trading_value)) & (volume_change_rate > 0.25)
    foreign_institution_momentum = accepted_mask & (foreign_net_buy > 0) & (institution_net_buy > 0) & (price_change_rate > 0)
    breakout = accepted_mask & (upper_limit_near | new_52week_high)
    reason_masks[high_value_volume] |= REASON_HIGH_VALUE_VOLUME
    reason_masks[foreign_institution_momentum] |= REASON_FOREIGN_INSTITUTION_MOMENTUM
    reason_masks[breakout] |= REASON_BREAKOUT_PRIORITY
    scores[high_value_volume] += np.float32(0.18)
    scores[foreign_institution_momentum] += np.float32(0.20)
    scores[breakout] += np.float32(0.12)
    reason_masks[accepted_mask & (reason_masks == 0)] |= REASON_BASELINE_LIQUIDITY
    reason_masks[~accepted_mask] |= hard_reject[~accepted_mask]

    score_ms = (time.perf_counter() - started_score) * 1000.0
    started_topk = time.perf_counter()
    accepted_indices = np.flatnonzero(accepted_mask).astype(np.int64, copy=False)
    rejected_indices = np.flatnonzero(~accepted_mask).astype(np.int64, copy=False)
    selected_indices = _select_top_indices(accepted_indices, scores, int(top_k))
    topk_ms = (time.perf_counter() - started_topk) * 1000.0

    return VectorizedScreeningResult(
        selected_indices=selected_indices,
        rejected_indices=rejected_indices,
        accepted_indices=accepted_indices,
        scores=scores,
        reason_masks=reason_masks,
        hard_reject_masks=hard_reject,
        profile={
            "backend": "python_numpy_vectorized",
            "candidate_screen_ms": round((time.perf_counter() - started_total) * 1000.0, 3),
            "candidate_score_ms": round(score_ms, 3),
            "candidate_topk_ms": round(topk_ms, 3),
            "input_count": count,
            "accepted_count": int(accepted_indices.shape[0]),
            "rejected_count": int(rejected_indices.shape[0]),
            "top_k": int(top_k),
        },
    )


def reason_mask_to_names(mask: int) -> tuple[str, ...]:
    return tuple(name for bit, name in REASON_NAMES.items() if int(mask) & bit)


def native_screening_available() -> bool:
    return _screening_core() is not None


def _try_native_screening(
    *,
    trading_value: np.ndarray,
    liquidity_score: np.ndarray,
    volume_change_rate: np.ndarray,
    price_change_rate: np.ndarray,
    foreign_net_buy: np.ndarray,
    institution_net_buy: np.ndarray,
    retail_net_buy: np.ndarray,
    program_net_buy: np.ndarray,
    upper_limit_near: np.ndarray,
    new_52week_high: np.ndarray,
    halt_status: np.ndarray,
    management_stock_status: np.ndarray,
    domestic_market: np.ndarray,
    min_trading_value: float,
    min_liquidity_score: float,
    top_k: int,
    started_total: float,
) -> VectorizedScreeningResult | None:
    if os.getenv("ONTOLOGY_FILTER1_NATIVE", "auto").strip().lower() in {"0", "false", "no", "off"}:
        return None
    core = _screening_core()
    if core is None:
        return None
    started_native = time.perf_counter()
    try:
        selected, rejected, accepted, scores, masks, hard_masks = core.score_topk(
            np.ascontiguousarray(trading_value, dtype=np.float32),
            np.ascontiguousarray(liquidity_score, dtype=np.float32),
            np.ascontiguousarray(volume_change_rate, dtype=np.float32),
            np.ascontiguousarray(price_change_rate, dtype=np.float32),
            np.ascontiguousarray(foreign_net_buy, dtype=np.float32),
            np.ascontiguousarray(institution_net_buy, dtype=np.float32),
            np.ascontiguousarray(retail_net_buy, dtype=np.float32),
            np.ascontiguousarray(program_net_buy, dtype=np.float32),
            np.ascontiguousarray(upper_limit_near, dtype=bool),
            np.ascontiguousarray(new_52week_high, dtype=bool),
            np.ascontiguousarray(halt_status, dtype=bool),
            np.ascontiguousarray(management_stock_status, dtype=bool),
            np.ascontiguousarray(domestic_market, dtype=bool),
            float(min_trading_value),
            float(min_liquidity_score),
            int(top_k),
        )
    except Exception:
        if os.getenv("ONTOLOGY_FILTER1_NATIVE", "auto").strip().lower() in {"1", "true", "yes", "on", "required"}:
            raise
        return None
    native_ms = (time.perf_counter() - started_native) * 1000.0
    return VectorizedScreeningResult(
        selected_indices=np.asarray(selected, dtype=np.int64),
        rejected_indices=np.asarray(rejected, dtype=np.int64),
        accepted_indices=np.asarray(accepted, dtype=np.int64),
        scores=np.asarray(scores, dtype=np.float32),
        reason_masks=np.asarray(masks, dtype=np.uint32),
        hard_reject_masks=np.asarray(hard_masks, dtype=np.uint32),
        profile={
            "backend": "rust_pyo3_native",
            "candidate_screen_ms": round((time.perf_counter() - started_total) * 1000.0, 3),
            "candidate_score_ms": round(native_ms, 3),
            "candidate_topk_ms": 0.0,
            "input_count": int(np.asarray(scores).shape[0]),
            "accepted_count": int(np.asarray(accepted).shape[0]),
            "rejected_count": int(np.asarray(rejected).shape[0]),
            "top_k": int(top_k),
        },
    )


def _screening_core():
    try:
        import screening_core
    except Exception:
        return None
    return screening_core


def _select_top_indices(accepted_indices: np.ndarray, scores: np.ndarray, top_k: int) -> np.ndarray:
    if accepted_indices.size == 0 or top_k <= 0:
        return np.zeros(0, dtype=np.int64)
    count = min(top_k, int(accepted_indices.size))
    candidate_scores = scores[accepted_indices]
    if count < accepted_indices.size:
        local = np.argpartition(-candidate_scores, count - 1)[:count]
    else:
        local = np.arange(accepted_indices.size, dtype=np.int64)
    selected = accepted_indices[local]
    # Preserve legacy deterministic tie behavior: sorted((score, ticker), reverse=True).
    order = np.lexsort((-selected, -scores[selected]))
    return selected[order].astype(np.int64, copy=False)


def _f32(values: np.ndarray) -> np.ndarray:
    return np.nan_to_num(np.asarray(values, dtype=np.float32), nan=0.0, posinf=0.0, neginf=0.0)
