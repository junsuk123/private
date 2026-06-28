use numpy::{IntoPyArray, PyArray1, PyReadonlyArray1};
use pyo3::prelude::*;

const REASON_INSUFFICIENT_LIQUIDITY: u32 = 1 << 0;
const REASON_HALT_STATUS: u32 = 1 << 1;
const REASON_MANAGEMENT_STATUS: u32 = 1 << 2;
const REASON_INFORMED_IMBALANCE: u32 = 1 << 3;
const REASON_INFORMED_DISTRIBUTION: u32 = 1 << 4;
const REASON_JOINT_BUYING: u32 = 1 << 5;
const REASON_JOINT_SELLING: u32 = 1 << 6;
const REASON_RETAIL_ABSORBED: u32 = 1 << 7;
const REASON_RETAIL_DEMAND_SELLING: u32 = 1 << 8;
const REASON_CROWDED_FLOW: u32 = 1 << 9;
const REASON_PRICE_CONFIRMATION: u32 = 1 << 10;
const REASON_PRICE_DIVERGENCE: u32 = 1 << 11;
const REASON_HIGH_VALUE_VOLUME: u32 = 1 << 12;
const REASON_FOREIGN_INSTITUTION_MOMENTUM: u32 = 1 << 13;
const REASON_BREAKOUT_PRIORITY: u32 = 1 << 14;
const REASON_BASELINE_LIQUIDITY: u32 = 1 << 15;

#[pyfunction]
#[allow(clippy::too_many_arguments)]
fn score_topk<'py>(
    py: Python<'py>,
    trading_value: PyReadonlyArray1<'py, f32>,
    liquidity_score: PyReadonlyArray1<'py, f32>,
    volume_change_rate: PyReadonlyArray1<'py, f32>,
    price_change_rate: PyReadonlyArray1<'py, f32>,
    foreign_net_buy: PyReadonlyArray1<'py, f32>,
    institution_net_buy: PyReadonlyArray1<'py, f32>,
    retail_net_buy: PyReadonlyArray1<'py, f32>,
    program_net_buy: PyReadonlyArray1<'py, f32>,
    upper_limit_near: PyReadonlyArray1<'py, bool>,
    new_52week_high: PyReadonlyArray1<'py, bool>,
    halt_status: PyReadonlyArray1<'py, bool>,
    management_stock_status: PyReadonlyArray1<'py, bool>,
    domestic_market: PyReadonlyArray1<'py, bool>,
    min_trading_value: f32,
    min_liquidity_score: f32,
    top_k: usize,
) -> PyResult<(
    &'py PyArray1<i64>,
    &'py PyArray1<i64>,
    &'py PyArray1<i64>,
    &'py PyArray1<f32>,
    &'py PyArray1<u32>,
    &'py PyArray1<u32>,
)> {
    let tv = trading_value.as_slice()?;
    let liquidity = liquidity_score.as_slice()?;
    let volume_rate = volume_change_rate.as_slice()?;
    let price_rate = price_change_rate.as_slice()?;
    let foreign = foreign_net_buy.as_slice()?;
    let institution = institution_net_buy.as_slice()?;
    let retail = retail_net_buy.as_slice()?;
    let program = program_net_buy.as_slice()?;
    let upper = upper_limit_near.as_slice()?;
    let high52 = new_52week_high.as_slice()?;
    let halted = halt_status.as_slice()?;
    let management = management_stock_status.as_slice()?;
    let domestic = domestic_market.as_slice()?;
    let len = tv.len();
    ensure_same_len(
        len,
        &[
            liquidity.len(),
            volume_rate.len(),
            price_rate.len(),
            foreign.len(),
            institution.len(),
            retail.len(),
            program.len(),
            upper.len(),
            high52.len(),
            halted.len(),
            management.len(),
            domestic.len(),
        ],
    )?;

    let (selected, rejected, accepted, scores, masks, hard_masks) = py.allow_threads(|| {
        let mut scores = vec![0.0_f32; len];
        let mut masks = vec![0_u32; len];
        let mut hard_masks = vec![0_u32; len];
        let mut accepted: Vec<i64> = Vec::with_capacity(len);
        let mut rejected: Vec<i64> = Vec::new();

        for i in 0..len {
            let mut hard = 0_u32;
            if tv[i] < min_trading_value || liquidity[i] < min_liquidity_score {
                hard |= REASON_INSUFFICIENT_LIQUIDITY;
            }
            if halted[i] {
                hard |= REASON_HALT_STATUS;
            }
            if management[i] {
                hard |= REASON_MANAGEMENT_STATUS;
            }
            hard_masks[i] = hard;

            let mut score = liquidity[i].min(1.0).max(0.0) * 0.35
                + volume_rate[i].max(0.0) * 0.18
                + price_rate[i].max(0.0) * 3.0;
            let mut mask = 0_u32;
            let norm = tv[i].abs().max(1.0);
            let f = round6(foreign[i] / norm);
            let inst = round6(institution[i] / norm);
            let r = round6(retail[i] / norm);
            let p = round6(program[i] / norm);
            let total = round6(f + inst + r + p);
            let informed = round6(0.55 * f + 0.45 * inst - 0.20 * r + 0.15 * p);
            let absorption = round6(-(r * (0.55 * f + 0.45 * inst)));
            let signed_eff = round6(price_rate[i] * informed);
            let vol_pressure = round6(volume_rate[i].max(0.0));
            if domestic[i] {
                let flow_raw =
                    16.0 * informed + 10.0 * absorption + 80.0 * signed_eff + 0.10 * vol_pressure.min(2.5);
                score += round4(flow_raw.clamp(-1.4, 1.4)) * 0.25;
                if informed >= 0.018 {
                    mask |= REASON_INFORMED_IMBALANCE;
                } else if informed <= -0.018 {
                    mask |= REASON_INFORMED_DISTRIBUTION;
                }
                if f > 0.0 && inst > 0.0 && price_rate[i] > 0.0 {
                    mask |= REASON_JOINT_BUYING;
                }
                if f < 0.0 && inst < 0.0 {
                    mask |= REASON_JOINT_SELLING;
                }
                if absorption >= 0.002 && informed > 0.0 {
                    mask |= REASON_RETAIL_ABSORBED;
                } else if absorption >= 0.002 && informed < 0.0 {
                    mask |= REASON_RETAIL_DEMAND_SELLING;
                } else if absorption <= -0.002 {
                    mask |= REASON_CROWDED_FLOW;
                }
                if signed_eff >= 0.0007 && vol_pressure >= 1.0 {
                    mask |= REASON_PRICE_CONFIRMATION;
                } else if signed_eff <= -0.0007 {
                    mask |= REASON_PRICE_DIVERGENCE;
                }
                let _lambda_proxy = if total.abs() > 0.002 { price_rate[i] / total } else { 0.0 };
            }
            if hard == 0 {
                if tv[i] >= min_trading_value && volume_rate[i] > 0.25 {
                    mask |= REASON_HIGH_VALUE_VOLUME;
                    score += 0.18;
                }
                if foreign[i] > 0.0 && institution[i] > 0.0 && price_rate[i] > 0.0 {
                    mask |= REASON_FOREIGN_INSTITUTION_MOMENTUM;
                    score += 0.20;
                }
                if upper[i] || high52[i] {
                    mask |= REASON_BREAKOUT_PRIORITY;
                    score += 0.12;
                }
                if mask == 0 {
                    mask |= REASON_BASELINE_LIQUIDITY;
                }
                accepted.push(i as i64);
            } else {
                mask |= hard;
                rejected.push(i as i64);
            }
            scores[i] = score;
            masks[i] = mask;
        }

        let mut selected = accepted.clone();
        selected.sort_by(|a, b| {
            scores[*b as usize]
                .partial_cmp(&scores[*a as usize])
                .unwrap_or(std::cmp::Ordering::Equal)
                .then_with(|| b.cmp(a))
        });
        selected.truncate(top_k.min(selected.len()));
        (selected, rejected, accepted, scores, masks, hard_masks)
    });

    Ok((
        selected.into_pyarray(py),
        rejected.into_pyarray(py),
        accepted.into_pyarray(py),
        scores.into_pyarray(py),
        masks.into_pyarray(py),
        hard_masks.into_pyarray(py),
    ))
}

fn ensure_same_len(expected: usize, lengths: &[usize]) -> PyResult<()> {
    if lengths.iter().all(|length| *length == expected) {
        Ok(())
    } else {
        Err(pyo3::exceptions::PyValueError::new_err("all input arrays must have the same length"))
    }
}

fn round6(value: f32) -> f32 {
    (value * 1_000_000.0).round() / 1_000_000.0
}

fn round4(value: f32) -> f32 {
    (value * 10_000.0).round() / 10_000.0
}

#[pymodule]
fn screening_core(_py: Python<'_>, module: &PyModule) -> PyResult<()> {
    module.add_function(wrap_pyfunction!(score_topk, module)?)?;
    Ok(())
}
