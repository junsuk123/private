"""One auditable market-derived policy for entry risk and position exits.

OWL assertions are projected/validated upstream. This pure numeric step never
queries a broker or trains a model. Bounds are engineering limits; operating
thresholds depend on measured noise, costs, liquidity, regime and uncertainty.
Neither a neural score nor an optimistic forecast can enlarge the safety limits.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, replace
from datetime import datetime, timedelta, timezone
from functools import lru_cache
import hashlib
import json
import math
import os
from pathlib import Path
from typing import Any, Mapping

POLICY_FAMILY_VERSION = "ontology-risk-v1"


@dataclass(frozen=True)
class PolicyLimits:
    maximum_daily_loss_rate: float = .01
    maximum_trade_loss_rate: float = .002
    maximum_position_weight_kr: float = .15
    maximum_position_weight_us: float = .12
    maximum_sector_weight: float = .30
    maximum_hard_stop_rate: float = .02
    maximum_emergency_stop_rate: float = .035
    maximum_holding_seconds: float = 7200
    maximum_quote_age_seconds: float = 20
    maximum_policy_age_seconds: float = 15
    maximum_trades_per_day: float = 24


@lru_cache(maxsize=1)
def policy_limits() -> PolicyLimits:
    import yaml
    path = Path(__file__).resolve().parents[3] / "config/ontology_risk_policy.yaml"
    raw = yaml.safe_load(path.read_text(encoding="utf-8")) if path.exists() else {}
    values = {name: float(raw.get(name, default)) for name, default in asdict(PolicyLimits()).items()}
    if any(not math.isfinite(value) or value <= 0 for value in values.values()):
        raise ValueError("ONTOLOGY_POLICY_LIMITS_INVALID")
    if any(value > 1 for name, value in values.items() if name.endswith(("rate", "weight", "weight_kr", "weight_us"))):
        raise ValueError("ONTOLOGY_POLICY_RATE_LIMIT_INVALID")
    return PolicyLimits(**values)


def _clip(value: float, low: float = 0., high: float = 1.) -> float:
    return min(high, max(low, float(value)))


def _number(values: Mapping[str, Any], name: str, default: float) -> float:
    try:
        value = float(values[name])
        return value if math.isfinite(value) else default
    except (KeyError, TypeError, ValueError):
        return default


def _ceiling(name: str, maximum: float) -> float:
    """Old launcher settings may tighten a ceiling, never pin a market trigger."""
    try:
        value = float(os.getenv(name, maximum))
        return min(maximum, value) if math.isfinite(value) and value > 0 else maximum
    except (TypeError, ValueError):
        return maximum


@dataclass(frozen=True)
class OntologyRiskPolicy:
    policy_id: str
    evidence_id: str
    market: str
    symbol: str
    as_of: datetime
    expires_at: datetime
    regime: str
    valid_for_entry: bool
    reason_codes: tuple[str, ...]
    stress: float
    confidence: float
    all_in_cost_rate: float
    noise_band_rate: float
    net_profit_floor_rate: float
    target_return_rate: float
    soft_stop_rate: float
    hard_stop_rate: float
    emergency_stop_rate: float
    trailing_stop_rate: float
    trailing_giveback: float
    profit_lock_arm_net: float
    minimum_holding_seconds: float
    maximum_holding_seconds: int
    early_exit_confirmations: int
    change_point_exit_probability: float
    ontology_sell_threshold: float
    negative_forecast_bps: float
    position_cap: float
    sector_cap: float
    trade_loss_budget_rate: float
    daily_loss_budget_rate: float
    minimum_cash_reserve: float
    max_trades_per_day: int
    max_spread_rate: float
    max_quote_age_seconds: float
    entry_band_rate: float
    minimum_reward_risk: float
    evidence: tuple[Mapping[str, Any], ...]

    def as_dict(self) -> dict[str, Any]:
        result = asdict(self)
        result["policy_family"] = POLICY_FAMILY_VERSION
        result["as_of"], result["expires_at"] = self.as_of.isoformat(), self.expires_at.isoformat()
        return result

    def is_current(self, now: datetime) -> bool:
        now = now if now.tzinfo else now.replace(tzinfo=timezone.utc)
        return self.as_of <= now <= self.expires_at

    def has_current_market_evidence(self, now: datetime) -> bool:
        """Entry unattractiveness is still actionable evidence for owned risk.

        A wide spread, adverse reward/risk or a no-entry regime must not make
        the exit controller ignore a newly observed deterioration. Missing,
        stale or invalid evidence remains unusable for discretionary changes.
        """
        entry_only = {
            "POLICY_NET_REWARD_INSUFFICIENT", "POLICY_NOISE_EXCEEDS_LOSS_BUDGET",
            "POLICY_SPREAD_TOO_WIDE", "POLICY_REGIME_NO_ENTRY",
        }
        return self.is_current(now) and (
            self.valid_for_entry or bool(self.reason_codes)
            and not (set(self.reason_codes) - entry_only)
        )

    def tighten_for_position(self, previous: "OntologyRiskPolicy | None") -> "OntologyRiskPolicy":
        """Once risk is owned, new volatility cannot move the loss barrier away."""
        if previous is None or previous.symbol != self.symbol or previous.market != self.market:
            return self
        return replace(
            self, soft_stop_rate=min(self.soft_stop_rate, previous.soft_stop_rate),
            hard_stop_rate=min(self.hard_stop_rate, previous.hard_stop_rate),
            emergency_stop_rate=min(self.emergency_stop_rate, previous.emergency_stop_rate),
            trailing_stop_rate=min(self.trailing_stop_rate, previous.trailing_stop_rate),
            maximum_holding_seconds=min(self.maximum_holding_seconds, previous.maximum_holding_seconds),
        )


def resolve_ontology_policy(
    projection: Any, *, symbol: str, all_in_cost_rate: float,
    forecast_gross_bps: float | None = None, requested_horizon_seconds: float = 900,
    limits: PolicyLimits | None = None,
) -> OntologyRiskPolicy:
    """Resolve rates in quote-currency returns, never KRW/USD absolute P&L.

    Volatility carries its measured horizon. Unavailable measurements produce
    no new-entry permission; the conservative exit fallback remains available.
    Formula coefficients are declared research priors, not fitted profitability.
    """
    limits = limits or policy_limits()
    now = projection.as_of
    now = now if now.tzinfo else now.replace(tzinfo=timezone.utc)
    values = projection.values
    market = str(projection.market)
    regime = str(projection.regime or "UNKNOWN").upper()
    reasons = list(projection.reason_codes) if not projection.usable else []
    required = ("realized_volatility", "spread_rate", "liquidity_score", "quote_age_seconds")
    missing = [key for key in required if key not in values]
    reasons.extend(f"POLICY_METRIC_MISSING:{key}" for key in missing)
    observations = tuple(projection.observations)
    vol_observation = next((item for item in observations if item.metric == "realized_volatility"), None)
    measured_horizon = getattr(vol_observation, "horizon_seconds", None)
    if measured_horizon is None or not math.isfinite(float(measured_horizon)) or measured_horizon <= 0:
        reasons.append("POLICY_VOLATILITY_HORIZON_UNKNOWN")
    try:
        cost = float(all_in_cost_rate)
        if not math.isfinite(cost) or cost < 0:
            raise ValueError
    except (TypeError, ValueError):
        cost = 0.0
        reasons.append("POLICY_COST_UNKNOWN")
    forecast = None
    if forecast_gross_bps is not None:
        try:
            forecast = float(forecast_gross_bps) / 10000.
            if not math.isfinite(forecast):
                raise ValueError
        except (TypeError, ValueError):
            forecast = None
            reasons.append("POLICY_FORECAST_INVALID")
    confidence = _clip(_number(values, "regime_confidence", 0.0))
    quality = _clip(_number(values, "data_quality_score", confidence))
    uncertainty = max(1.0 - confidence, _clip(_number(values, "model_uncertainty", 1.0 - confidence)))
    liquidity = _clip(_number(values, "liquidity_score", 0.0))
    spread = max(0., _number(values, "spread_rate", 0.0))
    breadth = _clip(_number(values, "market_breadth", 0.0), -1., 1.)
    trend = _clip(_number(values, "trend_strength", 0.0), -1., 1.)
    drawdown = max(0., _number(values, "drawdown_rate", 0.0))
    change = _clip(_number(values, "change_point_probability", 0.0))
    regime_stress = .85 if any(word in regime for word in ("STRESS", "RISK_OFF", "DISLOCATED", "SHOCK")) else .45 if "HIGH_VOL" in regime or "DOWN" in regime else .15
    stress = _clip(.25 * regime_stress + .20 * (1 - liquidity) + .15 * uncertainty + .15 * max(0., -breadth) + .10 * max(0., -trend) + .15 * change + min(.4, drawdown * 8))
    # Only provenance-validated cross-market links reach these global metrics.
    global_pressure = max(0., -_number(values, "global_risk_sentiment", 0.), -_number(values, "global_direction", 0.))
    model_downside = max(0., _number(values, "expected_downside_net_bps", 0.)) / 10000.
    stress = _clip(stress + .10 * min(1., global_pressure) + min(.15, model_downside * 10.))
    if not projection.usable or missing or regime == "UNKNOWN":
        stress = max(.85, stress)
    if regime == "UNKNOWN" or confidence <= 0:
        reasons.append("POLICY_MARKET_CONTEXT_UNKNOWN")
    try:
        requested_horizon = float(requested_horizon_seconds)
        if not math.isfinite(requested_horizon) or requested_horizon <= 0:
            raise ValueError
    except (TypeError, ValueError, OverflowError):
        requested_horizon = 60.
        reasons.append("POLICY_HOLDING_HORIZON_INVALID")
    horizon = _clip(requested_horizon, 60., limits.maximum_holding_seconds)
    effective_horizon = max(60., horizon * (.55 + .45 * max(0., trend)) * (1 - .5 * stress))
    sigma = max(0., _number(values, "realized_volatility", 0.0))
    # Normalize by an observed, explicit interval; never treat a per-tick sigma
    # as a per-minute sigma. Missing horizon only feeds the exit fallback.
    sigma_horizon = sigma * math.sqrt(effective_horizon / max(1., float(measured_horizon or effective_horizon)))
    sigma_horizon = min(.10, sigma_horizon)
    noise = max(spread * (1 + .5 * (1 - liquidity)), sigma_horizon * (.30 + .25 * uncertainty), cost * .15)
    emergency_ceiling = _ceiling("REALTIME_EMERGENCY_STOP_LOSS", limits.maximum_emergency_stop_rate)
    hard_ceiling = min(emergency_ceiling, _ceiling("REALTIME_HARD_STOP_LOSS", limits.maximum_hard_stop_rate))
    soft = min(hard_ceiling * .8, max(noise * 1.5, sigma_horizon * (1.8 - .8 * stress), cost * .4, 1e-5))
    hard = min(hard_ceiling, max(soft, soft * (1.35 - .20 * stress)))
    emergency = min(emergency_ceiling, max(hard, hard * (1.3 - .15 * stress)))
    if missing or "POLICY_VOLATILITY_HORIZON_UNKNOWN" in reasons:
        # No missing-data-induced near-zero automatic sell. Existing position
        # barriers remain tighter via tighten_for_position; the safety ceiling
        # remains usable if a position was recovered after a restart.
        soft, hard, emergency = hard_ceiling * .8, hard_ceiling, emergency_ceiling
    net_floor = max(cost * (.35 + .5 * uncertainty), noise * (.7 + .3 * stress), 1e-6)
    minimum_rr = 1.0 + .6 * stress + .3 * uncertainty
    target = cost + max(net_floor, soft * minimum_rr, sigma_horizon * (1.2 + .4 * max(0., trend)))
    if not missing and noise >= hard:
        reasons.append("POLICY_NOISE_EXCEEDS_LOSS_BUDGET")
    if forecast is not None:
        if forecast < cost + max(net_floor, soft * minimum_rr):
            reasons.append("POLICY_NET_REWARD_INSUFFICIENT")
        target = min(target, max(cost + net_floor, forecast))
    budget_factor = max(.1, (1 - stress) * (.3 + .7 * confidence) * (.4 + .6 * quality))
    market_cap = limits.maximum_position_weight_kr if market == "KR" else limits.maximum_position_weight_us
    market_cap = _ceiling("REALTIME_SMALL_ACCOUNT_MAX_POSITION_WEIGHT", market_cap)
    per_trade = limits.maximum_trade_loss_rate * budget_factor
    position_cap = min(market_cap * budget_factor, per_trade / max(hard + cost, 1e-6))
    daily = _ceiling("REALTIME_DAILY_REALIZED_LOSS_BUY_STOP_RATE", limits.maximum_daily_loss_rate) * budget_factor
    quote_age = max(1., limits.maximum_quote_age_seconds * (.35 + .65 * liquidity) * (1 - .4 * stress))
    if _number(values, "quote_age_seconds", float("inf")) > quote_age:
        reasons.append("POLICY_QUOTE_STALE")
    max_spread = max(cost * .75, sigma_horizon * (.4 + .3 * liquidity), 1e-6) * (1 - .35 * stress)
    if spread > max_spread:
        reasons.append("POLICY_SPREAD_TOO_WIDE")
    if market not in {"KR", "US"}:
        reasons.append("POLICY_MARKET_UNSUPPORTED")
    if any(word in regime for word in ("DISLOCATED", "HALTED", "NO_TRADE")):
        reasons.append("POLICY_REGIME_NO_ENTRY")
    valid = bool(projection.usable) and not reasons
    evidence = tuple({"metric": item.metric, "value": item.value, "observed_at": item.observed_at.isoformat(), "source": item.source, "horizon_seconds": item.horizon_seconds, "derived_from": list(item.derived_from)} for item in observations)
    fingerprint = hashlib.sha256(json.dumps({"market": market, "symbol": symbol, "context": projection.context_id, "at": now.isoformat(), "evidence": evidence, "cost": cost, "forecast": forecast, "family": POLICY_FAMILY_VERSION, "regime": regime, "requested_horizon": horizon, "limits": asdict(limits), "operating": [soft, hard, emergency, target, position_cap, daily, quote_age]}, sort_keys=True, allow_nan=False).encode()).hexdigest()[:20]
    policy = OntologyRiskPolicy(
        policy_id=f"orp-{fingerprint}", evidence_id=str(projection.context_id), market=market, symbol=symbol,
        as_of=now, expires_at=min(
            [now + timedelta(seconds=max(0., min(quote_age - _number(values, "quote_age_seconds", quote_age), limits.maximum_policy_age_seconds)))]
            + [item.observed_at + timedelta(seconds=item.max_age_seconds) for item in observations if item.metric in (
                *required, "regime_confidence", "data_quality_score", "model_uncertainty",
                "market_breadth", "trend_strength", "drawdown_rate", "change_point_probability",
                "global_risk_sentiment", "global_direction", "expected_downside_net_bps",
            )]
        ),
        regime=regime, valid_for_entry=valid, reason_codes=tuple(dict.fromkeys(reasons)), stress=stress, confidence=confidence,
        all_in_cost_rate=cost, noise_band_rate=noise, net_profit_floor_rate=net_floor, target_return_rate=target,
        soft_stop_rate=soft, hard_stop_rate=hard, emergency_stop_rate=emergency,
        trailing_stop_rate=max(noise, soft * (.5 + .25 * (1 - stress))),
        trailing_giveback=_clip(.20 + .40 * (1 - stress) * (.5 + .5 * liquidity), .1, .8),
        profit_lock_arm_net=net_floor + noise * (1 - .5 * stress),
        minimum_holding_seconds=min(effective_horizon * .4, 30 + 90 * (1 - stress) * uncertainty),
        maximum_holding_seconds=int(effective_horizon), early_exit_confirmations=max(1, int(round(1 + 3 * (1 - stress) * uncertainty))),
        change_point_exit_probability=_clip(.75 - .3 * stress + .1 * uncertainty, .35, .95),
        ontology_sell_threshold=-(.35 + .30 * (1 - stress)), negative_forecast_bps=max(noise, net_floor * (.5 + uncertainty)) * 10000,
        position_cap=position_cap if valid else 0., sector_cap=min(limits.maximum_sector_weight, market_cap * (1.2 + .8 * (1 - stress))),
        trade_loss_budget_rate=per_trade, daily_loss_budget_rate=daily, minimum_cash_reserve=_clip(.15 + .55 * stress + .10 * uncertainty, .1, .8),
        max_trades_per_day=max(1, int(limits.maximum_trades_per_day * budget_factor / (1 + cost / max(sigma_horizon, 1e-6)))),
        max_spread_rate=max_spread, max_quote_age_seconds=quote_age, entry_band_rate=min(max_spread, max(spread, noise * .5)),
        minimum_reward_risk=minimum_rr, evidence=evidence,
    )
    graph_metrics = {"model_uncertainty", "expected_downside_net_bps"}
    market_observations = tuple(item for item in observations
                                if not (item.source.split(":", 1)[0] == "validated_temporal_rgcn" and item.metric in graph_metrics))
    if len(market_observations) == len(observations):
        return policy
    # Forward shadow labels validate entry-frozen barriers, not subsequent live
    # tightening. Such a model may reduce risk, but must never widen a loss
    # barrier or grant permission that the observed market alone did not grant.
    from types import SimpleNamespace
    baseline = resolve_ontology_policy(
        SimpleNamespace(as_of=now, market=market, regime=regime, usable=projection.usable,
                        reason_codes=projection.reason_codes, context_id=projection.context_id,
                        observations=market_observations,
                        values={item.metric: item.value for item in market_observations}),
        symbol=symbol, all_in_cost_rate=all_in_cost_rate, forecast_gross_bps=forecast_gross_bps,
        requested_horizon_seconds=requested_horizon_seconds, limits=limits,
    )
    bounded = {name: min(getattr(policy, name), getattr(baseline, name)) for name in (
        "soft_stop_rate", "hard_stop_rate", "emergency_stop_rate", "trailing_stop_rate",
        "trailing_giveback", "profit_lock_arm_net", "minimum_holding_seconds",
        "maximum_holding_seconds", "early_exit_confirmations", "change_point_exit_probability",
        "negative_forecast_bps", "position_cap", "sector_cap", "trade_loss_budget_rate",
        "daily_loss_budget_rate", "max_trades_per_day", "max_spread_rate",
        "max_quote_age_seconds", "entry_band_rate",
    )}
    bounded.update({name: max(getattr(policy, name), getattr(baseline, name)) for name in (
        "net_profit_floor_rate", "minimum_reward_risk", "minimum_cash_reserve", "ontology_sell_threshold",
    )})
    bounded["target_return_rate"] = max(policy.target_return_rate,
        cost + max(bounded["net_profit_floor_rate"], bounded["soft_stop_rate"] * bounded["minimum_reward_risk"]))
    bounded_reasons = tuple(dict.fromkeys((*policy.reason_codes, *baseline.reason_codes)))
    if policy.noise_band_rate >= bounded["hard_stop_rate"]:
        bounded_reasons = tuple(dict.fromkeys((*bounded_reasons, "POLICY_NOISE_EXCEEDS_LOSS_BUDGET")))
        bounded["position_cap"] = 0.
    # Distinct identities also account for the bound; audit readers must not see
    # the same policy id attached to two different resolved geometries.
    fingerprint = hashlib.sha256(json.dumps(
        {"candidate": policy.policy_id, "baseline": baseline.policy_id, "bounds": bounded},
        sort_keys=True, allow_nan=False,
    ).encode()).hexdigest()[:20]
    return replace(policy, **bounded, policy_id=f"orp-{fingerprint}",
                   valid_for_entry=policy.valid_for_entry and baseline.valid_for_entry and not bounded_reasons,
                   reason_codes=bounded_reasons,
                   expires_at=min(policy.expires_at, baseline.expires_at))
