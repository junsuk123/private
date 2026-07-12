"""Unified, cost-aware net-profitability decision gate.

This is the single authoritative profitability decision surface for the whole
system. Strategy candidate generation, the RiskManager BUY path, the realtime
trading engine, and the GUI all consume the same :class:`ProfitabilityDecision`
so a BUY is judged by exactly one net-edge rule everywhere.

Core rule (net-profitability first — directional signal strength alone is NOT
sufficient):

    allow_buy = (
        expected_exit_price >= break_even_exit_price
        and net_expected_return >= required_min_net_return
        and spread_rate <= max_allowed_spread
        and spread_alpha_ratio <= max_spread_alpha_ratio
        and liquidity_score >= min_liquidity_score
        and cost_to_alpha_ratio <= max_cost_to_alpha_ratio
    )

`required_min_net_return` is dynamic — it rises with volatility, thin liquidity,
and small-account caution, so a marginal edge that clears cost in calm, liquid
conditions is correctly rejected in noisy/thin ones.

All cost math is delegated to :class:`app.cost.trading_cost_engine.TradingCostEngine`
so there is exactly one cost model in the codebase. This module adds the
policy-driven *decision* layer on top of it.

Configuration precedence (highest wins), all resolved values logged once:
    1. Environment variables (backward compatibility with the existing ~70 vars)
    2. ``config/profitability_policy.yaml``
    3. Built-in defaults below
"""

from __future__ import annotations

import logging
import os
from dataclasses import asdict, dataclass, field, replace
from pathlib import Path
from typing import Any

from app.cost.trading_cost_engine import CostBreakdown, TradingCostEngine

logger = logging.getLogger(__name__)

_EPSILON = 1e-9

# Rejection reason codes. Kept identical to the strings the RiskManager / candidate
# factory / GUI already emit so existing reason-text mappings keep working.
REASON_MISSING_EXIT = "MISSING_EXPECTED_EXIT_PRICE"
REASON_UNREALISTIC_EXIT = "UNREALISTIC_EXPECTED_EXIT_PRICE"
REASON_BELOW_BREAK_EVEN = "BELOW_BREAK_EVEN_WITH_MARGIN"
REASON_BELOW_MIN_NET = "BELOW_TARGET_NET_RETURN_AFTER_COST"
REASON_COST_BURDEN = "COST_BURDEN_HIGH"
REASON_SPREAD = "SPREAD_TOO_WIDE"
REASON_SPREAD_ALPHA = "SPREAD_CONSUMES_ALPHA"
REASON_LIQUIDITY = "LIQUIDITY_TOO_LOW"
REASON_SLIPPAGE = "SLIPPAGE_RISK_HIGH"
REASON_INVALID = "INVALID_ORDER_SIZE_OR_PRICE"


DEFAULT_PROFITABILITY_POLICY: dict[str, Any] = {
    # Minimum acceptable net-expected-return after ALL costs, by market.
    "min_required_net_return": {"default": 0.008, "KR": 0.008, "US": 0.012},
    # Extra net headroom demanded on top of pure break-even before a buy is worth it.
    "min_net_profit_buffer_rate": 0.001,
    # Spread / liquidity / cost ceilings.
    "max_spread_rate": 0.003,          # (ask-bid)/mid
    "max_slippage_rate": 0.003,        # expected entry slippage as a fraction of notional
    "max_spread_alpha_ratio": 0.35,    # spread may not eat > this fraction of gross alpha
    "max_cost_to_alpha_ratio": 0.5,
    "min_liquidity_score": 0.3,
    # Dynamic required-net-return buffers.
    "volatility_buffer_k": 0.5,        # required += k * realized_volatility_horizon
    "liquidity_buffer_max": 0.003,     # required += up to this as liquidity -> 0
    "account_buffer": {
        "small_account_equity_krw": 200000.0,
        "small_account_extra_net": 0.002,
    },
}


@dataclass(frozen=True)
class ProfitabilityPolicy:
    """Resolved (env > yaml > default) policy values, logged once at load."""

    min_required_net_return: dict[str, float]
    min_net_profit_buffer_rate: float
    max_spread_rate: float
    max_slippage_rate: float
    max_spread_alpha_ratio: float
    max_cost_to_alpha_ratio: float
    min_liquidity_score: float
    volatility_buffer_k: float
    liquidity_buffer_max: float
    small_account_equity_krw: float
    small_account_extra_net: float

    def min_net_for_market(self, market: str) -> float:
        key = _market_key(market)
        table = self.min_required_net_return
        return float(table.get(key, table.get("default", 0.008)))

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class ProfitabilityInput:
    """Everything the gate needs to judge one prospective order."""

    symbol: str
    action: str = "BUY"
    market: str = "KR"
    venue: str = "KRX"
    instrument_type: str = "domestic_stock"
    entry_price: float = 0.0
    expected_exit_price: float | None = None
    quantity: int = 1
    # Market microstructure (any that are known; the gate fills the rest from costs).
    spread_rate: float | None = None
    liquidity_score: float = 1.0
    realized_volatility: float = 0.0
    orderbook_snapshot: dict[str, Any] | None = None
    average_daily_trading_value: float | None = None
    # Account context for the small-account buffer.
    account_equity_krw: float = 0.0
    # Explicit override for the minimum net return (e.g. per-theory target).
    target_net_return: float | None = None


@dataclass(frozen=True)
class ProfitabilityBreakdown:
    """The cost/return math behind a decision (mirror of CostBreakdown + buffers)."""

    entry_price: float
    expected_exit_price: float
    break_even_exit_price: float
    gross_expected_return: float
    all_in_cost_rate: float
    net_expected_return: float
    cost_to_alpha_ratio: float
    spread_rate: float
    expected_slippage_rate: float
    market_impact_rate: float
    safety_margin_rate: float
    # Dynamic required-net-return decomposition.
    base_min_net_return: float
    volatility_buffer: float
    liquidity_buffer: float
    account_buffer: float

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class ProfitabilityDecision:
    """The one authoritative profitability decision object used everywhere."""

    allowed: bool
    action: str
    symbol: str
    entry_price: float
    expected_exit_price: float
    break_even_exit_price: float
    gross_expected_return: float
    all_in_cost_rate: float
    net_expected_return: float
    required_min_net_return: float
    spread_rate: float
    expected_slippage_rate: float
    market_impact_rate: float
    liquidity_score: float
    cost_to_alpha_ratio: float
    rejection_reasons: tuple[str, ...] = ()
    warnings: tuple[str, ...] = ()
    data_quality_flags: tuple[str, ...] = ()
    breakdown: ProfitabilityBreakdown | None = None
    policy_version: str = ""

    def as_dict(self) -> dict[str, Any]:
        payload = {
            "policy_version": self.policy_version,
            "allowed": self.allowed,
            "action": self.action,
            "symbol": self.symbol,
            "entry_price": self.entry_price,
            "expected_exit_price": self.expected_exit_price,
            "break_even_exit_price": self.break_even_exit_price,
            "gross_expected_return": self.gross_expected_return,
            "all_in_cost_rate": self.all_in_cost_rate,
            "net_expected_return": self.net_expected_return,
            "required_min_net_return": self.required_min_net_return,
            "spread_rate": self.spread_rate,
            "expected_slippage_rate": self.expected_slippage_rate,
            "market_impact_rate": self.market_impact_rate,
            "liquidity_score": self.liquidity_score,
            "cost_to_alpha_ratio": self.cost_to_alpha_ratio,
            "rejection_reasons": list(self.rejection_reasons),
            "warnings": list(self.warnings),
            "data_quality_flags": list(self.data_quality_flags),
        }
        if self.breakdown is not None:
            payload["breakdown"] = self.breakdown.as_dict()
        return payload


class ProfitabilityGate:
    """Authoritative net-profitability gate. Thread-safe for read; policy loaded once."""

    def __init__(
        self,
        *,
        cost_engine: TradingCostEngine | None = None,
        policy: ProfitabilityPolicy | None = None,
        config_path: Path | str = "config/profitability_policy.yaml",
    ) -> None:
        self.cost_engine = cost_engine or TradingCostEngine()
        self.policy = policy or load_policy(config_path)
        try:  # stamp every decision with the shared, versioned policy view
            from app.trading.trading_policy import TradingPolicySnapshot

            self.policy_version = TradingPolicySnapshot.from_environment().policy_version
        except Exception:  # noqa: BLE001 - versioning must never block a decision
            self.policy_version = ""

    def evaluate(self, request: ProfitabilityInput) -> ProfitabilityDecision:
        decision = self._evaluate(request)
        if self.policy_version and not decision.policy_version:
            return replace(decision, policy_version=self.policy_version)
        return decision

    def _evaluate(self, request: ProfitabilityInput) -> ProfitabilityDecision:
        symbol = request.symbol
        action = (request.action or "BUY").upper()
        entry_price = max(0.0, float(request.entry_price or 0.0))
        quantity = max(0, int(request.quantity or 0))
        expected_exit_price = request.expected_exit_price

        # --- Non-BUY actions are not blocked by the profitability gate ----------
        # Exits (SELL/REDUCE) are governed by DynamicExitPolicy and must never be
        # trapped by a buy-side profitability rule. We still return a populated
        # breakdown for observability, but always allowed=True.
        if action != "BUY":
            return self._informational_decision(request, entry_price, quantity)

        reasons: list[str] = []
        warnings: list[str] = []
        flags: list[str] = []

        # --- Basic validity ----------------------------------------------------
        if entry_price <= 0 or quantity <= 0:
            return self._reject(request, entry_price, 0.0, [REASON_INVALID])
        if expected_exit_price is None or float(expected_exit_price) <= 0:
            return self._reject(request, entry_price, float(expected_exit_price or 0.0), [REASON_MISSING_EXIT])
        expected_exit_price = float(expected_exit_price)

        # --- Cost math (single source of truth) --------------------------------
        cost: CostBreakdown = self.cost_engine.estimate(
            symbol=symbol,
            market=request.market,
            venue=request.venue,
            instrument_type=request.instrument_type,
            entry_price=entry_price,
            expected_exit_price=expected_exit_price,
            quantity=quantity,
            target_net_return=0.0,  # gate computes its own dynamic requirement below
            orderbook_snapshot=request.orderbook_snapshot,
            average_daily_trading_value=request.average_daily_trading_value,
        )
        policy_fees = self.cost_engine.policy_for(
            venue=request.venue,
            instrument_type=request.instrument_type,
            orderbook_snapshot=request.orderbook_snapshot,
        )

        # Spread: prefer an explicit rate; else derive from the cost policy's spread.
        notional = max(_EPSILON, cost.entry_price * cost.quantity)
        spread_rate = request.spread_rate
        if spread_rate is None:
            spread_rate = cost.spread_cost / notional
        spread_rate = max(0.0, float(spread_rate))
        expected_slippage_rate = cost.slippage_cost / notional
        market_impact_rate = cost.market_impact_cost / notional
        liquidity_score = max(0.0, min(1.0, float(request.liquidity_score)))

        # --- Dynamic required minimum net return -------------------------------
        # The configured market floor is a hard minimum; an explicit per-theory
        # target can only TIGHTEN it (spec: max(config.min_net_return, ...)).
        market_floor = self.policy.min_net_for_market(request.market)
        base_min = (
            max(market_floor, float(request.target_net_return))
            if request.target_net_return is not None
            else market_floor
        )
        vol_buffer = max(0.0, self.policy.volatility_buffer_k * float(request.realized_volatility or 0.0))
        # Liquidity buffer grows as liquidity_score -> 0.
        liq_buffer = self.policy.liquidity_buffer_max * (1.0 - liquidity_score)
        acct_buffer = 0.0
        equity = float(request.account_equity_krw or 0.0)
        if 0.0 < equity <= self.policy.small_account_equity_krw:
            acct_buffer = self.policy.small_account_extra_net
        required_min_net_return = max(
            base_min,
            self.policy.min_net_profit_buffer_rate + vol_buffer + liq_buffer + acct_buffer,
        )

        # --- The composite net-edge rule ---------------------------------------
        # 1. Exit price must clear break-even plus a minimum profit buffer.
        break_even_with_margin = cost.break_even_exit_price * (1.0 + self.policy.min_net_profit_buffer_rate)
        if expected_exit_price < break_even_with_margin - _EPSILON:
            reasons.append(REASON_BELOW_BREAK_EVEN)
        # 2. Net expected return must clear the (dynamic) requirement.
        if cost.net_expected_return < required_min_net_return - _EPSILON:
            reasons.append(REASON_BELOW_MIN_NET)
        # 3. Cost must not dominate the alpha.
        if cost.cost_to_alpha_ratio > self.policy.max_cost_to_alpha_ratio + _EPSILON:
            reasons.append(REASON_COST_BURDEN)
        # 4. Absolute spread ceiling.
        if spread_rate > self.policy.max_spread_rate + _EPSILON:
            reasons.append(REASON_SPREAD)
        # 5. Spread relative to alpha (execution quality).
        spread_alpha_ratio = spread_rate / max(abs(cost.gross_expected_return), _EPSILON)
        if spread_alpha_ratio > self.policy.max_spread_alpha_ratio + _EPSILON:
            reasons.append(REASON_SPREAD_ALPHA)
        # 6. Liquidity floor.
        if liquidity_score < self.policy.min_liquidity_score - _EPSILON:
            reasons.append(REASON_LIQUIDITY)
        # 7. Expected entry slippage ceiling (hard reject — preserves the original
        #    RiskManager slippage safety gate).
        if expected_slippage_rate > self.policy.max_slippage_rate + _EPSILON:
            reasons.append(REASON_SLIPPAGE)

        # An empty/invalid order book makes the cost engine report spread=1.0.
        if spread_rate >= 0.5:
            flags.append("EMPTY_OR_INVALID_ORDERBOOK")

        breakdown = ProfitabilityBreakdown(
            entry_price=entry_price,
            expected_exit_price=expected_exit_price,
            break_even_exit_price=cost.break_even_exit_price,
            gross_expected_return=cost.gross_expected_return,
            all_in_cost_rate=cost.total_cost_rate,
            net_expected_return=cost.net_expected_return,
            cost_to_alpha_ratio=cost.cost_to_alpha_ratio,
            spread_rate=spread_rate,
            expected_slippage_rate=expected_slippage_rate,
            market_impact_rate=market_impact_rate,
            safety_margin_rate=policy_fees.safety_margin_rate,
            base_min_net_return=base_min,
            volatility_buffer=vol_buffer,
            liquidity_buffer=liq_buffer,
            account_buffer=acct_buffer,
        )
        return ProfitabilityDecision(
            allowed=not reasons,
            action=action,
            symbol=symbol,
            entry_price=entry_price,
            expected_exit_price=expected_exit_price,
            break_even_exit_price=cost.break_even_exit_price,
            gross_expected_return=cost.gross_expected_return,
            all_in_cost_rate=cost.total_cost_rate,
            net_expected_return=cost.net_expected_return,
            required_min_net_return=required_min_net_return,
            spread_rate=spread_rate,
            expected_slippage_rate=expected_slippage_rate,
            market_impact_rate=market_impact_rate,
            liquidity_score=liquidity_score,
            cost_to_alpha_ratio=cost.cost_to_alpha_ratio,
            rejection_reasons=tuple(dict.fromkeys(reasons)),
            warnings=tuple(dict.fromkeys(warnings)),
            data_quality_flags=tuple(dict.fromkeys(flags)),
            breakdown=breakdown,
        )

    # -- helpers ------------------------------------------------------------------

    def _reject(
        self,
        request: ProfitabilityInput,
        entry_price: float,
        expected_exit_price: float,
        reasons: list[str],
    ) -> ProfitabilityDecision:
        return ProfitabilityDecision(
            allowed=False,
            action=(request.action or "BUY").upper(),
            symbol=request.symbol,
            entry_price=entry_price,
            expected_exit_price=expected_exit_price,
            break_even_exit_price=0.0,
            gross_expected_return=0.0,
            all_in_cost_rate=0.0,
            net_expected_return=0.0,
            required_min_net_return=self.policy.min_net_for_market(request.market),
            spread_rate=request.spread_rate or 0.0,
            expected_slippage_rate=0.0,
            market_impact_rate=0.0,
            liquidity_score=max(0.0, min(1.0, float(request.liquidity_score))),
            cost_to_alpha_ratio=0.0,
            rejection_reasons=tuple(reasons),
        )

    def _informational_decision(
        self, request: ProfitabilityInput, entry_price: float, quantity: int
    ) -> ProfitabilityDecision:
        exit_price = float(request.expected_exit_price or entry_price)
        breakdown = None
        gross = net = cost_rate = c2a = 0.0
        be = exit_price
        if entry_price > 0 and quantity > 0 and exit_price > 0:
            cost = self.cost_engine.estimate(
                symbol=request.symbol,
                market=request.market,
                venue=request.venue,
                instrument_type=request.instrument_type,
                entry_price=entry_price,
                expected_exit_price=exit_price,
                quantity=quantity,
                orderbook_snapshot=request.orderbook_snapshot,
                average_daily_trading_value=request.average_daily_trading_value,
            )
            gross, net, cost_rate, c2a, be = (
                cost.gross_expected_return,
                cost.net_expected_return,
                cost.total_cost_rate,
                cost.cost_to_alpha_ratio,
                cost.break_even_exit_price,
            )
        return ProfitabilityDecision(
            allowed=True,
            action=(request.action or "SELL").upper(),
            symbol=request.symbol,
            entry_price=entry_price,
            expected_exit_price=exit_price,
            break_even_exit_price=be,
            gross_expected_return=gross,
            all_in_cost_rate=cost_rate,
            net_expected_return=net,
            required_min_net_return=0.0,
            spread_rate=request.spread_rate or 0.0,
            expected_slippage_rate=0.0,
            market_impact_rate=0.0,
            liquidity_score=max(0.0, min(1.0, float(request.liquidity_score))),
            cost_to_alpha_ratio=c2a,
            warnings=("NON_BUY_ACTION_NOT_GATED",),
            breakdown=breakdown,
        )


def _market_key(market: str) -> str:
    name = str(market or "").strip().upper()
    if name in {"KR", "KRX", "KOSPI", "KOSDAQ", "KONEX"}:
        return "KR"
    if name in {"US", "USA", "NASD", "NASDAQ", "NYSE", "AMEX"}:
        return "US"
    return name or "default"


def _env_float(name: str, default: float) -> float:
    try:
        raw = os.getenv(name)
        return float(raw) if raw is not None else default
    except (TypeError, ValueError):
        return default


_POLICY_LOGGED = False


def load_policy(config_path: Path | str = "config/profitability_policy.yaml") -> ProfitabilityPolicy:
    """Resolve policy from defaults < yaml < env, and log the final values once.

    Env-var backward compatibility (existing ~70-var runtime keeps working):
      REALTIME_MIN_BUY_NET_RETURN_KR / _US  -> per-market min net return
      REALTIME_MIN_NET_PROFIT_BUFFER_RATE   -> min_net_profit_buffer_rate
    """
    global _POLICY_LOGGED
    merged = _deep_merge(DEFAULT_PROFITABILITY_POLICY, _load_yaml(config_path))

    min_net = dict(merged.get("min_required_net_return", {}))
    # Env overrides (backward compatibility).
    if os.getenv("REALTIME_MIN_BUY_NET_RETURN_KR") is not None:
        min_net["KR"] = _env_float("REALTIME_MIN_BUY_NET_RETURN_KR", min_net.get("KR", 0.008))
    if os.getenv("REALTIME_MIN_BUY_NET_RETURN_US") is not None:
        min_net["US"] = _env_float("REALTIME_MIN_BUY_NET_RETURN_US", min_net.get("US", 0.012))
    min_net.setdefault("default", min_net.get("KR", 0.008))

    account = merged.get("account_buffer", {}) or {}
    policy = ProfitabilityPolicy(
        min_required_net_return={k: float(v) for k, v in min_net.items()},
        min_net_profit_buffer_rate=_env_float(
            "REALTIME_MIN_NET_PROFIT_BUFFER_RATE", float(merged.get("min_net_profit_buffer_rate", 0.001))
        ),
        max_spread_rate=float(merged.get("max_spread_rate", 0.003)),
        max_slippage_rate=float(merged.get("max_slippage_rate", 0.003)),
        max_spread_alpha_ratio=float(merged.get("max_spread_alpha_ratio", 0.35)),
        max_cost_to_alpha_ratio=float(merged.get("max_cost_to_alpha_ratio", 0.5)),
        min_liquidity_score=float(merged.get("min_liquidity_score", 0.3)),
        volatility_buffer_k=float(merged.get("volatility_buffer_k", 0.5)),
        liquidity_buffer_max=float(merged.get("liquidity_buffer_max", 0.003)),
        small_account_equity_krw=float(account.get("small_account_equity_krw", 200000.0)),
        small_account_extra_net=_env_float(
            "REALTIME_SMALL_ACCOUNT_EXTRA_NET", float(account.get("small_account_extra_net", 0.002))
        ),
    )
    if not _POLICY_LOGGED:
        logger.info("ProfitabilityGate resolved policy: %s", policy.as_dict())
        _POLICY_LOGGED = True
    return policy


def _load_yaml(config_path: Path | str) -> dict[str, Any]:
    path = Path(config_path)
    if not path.exists():
        return {}
    try:
        import yaml  # local import: yaml is an existing dependency (theory_registry.yaml)

        loaded = yaml.safe_load(path.read_text(encoding="utf-8"))
        return loaded if isinstance(loaded, dict) else {}
    except Exception:  # noqa: BLE001 - a malformed policy file falls back to defaults+env.
        logger.warning("Failed to load %s; using defaults + env", path)
        return {}


def _deep_merge(base: dict[str, Any], overlay: dict[str, Any]) -> dict[str, Any]:
    merged = dict(base)
    for key, value in overlay.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = _deep_merge(merged[key], value)
        else:
            merged[key] = value
    return merged
