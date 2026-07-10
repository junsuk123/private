"""Execution-quality layer.

Short-horizon profitability is dominated by execution quality: a positive expected
net edge is destroyed if spread, slippage, or poor fills eat the alpha. This layer
evaluates spread burden, order-book pressure, expected slippage, and fill probability
BEFORE submission, and rejects a BUY when the expected fill would invalidate the net
edge. It also provides a limit-order price/timeout/reprice policy and consumes realized
slippage history (from :mod:`app.storage.execution_quality_store`) so a symbol with
persistently bad fills is down-scored or blocked.

Formulas (spec):
    spread_alpha_ratio        = spread_rate / max(gross_expected_return, eps)
    realized_slippage_rate    = |fill_price - decision_reference_price| / decision_reference_price
    execution_adjusted_net    = net_expected_return - expected_extra_slippage_rate
Reject when spread_alpha_ratio > max_spread_alpha_ratio, or when the execution-adjusted
net return drops below the required minimum.
"""

from __future__ import annotations

import logging
import os
from dataclasses import asdict, dataclass
from typing import Any

logger = logging.getLogger(__name__)

_EPSILON = 1e-9


@dataclass(frozen=True)
class ExecutionQualityInput:
    symbol: str
    strategy_family: str
    decision_reference_price: float
    gross_expected_return: float
    net_expected_return: float
    required_min_net_return: float
    best_bid: float = 0.0
    best_ask: float = 0.0
    bid_depth: float = 0.0
    ask_depth: float = 0.0
    order_quantity: int = 1
    side: str = "BUY"
    # Normalized exit motive for a SELL: ENTRY / TAKE_PROFIT / STOP_LOSS / HARD_STOP /
    # EMERGENCY / REDUCE / MODEL_EXIT / TIME_STOP. Drives whether a no-orderbook SELL
    # is allowed (urgent stops must still exit).
    action_reason: str = "ENTRY"
    # Age of the order book in seconds (None = unknown). A stale book is treated as
    # no book for a BUY when EXEC_REQUIRE_FRESH_ORDERBOOK_FOR_BUY is set.
    orderbook_age_sec: float | None = None


# Urgent SELL motives that must be allowed to exit even without a usable book.
_URGENT_SELL_REASONS = frozenset({"STOP_LOSS", "HARD_STOP", "EMERGENCY"})


@dataclass(frozen=True)
class ExecutionQualityAssessment:
    allowed: bool
    symbol: str
    spread_rate: float
    spread_alpha_ratio: float
    expected_slippage_rate: float
    orderbook_pressure: float          # >0 favours buyers (more bid depth), <0 favours sellers
    fill_probability: float
    execution_adjusted_net_return: float
    limit_price: float
    reject_reason: str | None = None
    warnings: tuple[str, ...] = ()

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


class ExecutionQualityEngine:
    def __init__(self, store: "Any | None" = None) -> None:
        self.store = store
        self.max_spread_alpha_ratio = _env_float("EXEC_MAX_SPREAD_ALPHA_RATIO", 0.35)
        self.max_expected_slippage_rate = _env_float("EXEC_MAX_EXPECTED_SLIPPAGE_RATE", 0.004)
        self.min_fill_probability = _env_float("EXEC_MIN_FILL_PROBABILITY", 0.3)
        # A symbol whose recent realized slippage exceeds this is blocked/penalized.
        self.max_recent_realized_slippage = _env_float("EXEC_MAX_RECENT_REALIZED_SLIPPAGE", 0.006)
        # No-orderbook / freshness policy. A BUY with no usable (or stale) book must
        # not be priced as if the spread were zero, so it is blocked by default.
        self.require_orderbook_for_buy = _env_bool("EXEC_REQUIRE_ORDERBOOK_FOR_BUY", True)
        self.require_fresh_orderbook_for_buy = _env_bool("EXEC_REQUIRE_FRESH_ORDERBOOK_FOR_BUY", True)
        self.max_orderbook_age_sec = _env_float("EXEC_MAX_ORDERBOOK_AGE_SEC", 3.0)
        # Assumed spread when the book is unknown — never 0.0 (which would hide the risk).
        self.unknown_spread_penalty_rate = _env_float("EXEC_UNKNOWN_SPREAD_PENALTY_RATE", 0.006)
        self.allow_no_orderbook_emergency_sell = _env_bool("EXEC_ALLOW_NO_ORDERBOOK_EMERGENCY_SELL", True)

    def assess(self, request: ExecutionQualityInput) -> ExecutionQualityAssessment:
        bid = max(0.0, float(request.best_bid))
        ask = max(0.0, float(request.best_ask))
        ref = max(_EPSILON, float(request.decision_reference_price))
        side = request.side.upper()
        action_reason = str(request.action_reason or "ENTRY").upper()
        warnings: list[str] = []

        has_book = bid > 0 and ask > 0 and ask >= bid
        stale_book = False
        if (
            has_book
            and side == "BUY"
            and self.require_fresh_orderbook_for_buy
            and request.orderbook_age_sec is not None
            and float(request.orderbook_age_sec) > self.max_orderbook_age_sec
        ):
            # A stale book is as untradeable as no book for a fresh BUY entry.
            stale_book = True
            warnings.append("STALE_ORDERBOOK")

        usable_book = has_book and not stale_book

        if not usable_book:
            # No usable book. Do NOT pretend the spread is zero — that hid execution
            # risk and let BUYs through blind. Block a BUY (unless disabled); allow an
            # urgent SELL stop to still exit; block a non-urgent no-book SELL.
            warnings.append("NO_ORDERBOOK")
            limit_price = ref
            if side == "BUY" and self.require_orderbook_for_buy:
                return self._blocked(request, spread_rate=self.unknown_spread_penalty_rate,
                                     reject_reason="EXEC_NO_ORDERBOOK_BLOCKED", limit_price=limit_price,
                                     warnings=warnings)
            if side == "SELL":
                urgent = action_reason in _URGENT_SELL_REASONS
                if urgent and self.allow_no_orderbook_emergency_sell:
                    warnings.append("NO_ORDERBOOK_EMERGENCY_SELL_ALLOWED")
                    return self._allowed_no_book(request, limit_price=limit_price, warnings=warnings)
                if not urgent:
                    return self._blocked(request, spread_rate=self.unknown_spread_penalty_rate,
                                         reject_reason="EXEC_NO_ORDERBOOK_SELL_BLOCKED", limit_price=limit_price,
                                         warnings=warnings)
                # Urgent sell but fallback disabled: still block, surfacing why.
                return self._blocked(request, spread_rate=self.unknown_spread_penalty_rate,
                                     reject_reason="EXEC_NO_ORDERBOOK_SELL_BLOCKED", limit_price=limit_price,
                                     warnings=warnings)
            # BUY with the requirement disabled: assess against a penalty spread.
            mid = ref
            spread_rate = self.unknown_spread_penalty_rate
        else:
            mid = (bid + ask) / 2.0
            spread_rate = (ask - bid) / mid

        # Order-book pressure in [-1, 1]; positive means bid-heavy (supports a buy).
        total_depth = float(request.bid_depth) + float(request.ask_depth)
        orderbook_pressure = (
            (float(request.bid_depth) - float(request.ask_depth)) / total_depth if total_depth > 0 else 0.0
        )

        gross = abs(float(request.gross_expected_return))
        spread_alpha_ratio = spread_rate / max(gross, _EPSILON)

        # Expected slippage: half-spread plus a depth-driven impact term, floored by any
        # persistently-bad realized history for this symbol.
        expected_slippage_rate = spread_rate / 2.0
        if request.side.upper() == "BUY" and orderbook_pressure < 0:
            # Thin bid relative to ask -> a buy is likelier to walk the book up.
            expected_slippage_rate += min(0.003, abs(orderbook_pressure) * spread_rate)
        recent_realized = self._recent_realized_slippage(request.symbol, request.strategy_family)
        if recent_realized is not None:
            expected_slippage_rate = max(expected_slippage_rate, recent_realized)

        # Fill probability heuristic: a marketable-ish limit at the ask with supportive
        # pressure fills easily; a wide spread with adverse pressure fills poorly.
        fill_probability = _clamp(0.9 - spread_alpha_ratio + 0.1 * orderbook_pressure, 0.0, 1.0)

        execution_adjusted_net = float(request.net_expected_return) - expected_slippage_rate

        # Side-aware limit price: a BUY caps at the ask (do not chase above it); a SELL
        # posts at the bid (marketable to buyers). The authoritative executable price is
        # produced by ExecutionPricingPolicy; this mirrors it for the assessment record.
        if side == "BUY":
            limit_price = ask if ask > 0 else ref
        else:
            limit_price = bid if bid > 0 else ref

        reject_reason: str | None = None
        # The spread/slippage/fill-probability rejections below are ENTRY (BUY) concerns:
        # they veto a buy whose net edge would be eaten by execution cost. An approved
        # SELL exit (take-profit / stop) must NOT be blocked by entry-edge math — that
        # would strand a position — so once a SELL has a usable book it is allowed.
        if side == "BUY":
            # Check the symbol's realized-slippage history first — it is the most actionable
            # reason (a symbol that persistently fills badly should be blocked by name).
            if recent_realized is not None and recent_realized > self.max_recent_realized_slippage:
                reject_reason = "EXEC_SYMBOL_SLIPPAGE_HISTORY_BAD"
            elif spread_alpha_ratio > self.max_spread_alpha_ratio + _EPSILON:
                reject_reason = "EXEC_SPREAD_CONSUMES_ALPHA"
            elif expected_slippage_rate > self.max_expected_slippage_rate + _EPSILON:
                reject_reason = "EXEC_EXPECTED_SLIPPAGE_TOO_HIGH"
            elif execution_adjusted_net < float(request.required_min_net_return) - _EPSILON:
                reject_reason = "EXEC_ADJUSTED_NET_BELOW_MIN"
            elif fill_probability < self.min_fill_probability:
                reject_reason = "EXEC_FILL_PROBABILITY_TOO_LOW"

        return ExecutionQualityAssessment(
            allowed=reject_reason is None,
            symbol=request.symbol,
            spread_rate=spread_rate,
            spread_alpha_ratio=spread_alpha_ratio,
            expected_slippage_rate=expected_slippage_rate,
            orderbook_pressure=orderbook_pressure,
            fill_probability=fill_probability,
            execution_adjusted_net_return=execution_adjusted_net,
            limit_price=limit_price,
            reject_reason=reject_reason,
            warnings=tuple(warnings),
        )

    def _blocked(
        self,
        request: ExecutionQualityInput,
        *,
        spread_rate: float,
        reject_reason: str,
        limit_price: float,
        warnings: list[str],
    ) -> ExecutionQualityAssessment:
        gross = abs(float(request.gross_expected_return))
        return ExecutionQualityAssessment(
            allowed=False,
            symbol=request.symbol,
            spread_rate=spread_rate,
            spread_alpha_ratio=spread_rate / max(gross, _EPSILON),
            expected_slippage_rate=spread_rate / 2.0,
            orderbook_pressure=0.0,
            fill_probability=0.0,
            execution_adjusted_net_return=float(request.net_expected_return) - spread_rate / 2.0,
            limit_price=limit_price,
            reject_reason=reject_reason,
            warnings=tuple(warnings),
        )

    def _allowed_no_book(
        self,
        request: ExecutionQualityInput,
        *,
        limit_price: float,
        warnings: list[str],
    ) -> ExecutionQualityAssessment:
        # An urgent SELL exit is allowed without a book; price via a penalty spread so
        # the record still reflects that execution quality is degraded.
        spread_rate = self.unknown_spread_penalty_rate
        gross = abs(float(request.gross_expected_return))
        return ExecutionQualityAssessment(
            allowed=True,
            symbol=request.symbol,
            spread_rate=spread_rate,
            spread_alpha_ratio=spread_rate / max(gross, _EPSILON),
            expected_slippage_rate=spread_rate / 2.0,
            orderbook_pressure=0.0,
            fill_probability=0.5,
            execution_adjusted_net_return=float(request.net_expected_return) - spread_rate / 2.0,
            limit_price=limit_price,
            reject_reason=None,
            warnings=tuple(warnings),
        )

    def record_fill(
        self,
        *,
        symbol: str,
        strategy_family: str,
        decision_reference_price: float,
        fill_price: float,
        side: str = "BUY",
        time_bucket: str | None = None,
    ) -> float:
        """Record a realized fill and return the realized slippage rate."""
        ref = max(_EPSILON, float(decision_reference_price))
        realized = abs(float(fill_price) - ref) / ref
        if self.store is not None:
            try:
                self.store.record(
                    symbol=symbol,
                    strategy_family=strategy_family,
                    realized_slippage_rate=realized,
                    side=side,
                    time_bucket=time_bucket,
                )
            except Exception:  # noqa: BLE001 - persistence is best-effort.
                logger.warning("execution-quality store record failed for %s", symbol)
        return realized

    def _recent_realized_slippage(self, symbol: str, strategy_family: str) -> float | None:
        if self.store is None:
            return None
        try:
            return self.store.recent_average(symbol=symbol, strategy_family=strategy_family)
        except Exception:  # noqa: BLE001
            return None


def _clamp(value: float, low: float, high: float) -> float:
    return max(low, min(high, float(value)))


def _env_float(name: str, default: float) -> float:
    raw = os.getenv(name)
    if raw is None:
        return float(default)
    try:
        return float(raw)
    except (TypeError, ValueError):
        return float(default)


def _env_bool(name: str, default: bool = False) -> bool:
    value = os.getenv(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}
