from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any


DEFAULT_TRADING_COST_CONFIG: dict[str, Any] = {
    "broker": "KoreaInvestment",
    "account_type": "BanKIS",
    "default_market": "KR",
    "domestic_stock": {
        "KRX": {"buy_fee_rate": 0.000140527, "sell_fee_rate": 0.000140527, "sell_tax_rate": 0.002},
        "NXT": {"buy_fee_rate": 0.000130527, "sell_fee_rate": 0.000130527, "sell_tax_rate": 0.002},
    },
    "overseas_stock": {
        "NASD": {"buy_fee_rate": 0.0025, "sell_fee_rate": 0.0025, "sell_tax_rate": 0.0},
        "NYSE": {"buy_fee_rate": 0.0025, "sell_fee_rate": 0.0025, "sell_tax_rate": 0.0},
        "AMEX": {"buy_fee_rate": 0.0025, "sell_fee_rate": 0.0025, "sell_tax_rate": 0.0},
        "SEHK": {"buy_fee_rate": 0.0030, "sell_fee_rate": 0.0030, "sell_tax_rate": 0.0013},
        "SHAA": {"buy_fee_rate": 0.0030, "sell_fee_rate": 0.0030, "sell_tax_rate": 0.0010},
        "SZAA": {"buy_fee_rate": 0.0030, "sell_fee_rate": 0.0030, "sell_tax_rate": 0.0010},
        "TKSE": {"buy_fee_rate": 0.0030, "sell_fee_rate": 0.0030, "sell_tax_rate": 0.0},
        "HASE": {"buy_fee_rate": 0.0040, "sell_fee_rate": 0.0040, "sell_tax_rate": 0.0010},
        "VNSE": {"buy_fee_rate": 0.0040, "sell_fee_rate": 0.0040, "sell_tax_rate": 0.0010},
    },
    "domestic_etf_etn_elw": {"default_fee_rate": 0.000146527, "sell_tax_rate": 0.0},
    "slippage": {"default_slippage_rate": 0.0005, "use_dynamic_slippage_if_orderbook_exists": True},
    "spread": {"default_spread_rate": 0.0},
    "market_impact": {"default_market_impact_rate": 0.0},
    "safety_margin": {"default_safety_margin_rate": 0.001},
    "gate": {
        "max_cost_to_alpha_ratio": 0.5,
        "default_target_net_return": 0.0,
        "max_spread_rate": 0.003,
        "max_slippage_rate": 0.003,
    },
}


@dataclass(frozen=True)
class FeePolicy:
    buy_fee_rate: float
    sell_fee_rate: float
    sell_tax_rate: float
    slippage_rate: float
    spread_rate: float
    market_impact_rate: float
    safety_margin_rate: float
    max_cost_to_alpha_ratio: float


@dataclass(frozen=True)
class CostBreakdown:
    symbol: str
    market: str
    venue: str
    instrument_type: str
    entry_price: float
    expected_exit_price: float
    quantity: int
    gross_expected_profit: float
    gross_expected_return: float
    buy_fee: float
    sell_fee: float
    sell_tax: float
    slippage_cost: float
    spread_cost: float
    market_impact_cost: float
    total_cost: float
    total_cost_rate: float
    break_even_return: float
    break_even_exit_price: float
    required_exit_price: float
    net_expected_profit: float
    net_expected_return: float
    cost_to_alpha_ratio: float
    excess_return_after_cost: float
    tradable: bool
    reject_reason: str | None = None

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


class TradingCostEngine:
    def __init__(self, config_path: Path | str = "config/trading_costs.json") -> None:
        self.config_path = Path(config_path)
        self.config = self._load_config()

    def estimate(
        self,
        *,
        symbol: str,
        market: str = "KR",
        venue: str = "KRX",
        instrument_type: str = "domestic_stock",
        account_type: str = "BanKIS",
        entry_price: float,
        expected_exit_price: float,
        quantity: int,
        target_net_return: float = 0.0,
        orderbook_snapshot: dict[str, Any] | None = None,
        average_daily_trading_value: float | None = None,
    ) -> CostBreakdown:
        del account_type
        quantity = max(0, int(quantity))
        entry_price = max(0.0, float(entry_price))
        expected_exit_price = max(0.0, float(expected_exit_price))
        policy = self.policy_for(venue=venue, instrument_type=instrument_type, orderbook_snapshot=orderbook_snapshot)

        notional = entry_price * quantity
        gross_received = expected_exit_price * quantity
        gross_expected_profit = gross_received - notional
        gross_expected_return = _safe_div(gross_expected_profit, max(notional, 1e-9))

        buy_fee = notional * policy.buy_fee_rate
        sell_fee = gross_received * policy.sell_fee_rate
        sell_tax = gross_received * policy.sell_tax_rate
        slippage_cost = notional * policy.slippage_rate
        spread_cost = notional * policy.spread_rate
        market_impact_cost = notional * self._market_impact_rate(quantity, entry_price, average_daily_trading_value, policy)
        total_cost = buy_fee + sell_fee + sell_tax + slippage_cost + spread_cost + market_impact_cost
        total_cost_rate = _safe_div(total_cost, max(notional, 1e-9))

        sell_deduction_rate = min(0.99, policy.sell_fee_rate + policy.sell_tax_rate)
        entry_multiplier = 1 + policy.buy_fee_rate + policy.slippage_rate + policy.spread_rate + policy.market_impact_rate
        break_even_exit_price = (
            entry_price * entry_multiplier / max(1e-9, 1 - sell_deduction_rate)
            if entry_price > 0
            else 0.0
        )
        required_exit_price = break_even_exit_price * (1 + max(0.0, target_net_return))
        break_even_return = _safe_div(break_even_exit_price - entry_price, max(entry_price, 1e-9))
        sell_received_amount = gross_received - sell_fee - sell_tax
        buy_total_cost = notional + buy_fee
        net_expected_profit = sell_received_amount - buy_total_cost - slippage_cost - spread_cost - market_impact_cost
        net_expected_return = _safe_div(net_expected_profit, max(buy_total_cost, 1e-9))
        cost_to_alpha_ratio = _safe_div(total_cost_rate, max(abs(gross_expected_return), 1e-9))
        excess_return_after_cost = gross_expected_return - total_cost_rate - policy.safety_margin_rate

        reject_reason = None
        if quantity <= 0 or entry_price <= 0:
            reject_reason = "INVALID_ORDER_SIZE_OR_PRICE"
        elif net_expected_return <= 0:
            reject_reason = "NET_RETURN_NOT_POSITIVE"
        elif gross_expected_return < break_even_return + policy.safety_margin_rate:
            reject_reason = "BELOW_BREAK_EVEN_WITH_MARGIN"
        elif target_net_return and net_expected_return < target_net_return:
            reject_reason = "BELOW_TARGET_NET_RETURN"
        elif cost_to_alpha_ratio > policy.max_cost_to_alpha_ratio:
            reject_reason = "COST_BURDEN_HIGH"

        return CostBreakdown(
            symbol=symbol,
            market=market,
            venue=venue,
            instrument_type=instrument_type,
            entry_price=entry_price,
            expected_exit_price=expected_exit_price,
            quantity=quantity,
            gross_expected_profit=gross_expected_profit,
            gross_expected_return=gross_expected_return,
            buy_fee=buy_fee,
            sell_fee=sell_fee,
            sell_tax=sell_tax,
            slippage_cost=slippage_cost,
            spread_cost=spread_cost,
            market_impact_cost=market_impact_cost,
            total_cost=total_cost,
            total_cost_rate=total_cost_rate,
            break_even_return=break_even_return,
            break_even_exit_price=break_even_exit_price,
            required_exit_price=required_exit_price,
            net_expected_profit=net_expected_profit,
            net_expected_return=net_expected_return,
            cost_to_alpha_ratio=cost_to_alpha_ratio,
            excess_return_after_cost=excess_return_after_cost,
            tradable=reject_reason is None,
            reject_reason=reject_reason,
        )

    def policy_for(
        self,
        *,
        venue: str = "KRX",
        instrument_type: str = "domestic_stock",
        orderbook_snapshot: dict[str, Any] | None = None,
    ) -> FeePolicy:
        venue = (venue or "KRX").upper()
        instrument_type = instrument_type or "domestic_stock"
        if instrument_type == "overseas_stock":
            overseas = self.config.get("overseas_stock", {})
            policy = overseas.get(venue) or overseas.get("NASD") or DEFAULT_TRADING_COST_CONFIG["overseas_stock"]["NASD"]
            buy_fee_rate = float(policy.get("buy_fee_rate", 0.0025))
            sell_fee_rate = float(policy.get("sell_fee_rate", 0.0025))
            sell_tax_rate = float(policy.get("sell_tax_rate", 0.0))
        elif instrument_type in {"domestic_etf", "domestic_etn", "domestic_elw", "etf", "etn", "elw"}:
            product = self.config.get("domestic_etf_etn_elw", {})
            buy_fee_rate = sell_fee_rate = float(product.get("default_fee_rate", 0.000146527))
            sell_tax_rate = float(product.get("sell_tax_rate", 0.0))
        else:
            domestic = self.config.get("domestic_stock", {})
            policy = domestic.get(venue) or domestic.get("KRX") or DEFAULT_TRADING_COST_CONFIG["domestic_stock"]["KRX"]
            buy_fee_rate = float(policy.get("buy_fee_rate", 0.000140527))
            sell_fee_rate = float(policy.get("sell_fee_rate", 0.000140527))
            sell_tax_rate = float(policy.get("sell_tax_rate", 0.002))

        slippage_config = self.config.get("slippage", {})
        slippage_rate = float(slippage_config.get("default_slippage_rate", 0.0005))
        spread_rate = float(self.config.get("spread", {}).get("default_spread_rate", 0.0))
        if slippage_config.get("use_dynamic_slippage_if_orderbook_exists", True):
            dynamic_spread = self._spread_rate_from_orderbook(orderbook_snapshot)
            if dynamic_spread is not None:
                spread_rate = max(spread_rate, dynamic_spread)
                slippage_rate = max(slippage_rate, dynamic_spread / 2)

        return FeePolicy(
            buy_fee_rate=buy_fee_rate,
            sell_fee_rate=sell_fee_rate,
            sell_tax_rate=sell_tax_rate,
            slippage_rate=slippage_rate,
            spread_rate=spread_rate,
            market_impact_rate=float(self.config.get("market_impact", {}).get("default_market_impact_rate", 0.0)),
            safety_margin_rate=float(self.config.get("safety_margin", {}).get("default_safety_margin_rate", 0.001)),
            max_cost_to_alpha_ratio=float(self.config.get("gate", {}).get("max_cost_to_alpha_ratio", 0.5)),
        )

    def _load_config(self) -> dict[str, Any]:
        if not self.config_path.exists():
            return dict(DEFAULT_TRADING_COST_CONFIG)
        try:
            loaded = json.loads(self.config_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return dict(DEFAULT_TRADING_COST_CONFIG)
        return _deep_merge(DEFAULT_TRADING_COST_CONFIG, loaded)

    @staticmethod
    def _spread_rate_from_orderbook(orderbook_snapshot: dict[str, Any] | None) -> float | None:
        if not orderbook_snapshot:
            return None
        bid = _to_float(orderbook_snapshot.get("bid_price") or orderbook_snapshot.get("best_bid"))
        ask = _to_float(orderbook_snapshot.get("ask_price") or orderbook_snapshot.get("best_ask"))
        mid = (bid + ask) / 2 if bid > 0 and ask > 0 else 0.0
        if mid <= 0 or ask < bid:
            return None
        return (ask - bid) / mid

    @staticmethod
    def _market_impact_rate(
        quantity: int,
        entry_price: float,
        average_daily_trading_value: float | None,
        policy: FeePolicy,
    ) -> float:
        if not average_daily_trading_value or average_daily_trading_value <= 0:
            return policy.market_impact_rate
        participation = (quantity * entry_price) / average_daily_trading_value
        return max(policy.market_impact_rate, min(0.01, participation * 0.15))


def _deep_merge(base: dict[str, Any], overlay: dict[str, Any]) -> dict[str, Any]:
    merged = dict(base)
    for key, value in overlay.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = _deep_merge(merged[key], value)
        else:
            merged[key] = value
    return merged


def _safe_div(numerator: float, denominator: float) -> float:
    if abs(denominator) < 1e-12:
        return 0.0
    return numerator / denominator


def _to_float(value: Any) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return 0.0
