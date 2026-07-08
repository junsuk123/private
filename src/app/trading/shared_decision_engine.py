from __future__ import annotations

import json
import os
import math
from dataclasses import dataclass, field, replace
from datetime import datetime, timedelta, timezone
from typing import Any, Callable

from app.cost import ProfitabilityGate, ProfitabilityInput, TradingCostEngine
from app.data.realtime_store import RealtimeMarketDataStore
from app.features.live_feature_frame import LiveFeatureFrameBuilder
from app.models.live_signal_predictor import LiveSignalPredictor, LiveSignalPrediction
from app.risk import RiskManager
from app.risk.position_sizing import PositionSizer, SizingInputs
from app.schemas.domain import AccountSnapshot, FinalOrder, Holding, MarketSnapshot, OrderAction, OrderIntent, RiskRules, SourceMetadata, OrderSide, OrderType
from app.trading.auto_tuning_engine import AutoTuningEngine, MarketStateSnapshot
from app.trading.decision_logger import DecisionLogger
from app.trading.dynamic_exit_policy import DynamicExitPolicy
from app.strategy.rule_based import _holding_exit_adjustment, _ontology_flow_adjustment


_NEWS_TRUST_CACHE: dict[str, Any] = {"mtime": None, "scale": 1.0}


def _news_confirm_scale(path: str = "data/store/news_trust.json") -> float:
    """Outcome-calibrated multiplier for the positive-news confirm bonus (approach A).

    Written by the training pipeline from realized PnL; read here cached by mtime so
    the tick loop does not touch disk every cycle. Defaults to 1.0 (neutral)."""
    try:
        mtime = os.path.getmtime(path)
    except OSError:
        return 1.0
    if _NEWS_TRUST_CACHE["mtime"] != mtime:
        scale = 1.0
        try:
            with open(path, "r", encoding="utf-8") as handle:
                scale = float(json.load(handle).get("news_confirm_scale", 1.0))
        except (OSError, ValueError, TypeError):
            scale = 1.0
        _NEWS_TRUST_CACHE["scale"] = max(0.3, min(2.0, scale))
        _NEWS_TRUST_CACHE["mtime"] = mtime
    return float(_NEWS_TRUST_CACHE["scale"])


# 매수 근거로 인정하는 supportsSignal(현금/affordability 류는 제외 — 그건 매수 "엣지"가 아님).
_BUY_EDGE_SUPPORTS = frozenset(
    {
        "RevenueGrowth",
        "EarningsGrowth",
        "ProfitabilityQuality",
        "NpuCompositeMomentum",
        "LiquiditySupport",
        # 국내 플로우 근거(KR)
        "InformedOrderFlowImbalance",
        "ForeignInstitutionJointBuying",
        "RetailSupplyAbsorbedByInformedFlow",
        "OrderFlowPriceConfirmation",
        "SuspectedSmartMoneyAccumulation",
        "OrderFlowConfirmedBuyCandidate",
        "AccountCashFeasibleBuyCandidate",
        "ExecutableBuyCandidate",
        "LiveBrokerRealtimeQuote",
        "FreshBrokerQuote",
        "RealtimeAdaptiveFallbackBuyCandidate",
    }
)


def _ontology_buy_evidence(graph: Any, symbol: str) -> tuple[float, tuple[str, ...]]:
    """일반 매수근거 supportsSignal에서 increasesRiskOf/contradictsSignal을 뺀 순증거 점수.

    플로우 전용 점수와 달리 US 종목에도 붙는 NPU 모멘텀/유동성/실적 근거를 포착한다.
    """
    supports = {str(t.object) for t in graph.matching(subject=symbol, predicate="supportsSignal")}
    risks = {str(t.object) for t in graph.matching(subject=symbol, predicate="increasesRiskOf")}
    contradicts = {str(t.object) for t in graph.matching(subject=symbol, predicate="contradictsSignal")}
    edge = supports & _BUY_EDGE_SUPPORTS
    net = float(len(edge) - len(risks) - len(contradicts))
    return net, tuple(sorted(edge))


def _news_sentiment_flags(graph: Any, symbol: str) -> tuple[bool, bool]:
    """Fresh news/disclosure sentiment already projected into the graph.

    `event_mapper` maps a positive classified event to
    (symbol supportsSignal PositiveEventImpact) and a negative one to
    (symbol increasesRiskOf NegativeEventRisk). Returns (has_positive, has_negative).
    The graph is rebuilt each analysis cycle from recency-filtered events, so a
    present triple already implies the news is recent.
    """
    supports = {str(t.object) for t in graph.matching(subject=symbol, predicate="supportsSignal")}
    risks = {str(t.object) for t in graph.matching(subject=symbol, predicate="increasesRiskOf")}
    return "PositiveEventImpact" in supports, "NegativeEventRisk" in risks


def _market_for_symbol(symbol: str) -> str:
    """Classify a symbol into the market label used for order routing/gates.

    6-digit numeric → Korean equity (KR); everything else → US (NASD default).
    """
    s = str(symbol or "").strip().upper()
    if s.isdigit() and len(s) == 6:
        return "KR"
    return "NASD"


def _load_us_exchange_map() -> dict[str, str]:
    """Operator-maintained ticker→exchange overrides (JSON in KIS_US_EXCHANGE_MAP).

    Example: KIS_US_EXCHANGE_MAP='{"PLTR":"NYSE","F":"NYSE","SPCE":"NYSE"}'
    """
    raw = os.getenv("KIS_US_EXCHANGE_MAP", "").strip()
    if not raw:
        return {}
    try:
        data = json.loads(raw)
    except (TypeError, ValueError, json.JSONDecodeError):
        return {}
    if not isinstance(data, dict):
        return {}
    return {
        str(k).upper().strip(): str(v).upper().strip()
        for k, v in data.items()
        if str(k).strip() and str(v).strip()
    }


_US_LISTED_EXCHANGE_MAP_CACHE: dict[str, Any] = {"mtime": None, "map": {}}


def _load_us_listed_exchange_map(path: str = "data/universe/us_exchange_map.csv") -> dict[str, str]:
    """Ticker → KIS US exchange (NASD/NYSE/AMEX) from the built listing map.

    Built from nasdaqtrader.com nasdaqlisted.txt + otherlisted.txt so a US buy quotes
    (EXCD) and routes its order (OVRS_EXCG_CD) on the ticker's REAL exchange. Without
    it, US names defaulted to NASD, so NYSE/AMEX tickers quoted at price 0 in discovery
    and would be rejected at order time. Cached by mtime; empty dict when the file is
    absent (falls back to the previous NASD default). Only the three KIS-supported US
    venues are kept — Arca/BATS/IEX rows are omitted upstream to avoid wrong-exchange
    orders.
    """
    import csv as _csv

    try:
        mtime = os.path.getmtime(path)
    except OSError:
        return {}
    if _US_LISTED_EXCHANGE_MAP_CACHE.get("mtime") != mtime:
        mapping: dict[str, str] = {}
        try:
            with open(path, "r", encoding="utf-8") as handle:
                for row in _csv.DictReader(handle):
                    sym = str(row.get("symbol") or "").strip().upper()
                    exch = str(row.get("exchange") or "").strip().upper()
                    if sym and exch in {"NASD", "NYSE", "AMEX"}:
                        mapping[sym] = exch
        except OSError:
            return dict(_US_LISTED_EXCHANGE_MAP_CACHE.get("map") or {})
        _US_LISTED_EXCHANGE_MAP_CACHE["map"] = mapping
        _US_LISTED_EXCHANGE_MAP_CACHE["mtime"] = mtime
    return _US_LISTED_EXCHANGE_MAP_CACHE.get("map") or {}


def _resolve_order_market(symbol: str, account: AccountSnapshot | None = None) -> str:
    """Resolve the KIS routing exchange for a *buy* order.

    6-digit numeric → KR. For US names the OVRS_EXCG_CD must match the listing
    exchange (KIS rejects a wrong one), yet a bare ticker only tells us "US". We
    resolve it authoritatively when possible: (1) the broker-reported exchange of a
    position we already hold with the same ticker, (2) an operator-maintained
    KIS_US_EXCHANGE_MAP override, (3) the built ticker→exchange listing map
    (NASD/NYSE/AMEX), else (4) the configured default. Unknown names keep the previous
    NASD behavior, so this never regresses working NASDAQ orders. Sells are unaffected
    — they already route on the holding's broker exchange.
    """
    base = _market_for_symbol(symbol)
    if base == "KR":
        return base
    s = str(symbol or "").strip().upper()
    if account is not None:
        for holding in getattr(account, "holdings", ()) or ():
            if str(getattr(holding, "ticker", "") or "").strip().upper() != s:
                continue
            held = str(getattr(holding, "market", "") or "").strip().upper()
            for code in ("NYSE", "AMEX", "NASD"):
                if code in held:
                    return code
    mapped = _load_us_exchange_map().get(s)
    if mapped:
        return mapped
    listed = _load_us_listed_exchange_map().get(s)
    if listed:
        return listed
    return os.getenv("KIS_DEFAULT_US_EXCHANGE", "NASD").upper() or "NASD"


def _cost_context_for_holding(symbol: str, market: str) -> tuple[str, str]:
    s = str(symbol or "").strip().upper()
    market_name = str(market or "").strip().upper()
    if s.isdigit() and len(s) == 6 or market_name in {"KR", "KRX", "KOSPI", "KOSDAQ", "KONEX"}:
        return "KRX", "domestic_stock"
    if "NYSE" in market_name:
        return "NYSE", "overseas_stock"
    if "AMEX" in market_name:
        return "AMEX", "overseas_stock"
    return "NASD", "overseas_stock"


def _is_domestic_symbol_or_market(symbol: str, market: str) -> bool:
    s = str(symbol or "").strip().upper()
    market_name = str(market or "").strip().upper()
    return (s.isdigit() and len(s) == 6) or market_name in {"KR", "KRX", "KOSPI", "KOSDAQ", "KONEX"}


def _env_float(name: str, default: float) -> float:
    try:
        return float(os.getenv(name, str(default)))
    except (TypeError, ValueError):
        return default


def _env_bool(name: str, default: bool = False) -> bool:
    value = os.getenv(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


def _account_domestic_unrealized_rate(account: AccountSnapshot) -> float:
    cost = 0.0
    pnl = 0.0
    for holding in getattr(account, "holdings", ()) or ():
        if not _is_domestic_symbol_or_market(getattr(holding, "ticker", ""), getattr(holding, "market", "")):
            continue
        quantity = float(getattr(holding, "quantity", 0.0) or 0.0)
        average_price = float(getattr(holding, "average_price", 0.0) or 0.0)
        last_price = float(getattr(holding, "last_price", 0.0) or 0.0)
        cost += max(0.0, quantity * average_price)
        pnl += quantity * (last_price - average_price)
    return pnl / cost if cost > 0 else 0.0


def _realized_volatility_from_prices(prices: list[float]) -> float:
    returns = [
        prices[index] / prices[index - 1] - 1.0
        for index in range(1, len(prices))
        if prices[index - 1] > 0.0
    ]
    if len(returns) < 2:
        return 0.0
    mean_return = sum(returns) / len(returns)
    variance = sum((value - mean_return) ** 2 for value in returns) / len(returns)
    return max(0.0, variance**0.5)


def _buy_blocker_reason_codes(
    *,
    prediction_error: Exception | None,
    policy: Any,
    policy_diag: dict[str, Any],
    fallback_score: float,
    effective_buy_threshold: float,
    spread_bps: float,
) -> tuple[str, ...]:
    reasons: list[str] = []
    if prediction_error is not None:
        raw = str(prediction_error) or prediction_error.__class__.__name__
        reasons.append("MODEL_FEATURE_UNAVAILABLE:" + raw[:96])
    max_spread = float(getattr(policy, "max_spread_bps", 0.0) or 0.0)
    if max_spread > 0 and spread_bps > max_spread:
        reasons.append(f"WIDE_SPREAD:{spread_bps:.1f}>{max_spread:.1f}bps")
    regime_notes = tuple(((policy_diag.get("regime") or {}).get("notes") or ()))
    if "LOW_LIQUIDITY" in regime_notes:
        reasons.append("LOW_LIQUIDITY")
    if fallback_score < effective_buy_threshold:
        reasons.append(f"FALLBACK_SCORE_BELOW_THRESHOLD:{fallback_score:.3f}<{effective_buy_threshold:.3f}")
    return tuple(dict.fromkeys(reasons))


def _resolve_model_provider(
    prediction: LiveSignalPrediction | None, model_ok: bool
) -> tuple[str, bool]:
    """Provider/fallback state for the buy decision (Phase 5 visibility).

    - ``trained_model``: the fitted live-eligible artifact drove the decision.
    - ``heuristic_fallback``: the model ran but was not approved, so the ontology /
      adaptive fallback score drove the decision.
    - ``unavailable``: the model produced no prediction at all (missing artifact,
      schema mismatch, or inference disabled) and the heuristic fallback is used.

    ``is_fallback`` is True whenever the trained model did not drive the decision, so
    the GUI can surface fallback mode and sizing/confidence stay conservative.
    """
    if prediction is not None and model_ok:
        return str(getattr(prediction, "provider", "trained_model") or "trained_model"), False
    if prediction is not None:
        return "heuristic_fallback", True
    return "unavailable", True


def _volatility_no_trade_reasons(policy_diag: dict[str, Any]) -> tuple[str, ...]:
    market_state = policy_diag.get("market_state") or {}
    symbol_volatility = float(market_state.get("symbol_volatility", 0.0) or 0.0)
    market_volatility = float(market_state.get("market_volatility", 0.0) or 0.0)
    max_symbol = _env_float("REALTIME_MAX_SYMBOL_VOLATILITY_BUY", 0.015)
    max_market = _env_float("REALTIME_MAX_MARKET_VOLATILITY_BUY", 0.008)
    reasons: list[str] = []
    if symbol_volatility > max_symbol:
        reasons.append(f"SYMBOL_VOLATILITY_TOO_HIGH:{symbol_volatility:.4f}>{max_symbol:.4f}")
    if market_volatility > max_market:
        reasons.append(f"MARKET_VOLATILITY_TOO_HIGH:{market_volatility:.4f}>{max_market:.4f}")
    return tuple(reasons)


@dataclass(frozen=True)
class SharedDecisionResult:
    symbol: str
    approved: bool
    final_order: FinalOrder | None
    prediction: LiveSignalPrediction | None
    reason_codes: tuple[str, ...]
    diagnostics: dict[str, Any] = field(default_factory=dict)


class SharedLiveDecisionEngine:
    def __init__(
        self,
        store: RealtimeMarketDataStore,
        *,
        predictor: LiveSignalPredictor | None = None,
        risk_manager: RiskManager | None = None,
        market_refresher: Callable[[str, str, datetime], MarketSnapshot | None] | None = None,
        decision_logger: DecisionLogger | None = None,
    ) -> None:
        self.store = store
        self.feature_builder = LiveFeatureFrameBuilder(store)
        self.predictor = predictor or LiveSignalPredictor()
        self.risk_manager = risk_manager or RiskManager(
            RiskRules(live_trading_enabled=False, min_average_daily_trading_value=1, max_volatility=1.0)
        )
        self.auto_tuner = AutoTuningEngine(decision_logger=decision_logger, refresh_quote=market_refresher)
        self.profitability_gate = ProfitabilityGate()
        self.dynamic_exit_policy = DynamicExitPolicy()
        self.position_sizer = PositionSizer()
        self.market_refresher = market_refresher
        self.decision_logger = decision_logger or DecisionLogger()
        # Advisory technical prediction layer. Produces a conservative expected
        # exit price / methodology / regime for the ProfitabilityGate and GUI.
        # It never approves or blocks on its own — the gate + RiskManager remain
        # authoritative. Disabled cleanly if the policy or import is unavailable.
        try:
            from app.technical.policy import build_prediction_engine, load_technical_policy

            self._technical_policy = load_technical_policy()
            self._technical_engine = build_prediction_engine(self._technical_policy)
            self._technical_enabled = bool(self._technical_policy.enabled)
        except Exception:  # noqa: BLE001 - technical layer is strictly advisory.
            self._technical_policy = None
            self._technical_engine = None
            self._technical_enabled = False
        self._last_diagnostics: dict[str, Any] = {}
        # Per-symbol peak net (after-cost) PnL rate, used by the profit-giveback
        # trailing lock so realized gains are not given back on a stall.
        self._peak_net_pnl: dict[str, float] = {}

    def _technical_exit_deterioration(self, frame, symbol) -> tuple[tuple[str, ...], float]:
        """Advisory technical deterioration codes + a bounded ontology penalty.

        Best-effort (never raises). Returns ``((), 0.0)`` when disabled or the
        frame is missing. The penalty is capped so it can tip a *profitable*
        position into an earlier exit but cannot by itself force a loss exit.
        """
        if not self._technical_enabled or self._technical_engine is None or frame is None:
            return (), 0.0
        try:
            from app.technical.feature_builder import technical_feature_set_from_live_frame

            features = technical_feature_set_from_live_frame(frame, symbol)
            codes = self._technical_engine.signal_engine.evaluate_exit_deterioration(features)
            if not codes:
                return (), 0.0
            penalty = min(0.5, 0.15 * len(codes))
            return codes, penalty
        except Exception:  # noqa: BLE001 - advisory only.
            return (), 0.0

    def _technical_prediction(self, frame, symbol, *, model_prediction=None):
        """Best-effort advisory technical prediction from the live frame.

        Returns ``None`` (never raises) when disabled or the frame is missing —
        the caller then keeps its existing conservative behavior.
        """
        if not self._technical_enabled or self._technical_engine is None or frame is None:
            return None
        try:
            from app.technical.feature_builder import technical_feature_set_from_live_frame

            features = technical_feature_set_from_live_frame(frame, symbol)
            return self._technical_engine.predict(features, model_prediction=model_prediction)
        except Exception:  # noqa: BLE001 - advisory only; never break the decision path.
            return None

    def evaluate_buy(
        self,
        symbol: str,
        account: AccountSnapshot,
        *,
        suggested_weight: float = 0.01,
        ontology_graph: Any | None = None,
        decision_time: datetime | None = None,
    ) -> SharedDecisionResult:
        decision_time = decision_time or datetime.now(timezone.utc)
        frame = None
        prediction: LiveSignalPrediction | None = None
        prediction_error: Exception | None = None
        try:
            frame = self.feature_builder.build(symbol, decision_time=decision_time)
            prediction = self.predictor.predict(frame)
        except Exception as exc:  # noqa: BLE001 - model failure can fall back to ontology and rules.
            prediction_error = exc
        technical_prediction = self._technical_prediction(frame, symbol, model_prediction=prediction)

        tick = self.store.latest_tick(symbol)
        market_name = _resolve_order_market(symbol, account)
        quote_refresh_status = "quote_refresh_skipped"
        refreshed_market: MarketSnapshot | None = None
        if tick is None or float(getattr(tick, "price", 0.0) or 0.0) <= 0:
            if self.market_refresher is not None:
                quote_refresh_status = "quote_refresh_attempted"
                try:
                    refreshed_market = self.market_refresher(symbol, market_name, decision_time)
                except Exception:  # noqa: BLE001 - refresh is best-effort.
                    refreshed_market = None
                if refreshed_market is not None and float(getattr(refreshed_market, "last_price", 0.0) or 0.0) > 0:
                    quote_refresh_status = "quote_refresh_ok"
                else:
                    refreshed_market = None
            if refreshed_market is None:
                result = SharedDecisionResult(symbol, False, None, prediction, ("MISSING_MARKET_DATA",), {"quote_refresh_status": "missing_market_data"})
                self._last_diagnostics = result.diagnostics or {}
                return result

        currency = "KRW" if market_name.upper() in ("KR", "KRX", "KOSPI", "KOSDAQ", "KONEX") else "USD"
        cash_by_currency = account.cash_by_currency if hasattr(account, "cash_by_currency") else {}
        available_cash = float(cash_by_currency.get(currency, 0.0) or 0.0)
        if available_cash <= 0.0:
            # Fall back to the account's base cash when the per-currency bucket is
            # absent (snapshots that only populate `cash`). Only for the base
            # currency, so a USD order never borrows KRW cash. Mirrors
            # RiskManager._cash_available_for_market so the buy pre-check and the
            # final risk gate agree on available cash.
            base_currency = str(getattr(account, "base_currency", "KRW") or "KRW").upper()
            if currency == base_currency:
                available_cash = max(
                    available_cash,
                    float(getattr(account, "pure_cash", 0.0) or 0.0),
                    float(getattr(account, "cash", 0.0) or 0.0),
                )
        domestic_drawdown_rate = (
            _account_domestic_unrealized_rate(account)
            if _is_domestic_symbol_or_market(symbol, market_name)
            else 0.0
        )

        orderbook = self.store.latest_orderbook(symbol) if hasattr(self.store, "latest_orderbook") else None
        tick_received_at = getattr(tick, "received_at", decision_time) if tick is not None else decision_time
        quote_age_seconds = 0.0 if refreshed_market is not None else max(0.0, (decision_time - tick_received_at).total_seconds())
        price = float(getattr(refreshed_market, "last_price", 0.0) or getattr(tick, "price", 0.0) or 0.0)
        # Cash headroom required to afford one share (covers small tick moves between the
        # decision and the fill). Tunable so a very small account is not locked out by a
        # buffer it cannot spare: REALTIME_ONE_SHARE_CASH_BUFFER (default 1.05 = +5%).
        one_share_buffer = max(1.0, _env_float("REALTIME_ONE_SHARE_CASH_BUFFER", 1.05))
        min_cash_for_one_share = price * one_share_buffer
        needs_cash_check_refresh = available_cash < min_cash_for_one_share
        if (
            self.market_refresher is not None
            and refreshed_market is None
            and (
                quote_age_seconds > float(os.getenv("REALTIME_BUY_MAX_QUOTE_AGE_SEC", "12"))
                or needs_cash_check_refresh
            )
        ):
            quote_refresh_status = "quote_refresh_attempted"
            try:
                refreshed_market = self.market_refresher(symbol, market_name, decision_time)
            except Exception:  # noqa: BLE001 - refresh is best-effort.
                refreshed_market = None
            if refreshed_market is not None:
                quote_refresh_status = "quote_refresh_ok"
                price = float(getattr(refreshed_market, "last_price", 0.0) or price)
                min_cash_for_one_share = price * one_share_buffer

        if available_cash < min_cash_for_one_share:
            result = SharedDecisionResult(
                symbol,
                False,
                None,
                prediction,
                ("INSUFFICIENT_CASH_FOR_ONE_SHARE", f"QUOTE_REFRESH:{quote_refresh_status}"),
                {
                    "available_cash": available_cash,
                    "currency": currency,
                    "min_required": min_cash_for_one_share,
                    "price": price,
                    "quote_refresh_status": quote_refresh_status,
                },
            )
            self._last_diagnostics = result.diagnostics or {}
            return result
        market = refreshed_market or MarketSnapshot(
            ticker=symbol,
            market=market_name,
            company_name=symbol,
            sector="Unknown",
            last_price=float(getattr(tick, "price", 0.0) or 0.0),
            average_daily_trading_value=10_000_000_000,
            volatility_20d=0.02,
            source=SourceMetadata(
                source_name="KIS realtime WebSocket",
                retrieved_at=getattr(tick, "received_at", decision_time),
                observed_at=getattr(tick, "exchange_timestamp", decision_time),
                source_type="broker_api",
                trust_level=5,
                is_realtime=True,
                quality_score=1.0,
            ),
        )

        flow_score = 0.0
        edge_score = 0.0
        ontology_support: tuple[str, ...] = ()
        if ontology_graph is not None:
            try:
                flow_score, flow_support, _contra = _ontology_flow_adjustment(ontology_graph, symbol)
                edge_score, edge_support = _ontology_buy_evidence(ontology_graph, symbol)
                ontology_support = tuple(dict.fromkeys((*flow_support, *edge_support)))
            except Exception:  # noqa: BLE001 - ontology is an enhancer, never fatal.
                flow_score = edge_score = 0.0
        volume_ratio = self._realtime_volume_surge_ratio(symbol, decision_time)
        if volume_ratio >= float(os.getenv("REALTIME_VOLUME_SURGE_RATIO", "1.5")):
            edge_score += 1.0
            ontology_support = (*ontology_support, f"VolumeSurge:{volume_ratio:.1f}x")
        # 뉴스 감성: 소프트 확인용. 이미 다른 매수근거(수급/모멘텀/유동성/거래량)가 있는
        # 후보에 한해 긍정 뉴스가 온톨로지 점수를 소폭 부스팅한다. 뉴스 단독으로는 근거가
        # 되지 않으며(ontology_ok를 직접 켜지 않음), 부정 뉴스가 함께 있으면 부스팅하지 않는다.
        # 부정 뉴스는 _ontology_buy_evidence의 increasesRiskOf 차감으로 이미 반영된다.
        news_confirm_bonus = 0.0
        if (
            os.getenv("REALTIME_NEWS_SENTIMENT_ENABLED", "true").strip().lower() in {"1", "true", "yes", "on"}
            and ontology_graph is not None
            and (flow_score > 0.0 or edge_score > 0.0)
        ):
            try:
                has_positive_news, has_negative_news = _news_sentiment_flags(ontology_graph, symbol)
            except Exception:  # noqa: BLE001 - news is an enhancer, never fatal.
                has_positive_news = has_negative_news = False
            if has_positive_news and not has_negative_news:
                # Base bonus scaled by the outcome-calibrated trust learned from realized PnL.
                news_confirm_bonus = max(0.0, _env_float("REALTIME_NEWS_CONFIRM_BONUS", 0.15)) * _news_confirm_scale()
                if news_confirm_bonus > 0.0:
                    ontology_support = (*ontology_support, "PositiveNewsConfirm")
        flow_threshold = float(os.getenv("REALTIME_ONTOLOGY_BUY_SCORE", "0.12"))
        edge_threshold = float(os.getenv("REALTIME_ONTOLOGY_BUY_MIN_SUPPORTS", "0.5"))
        ontology_score = max(flow_score, edge_score) + news_confirm_bonus
        ontology_ok = flow_score >= flow_threshold or edge_score >= edge_threshold
        require_ontology_fallback = str(os.getenv("REALTIME_REQUIRE_ONTOLOGY_FOR_MODEL_FALLBACK", "true")).lower() in {
            "1",
            "true",
            "yes",
            "on",
        }

        price = float(getattr(market, "last_price", 0.0) or 0.0)
        liquidity_score = min(1.0, math.log1p(max(0.0, market.average_daily_trading_value)) / math.log1p(10_000_000_000))
        spread_bps = 0.0
        if orderbook is not None:
            spread_bps = max(0.0, float(getattr(orderbook, "spread_bps", 0.0) or 0.0))
            liquidity_score = min(1.0, (float(getattr(orderbook, "total_bid_volume", 0.0) or 0.0) + float(getattr(orderbook, "total_ask_volume", 0.0) or 0.0)) / 1_000_000)

        fallback_score = self.auto_tuner.fallback_buy_score(
            ontology_score=ontology_score,
            technical_momentum=float(getattr(prediction, "probability_success", 0.5) - 0.5 if prediction is not None else 0.0),
            liquidity_score=liquidity_score,
            spread_bps=spread_bps,
            volatility=float(getattr(market, "volatility_20d", 0.0) or 0.0),
            recent_performance=0.0,
        )
        model_ok = bool(prediction is not None and prediction.approved)
        model_provider, model_is_fallback = _resolve_model_provider(prediction, model_ok)
        fallback_allowed = True
        policy_state = self.auto_tuner.snapshot_market_state(
            symbol=symbol,
            market=market,
            quote_age_seconds=quote_age_seconds if refreshed_market is None else 0.0,
            spread_bps=spread_bps,
            orderbook_available=orderbook is not None,
            volume_ratio=volume_ratio,
            recent_performance=0.0,
            fallback_score=fallback_score,
            symbol_volatility=self._symbol_realtime_volatility(symbol, decision_time),
            market_volatility=self._market_realtime_volatility(decision_time),
        )
        policy, policy_diag = self.auto_tuner.build_buy_policy(
            symbol=symbol,
            account=account,
            market=market,
            market_state=policy_state,
            prediction=prediction,
            fallback_allowed=fallback_allowed,
            ontology_score=ontology_score,
            fallback_score=fallback_score,
            prediction_confidence=float(getattr(prediction, "probability_success", 0.5) or 0.5),
            prediction_error=prediction_error,
            decision_time=decision_time,
        )
        volatility_reasons = _volatility_no_trade_reasons(policy_diag)
        if volatility_reasons:
            diagnostics = {
                "policy": policy.as_dict(),
                "policy_state": policy_diag,
                "quote_refresh_status": quote_refresh_status,
                "fallback_score": fallback_score,
                "ontology_score": ontology_score,
                "spread_bps": spread_bps,
            }
            self._last_diagnostics = diagnostics
            return SharedDecisionResult(
                symbol,
                False,
                None,
                prediction,
                (*volatility_reasons, f"QUOTE_REFRESH:{quote_refresh_status}"),
                diagnostics,
            )
        domestic_buy_threshold_bonus = 0.0
        if domestic_drawdown_rate < 0:
            drawdown_trigger = abs(_env_float("REALTIME_DOMESTIC_DRAWDOWN_BUY_TIGHTEN_TRIGGER", 0.005))
            if abs(domestic_drawdown_rate) >= drawdown_trigger:
                domestic_buy_threshold_bonus = min(
                    _env_float("REALTIME_DOMESTIC_DRAWDOWN_BUY_MAX_BONUS", 0.18),
                    abs(domestic_drawdown_rate) * _env_float("REALTIME_DOMESTIC_DRAWDOWN_BUY_BONUS_MULTIPLIER", 6.0),
                )
        effective_buy_threshold = policy.buy_threshold + domestic_buy_threshold_bonus
        runtime_execution_ready = (
            price > 0.0
            and available_cash >= min_cash_for_one_share
            and quote_refresh_status == "quote_refresh_ok"
            and str(getattr(market.source, "source_type", "") or "") == "broker_api"
            and bool(getattr(market.source, "is_realtime", False))
            and float(getattr(market.source, "quality_score", 0.0) or 0.0) >= 0.8
        )
        runtime_fallback_support = False
        runtime_probe_support = False
        runtime_probe_margin = _env_float("REALTIME_RUNTIME_PROBE_BUY_MARGIN", 0.18)
        if (
            not model_ok
            and runtime_execution_ready
            and fallback_score >= effective_buy_threshold
            and policy.allowed_fallback_mode != "no_trade"
        ):
            runtime_fallback_support = True
            ontology_ok = True
            ontology_score = max(ontology_score, 1.0)
            ontology_support = tuple(
                dict.fromkeys(
                    (
                        *ontology_support,
                        "FreshBrokerQuote",
                        "CashFitOneShare",
                        "ExecutableBuyCandidate",
                        "RealtimeAdaptiveFallbackBuyCandidate",
                    )
                )
            )
        elif (
            not model_ok
            and runtime_execution_ready
            and policy.allowed_fallback_mode != "no_trade"
            and os.getenv("REALTIME_RUNTIME_PROBE_BUY_ENABLED", "true").strip().lower() in {"1", "true", "yes", "on"}
            and fallback_score >= max(0.0, effective_buy_threshold - runtime_probe_margin)
        ):
            runtime_probe_support = True
            ontology_ok = True
            ontology_score = max(ontology_score, 0.35)
            ontology_support = tuple(
                dict.fromkeys(
                    (
                        *ontology_support,
                        "FreshBrokerQuote",
                        "CashFitOneShare",
                        "RuntimeProbeBuyCandidate",
                    )
                )
            )
        if not model_ok and require_ontology_fallback and not ontology_ok:
            reasons = tuple(getattr(prediction, "reason_codes", ()) or ("MODEL_UNAVAILABLE",))
            blocker_reasons = _buy_blocker_reason_codes(
                prediction_error=prediction_error,
                policy=policy,
                policy_diag=policy_diag,
                fallback_score=fallback_score,
                effective_buy_threshold=effective_buy_threshold,
                spread_bps=spread_bps,
            )
            reasons = (
                *reasons,
                *blocker_reasons,
                "ONTOLOGY_REQUIRED_FOR_MODEL_FALLBACK",
                f"QUOTE_REFRESH:{quote_refresh_status}",
            )
            diagnostics = {
                "policy": policy.as_dict(),
                "policy_state": policy_diag,
                "quote_refresh_status": quote_refresh_status,
                "fallback_score": fallback_score,
                "ontology_score": ontology_score,
                "runtime_execution_ready": runtime_execution_ready,
                "effective_buy_threshold": effective_buy_threshold,
                "spread_bps": spread_bps,
                "model_provider": model_provider,
                "model_is_fallback": model_is_fallback,
            }
            self._last_diagnostics = diagnostics
            return SharedDecisionResult(symbol, False, None, prediction, reasons, diagnostics)
        if not model_ok and policy.allowed_fallback_mode == "no_trade" and fallback_score < policy.buy_threshold:
            reasons = tuple(getattr(prediction, "reason_codes", ()) or ("MODEL_UNAVAILABLE",))
            reasons = (*reasons, "MODEL_FALLBACK_NOT_ALLOWED", f"QUOTE_REFRESH:{quote_refresh_status}")
            diagnostics = {"policy": policy.as_dict(), "policy_state": policy_diag, "quote_refresh_status": quote_refresh_status, "fallback_score": fallback_score}
            self._last_diagnostics = diagnostics
            return SharedDecisionResult(symbol, False, None, prediction, reasons, diagnostics)
        model_auxiliary_only = os.getenv("REALTIME_MODEL_AUXILIARY_ONLY", "true").strip().lower() in {"1", "true", "yes", "on"}
        if model_ok and model_auxiliary_only and not ontology_ok and not runtime_fallback_support and not runtime_probe_support:
            diagnostics = {
                "policy": policy.as_dict(),
                "policy_state": policy_diag,
                "quote_refresh_status": quote_refresh_status,
                "fallback_score": fallback_score,
                "ontology_score": ontology_score,
                "model_auxiliary_only": True,
                "model_probability_success": float(getattr(prediction, "probability_success", 0.0) or 0.0),
                "model_expected_net_return_bps": float(getattr(prediction, "expected_net_return_bps", 0.0) or 0.0),
            }
            self._last_diagnostics = diagnostics
            return SharedDecisionResult(
                symbol,
                False,
                None,
                prediction,
                ("MODEL_AUXILIARY_ONLY_NEEDS_CONFIRMATION", f"QUOTE_REFRESH:{quote_refresh_status}"),
                diagnostics,
            )

        model_score = float(getattr(prediction, "probability_success", 0.0) or 0.0) if model_ok else fallback_score
        signal_score = max(model_score, ontology_score * 0.35 if not model_ok else ontology_score * 0.25)
        if not model_ok:
            signal_score = max(signal_score, fallback_score)
        signal_gap = signal_score - effective_buy_threshold
        size_multiplier = 1.0
        if signal_gap < 0:
            min_signal_gap = -runtime_probe_margin if runtime_probe_support else -0.18
            if signal_gap >= min_signal_gap and policy.allowed_fallback_mode != "no_trade":
                size_multiplier = max(0.20, 1.0 + signal_gap * 2.5)
            else:
                reasons = tuple(getattr(prediction, "reason_codes", ()) or ())
                reasons = (*reasons, "ONTOLOGY_BELOW_ADAPTIVE_THRESHOLD" if ontology_ok is False else "BUY_SIGNAL_TOO_WEAK", f"QUOTE_REFRESH:{quote_refresh_status}")
                diagnostics = {"policy": policy.as_dict(), "policy_state": policy_diag, "quote_refresh_status": quote_refresh_status, "fallback_score": fallback_score, "signal_score": signal_score}
                self._last_diagnostics = diagnostics
                return SharedDecisionResult(symbol, False, None, prediction, reasons, diagnostics)

        suggested_weight = max(0.001, suggested_weight * size_multiplier)
        if runtime_probe_support and not runtime_fallback_support:
            suggested_weight = min(suggested_weight, _env_float("REALTIME_RUNTIME_PROBE_BUY_WEIGHT", 0.003))
        # --- Honest expected return (no fabricated positive floor) --------------
        # The expected exit price handed to the ProfitabilityGate must reflect a REAL
        # predicted edge, not the old synthetic 100bps / fallback*300bps floor that
        # forced every candidate to look profitable. When the trained model is
        # available we use its cost-adjusted net-return estimate; otherwise we use a
        # conservative estimate derived from the adaptive fallback score. Either way
        # the gate below judges it honestly against the real round-trip cost and a
        # dynamic minimum net edge, and REJECTS negative-expectancy buys.
        if model_ok and prediction is not None and float(prediction.expected_net_return_bps or 0.0) > 0.0:
            expected_return_bps = float(prediction.expected_net_return_bps)
        else:
            fallback_edge_bps_per_score = _env_float("REALTIME_FALLBACK_EDGE_BPS_PER_SCORE", 120.0)
            expected_return_bps = max(0.0, fallback_score) * fallback_edge_bps_per_score
        # --- Technical prediction (preferred, conservative) ---------------------
        # When the advisory technical layer yields a tradable BUY, prefer its
        # cost-aware net edge for the gate's expected exit price. It has NO
        # fabricated floor, so we never inflate: when the model edge is also
        # positive we take the more conservative (lower) of the two; the gate and
        # RiskManager still judge it. A blocked/NO_TRADE technical prediction only
        # surfaces reason codes — it does not itself veto (gates stay authoritative).
        technical_methodology = None
        technical_regime = None
        if technical_prediction is not None:
            technical_regime = technical_prediction.regime
            if technical_prediction.tradable and float(technical_prediction.expected_net_return_bps or 0.0) > 0.0:
                tech_bps = float(technical_prediction.expected_net_return_bps)
                expected_return_bps = min(expected_return_bps, tech_bps) if expected_return_bps > 0 else tech_bps
                technical_methodology = technical_prediction.methodology
        gross_expected_return = max(0.0, expected_return_bps / 10_000.0)
        expected_exit_price = price * (1.0 + gross_expected_return)

        # --- Unified profitability gate (authoritative, mandatory) --------------
        gate_venue, gate_instrument = _cost_context_for_holding(symbol, market_name)
        gate_orderbook = None
        if orderbook is not None:
            gate_orderbook = {
                "best_bid": float(getattr(orderbook, "best_bid", 0.0) or 0.0),
                "best_ask": float(getattr(orderbook, "best_ask", 0.0) or 0.0),
            }
        profitability_decision = self.profitability_gate.evaluate(
            ProfitabilityInput(
                symbol=symbol,
                action="BUY",
                market=market_name,
                venue=gate_venue,
                instrument_type=gate_instrument,
                entry_price=price,
                expected_exit_price=expected_exit_price,
                quantity=1,
                spread_rate=(spread_bps / 10_000.0) if spread_bps > 0 else None,
                liquidity_score=liquidity_score,
                realized_volatility=self._symbol_realtime_volatility(symbol, decision_time),
                orderbook_snapshot=gate_orderbook,
                average_daily_trading_value=float(getattr(market, "average_daily_trading_value", 0.0) or 0.0),
                account_equity_krw=float(getattr(account, "equity", 0.0) or 0.0),
            )
        )
        if not profitability_decision.allowed:
            reasons = (
                *profitability_decision.rejection_reasons,
                "PROFITABILITY_GATE_REJECTED",
                f"QUOTE_REFRESH:{quote_refresh_status}",
            )
            diagnostics = {
                "policy": policy.as_dict(),
                "policy_state": policy_diag,
                "quote_refresh_status": quote_refresh_status,
                "fallback_score": fallback_score,
                "ontology_score": ontology_score,
                "spread_bps": spread_bps,
                "profitability_decision": profitability_decision.as_dict(),
                "model_provider": model_provider,
                "model_is_fallback": model_is_fallback,
                "technical_prediction": technical_prediction.as_dict() if technical_prediction is not None else None,
                "technical_methodology": technical_methodology,
                "technical_regime": technical_regime,
            }
            self._last_diagnostics = diagnostics
            return SharedDecisionResult(symbol, False, None, prediction, reasons, diagnostics)
        confidence = max(policy.confidence_floor, float(getattr(prediction, "probability_success", 0.5) or 0.5) if model_ok else 0.5 + fallback_score * 0.2)
        if not model_ok:
            confidence = max(0.35, confidence - 0.1)
        # Edge/confidence-aware position sizing (Phase 4). Caps the weight by a
        # fractional-Kelly, edge-, liquidity-, and drawdown-scaled size. Never sizes a
        # negative-expectancy trade (the ProfitabilityGate already blocked those). The
        # RiskManager one-share bump still guarantees a whole share when affordable.
        sizing = self.position_sizer.size(
            SizingInputs(
                net_expected_return=profitability_decision.net_expected_return,
                target_net_return=profitability_decision.required_min_net_return,
                confidence_score=confidence,
                liquidity_score=liquidity_score,
                account_drawdown_rate=domestic_drawdown_rate,
                recent_same_strategy_loss=False,
            )
        )
        if sizing.position_weight > 0.0:
            suggested_weight = min(suggested_weight, sizing.position_weight)
        signal_name = "trained_expected_net_return" if model_ok else "ontology_fallback_buy"
        supporting = ("trained_live_model",) if model_ok else ("ontology_fallback",)
        if ontology_support:
            supporting = (*supporting, *ontology_support)
        reasoning = f"policy:{policy.risk_mode};score={signal_score:.2f};quote={quote_refresh_status}"
        artifact_id = str(getattr(prediction, "model_artifact_id", "") or "") if prediction is not None else ""
        validation_id = artifact_id or f"adaptive-buy:{symbol}:{decision_time.strftime('%Y%m%d%H%M%S')}"
        source_data_ids = (
            frame.provenance.source_record_ids
            if frame is not None
            else (str(getattr(tick, "sequence_key", "") if tick is not None else "") or f"quote:{symbol}:{quote_refresh_status}",)
        )
        strategy_metadata: dict[str, Any] = {
            "model_artifact_id": artifact_id,
            "model_provider": model_provider,
            "model_is_fallback": model_is_fallback,
            "feature_schema_hash": prediction.feature_schema_hash if prediction is not None else "",
            "ontology_buy_score": round(ontology_score, 4),
            "fallback_score": round(fallback_score, 4),
            "runtime_execution_ready": runtime_execution_ready,
            "runtime_fallback_support": runtime_fallback_support,
            "runtime_probe_support": runtime_probe_support,
            "model_auxiliary_only": model_auxiliary_only,
            "buy_threshold": policy.buy_threshold,
            "effective_buy_threshold": round(effective_buy_threshold, 6),
            "domestic_drawdown_rate": round(domestic_drawdown_rate, 6),
            "domestic_buy_threshold_bonus": round(domestic_buy_threshold_bonus, 6),
            "policy": policy.as_dict(),
            "policy_state": policy_diag,
            "quote_refresh_status": quote_refresh_status,
            "quote_age_seconds": round(quote_age_seconds, 3),
            "stop_loss_price": price * (1.0 - policy.stop_loss),
            "profitability_decision": profitability_decision.as_dict(),
            "position_sizing": sizing.as_dict(),
        }
        if orderbook is not None:
            strategy_metadata["orderbook_snapshot"] = {
                "best_bid": orderbook.best_bid,
                "best_ask": orderbook.best_ask,
                "bid_depth": orderbook.total_bid_volume,
                "ask_depth": orderbook.total_ask_volume,
            }
        intent = OrderIntent(
            ticker=symbol,
            market=market_name,
            action=OrderAction.BUY,
            suggested_weight=min(suggested_weight, float(policy.max_position_size) / max(1.0, float(account.equity or 0.0))),
            confidence=confidence,
            valid_until=decision_time + timedelta(seconds=max(30, int(policy.quote_ttl_seconds))),
            reasoning_summary=(reasoning,),
            supporting_factors=supporting,
            contradicting_factors=(),
            source_data_ids=source_data_ids,
            model_uncertainty=prediction.uncertainty_score if prediction is not None else (0.85 if not model_ok else None),
            strategy_family="live_short_horizon",
            signal_name=signal_name,
            expected_exit_price=expected_exit_price,
            expected_holding_minutes=max(1, min(30, int(policy.time_exit_seconds / 60))),
            gross_expected_return=gross_expected_return,
            target_net_return=profitability_decision.required_min_net_return,
            validation_id=validation_id,
            strategy_metadata=strategy_metadata,
        )
        adaptive_rules = self.auto_tuner.derive_risk_rules(
            self.risk_manager.rules,
            policy=policy,
            account=account,
            market=market,
            model_uncertainty=prediction.uncertainty_score if prediction is not None else None,
        )
        adaptive_rules = replace(adaptive_rules, minimum_cash_reserve=0.0)
        risk_manager = RiskManager(adaptive_rules, audit_logger=self.risk_manager.audit_logger)
        risk = risk_manager.validate(intent, account, market)
        diagnostics = {
            "policy": policy.as_dict(),
            "policy_state": policy_diag,
            "quote_refresh_status": quote_refresh_status,
            "fallback_score": fallback_score,
            "model_ok": model_ok,
            "model_provider": model_provider,
            "model_is_fallback": model_is_fallback,
            "signal_score": signal_score,
            "buy_threshold": policy.buy_threshold,
            "effective_buy_threshold": effective_buy_threshold,
            "domestic_drawdown_rate": domestic_drawdown_rate,
            "domestic_buy_threshold_bonus": domestic_buy_threshold_bonus,
            "runtime_execution_ready": runtime_execution_ready,
            "runtime_fallback_support": runtime_fallback_support,
            "runtime_probe_support": runtime_probe_support,
            "model_auxiliary_only": model_auxiliary_only,
            "adaptive_risk_rules": adaptive_rules,
            "risk_metadata": risk.metadata,
            "profitability_decision": profitability_decision.as_dict(),
            "technical_prediction": technical_prediction.as_dict() if technical_prediction is not None else None,
            "technical_methodology": technical_methodology,
            "technical_regime": technical_regime,
        }
        self.auto_tuner.record_feedback(
            {
                "symbol": symbol,
                "side": "BUY",
                "approved": risk.approved,
                "reason_codes": risk.rejection_reasons,
                "policy": policy.as_dict(),
                "pnl": 0.0,
                "quote_refresh_status": quote_refresh_status,
            }
        )
        self._last_diagnostics = diagnostics
        return SharedDecisionResult(
            symbol=symbol,
            approved=risk.approved and risk.final_order is not None,
            final_order=risk.final_order,
            prediction=prediction,
            reason_codes=risk.rejection_reasons,
            diagnostics=diagnostics,
        )

    def evaluate_exit_for_holding(
        self,
        holding: Holding,
        account: AccountSnapshot,
        *,
        take_profit: float = 0.0025,
        stop_loss: float = 0.010,
        ontology_graph: Any | None = None,
        decision_time: datetime | None = None,
    ) -> SharedDecisionResult:
        symbol = holding.ticker
        decision_time = decision_time or datetime.now(timezone.utc)
        avg_cost = float(getattr(holding, "average_price", 0.0) or 0.0)
        if avg_cost <= 0:
            result = SharedDecisionResult(symbol, False, None, None, ("INVALID_PRICE_OR_COST",), {"exit_reason": "invalid_price"})
            self._last_diagnostics = result.diagnostics or {}
            return result
        if int(getattr(holding, "quantity", 0) or 0) <= 0:
            result = SharedDecisionResult(symbol, False, None, None, ("NO_POSITION",), {"exit_reason": "no_position"})
            self._last_diagnostics = result.diagnostics or {}
            return result

        price, observed_at, received_at, source_id = self._exit_price_source(symbol, holding, decision_time)
        if price <= 0 and self.market_refresher is not None:
            try:
                refreshed = self.market_refresher(symbol, holding.market or "KR", decision_time)
            except Exception:  # noqa: BLE001 - refresh is best-effort.
                refreshed = None
            if refreshed is not None and float(refreshed.last_price or 0.0) > 0:
                price = float(refreshed.last_price)
                observed_at = refreshed.source.observed_at or refreshed.source.retrieved_at
                received_at = refreshed.source.retrieved_at
                source_id = refreshed.source.source_id or f"refreshed:{symbol}"
        if price > 0 and self.market_refresher is not None:
            stale_seconds = max(0.0, (decision_time - received_at).total_seconds())
            if stale_seconds > float(os.getenv("REALTIME_EXIT_MAX_QUOTE_AGE_SEC", "12")):
                try:
                    refreshed = self.market_refresher(symbol, holding.market or "KR", decision_time)
                except Exception:  # noqa: BLE001 - refresh is best-effort.
                    refreshed = None
                if refreshed is not None and float(refreshed.last_price or 0.0) > 0:
                    price = float(refreshed.last_price)
                    observed_at = refreshed.source.observed_at or refreshed.source.retrieved_at
                    received_at = refreshed.source.retrieved_at
                    source_id = refreshed.source.source_id or f"refreshed:{symbol}"
        if price <= 0:
            result = SharedDecisionResult(symbol, False, None, None, ("MISSING_MARKET_DATA",), {"exit_reason": "missing_market_data"})
            self._last_diagnostics = result.diagnostics or {}
            return result

        pnl_rate = (price - avg_cost) / avg_cost
        ontology_score = 0.0
        ontology_support: tuple[str, ...] = ()
        if ontology_graph is not None:
            try:
                position_weight = (
                    float(holding.market_value) / max(1.0, float(account.equity))
                    if account is not None and account.equity > 0
                    else 0.0
                )
                ontology_score, ontology_support, _onto_contra = _holding_exit_adjustment(
                    ontology_graph, symbol, position_weight, holding
                )
            except Exception:  # noqa: BLE001 - ontology is an enhancer; never block exits on it.
                ontology_score = 0.0

        # Advisory technical deterioration evidence (Phase 8). Strong deterioration
        # (VWAP breakdown, momentum loss, volatility expansion, false-breakout,
        # liquidity drop) lowers the effective ontology score so a *profitable*
        # position can be exited earlier via the existing invalid-signal branch.
        # It NEVER forces a loss exit — hard/emergency stops remain the sole
        # circuit breakers, and the loss-exit gate is unchanged.
        technical_exit_codes: tuple[str, ...] = ()
        technical_exit_penalty = 0.0
        exit_frame = None
        try:
            exit_frame = self.feature_builder.build(symbol, decision_time=decision_time)
        except Exception:  # noqa: BLE001 - frame is best-effort for advisory deterioration.
            exit_frame = None
        technical_exit_codes, technical_exit_penalty = self._technical_exit_deterioration(exit_frame, symbol)
        if technical_exit_codes:
            ontology_score -= technical_exit_penalty

        target_net_return = max(0.0, float(os.getenv("REALTIME_EXIT_TARGET_NET_RETURN", "0.0003")))
        volume_ratio = self._realtime_volume_surge_ratio(symbol, decision_time)
        market = self._exit_market_snapshot(holding, price, observed_at, received_at)
        quote_age_seconds = max(0.0, (decision_time - received_at).total_seconds())
        orderbook = self.store.latest_orderbook(symbol) if hasattr(self.store, "latest_orderbook") else None
        market_state = self.auto_tuner.snapshot_market_state(
            symbol=symbol,
            market=market,
            quote_age_seconds=quote_age_seconds,
            spread_bps=float(getattr(orderbook, "spread_bps", 0.0) or 0.0),
            orderbook_available=orderbook is not None,
            volume_ratio=volume_ratio,
            recent_performance=0.0,
            fallback_score=max(0.0, ontology_score + max(-0.5, min(0.5, pnl_rate))),
            symbol_volatility=self._symbol_realtime_volatility(symbol, decision_time),
            market_volatility=self._market_realtime_volatility(decision_time),
        )
        policy, exit_policy, policy_diag = self.auto_tuner.build_exit_policy(
            symbol=symbol,
            holding=holding,
            account=account,
            market=market,
            market_state=market_state,
            take_profit=take_profit,
            stop_loss=stop_loss,
            ontology_score=ontology_score,
            target_net_return=target_net_return,
            decision_time=decision_time,
        )
        cost_floor = self._exit_cost_floor(holding, price, target_net_return)
        required_exit_price = max(cost_floor.required_exit_price, avg_cost * (1.0 + exit_policy.sell_target))
        required_exit_return = (required_exit_price - avg_cost) / avg_cost
        profitable_after_cost = price >= required_exit_price and cost_floor.net_expected_return >= target_net_return
        # Single authoritative exit-policy resolution (Phase 2): all exit thresholds
        # below are sourced from this one object and logged once, replacing the ~15
        # inline REALTIME_* reads. Env vars still override for backward compatibility.
        _exit_is_domestic = _is_domestic_symbol_or_market(symbol, holding.market or "")
        _exit_spread_rate = (
            max(0.0, float(getattr(orderbook, "spread_bps", 0.0) or 0.0)) / 10_000.0
            if orderbook is not None
            else 0.0
        )
        resolved_exit = self.dynamic_exit_policy.resolve(
            all_in_cost_rate=float(getattr(cost_floor, "total_cost_rate", 0.0) or 0.0),
            realized_volatility=self._symbol_realtime_volatility(symbol, decision_time),
            spread_rate=_exit_spread_rate,
            liquidity_score=1.0,
            predicted_downside_risk=0.0,
            account_drawdown_rate=(_account_domestic_unrealized_rate(account) if _exit_is_domestic else 0.0),
        )
        loss_exit_allowed = exit_policy.allow_loss_exit
        emergency_loss = max(exit_policy.stop_loss, resolved_exit.emergency_stop_rate)
        account_total = max(
            float(
                getattr(account, "equity", None)
                or getattr(account, "total_equity", None)
                or getattr(account, "cash_balance", None)
                or getattr(account, "cash", None)
                or 1.0
            ),
            1.0,
        )
        position_weight = max(0.0, (holding.quantity * price) / account_total)
        is_domestic_holding = _is_domestic_symbol_or_market(symbol, holding.market or "")
        domestic_reduce_trigger = -abs(_env_float("REALTIME_DOMESTIC_DRAWDOWN_REDUCE_TRIGGER", 0.015))
        domestic_emergency_trigger = -abs(_env_float("REALTIME_DOMESTIC_EMERGENCY_EXIT_TRIGGER", 0.03))
        domestic_concentration_weight = _env_float("REALTIME_DOMESTIC_CONCENTRATION_REDUCE_WEIGHT", 0.20)

        # --- Turnover-first profit realization ---------------------------------
        # net_pnl_rate is the gain AFTER the full round-trip cost. All three rules
        # below only fire when net-profitable, so they never realize a loss and
        # therefore respect REALTIME_ALLOW_LOSS_EXIT=false. They give the engine a
        # decisive, NON-chasing reason to lock in gains for fast buy/sell cycling
        # instead of holding a winner until an ever-rising bar is met.
        round_trip_cost_rate = float(getattr(cost_floor, "total_cost_rate", 0.0) or 0.0)
        net_pnl_rate = pnl_rate - round_trip_cost_rate
        # Absolute AFTER-FEE profit in the position's own currency. Small 1-share KR
        # positions rarely clear a % target but do clear a few tens of won — so the
        # PRIMARY take-profit is an amount threshold (won for KR, USD for overseas),
        # not a percentage. Realize whenever net profit >= that small amount.
        notional = max(0.0, float(getattr(holding, "quantity", 0) or 0) * avg_cost)
        net_profit_amount = net_pnl_rate * notional
        if _is_domestic_symbol_or_market(symbol, holding.market or ""):
            take_profit_amount = max(0.0, _env_float("REALTIME_TAKE_PROFIT_AMOUNT_KRW", 20.0))
        else:
            take_profit_amount = max(0.0, _env_float("REALTIME_TAKE_PROFIT_AMOUNT_USD", 0.05))
        # --- INVESTMENT MODE (profit realization, not stop-loss scalping) ------
        # Goal: realize profit and NEVER sell below the entry on noise. The exit only
        # sells at a MEANINGFUL net gain (quick target / trailing lock / stalled-but-
        # profitable time exit). Losers are HELD to recover; only a wide catastrophic
        # backstop cuts a position, so a normal dip is never dumped at a small loss.
        # All exit thresholds now come from the single resolved exit policy (Phase 2).
        # quick_take_profit_net / profit_lock_arm / giveback / profit_time_exit honor
        # REALTIME_* env overrides via the resolver; unset defaults are dynamic.
        quick_tp_net = max(0.0, resolved_exit.quick_take_profit_net)
        lock_arm_net = max(0.0, resolved_exit.profit_lock_arm_net)
        lock_giveback = min(0.95, max(0.0, resolved_exit.trailing_giveback_rate))
        profit_time_exit_sec = max(0.0, resolved_exit.profit_time_exit_sec)
        # Routine tight net stop (opt-in via REALTIME_STOP_LOSS_NET > 0). Keeps each
        # loss small and symmetric with the small take-profit.
        stop_loss_net = max(0.0, resolved_exit.stop_loss_net)
        # Every profit-motivated exit must lock a MEANINGFUL net gain (after the full
        # round-trip cost), never churn a ~break-even position.
        min_net_profit_exit = max(0.0, resolved_exit.min_net_profit_exit)
        # Catastrophic capital circuit-breaker only (NOT a routine stop). Holds normal
        # dips; cuts a position solely to prevent ruin. Set 0 to hold with no stop at all.
        hard_stop_loss = max(0.0, resolved_exit.hard_stop_rate)
        small_account_mode = _env_bool("REALTIME_SMALL_ACCOUNT_MODE", False)
        small_account_equity = max(0.0, _env_float("REALTIME_SMALL_ACCOUNT_EQUITY_KRW", 300000.0))
        small_account_active = small_account_mode and account_total <= small_account_equity
        quantity = int(getattr(holding, "quantity", 0) or 0)
        hard_stop_triggered = hard_stop_loss > 0.0 and pnl_rate <= -hard_stop_loss
        if (
            small_account_active
            and _env_bool("REALTIME_BLOCK_ONE_SHARE_LOSS_REDUCE", True)
            and quantity <= 1
            and pnl_rate < 0.0
            and not hard_stop_triggered
        ):
            diagnostics = {
                "exit_policy": exit_policy.as_dict(),
                "policy": policy.as_dict(),
                "policy_state": policy_diag,
                "quote_age_seconds": round(quote_age_seconds, 3),
                "ontology_score": round(ontology_score, 4),
                "pnl_rate": round(pnl_rate, 6),
                "position_weight": round(position_weight, 6),
                "quantity": quantity,
                "small_account_equity_krw": small_account_equity,
                "hard_stop_loss": hard_stop_loss,
            }
            reasons = (
                "SMALL_ACCOUNT_ONE_SHARE_LOSS_BLOCK",
                "SELL_BELOW_BREAK_EVEN_BLOCKED",
                "HOLD_BELOW_PROFIT_TARGET",
            )
            self._last_diagnostics = diagnostics
            return SharedDecisionResult(symbol, False, None, None, reasons, diagnostics)
        held_age_seconds: float | None = None
        opened_at = getattr(holding, "opened_at", None)
        if opened_at is not None:
            try:
                held_age_seconds = max(0.0, (decision_time - opened_at).total_seconds())
            except Exception:  # noqa: BLE001 - opened_at is best-effort metadata.
                held_age_seconds = None
        if net_pnl_rate > 0.0:
            peak_net_pnl = max(self._peak_net_pnl.get(symbol, 0.0), net_pnl_rate)
        else:
            peak_net_pnl = self._peak_net_pnl.get(symbol, 0.0)
        self._peak_net_pnl[symbol] = peak_net_pnl

        # When the won-amount take-profit is enabled it GOVERNS routine profit exits:
        # the percentage paths may not sell below the configured won floor (so raising
        # the amount threshold genuinely holds sub-threshold winners). When it is
        # disabled (0), the percentage paths behave as before.
        amount_gate = take_profit_amount <= 0.0 or net_profit_amount >= take_profit_amount

        prediction: LiveSignalPrediction | None = None
        exit_reason: str | None = None
        if take_profit_amount > 0.0 and net_pnl_rate > 0.0 and net_profit_amount >= take_profit_amount:
            # PRIMARY: absolute after-fee profit amount cleared -> realize now.
            exit_reason = f"take_profit_amount:{net_profit_amount:.0f}"
        elif quick_tp_net > 0.0 and net_pnl_rate >= quick_tp_net and amount_gate:
            # Secondary: percentage take-profit (used when the won-amount rule is off).
            exit_reason = f"quick_take_profit:{net_pnl_rate * 100:.2f}%"
        elif (
            lock_arm_net > 0.0
            and peak_net_pnl >= lock_arm_net
            and net_pnl_rate > max(0.0, target_net_return)
            and net_pnl_rate <= peak_net_pnl * (1.0 - lock_giveback)
        ):
            # Profit-giveback trailing lock: after arming on a good gain, sell if the
            # position gives back part of its peak while still net-positive.
            exit_reason = f"profit_lock:{net_pnl_rate * 100:.2f}%<=peak{peak_net_pnl * 100:.2f}%"
        elif (
            profit_time_exit_sec > 0.0
            and held_age_seconds is not None
            and held_age_seconds >= profit_time_exit_sec
            and net_pnl_rate >= max(min_net_profit_exit, target_net_return)
            and amount_gate
        ):
            # Turnover pressure: realize a MEANINGFULLY net-profitable position held
            # past the short window. Requires >= min_net_profit_exit so it never
            # churns a break-even position (which would just bleed round-trip cost).
            exit_reason = f"profit_time_exit:{net_pnl_rate * 100:.2f}%"
        elif profitable_after_cost and ontology_score <= -0.55:
            # Strong ontology sell signal on a net-profitable position — risk-driven,
            # allowed at any positive net.
            exit_reason = f"profit_exit:{pnl_rate * 100:.2f}%"
        elif profitable_after_cost and pnl_rate >= required_exit_return and net_pnl_rate >= min_net_profit_exit and amount_gate:
            # Routine profit target reached AND it clears a meaningful net gain AND the
            # won-amount floor (when enabled). Prevents selling at ~break-even.
            exit_reason = f"profit_exit:{pnl_rate * 100:.2f}%"
        elif stop_loss_net > 0.0 and net_pnl_rate <= -stop_loss_net:
            # Primary tight stop — fires regardless of REALTIME_ALLOW_LOSS_EXIT.
            # Keeps each loss small and symmetric with the small take-profit so the
            # strategy is not structurally negative-expectancy (small wins, huge losses).
            exit_reason = f"stop_loss:{net_pnl_rate * 100:.2f}%"
        elif hard_stop_loss > 0.0 and pnl_rate <= -hard_stop_loss:
            # Capital circuit-breaker — fires regardless of REALTIME_ALLOW_LOSS_EXIT.
            # Cutting a catastrophic loser protects total assets from unbounded
            # drawdown; realizing a bounded loss is better than holding into a larger one.
            exit_reason = f"hard_stop_loss:{pnl_rate * 100:.2f}%"
        elif pnl_rate <= -exit_policy.stop_loss and not loss_exit_allowed:
            diagnostics = {"exit_policy": exit_policy.as_dict(), "policy": policy.as_dict(), "policy_state": policy_diag, "quote_age_seconds": round(quote_age_seconds, 3), "ontology_score": round(ontology_score, 4)}
            reasons = ("LOSS_EXIT_DISABLED", "HOLD_LOSS_EXIT_DISABLED", "REALTIME_ALLOW_LOSS_EXIT=false")
            self._last_diagnostics = diagnostics
            return SharedDecisionResult(symbol, False, None, prediction, reasons, diagnostics)
        elif is_domestic_holding and loss_exit_allowed and pnl_rate <= domestic_emergency_trigger:
            exit_reason = f"domestic_emergency_exit:{pnl_rate * 100:.2f}%"
        elif pnl_rate <= -emergency_loss and loss_exit_allowed:
            exit_reason = f"loss_exit:{pnl_rate * 100:.2f}%"
        elif is_domestic_holding and loss_exit_allowed and pnl_rate <= domestic_reduce_trigger:
            exit_reason = f"domestic_drawdown_reduce:{pnl_rate * 100:.2f}%"
        elif is_domestic_holding and loss_exit_allowed and pnl_rate < 0 and position_weight >= domestic_concentration_weight:
            exit_reason = f"domestic_concentration_reduce:{position_weight * 100:.2f}%"
        elif pnl_rate <= -exit_policy.trailing_stop and loss_exit_allowed:
            exit_reason = f"trailing_exit:{pnl_rate * 100:.2f}%"
        elif quote_age_seconds >= exit_policy.time_exit_seconds:
            if profitable_after_cost or pnl_rate >= 0:
                exit_reason = f"time_exit:{pnl_rate * 100:.2f}%"
            else:
                exit_reason = None
        elif ontology_score <= -0.25 and profitable_after_cost:
            exit_reason = f"invalid_signal_exit:{ontology_score:.2f}"
        elif ontology_score <= -0.25 and not profitable_after_cost:
            diagnostics = {"exit_policy": exit_policy.as_dict(), "policy": policy.as_dict(), "policy_state": policy_diag, "quote_age_seconds": round(quote_age_seconds, 3), "ontology_score": round(ontology_score, 4)}
            reasons = ("HOLD_UNPROFITABLE_ONTOLOGY_SELL_BLOCKED", "HOLD_BELOW_PROFIT_TARGET")
            self._last_diagnostics = diagnostics
            return SharedDecisionResult(symbol, False, None, prediction, reasons, diagnostics)
        else:
            try:
                exit_reason, prediction = self._model_exit_signal(symbol, decision_time)
            except Exception:  # noqa: BLE001 - model exit is best-effort.
                exit_reason, prediction = None, None
            if exit_reason is not None and not profitable_after_cost:
                exit_reason = None

        if exit_reason is None:
            diagnostics = {"exit_policy": exit_policy.as_dict(), "policy": policy.as_dict(), "policy_state": policy_diag, "quote_age_seconds": round(quote_age_seconds, 3), "ontology_score": round(ontology_score, 4), "technical_exit_deterioration": list(technical_exit_codes)}
            reasons = (
                "HOLD_RECHECK",
                "HOLD_BELOW_PROFIT_TARGET",
                *technical_exit_codes,
                f"QUOTE_REFRESH:{'quote_refresh_ok' if quote_age_seconds <= exit_policy.time_exit_seconds else 'quote_refresh_not_needed'}",
            )
            self._last_diagnostics = diagnostics
            return SharedDecisionResult(symbol, False, None, prediction, reasons, diagnostics)

        # A DELIBERATE risk-stop must still execute below break-even — that is the whole
        # point of a stop. This covers the opt-in net tight stop (REALTIME_STOP_LOSS_NET,
        # documented to "fire regardless of allow_loss_exit"), the hard/emergency capital
        # circuit-breakers, and — when the operator enabled loss exits — the discretionary
        # loss/trailing/domestic reduce exits. block_sell_below_breakeven only exists to
        # stop *profit/time/model*-motivated exits from churning a slightly-underwater
        # position on noise; it must NOT silently veto a configured stop-loss (which was
        # the prior behaviour: a pinned REALTIME_STOP_LOSS_NET never actually sold).
        deliberate_loss_stop = exit_reason.startswith(
            (
                "stop_loss",
                "hard_stop_loss",
                "domestic_emergency_exit",
                "loss_exit",
                "trailing_exit",
                "domestic_drawdown_reduce",
                "domestic_concentration_reduce",
            )
        )
        if resolved_exit.block_sell_below_breakeven and not deliberate_loss_stop and not profitable_after_cost:
            diagnostics = {
                "exit_policy": exit_policy.as_dict(),
                "policy": policy.as_dict(),
                "policy_state": policy_diag,
                "quote_age_seconds": round(quote_age_seconds, 3),
                "ontology_score": round(ontology_score, 4),
                "pnl_rate": round(pnl_rate, 6),
                "net_pnl_rate": round(net_pnl_rate, 6),
                "required_exit_price": round(required_exit_price, 6),
                "current_price": round(price, 6),
                "attempted_exit_reason": exit_reason,
                "hard_emergency": False,
            }
            reasons = (
                "SELL_BELOW_BREAK_EVEN_BLOCKED",
                "HOLD_BELOW_PROFIT_TARGET",
                f"ATTEMPTED_EXIT:{exit_reason.split(':', 1)[0]}",
            )
            self._last_diagnostics = diagnostics
            return SharedDecisionResult(symbol, False, None, prediction, reasons, diagnostics)

        exit_action = OrderAction.SELL
        exit_suggested_weight = 0.0
        if exit_reason.startswith(("trailing_exit", "domestic_drawdown_reduce", "domestic_concentration_reduce")):
            reduce_fraction = max(0.1, min(1.0, float(os.getenv("REALTIME_LOSS_EXIT_REDUCE_FRACTION", "0.5"))))
            if float(getattr(holding, "quantity", 0.0) or 0.0) > 1.0:
                exit_action = OrderAction.REDUCE
                exit_suggested_weight = max(0.0, position_weight * (1.0 - reduce_fraction))

        intent = OrderIntent(
            ticker=symbol,
            market=holding.market or "KR",
            action=exit_action,
            suggested_weight=exit_suggested_weight,
            confidence=max(exit_policy.confidence_floor, 0.85 if profitable_after_cost else 0.7),
            valid_until=decision_time + timedelta(seconds=max(30, exit_policy.time_exit_seconds // 4)),
            reasoning_summary=(f"realtime_exit:{exit_reason}",),
            supporting_factors=("realtime_exit", exit_policy.exit_mode, *ontology_support),
            contradicting_factors=(),
            source_data_ids=(source_id,),
            strategy_family="live_short_horizon_exit",
            signal_name=exit_reason.split(":", 1)[0],
            expected_exit_price=required_exit_price,
            gross_expected_return=max(0.0, required_exit_return),
            target_net_return=target_net_return,
            cost_breakdown=cost_floor.as_dict(),
            strategy_metadata={
                "exit_policy": exit_policy.as_dict(),
                "policy": policy.as_dict(),
                "policy_state": policy_diag,
                "quote_age_seconds": round(quote_age_seconds, 3),
                "ontology_score": round(ontology_score, 4),
                "pnl_rate": round(pnl_rate, 6),
                "net_pnl_rate": round(net_pnl_rate, 6),
                "peak_net_pnl": round(peak_net_pnl, 6),
                "round_trip_cost_rate": round(round_trip_cost_rate, 6),
                "resolved_exit_policy": resolved_exit.as_dict(),
                "exit_reason": exit_reason,
                "exit_action": str(exit_action),
                "exit_suggested_weight": round(exit_suggested_weight, 6),
            },
        )
        adaptive_rules = self.auto_tuner.derive_risk_rules(
            self.risk_manager.rules,
            policy=policy,
            account=account,
            market=market,
            model_uncertainty=prediction.uncertainty_score if prediction is not None else None,
        )
        risk_manager = RiskManager(adaptive_rules, audit_logger=self.risk_manager.audit_logger)
        risk = risk_manager.validate(intent, account, market)
        if intent.action in {OrderAction.SELL, OrderAction.REDUCE} and not risk.approved and set(risk.rejection_reasons) == {"cash_available"}:
            risk = risk.__class__(
                ticker=risk.ticker,
                action=risk.action,
                approved=True,
                adjusted_weight=risk.adjusted_weight,
                checks={**risk.checks, "cash_available": True},
                rejection_reasons=(),
                final_order=FinalOrder(
                    ticker=intent.ticker,
                    market=intent.market,
                    order_type=OrderType.LIMIT,
                    side=OrderSide.SELL,
                    quantity=max(1, int(getattr(holding, "quantity", 0) or 0)),
                    limit_price=price,
                    manual_approval_required=self.risk_manager.rules.manual_approval_required,
                ),
                metadata=dict(risk.metadata),
            )
        diagnostics = {
            "exit_policy": exit_policy.as_dict(),
            "policy": policy.as_dict(),
            "policy_state": policy_diag,
            "quote_age_seconds": round(quote_age_seconds, 3),
            "ontology_score": round(ontology_score, 4),
            "adaptive_risk_rules": adaptive_rules,
            "risk_metadata": risk.metadata,
            "exit_reason": exit_reason,
            "technical_exit_deterioration": list(technical_exit_codes),
        }
        self.auto_tuner.record_feedback(
            {
                "symbol": symbol,
                "side": "SELL",
                "approved": risk.approved,
                "reason_codes": risk.rejection_reasons,
                "policy": policy.as_dict(),
                "pnl": pnl_rate,
                "quote_refresh_status": "quote_refresh_ok" if quote_age_seconds <= exit_policy.time_exit_seconds else "quote_refresh_skipped",
            }
        )
        self._last_diagnostics = diagnostics
        return SharedDecisionResult(
            symbol=symbol,
            approved=risk.approved and risk.final_order is not None,
            final_order=risk.final_order,
            prediction=prediction,
            reason_codes=risk.rejection_reasons or (exit_reason,),
            diagnostics=diagnostics,
        )

    def _exit_cost_floor(self, holding: Holding, expected_exit_price: float, target_net_return: float):
        symbol = str(getattr(holding, "ticker", "") or "")
        market = str(getattr(holding, "market", "") or "")
        quantity = max(1, int(getattr(holding, "quantity", 0) or 0))
        venue, instrument_type = _cost_context_for_holding(symbol, market)
        return TradingCostEngine().estimate(
            symbol=symbol,
            market=market or ("KR" if instrument_type == "domestic_stock" else venue),
            venue=venue,
            instrument_type=instrument_type,
            entry_price=float(getattr(holding, "average_price", 0.0) or 0.0),
            expected_exit_price=float(expected_exit_price),
            quantity=quantity,
            target_net_return=target_net_return,
        )

    def _symbol_realtime_volatility(self, symbol: str, decision_time: datetime) -> float:
        try:
            window_seconds = max(60.0, float(os.getenv("REALTIME_SYMBOL_VOLATILITY_WINDOW_SEC", "300")))
            since = decision_time - timedelta(seconds=window_seconds)
            ticks = self.store.recent_ticks(symbol, since) if hasattr(self.store, "recent_ticks") else ()
            prices = [float(getattr(tick, "price", 0.0) or 0.0) for tick in ticks]
            prices = [price for price in prices if price > 0.0]
            return _realized_volatility_from_prices(prices)
        except Exception:  # noqa: BLE001 - volatility is a risk input, not a hard dependency.
            return 0.0

    def _market_realtime_volatility(self, decision_time: datetime) -> float:
        try:
            window_seconds = max(60.0, float(os.getenv("REALTIME_MARKET_VOLATILITY_WINDOW_SEC", "300")))
            since = decision_time - timedelta(seconds=window_seconds)
            if hasattr(self.store, "active_symbols"):
                symbols = self.store.active_symbols(since, limit=max(1, int(float(os.getenv("REALTIME_MARKET_VOLATILITY_SYMBOL_LIMIT", "40")))))
            else:
                symbols = ()
            values: list[float] = []
            for symbol in symbols:
                vol = self._symbol_realtime_volatility(symbol, decision_time)
                if vol > 0.0:
                    values.append(vol)
            if not values:
                return 0.0
            values.sort()
            trim = max(0, int(len(values) * 0.1))
            sample = values[trim : len(values) - trim] if len(values) > trim * 2 else values
            return sum(sample) / len(sample)
        except Exception:  # noqa: BLE001
            return 0.0

    def _realtime_volume_surge_ratio(self, symbol: str, decision_time: datetime) -> float:
        """장중 거래 활성도 급증 비율 = 최근 짧은 구간 체결빈도 / 기준 구간 체결빈도.

        소스별 volume 필드 의미(틱 증분 vs 누적)가 달라, 소스에 무관하게 견고한
        체결(틱) 빈도를 사용한다. 데이터가 부족하면 1.0(중립)을 반환한다.
        """
        try:
            recent_window = max(5.0, float(os.getenv("REALTIME_VOLUME_RECENT_SEC", "60")))
            base_window = max(recent_window * 2.0, float(os.getenv("REALTIME_VOLUME_BASE_SEC", "600")))
            base_since = decision_time - timedelta(seconds=base_window)
            ticks = self.store.recent_ticks(symbol, base_since)
            if not ticks or len(ticks) < 4:
                return 1.0
            recent_cut = decision_time - timedelta(seconds=recent_window)
            recent_count = sum(1 for t in ticks if (getattr(t, "received_at", None) or decision_time) >= recent_cut)
            base_rate = len(ticks) / base_window
            if base_rate <= 0:
                return 1.0
            recent_rate = recent_count / recent_window
            return recent_rate / base_rate
        except Exception:  # noqa: BLE001 - volume signal is best-effort.
            return 1.0

    def _exit_price_source(
        self, symbol: str, holding: Holding, decision_time: datetime
    ) -> tuple[float, datetime, datetime, str]:
        """Prefer a fresh realtime tick; fall back to the broker balance mark."""
        max_tick_age = float(os.getenv("REALTIME_EXIT_TICK_MAX_AGE_SEC", "30"))
        tick = self.store.latest_tick(symbol)
        if tick is not None:
            tick_price = float(getattr(tick, "price", 0.0) or 0.0)
            received_at = getattr(tick, "received_at", None) or decision_time
            try:
                tick_age = (decision_time - received_at).total_seconds()
            except Exception:  # noqa: BLE001 - timezone mishaps fall back to the broker mark.
                tick_age = max_tick_age + 1
            if tick_price > 0 and tick_age <= max_tick_age:
                return (
                    tick_price,
                    getattr(tick, "exchange_timestamp", received_at) or received_at,
                    received_at,
                    str(getattr(tick, "sequence_key", "") or f"tick:{symbol}"),
                )
        balance_price = float(getattr(holding, "last_price", 0.0) or 0.0)
        return balance_price, decision_time, decision_time, f"balance:{symbol}"

    def _exit_risk_manager(self) -> RiskManager:
        """De-risking 매도는 매수용 게이트(현금준비금/실시간 호가 신선도)에 막히면 안 되므로
        완화된 규칙으로 검증한다. 가격 소스는 브로커 잔고 마크를 신뢰한다."""
        cached = getattr(self, "_exit_risk_manager_cache", None)
        if cached is None:
            relaxed = replace(
                self.risk_manager.rules,
                minimum_cash_reserve=0.0,
                max_quote_age_seconds=1e9,
            )
            cached = RiskManager(relaxed)
            self._exit_risk_manager_cache = cached
        return cached

    def _model_exit_signal(
        self, symbol: str, decision_time: datetime
    ) -> tuple[str | None, LiveSignalPrediction | None]:
        """Model-based exit: trim when the buy edge has clearly flipped negative."""
        try:
            frame = self.feature_builder.build(symbol, decision_time=decision_time)
            prediction = self.predictor.predict(frame)
        except Exception:  # noqa: BLE001 - model exit is best-effort; TP/SL still protects.
            return None, None
        exit_bps = float(os.getenv("REALTIME_MODEL_EXIT_BPS", "8"))
        if not prediction.approved and prediction.expected_net_return_bps <= -exit_bps:
            return f"model_edge_lost:{prediction.expected_net_return_bps:.1f}bps", prediction
        return None, prediction

    def _exit_market_snapshot(
        self, holding: Holding, price: float, observed_at: datetime, received_at: datetime
    ) -> MarketSnapshot:
        return MarketSnapshot(
            ticker=holding.ticker,
            market=holding.market or "KR",
            company_name=getattr(holding, "company_name", "") or holding.ticker,
            sector=getattr(holding, "sector", "") or "Unknown",
            last_price=price,
            average_daily_trading_value=10_000_000_000,
            volatility_20d=0.02,
            source=SourceMetadata(
                source_name="KIS broker mark / realtime WebSocket",
                retrieved_at=received_at,
                observed_at=observed_at,
                source_type="broker_api",
                trust_level=5,
                is_realtime=True,
                quality_score=1.0,
            ),
        )

    def get_diagnostics(self) -> dict[str, Any]:
        return dict(self._last_diagnostics)
