"""Micro symbol ontology reasoning.

MicroSymbolReasoner evaluates ONE macro-selected symbol: entry/exit timing,
conservative expected entry/exit price, expected net return after cost, downside
risk, and execution quality. It reuses the advisory technical prediction layer
(`app.technical`) and layers the macro strategy-permission gate on top.

ADVISORY ONLY: it may emit OrderIntent-like evidence but never submits an order.
Expected net return is required for a BUY candidate — a missing or non-positive
net edge yields HOLD_OR_WATCH / BLOCKED, never BUY_CANDIDATE.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Mapping

from app.graph.macro_micro_common import (
    EXPECTED_NET_RETURN_NON_POSITIVE,
    LOW_LIQUIDITY,
    MICRO_BREAKOUT_CONFIRMED,
    MICRO_CONFIDENCE_TOO_LOW,
    MICRO_EXIT_DETERIORATION,
    MICRO_EXPECTED_NET_RETURN_MISSING,
    MICRO_MEAN_REVERSION_CANDIDATE,
    MICRO_MOMENTUM_CONFIRMED,
    MICRO_SIGNAL_UNAVAILABLE,
    MICRO_STRATEGY_BLOCKED_BY_MACRO,
    MICRO_TECHNICAL_HISTORY_INSUFFICIENT,
    SPREAD_CONSUMES_ALPHA,
    STALE_QUOTE,
    EntrySignal,
    ExecutionQuality,
    ExitSignal,
    MicroRegime,
    SelectedStrategy,
    explanation,
)
from app.technical.prediction import PredictionAction, TechnicalPredictionEngine
from app.technical.signals import (
    CompositeTechnicalSignalEngine,
    SignalDirection,
    TechnicalFeatureSet,
)

# Which macro strategy tokens correspond to each micro strategy (for permission).
_STRATEGY_MACRO_TOKENS: dict[str, tuple[str, ...]] = {
    "momentum": ("momentum", "low_volume_momentum_buy"),
    "breakout": ("breakout", "weak_breakout_buy", "late_breakout_chasing"),
    "mean_reversion": ("mean_reversion", "aggressive_countertrend_reversion"),
    "vwap_reversion": ("vwap_reversion", "vwap_pullback"),
}

_METHODOLOGY_TO_STRATEGY = {
    "momentum_trend_following": "momentum",
    "breakout_trading_range_break": "breakout",
    "mean_reversion": "mean_reversion",
    "vwap_volume_liquidity": "vwap_reversion",
}

_STRATEGY_TO_MICRO_REGIME = {
    "momentum": MicroRegime.MOMENTUM_CANDIDATE,
    "breakout": MicroRegime.BREAKOUT_CANDIDATE,
    "mean_reversion": MicroRegime.MEAN_REVERSION_CANDIDATE,
    "vwap_reversion": MicroRegime.VWAP_REVERSION_CANDIDATE,
}


@dataclass(frozen=True)
class MicroReasonerConfig:
    minimum_micro_confidence: float = 0.55
    minimum_expected_net_return_bps: float = 0.0
    block_if_spread_consumes_alpha: bool = True
    block_if_stale_quote: bool = True
    block_if_low_liquidity: bool = True
    low_liquidity_score: float = 0.35
    max_quote_age_seconds: float = 90.0

    @classmethod
    def from_env(cls) -> "MicroReasonerConfig":
        def _f(name: str, default: float) -> float:
            raw = os.getenv(name)
            try:
                return float(raw) if raw not in (None, "") else default
            except ValueError:
                return default

        return cls(
            minimum_micro_confidence=_f("MICRO_MIN_CONFIDENCE", cls.minimum_micro_confidence),
            minimum_expected_net_return_bps=_f("MICRO_MIN_NET_BPS", cls.minimum_expected_net_return_bps),
        )


@dataclass(frozen=True)
class MicroReasoningInput:
    timestamp: datetime
    symbol: str
    market: str = "KR"
    macro_result_ref: str | None = None
    allowed_micro_strategies: tuple[str, ...] = ()
    blocked_micro_strategies: tuple[str, ...] = ()
    technical_features: TechnicalFeatureSet | None = None
    short_horizon_prediction: Any = None            # LiveSignalPrediction-like or None
    realtime_tick: Any = None
    orderbook: Any = None
    broker_quote: Any = None
    live_feature_frame: Any = None
    holding_state: Mapping[str, Any] | None = None  # {quantity, average_price, ...} or None
    cash_context: Mapping[str, Any] = field(default_factory=dict)
    all_in_cost_bps: float | None = None
    quote_age_seconds: float | None = None
    source_freshness: Mapping[str, Any] = field(default_factory=dict)
    provenance: Mapping[str, Any] = field(default_factory=dict)

    @property
    def is_holding(self) -> bool:
        return bool(self.holding_state and int(self.holding_state.get("quantity", 0) or 0) > 0)


@dataclass(frozen=True)
class MicroReasoningResult:
    timestamp: datetime
    symbol: str
    micro_regime: MicroRegime
    selected_strategy: SelectedStrategy
    entry_signal: EntrySignal
    exit_signal: ExitSignal
    expected_entry_price: float | None
    expected_exit_price: float | None
    expected_gross_return_bps: float | None
    expected_net_return_bps: float | None
    downside_risk_bps: float | None
    confidence: float
    execution_quality: ExecutionQuality
    reason_codes: tuple[str, ...]
    explanation_paths: tuple[dict, ...]
    rdf_graph_id: str | None = None
    diagnostics: Mapping[str, Any] = field(default_factory=dict)

    @property
    def is_buy_candidate(self) -> bool:
        return self.entry_signal == EntrySignal.BUY_CANDIDATE

    @property
    def is_exit_candidate(self) -> bool:
        return self.exit_signal in (ExitSignal.SELL_CANDIDATE, ExitSignal.RISK_REDUCE,
                                    ExitSignal.TAKE_PROFIT, ExitSignal.TRAILING_STOP)

    def as_dict(self) -> dict:
        def _r(v, n=3):
            return round(v, n) if isinstance(v, (int, float)) else v

        return {
            "timestamp": self.timestamp.isoformat(),
            "symbol": self.symbol,
            "micro_regime": self.micro_regime.value,
            "selected_strategy": self.selected_strategy.value,
            "entry_signal": self.entry_signal.value,
            "exit_signal": self.exit_signal.value,
            "expected_entry_price": _r(self.expected_entry_price, 4),
            "expected_exit_price": _r(self.expected_exit_price, 4),
            "expected_gross_return_bps": _r(self.expected_gross_return_bps),
            "expected_net_return_bps": _r(self.expected_net_return_bps),
            "downside_risk_bps": _r(self.downside_risk_bps),
            "confidence": _r(self.confidence, 4),
            "execution_quality": self.execution_quality.value,
            "reason_codes": list(self.reason_codes),
            "explanation_paths": list(self.explanation_paths),
            "rdf_graph_id": self.rdf_graph_id,
            "diagnostics": dict(self.diagnostics),
        }


def _strategy_permitted(strategy: str, allowed: tuple[str, ...], blocked: tuple[str, ...]) -> tuple[bool, str | None]:
    blocked_set = set(blocked)
    if "new_buy" in blocked_set:
        return False, "macro blocks all new buys"
    tokens = set(_STRATEGY_MACRO_TOKENS.get(strategy, (strategy,)))
    if tokens & blocked_set:
        return False, f"macro blocks strategy tokens {sorted(tokens & blocked_set)}"
    if allowed:  # when an allow-list is present, the strategy must be in it
        allowed_set = set(allowed)
        if not (tokens & allowed_set) and strategy not in allowed_set:
            return False, f"strategy '{strategy}' not in macro allow-list"
    return True, None


class MicroSymbolReasoner:
    def __init__(
        self,
        config: MicroReasonerConfig | None = None,
        *,
        signal_engine: CompositeTechnicalSignalEngine | None = None,
        prediction_engine: TechnicalPredictionEngine | None = None,
    ) -> None:
        self.config = config or MicroReasonerConfig()
        self.signal_engine = signal_engine or CompositeTechnicalSignalEngine()
        self.prediction_engine = prediction_engine or TechnicalPredictionEngine(signal_engine=self.signal_engine)

    def reason(self, data: MicroReasoningInput) -> MicroReasoningResult:
        cfg = self.config
        features = data.technical_features
        reasons: list[str] = []
        paths: list[dict] = []

        if features is None:
            live_quote = data.realtime_tick is not None or data.broker_quote is not None
            if live_quote:
                return self._result(
                    data,
                    MicroRegime.HOLD_OR_WATCH,
                    SelectedStrategy.HOLD,
                    EntrySignal.WAIT_CONFIRMATION,
                    ExitSignal.NONE,
                    ExecutionQuality.ACCEPTABLE,
                    reasons=[MICRO_TECHNICAL_HISTORY_INSUFFICIENT],
                    paths=[
                        explanation(
                            MICRO_TECHNICAL_HISTORY_INSUFFICIENT,
                            "Live quote is available, but technical history is not sufficient for a micro signal.",
                        )
                    ],
                    confidence=0.25,
                    expected_entry_price=_quote_price(data.broker_quote) or _quote_price(data.realtime_tick),
                )
            return self._result(data, MicroRegime.NO_TRADE_SYMBOL, SelectedStrategy.HOLD,
                                EntrySignal.NONE, ExitSignal.NONE, ExecutionQuality.BLOCKED,
                                reasons=[MICRO_SIGNAL_UNAVAILABLE], paths=[explanation(MICRO_SIGNAL_UNAVAILABLE, "No technical features for symbol.")],
                                confidence=0.0)

        # --- Held-position exit deterioration (SELL/REDUCE evidence) comes first. ---
        exit_signal = ExitSignal.NONE
        exit_codes = self.signal_engine.evaluate_exit_deterioration(features) if data.is_holding else ()
        if data.is_holding and exit_codes:
            reasons.extend(exit_codes)
            paths.append(explanation(MICRO_EXIT_DETERIORATION, "Exit deterioration evidence on held position.", {"codes": list(exit_codes)}))
            exit_signal = ExitSignal.RISK_REDUCE
            return self._result(data, MicroRegime.EXIT_DETERIORATION, SelectedStrategy.REDUCE_RISK,
                                EntrySignal.NONE, exit_signal, ExecutionQuality.ACCEPTABLE,
                                reasons=reasons, paths=paths, confidence=0.6,
                                expected_exit_price=(features.vwap if features.vwap else features.price))

        # --- Freshness gate (stale -> BLOCKED/HOLD, never BUY). ---
        age = data.quote_age_seconds
        if cfg.block_if_stale_quote and age is not None and age > cfg.max_quote_age_seconds:
            reasons.append(STALE_QUOTE)
            paths.append(explanation(STALE_QUOTE, f"Quote age {age:.1f}s exceeds {cfg.max_quote_age_seconds:.0f}s."))
            return self._result(data, MicroRegime.NO_TRADE_SYMBOL, SelectedStrategy.HOLD,
                                EntrySignal.BLOCKED, ExitSignal.NONE, ExecutionQuality.BLOCKED,
                                reasons=reasons, paths=paths, confidence=0.0)

        # --- Technical composite + conservative prediction (reused, advisory). ---
        composite = self.signal_engine.evaluate(features)
        prediction = self.prediction_engine.predict(
            features, composite=composite,
            model_prediction=data.short_horizon_prediction,
            all_in_cost_bps=data.all_in_cost_bps,
        )

        strategy_name = _METHODOLOGY_TO_STRATEGY.get(composite.selected_methodology, "hold")
        micro_regime = _STRATEGY_TO_MICRO_REGIME.get(strategy_name, MicroRegime.HOLD_OR_WATCH)

        # --- Composite-level BUY block (risk regime / no edge / HOLD). ---
        if composite.blocks_buy or composite.direction != SignalDirection.BUY:
            reasons.extend(composite.reason_codes)
            return self._result(data, MicroRegime.HOLD_OR_WATCH if not composite.blocks_buy else MicroRegime.NO_TRADE_SYMBOL,
                                SelectedStrategy.HOLD, EntrySignal.NONE, ExitSignal.NONE,
                                ExecutionQuality.WEAK if not composite.blocks_buy else ExecutionQuality.BLOCKED,
                                reasons=reasons, paths=paths + list(_paths_from_codes(composite.reason_codes)),
                                confidence=composite.confidence)

        # --- Macro strategy permission gate. ---
        permitted, why = _strategy_permitted(strategy_name, data.allowed_micro_strategies, data.blocked_micro_strategies)
        if not permitted:
            reasons.append(MICRO_STRATEGY_BLOCKED_BY_MACRO)
            paths.append(explanation(MICRO_STRATEGY_BLOCKED_BY_MACRO, why or "blocked by macro", {"strategy": strategy_name}))
            return self._result(data, MicroRegime.NO_TRADE_SYMBOL, _to_strategy(strategy_name),
                                EntrySignal.BLOCKED, ExitSignal.NONE, ExecutionQuality.BLOCKED,
                                reasons=reasons, paths=paths, confidence=composite.confidence)

        # --- Execution quality (spread / liquidity vs edge). ---
        exec_quality, exec_reason = self._execution_quality(features, composite)
        if exec_reason:
            reasons.append(exec_reason)
            paths.append(explanation(exec_reason, "execution quality degraded"))

        # --- Expected net return is REQUIRED for a BUY candidate. ---
        if prediction.action != PredictionAction.BUY or not prediction.tradable:
            reasons.append(MICRO_EXPECTED_NET_RETURN_MISSING if prediction.expected_net_return_bps is None else EXPECTED_NET_RETURN_NON_POSITIVE)
            reasons.extend(prediction.reason_codes)
            return self._result(data, MicroRegime.HOLD_OR_WATCH, _to_strategy(strategy_name),
                                EntrySignal.WAIT_CONFIRMATION, ExitSignal.NONE,
                                exec_quality if exec_quality != ExecutionQuality.BLOCKED else ExecutionQuality.WEAK,
                                reasons=reasons, paths=paths, confidence=prediction.confidence,
                                expected_exit_price=prediction.expected_exit_price,
                                expected_net_return_bps=prediction.expected_net_return_bps)

        net_bps = float(prediction.expected_net_return_bps or 0.0)
        if net_bps <= cfg.minimum_expected_net_return_bps:
            reasons.append(EXPECTED_NET_RETURN_NON_POSITIVE)
            return self._result(data, MicroRegime.HOLD_OR_WATCH, _to_strategy(strategy_name),
                                EntrySignal.WAIT_CONFIRMATION, ExitSignal.NONE, ExecutionQuality.WEAK,
                                reasons=reasons, paths=paths, confidence=prediction.confidence,
                                expected_exit_price=prediction.expected_exit_price,
                                expected_net_return_bps=net_bps)
        if prediction.confidence < cfg.minimum_micro_confidence:
            reasons.append(MICRO_CONFIDENCE_TOO_LOW)
            return self._result(data, MicroRegime.HOLD_OR_WATCH, _to_strategy(strategy_name),
                                EntrySignal.WAIT_CONFIRMATION, ExitSignal.NONE, exec_quality,
                                reasons=reasons, paths=paths, confidence=prediction.confidence,
                                expected_exit_price=prediction.expected_exit_price,
                                expected_net_return_bps=net_bps)
        if exec_quality == ExecutionQuality.BLOCKED:
            return self._result(data, MicroRegime.HOLD_OR_WATCH, _to_strategy(strategy_name),
                                EntrySignal.BLOCKED, ExitSignal.NONE, ExecutionQuality.BLOCKED,
                                reasons=reasons, paths=paths, confidence=prediction.confidence,
                                expected_exit_price=prediction.expected_exit_price,
                                expected_net_return_bps=net_bps)

        # --- Confirmed BUY candidate (advisory). ---
        reasons.append(_confirm_code(strategy_name))
        reasons.extend(composite.reason_codes)
        paths.append(explanation(_confirm_code(strategy_name), prediction.explanation,
                                 {"strategy": strategy_name, "net_bps": net_bps}))
        return self._result(
            data, micro_regime, _to_strategy(strategy_name),
            EntrySignal.BUY_CANDIDATE, ExitSignal.NONE, exec_quality,
            reasons=reasons, paths=paths, confidence=prediction.confidence,
            expected_entry_price=prediction.entry_price,
            expected_exit_price=prediction.expected_exit_price,
            expected_gross_return_bps=(prediction.expected_gross_return * 10_000.0) if prediction.expected_gross_return is not None else None,
            expected_net_return_bps=net_bps,
            downside_risk_bps=prediction.downside_risk_bps,
        )

    # ------------------------------------------------------------------ #
    def _execution_quality(self, features: TechnicalFeatureSet, composite) -> tuple[ExecutionQuality, str | None]:
        cfg = self.config
        if cfg.block_if_low_liquidity and features.liquidity_score is not None and features.liquidity_score < cfg.low_liquidity_score:
            return ExecutionQuality.BLOCKED, LOW_LIQUIDITY
        from app.technical import reason_codes as trc
        if cfg.block_if_spread_consumes_alpha and trc.SPREAD_CONSUMES_TECHNICAL_ALPHA in composite.reason_codes:
            return ExecutionQuality.BLOCKED, SPREAD_CONSUMES_ALPHA
        spread = features.spread_bps
        if spread is None:
            return ExecutionQuality.ACCEPTABLE, None
        if spread <= 5:
            return ExecutionQuality.GOOD, None
        if spread <= 20:
            return ExecutionQuality.ACCEPTABLE, None
        return ExecutionQuality.WEAK, None

    def _result(self, data: MicroReasoningInput, micro_regime, strategy, entry, exit_, exec_quality, *,
                reasons, paths, confidence,
                expected_entry_price=None, expected_exit_price=None,
                expected_gross_return_bps=None, expected_net_return_bps=None,
                downside_risk_bps=None) -> MicroReasoningResult:
        return MicroReasoningResult(
            timestamp=data.timestamp,
            symbol=data.symbol,
            micro_regime=micro_regime,
            selected_strategy=strategy,
            entry_signal=entry,
            exit_signal=exit_,
            expected_entry_price=expected_entry_price,
            expected_exit_price=expected_exit_price,
            expected_gross_return_bps=expected_gross_return_bps,
            expected_net_return_bps=expected_net_return_bps,
            downside_risk_bps=downside_risk_bps,
            confidence=max(0.0, min(1.0, confidence)),
            execution_quality=exec_quality,
            reason_codes=tuple(dict.fromkeys(reasons)),
            explanation_paths=tuple(paths),
            diagnostics={"macro_result_ref": data.macro_result_ref},
        )


def _to_strategy(name: str) -> SelectedStrategy:
    try:
        return SelectedStrategy(name)
    except ValueError:
        return SelectedStrategy.HOLD


def _confirm_code(strategy: str) -> str:
    return {
        "momentum": MICRO_MOMENTUM_CONFIRMED,
        "breakout": MICRO_BREAKOUT_CONFIRMED,
        "mean_reversion": MICRO_MEAN_REVERSION_CANDIDATE,
        "vwap_reversion": MICRO_MEAN_REVERSION_CANDIDATE,
    }.get(strategy, MICRO_MOMENTUM_CONFIRMED)


def _quote_price(quote: Any) -> float | None:
    if quote is None:
        return None
    for name in ("last_price", "price", "mark_price"):
        try:
            value = float(getattr(quote, name, 0.0) or 0.0)
        except (TypeError, ValueError):
            value = 0.0
        if value > 0.0:
            return value
    return None


def _paths_from_codes(codes) -> list[dict]:
    return [explanation(c, c) for c in codes]
