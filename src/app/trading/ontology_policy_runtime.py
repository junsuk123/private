"""Bounded store/context adapter for formal ontology-derived risk policies.

Call from the decision worker, never the WebSocket receive callback. It only
reads the local store and cached contexts. No broker API, graph materialization,
model training, implicit data warmup or disk writes occur in ``resolve``.
"""
from __future__ import annotations

import math
import statistics
import threading
from collections import OrderedDict
from dataclasses import replace
from datetime import datetime, timedelta, timezone
from typing import Any, Callable, Mapping

from app.ontology.policy_evidence import (
    EvidenceProjection, PolicyObservation, materialize_evidence_graph,
    project_context_evidence, project_market_evidence,
)
from app.risk.ontology_thresholds import OntologyRiskPolicy, resolve_ontology_policy

_REQUIRED = ("realized_volatility", "spread_rate", "liquidity_score", "quote_age_seconds", "regime_confidence")


def _get(value: Any, key: str, default: Any = None) -> Any:
    return value.get(key, default) if isinstance(value, Mapping) else getattr(value, key, default)


def _number(value: Any) -> float | None:
    if value is None or isinstance(value, bool):
        return None
    try:
        result = float(value)
    except (TypeError, ValueError, OverflowError):
        return None
    return result if math.isfinite(result) else None


def _time(value: Any) -> datetime | None:
    if isinstance(value, str):
        try:
            value = datetime.fromisoformat(value.replace("Z", "+00:00"))
        except ValueError:
            return None
    if not isinstance(value, datetime) or value.tzinfo is None or value.utcoffset() is None:
        return None
    return value.astimezone(timezone.utc)


def _group(value: Any) -> str:
    from app.data.market_capabilities import normalize_market_group
    text = getattr(value, "value", value)
    group = normalize_market_group(text)
    return group.value if group is not None else str(text or "").upper()


def _meta_identity(item: Any) -> tuple[str, str, str, str]:
    meta = _get(item, "meta")
    return (_group(_get(meta, "market_group")), str(_get(meta, "exchange", "")).upper(),
            str(getattr(_get(meta, "session"), "value", _get(meta, "session", ""))),
            str(_get(meta, "stream_id", "")))


class OntologyPolicyRuntime:
    """Resolve fresh policies and retain a bounded inspectable decision snapshot."""

    def __init__(
        self, store: Any, *, context_provider: Callable[[], Mapping[str, Any]],
        graph_advisory_provider: Callable[[str, datetime], Mapping[str, Any] | None] | None = None,
        max_cache_entries: int = 256, bar_cache_seconds: float = 10.0,
    ) -> None:
        self._store = store
        self._context_provider = context_provider
        self._graph_advisory_provider = graph_advisory_provider
        self._maximum = max(1, min(2048, int(max_cache_entries)))
        self._bar_cache_seconds = max(0.0, min(30.0, float(bar_cache_seconds)))
        self._lock = threading.RLock()
        self._bars: OrderedDict[tuple[str, str], tuple[datetime, tuple[Any, ...]]] = OrderedDict()
        self._latest: OrderedDict[tuple[str, str], tuple[OntologyRiskPolicy, EvidenceProjection]] = OrderedDict()

    def resolve(
        self, *, symbol: str, market: str, now: datetime, all_in_cost_rate: float,
        forecast_gross_bps: float | None = None, requested_horizon_seconds: float = 900,
        account: Any = None,
    ) -> OntologyRiskPolicy:
        moment = _time(now)
        if moment is None:
            raise ValueError("now must be a timezone-aware datetime")
        group, symbol = _group(market), str(symbol).strip().upper()
        records: list[PolicyObservation] = []
        adapter_reasons: list[str] = []
        try:
            contexts = self._context_provider() or {}
            cycle = contexts.get(group)
        except Exception:
            cycle = None
            adapter_reasons.append("ONTO_POLICY_CONTEXT_UNAVAILABLE")
        regime = _get(cycle, "regime")
        cycle_id = str(_get(cycle, "cycle_id", "") or "")
        decisions = _get(cycle, "decisions", ()) or ()
        trace = next((item for item in decisions if _get(item, "ticker") == symbol), decisions[0] if decisions else None)
        # Regime snapshots are per-market, but their volatility is not a known
        # bar interval. Local symbol sigma below is the only sizing volatility.
        context = project_context_evidence(
            group, as_of=moment, regime=regime, temporal=_get(cycle, "temporal"),
            domestic=_get(trace, "domestic_context") or None,
            global_context=_get(trace, "global_context") or None,
            context_id=cycle_id,
        )
        # Liquidity has an executable book-level measurement below; retaining a
        # market-average value with a newer cycle time would override that fact.
        records.extend(item for item in context.observations if item.metric not in {"realized_volatility", "liquidity_score"})
        try:
            bars = self._completed_bars(symbol, group, moment)
        except Exception:
            bars = ()
            adapter_reasons.append("ONTO_POLICY_BAR_READ_FAILED")
        bar_records = self._bar_observations(symbol, group, moment, bars)
        records.extend(bar_records)
        try:
            book = self._store.latest_orderbook(symbol)
        except Exception:
            book = None
            adapter_reasons.append("ONTO_POLICY_BOOK_READ_FAILED")
        records.extend(self._book_observations(symbol, group, moment, book, bars))
        if account is not None:
            # AccountSnapshot has no cash-flow-adjusted high-watermark ledger.
            # Only an explicitly supplied measured drawdown may be represented.
            drawdown = _number(_get(account, "drawdown_rate"))
            captured = _time(_get(account, "captured_at"))
            if drawdown is not None and captured is not None:
                records.append(PolicyObservation(
                    "drawdown_rate", drawdown, group, captured, "account_snapshot", 120,
                    derived_from=("account:" + captured.isoformat(),),
                ))
        if self._graph_advisory_provider is not None:
            try:
                advisory = self._graph_advisory_provider(symbol, moment)
            except Exception:
                advisory = None
                adapter_reasons.append("ONTO_POLICY_GRAPH_ADVISORY_UNAVAILABLE")
            records.extend(self._graph_observations(symbol, group, moment, advisory))
        projected = project_market_evidence(
            group, as_of=moment, observations=records, required_metrics=_REQUIRED,
            regime=context.regime, context_id=cycle_id,
        )
        # Missing required observations already fail closed. Optional advisory
        # errors remain diagnostics, not fabricated neutral readings.
        if adapter_reasons:
            projected = replace(projected, reason_codes=tuple(dict.fromkeys((*projected.reason_codes, *adapter_reasons))))
        policy = resolve_ontology_policy(
            projected, symbol=symbol, all_in_cost_rate=all_in_cost_rate,
            forecast_gross_bps=forecast_gross_bps,
            requested_horizon_seconds=requested_horizon_seconds,
        )
        # A measured source may expire earlier than the general policy lifetime.
        # A 19-second-old quote with a 20-second allowance has one second left.
        expirations = [item.observed_at + timedelta(seconds=item.max_age_seconds)
                       for item in projected.observations]
        expirations.extend(
            item.observed_at + timedelta(seconds=policy.max_quote_age_seconds)
            for item in projected.observations if item.metric == "quote_age_seconds"
        )
        if expirations:
            policy = replace(policy, expires_at=min(policy.expires_at, *expirations))
        key = (group, symbol)
        with self._lock:
            self._latest[key] = (policy, projected)
            self._latest.move_to_end(key)
            while len(self._latest) > self._maximum:
                self._latest.popitem(last=False)
        return policy

    def _completed_bars(self, symbol: str, market: str, now: datetime) -> tuple[Any, ...]:
        key = (market, symbol)
        with self._lock:
            cached = self._bars.get(key)
            # Never reuse a later decision's data in a historical decision.
            if cached is not None and 0 <= (now - cached[0]).total_seconds() <= self._bar_cache_seconds and cached[0].replace(second=0, microsecond=0) == now.replace(second=0, microsecond=0):
                self._bars.move_to_end(key)
                return cached[1]
        raw = tuple(self._store.recent_minute_bars(symbol, now - timedelta(minutes=66), limit=64))
        candidates: dict[datetime, Any] = {}
        conflicts: set[datetime] = set()
        for bar in raw[-64:]:
            start = _time(_get(bar, "minute_start"))
            if start is None or start.second or start.microsecond or start + timedelta(seconds=60) > now:
                continue
            if _get(bar, "symbol") != symbol or _meta_identity(bar)[0] != market:
                continue
            if start in candidates and (float(_get(bar, "close", 0)), _meta_identity(bar)) != (float(_get(candidates[start], "close", 0)), _meta_identity(candidates[start])):
                conflicts.add(start)
            candidates[start] = bar
        ordered = sorted(candidates.items())
        contiguous: list[Any] = []
        previous: datetime | None = None
        identity: tuple[str, str, str, str] | None = None
        for start, bar in reversed(ordered):
            current_identity = _meta_identity(bar)
            meta = _get(bar, "meta")
            close = _number(_get(bar, "close"))
            if start in conflicts or close is None or close <= 0 or not _get(bar, "source_record_ids") or _get(meta, "metadata_inferred", True):
                break
            if not current_identity[1] or current_identity[2] in {"", "UNKNOWN"} or not current_identity[3]:
                break
            if previous is not None and ((previous - start).total_seconds() != 60 or current_identity != identity):
                break
            contiguous.append(bar)
            previous, identity = start, current_identity
        result = tuple(reversed(contiguous))
        with self._lock:
            self._bars[key] = (now, result)
            self._bars.move_to_end(key)
            while len(self._bars) > self._maximum:
                self._bars.popitem(last=False)
        return result

    @staticmethod
    def _bar_observations(symbol: str, market: str, now: datetime, bars: tuple[Any, ...]) -> list[PolicyObservation]:
        if len(bars) < 6:
            return []
        prices = [float(_get(bar, "close")) for bar in bars]
        returns = [math.log(second / first) for first, second in zip(prices, prices[1:])]
        if len(returns) < 5 or not all(math.isfinite(value) for value in returns):
            return []
        observed = _time(_get(bars[-1], "minute_start")) + timedelta(seconds=60)
        references = tuple("bar:" + symbol + ":" + _get(bar, "minute_start").isoformat() + ":" + str(_get(bar, "source_record_ids")[0]) for bar in bars)
        return [PolicyObservation(
            metric, value, market, observed, "live_market_snapshot", 120,
            derived_from=references, horizon_seconds=60,
        ) for metric, value in (
            ("realized_volatility", statistics.stdev(returns)),
            ("downside_volatility", math.sqrt(sum(min(0.0, value) ** 2 for value in returns) / len(returns))),
        )]

    @staticmethod
    def _book_observations(symbol: str, market: str, now: datetime, book: Any, bars: tuple[Any, ...]) -> list[PolicyObservation]:
        from app.data.realtime_types import KIS_REALTIME_SOURCE
        if book is None or _get(book, "source") != KIS_REALTIME_SOURCE or _get(book, "symbol") != symbol:
            return []
        meta = _get(book, "meta")
        eligibility = getattr(meta, "is_live_buy_eligible", None)
        if eligibility is None or not eligibility()[0] or _meta_identity(book)[0] != market:
            return []
        event, received = _time(_get(book, "exchange_timestamp")), _time(_get(book, "received_at"))
        if event is None or received is None or event > now or received > now:
            return []
        bid, ask = _number(_get(book, "best_bid")), _number(_get(book, "best_ask"))
        if bid is None or ask is None or bid <= 0 or ask < bid:
            return []
        observed = min(event, received)
        references = (str(_get(book, "record_id", "")), "exchange:" + event.isoformat(), "received:" + received.isoformat())
        records = [PolicyObservation(
            "spread_rate", (ask - bid) / ((ask + bid) / 2), market, observed,
            "kis_realtime", 20, derived_from=references,
        ), PolicyObservation(
            "quote_age_seconds", (now - observed).total_seconds(), market, observed,
            "kis_realtime", 20, unit="seconds", derived_from=references,
        )]
        levels = _get(book, "levels", ())
        if bars and levels and _meta_identity(bars[-1])[:3] == _meta_identity(book)[:3]:
            volumes = [_number(_get(bar, "volume")) for bar in bars]
            positive = [volume for volume in volumes if volume is not None and volume > 0]
            bid_size, ask_size = _number(_get(levels[0], "bid_size")), _number(_get(levels[0], "ask_size"))
            if positive and bid_size is not None and ask_size is not None and min(bid_size, ask_size) > 0:
                depth = min(bid_size, ask_size)
                turnover = statistics.median(positive)
                bar_end = _time(_get(bars[-1], "minute_start")) + timedelta(seconds=60)
                if (now - bar_end).total_seconds() > 120:
                    return records
                records.append(PolicyObservation(
                    "liquidity_score", depth / (depth + turnover), market,
                    observed, "orderbook_snapshot", 20,
                    derived_from=(*references, "minute-volume-median:" + str(len(positive)), "volume-window-ended:" + bar_end.isoformat()),
                ))
        return records

    @staticmethod
    def _graph_observations(symbol: str, market: str, now: datetime, advisory: Any) -> list[PolicyObservation]:
        if advisory is None:
            return []
        if hasattr(advisory, "as_dict"):
            advisory = advisory.as_dict()
        if not isinstance(advisory, Mapping) or _group(advisory.get("market")) != market:
            return []
        if advisory.get("symbol", symbol) != symbol or advisory.get("source") != "validated_temporal_rgcn":
            return []
        if (advisory.get("label_execution_policy") != "ontology-risk-v1-entry-frozen-shadow"
                or advisory.get("authority") != "bounded_risk_advisory_only"):
            return []
        if advisory.get("available") is False or advisory.get("validated") is False:
            return []
        checkpoint, snapshot = str(advisory.get("checkpoint_hash") or ""), str(advisory.get("ontology_snapshot_id") or "")
        observed = _time(advisory.get("observed_at") or advisory.get("as_of"))
        valid_until = _time(advisory.get("valid_until"))
        if not checkpoint or not snapshot or observed is None or valid_until is None or not observed <= now <= valid_until:
            return []
        maximum_age = min(30.0, (valid_until - observed).total_seconds())
        if maximum_age <= 0:
            return []
        records = []
        # Entry-frozen shadow payoffs are not the realised payoffs of the live
        # policy, which may tighten its exits. Only bounded risk signals cross.
        for metric in ("model_uncertainty", "expected_downside_net_bps"):
            value = advisory.get(metric)
            if value is not None:
                records.append(PolicyObservation(
                    metric, value, market, observed, "validated_temporal_rgcn", maximum_age,
                    unit="bps" if metric.endswith("_bps") else "ratio",
                    derived_from=(checkpoint, snapshot, advisory["label_execution_policy"], advisory["authority"]),
                ))
        return records

    def snapshot(self) -> dict[str, Any]:
        """Cached diagnostics; no database read or model inference."""
        with self._lock:
            values = tuple(self._latest.items())
            bar_entries = len(self._bars)
        return {
            "policy_count": len(values), "bar_cache_entries": bar_entries,
            "maximum_cache_entries": self._maximum,
            "policies": {market + ":" + symbol: {"policy": policy.as_dict(), "projection": evidence.as_dict()}
                         for (market, symbol), (policy, evidence) in values},
        }

    def evidence_graph(self, symbol: str, market: str) -> Any:
        """Explicit audit export; callers must run it outside quote/order workers."""
        with self._lock:
            cached = self._latest.get((_group(market), str(symbol).strip().upper()))
        if cached is None:
            raise KeyError("No resolved policy for market/symbol")
        policy, projection = cached
        thresholds = {name: value for name, value in policy.as_dict().items()
                      if isinstance(value, (float, int)) and not isinstance(value, bool)}
        from app.models.strategy_utility.strategy_graph import materialize_strategy_graph

        graph = materialize_evidence_graph(projection, threshold_values=thresholds, instrument_symbol=policy.symbol)
        graph += materialize_strategy_graph(policy.market)
        return graph
