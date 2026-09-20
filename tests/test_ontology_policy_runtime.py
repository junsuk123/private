from __future__ import annotations

from dataclasses import replace
from datetime import datetime, timedelta, timezone
from types import SimpleNamespace

import pytest
from rdflib import Namespace, RDF

from app.data.market_capabilities import FeedScope, MarketGroup, SessionId, Venue
from app.data.realtime_types import FeedMetadata, KIS_REALTIME_SOURCE, OrderbookLevel, RealtimeMinuteBar, RealtimeOrderbookSnapshot
from app.ontology.policy_evidence import POLICY_NAMESPACE, validate_evidence_graph
from app.trading.ontology_policy_runtime import OntologyPolicyRuntime

NOW = datetime(2026, 9, 21, 1, 30, 10, tzinfo=timezone.utc)
META = FeedMetadata(market_group=MarketGroup.KR, exchange="KRX", venue=Venue.KRX,
                    session=SessionId.KRX_REGULAR, currency="KRW", feed_scope=FeedScope.VENUE_SPECIFIC,
                    tr_id="H0STCNT0", is_tradeable=True, metadata_inferred=False)


def bars(symbol="005930"):
    prices = (10000, 10020, 10010, 10045, 10030, 10065, 10080)
    return tuple(RealtimeMinuteBar(
        symbol=symbol, minute_start=NOW.replace(second=0) - timedelta(minutes=7-index),
        open=price, high=price, low=price, close=price, volume=1000, vwap=price,
        trade_count=100, spread_bps=10, orderbook_imbalance=0.0, liquidity_score=.9,
        volatility=.999, last_update_age_ms=0, source_record_ids=("trade-" + str(index),), meta=META,
    ) for index, price in enumerate(prices))


def book(symbol="005930", **changes):
    return replace(RealtimeOrderbookSnapshot(
        symbol=symbol, exchange_timestamp=NOW - timedelta(seconds=1),
        received_at=NOW, source=KIS_REALTIME_SOURCE,
        levels=(OrderbookLevel(10079, 2000, 10081, 2000),),
        meta=replace(META, tr_id="H0STASP0"),
    ), **changes)


def context(at=NOW):
    regime = SimpleNamespace(evaluated_at=at, confidence=.9, routing_regime="TREND_LOW_VOL",
                              feature_snapshot={"direction": .6, "breadth": .4, "liquidity": .9,
                                                "volatility": .7})
    return {"KR": SimpleNamespace(cycle_id="kr-cycle-1", captured_at=at, regime=regime, temporal=None)}


class Store:
    def __init__(self):
        self.bars = bars()
        self.book = book()
        self.bar_reads = 0
        self.book_reads = 0

    def recent_minute_bars(self, symbol, since, *, limit):
        self.bar_reads += 1
        assert limit == 64
        return self.bars

    def latest_orderbook(self, symbol):
        self.book_reads += 1
        return self.book


def resolve(runtime, **changes):
    args = dict(symbol="005930", market="KR", now=NOW, all_in_cost_rate=.0005)
    args.update(changes)
    return runtime.resolve(**args)


def test_runtime_uses_measured_bar_sigma_instead_of_unscaled_context_or_tick_sigma():
    store = Store()
    runtime = OntologyPolicyRuntime(store, context_provider=context)
    policy = resolve(runtime)
    assert policy.valid_for_entry, policy.reason_codes
    evidence = {item["metric"]: item for item in policy.evidence}
    assert 0.0 < evidence["realized_volatility"]["value"] < .01
    assert evidence["realized_volatility"]["horizon_seconds"] == 60
    assert len(evidence["realized_volatility"]["derived_from"]) == 7
    assert evidence["quote_age_seconds"]["value"] == 1
    assert evidence["liquidity_score"]["value"] == pytest.approx(2/3)
    assert evidence["spread_rate"]["value"] == pytest.approx(2/10080)
    graph = runtime.evidence_graph("005930", "KR")
    assert validate_evidence_graph(graph)[0]
    pol = Namespace(POLICY_NAMESPACE)
    assert list(graph.subjects(RDF.type, pol.Strategy))
    assert list(graph.subjects(RDF.type, pol.Instrument))
    assert list(graph.triples((None, pol.sameMethodologyFamily, None)))


@pytest.mark.parametrize("corruption", ["gap", "too_short", "wrong_market", "wrong_session", "unknown_meta", "future_only", "conflicting_duplicate", "nonfinite", "no_provenance"])
def test_incomplete_or_incompatible_bar_series_never_create_sizing_volatility(corruption):
    store = Store()
    if corruption == "gap":
        store.bars = store.bars[:4] + store.bars[5:]
    elif corruption == "too_short":
        store.bars = store.bars[-5:]
    elif corruption == "wrong_market":
        store.bars = tuple(replace(item, meta=replace(META, market_group=MarketGroup.US)) for item in store.bars)
    elif corruption == "wrong_session":
        store.bars = (*store.bars[:-2], *(replace(item, meta=replace(META, session=SessionId.NXT_PRE)) for item in store.bars[-2:]))
    elif corruption == "unknown_meta":
        store.bars = tuple(replace(item, meta=FeedMetadata()) for item in store.bars)
    elif corruption == "future_only":
        store.bars = tuple(replace(item, minute_start=item.minute_start + timedelta(minutes=20)) for item in store.bars)
    elif corruption == "conflicting_duplicate":
        store.bars = (*store.bars, replace(store.bars[-1], close=20000))
    elif corruption == "nonfinite":
        store.bars = (*store.bars[:-1], replace(store.bars[-1], close=float("nan")))
    elif corruption == "no_provenance":
        store.bars = tuple(replace(item, source_record_ids=()) for item in store.bars)
    policy = resolve(OntologyPolicyRuntime(store, context_provider=context))
    assert not policy.valid_for_entry
    assert "realized_volatility" not in {item["metric"] for item in policy.evidence}
    assert "POLICY_METRIC_MISSING:realized_volatility" in policy.reason_codes


def test_in_progress_bar_does_not_change_completed_bar_sigma():
    store = Store()
    runtime = OntologyPolicyRuntime(store, context_provider=context)
    original = resolve(runtime)
    store.bars = (*store.bars, replace(store.bars[-1], minute_start=NOW.replace(second=0), close=1))
    updated = resolve(OntologyPolicyRuntime(store, context_provider=context))
    first = next(item["value"] for item in original.evidence if item["metric"] == "realized_volatility")
    second = next(item["value"] for item in updated.evidence if item["metric"] == "realized_volatility")
    assert first == second


@pytest.mark.parametrize("corruption", ["event_stale", "receive_stale", "future_event", "future_receive", "crossed", "empty", "rest", "wrong_market", "inferred"])
def test_quote_requires_live_source_identity_both_clocks_and_real_two_sided_prices(corruption):
    store = Store()
    if corruption == "event_stale":
        store.book = book(exchange_timestamp=NOW-timedelta(minutes=5))
    elif corruption == "receive_stale":
        store.book = book(received_at=NOW-timedelta(minutes=5))
    elif corruption == "future_event":
        store.book = book(exchange_timestamp=NOW+timedelta(seconds=1))
    elif corruption == "future_receive":
        store.book = book(received_at=NOW+timedelta(seconds=1))
    elif corruption == "crossed":
        store.book = book(levels=(OrderbookLevel(10100, 1000, 10000, 1000),))
    elif corruption == "empty":
        store.book = book(levels=())
    elif corruption == "rest":
        store.book = book(source="kis_rest_snapshot")
    elif corruption == "wrong_market":
        store.book = book(meta=replace(META, market_group=MarketGroup.US))
    elif corruption == "inferred":
        store.book = book(meta=replace(META, metadata_inferred=True))
    policy = resolve(OntologyPolicyRuntime(store, context_provider=context))
    assert not policy.valid_for_entry
    assert "spread_rate" not in {item["metric"] for item in policy.evidence}


def test_context_is_market_partitioned_and_stale_cycles_cannot_be_renewed():
    store = Store()
    for contexts in ({"US": context()["KR"]}, context(NOW-timedelta(minutes=5))):
        policy = resolve(OntologyPolicyRuntime(store, context_provider=lambda: contexts))
        assert not policy.valid_for_entry
        assert "regime_confidence" not in {item["metric"] for item in policy.evidence}


def test_cycle_global_context_uses_verified_cross_market_relations_and_original_time():
    contexts = context()
    observed = NOW-timedelta(hours=20)
    global_context = {"context_id": "global-1", "market": "KR", "captured_at": NOW.isoformat(),
                      "direction": -.4, "risk_sentiment": -.5,
                      "indicator_relations": [{"indicator": "SP500", "source": "fred",
                         "target_market": "KR", "origin_market": "US", "usable": True,
                         "observed_at": observed.isoformat()}]}
    contexts["KR"].decisions = (SimpleNamespace(ticker="005930", global_context=global_context, domestic_context={}),)
    runtime = OntologyPolicyRuntime(Store(), context_provider=lambda: contexts)
    policy = resolve(runtime)
    metrics = {item["metric"]: item for item in policy.evidence}
    assert metrics["global_direction"]["value"] == -.4
    assert metrics["global_direction"]["observed_at"] == observed.isoformat()
    assert metrics["realized_volatility"]["source"] == "live_market_snapshot"
    global_context["indicator_relations"][0]["source"] = "random_blog"
    rejected = resolve(runtime)
    assert "global_direction" not in {item["metric"] for item in rejected.evidence}


def test_bar_cache_is_bounded_quotes_refresh_and_same_minute_freshness_ages():
    store = Store()
    runtime = OntologyPolicyRuntime(store, context_provider=context, max_cache_entries=2)
    resolve(runtime)
    resolve(runtime, now=NOW+timedelta(seconds=3))
    assert store.bar_reads == 1
    assert store.book_reads == 2
    for symbol in ("000001", "000002", "000003"):
        resolve(runtime, symbol=symbol)
    snapshot = runtime.snapshot()
    assert snapshot["policy_count"] == 2
    assert snapshot["bar_cache_entries"] == 2
    with pytest.raises(KeyError):
        runtime.evidence_graph("005930", "KR")


def test_cached_future_bars_are_not_reused_for_an_earlier_decision():
    store = Store()
    runtime = OntologyPolicyRuntime(store, context_provider=context)
    resolve(runtime)
    earlier = resolve(runtime, now=NOW-timedelta(minutes=10))
    assert store.bar_reads == 2
    assert not earlier.valid_for_entry
    assert "realized_volatility" not in {item["metric"] for item in earlier.evidence}


def test_conflicting_duplicate_bar_volume_cannot_change_executable_liquidity():
    store = Store()
    store.bars = (*store.bars, replace(store.bars[-1], volume=1))
    policy = resolve(OntologyPolicyRuntime(store, context_provider=context))
    assert not policy.valid_for_entry
    assert "liquidity_score" not in {item["metric"] for item in policy.evidence}


def test_graph_advisory_exact_validity_and_checkpoint_provenance_are_preserved():
    advisory = {"symbol": "005930", "market": "KR", "as_of": NOW, "valid_until": NOW+timedelta(seconds=5),
                "available": True, "validated": True,
                "source": "validated_temporal_rgcn", "checkpoint_hash": "checkpoint", "ontology_snapshot_id": "graph1",
                "label_execution_policy": "ontology-risk-v1-entry-frozen-shadow", "authority": "bounded_risk_advisory_only",
                "model_uncertainty": .25, "expected_net_return_bps": -10, "expected_downside_net_bps": 25,
                "probability_success": .4}
    runtime = OntologyPolicyRuntime(Store(), context_provider=context, graph_advisory_provider=lambda symbol, now: advisory)
    policy = resolve(runtime)
    metrics = {item["metric"]: item for item in policy.evidence}
    assert "expected_net_return_bps" not in metrics
    assert "probability_success" not in metrics
    assert metrics["expected_downside_net_bps"]["value"] == 25
    assert metrics["expected_downside_net_bps"]["source"] == "validated_temporal_rgcn"
    assert policy.expires_at <= NOW+timedelta(seconds=5)
    later = resolve(runtime, now=NOW+timedelta(seconds=6))
    assert "model_uncertainty" not in {item["metric"] for item in later.evidence}
    for key in ("symbol", "available", "validated", "checkpoint_hash", "ontology_snapshot_id", "valid_until", "label_execution_policy", "authority"):
        incomplete = {name: value for name, value in advisory.items() if name != key}
        assert not runtime._graph_observations("005930", "KR", NOW, incomplete)


def test_policy_expiry_subtracts_already_elapsed_quote_age_from_dynamic_budget():
    store = Store()
    store.book = book(exchange_timestamp=NOW-timedelta(seconds=4))
    runtime = OntologyPolicyRuntime(store, context_provider=context)
    policy = resolve(runtime)
    assert policy.expires_at <= NOW-timedelta(seconds=4)+timedelta(seconds=policy.max_quote_age_seconds)
    assert policy.expires_at - NOW < timedelta(seconds=policy.max_quote_age_seconds)


def test_missing_store_or_context_fail_closed_but_protective_exits_have_nonzero_barrier():
    class Broken:
        def latest_orderbook(self, *args, **kwargs):
            raise OSError("offline")
        recent_minute_bars = latest_orderbook
    runtime = OntologyPolicyRuntime(Broken(), context_provider=lambda: {})
    policy = resolve(runtime)
    assert not policy.valid_for_entry
    assert policy.position_cap == 0
    assert policy.hard_stop_rate > 0
    assert "ONTO_POLICY_BOOK_READ_FAILED" in policy.reason_codes


def test_real_store_adapter_read_and_snapshot_have_no_broker_dependency(tmp_path):
    from app.data.realtime_store import RealtimeMarketDataStore
    store = RealtimeMarketDataStore(tmp_path / "realtime.sqlite3")
    store.save_minute_bars(bars())
    store.save_orderbooks((book(),))
    runtime = OntologyPolicyRuntime(store, context_provider=context)
    policy = resolve(runtime)
    assert policy.valid_for_entry, policy.reason_codes
    assert runtime.snapshot()["policy_count"] == 1
