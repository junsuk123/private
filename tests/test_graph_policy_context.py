from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timedelta, timezone
from types import SimpleNamespace

import pytest

from app.features.strategy_graph_context import STRATEGY_GRAPH_CONTEXT_DIM, STRATEGY_GRAPH_CONTEXT_SCHEMA, context_index
from app.models.strategy_utility.policy_context import (
    GraphPolicyContextCache, graph_training_snapshot_id, validate_graph_training_context,
)

NOW = datetime(2026, 9, 21, 4, 30, tzinfo=timezone.utc)


def snapshot(symbol="005930", **changes):
    raw = [0.0] * STRATEGY_GRAPH_CONTEXT_DIM
    raw[context_index("is_krx")] = float(symbol[0].isdigit() and len(symbol) == 6)
    values = dict(symbol=symbol, snapshot_id=f"live-strategy:{symbol}:observed-book-record",
                  feature_snapshot_id=f"live:{STRATEGY_GRAPH_CONTEXT_SCHEMA}:observed-book-record",
                  as_of=NOW, valid_until=NOW+timedelta(seconds=5), features=tuple(raw),
                  data_fresh=True, tradable=True, feature_schema_name=STRATEGY_GRAPH_CONTEXT_SCHEMA)
    values.update(changes)
    return SimpleNamespace(**values)


def test_raw_canonical_context_and_original_provenance_survive_detached_capture():
    cache = GraphPolicyContextCache(clock=lambda: NOW)
    frame = snapshot()
    assert cache.record(frame)
    result = cache.latest("005930", NOW)
    assert result is not None
    assert len(result["features"]) == STRATEGY_GRAPH_CONTEXT_DIM == 49
    assert result["source"] == "live_strategy_graph_context"
    assert result["source_snapshot_id"] == frame.snapshot_id
    assert result["source_feature_snapshot_id"] == frame.feature_snapshot_id
    assert result["source_provenance"] == [frame.snapshot_id, frame.feature_snapshot_id]
    assert result["snapshot_id"] == result["feature_snapshot_id"] == graph_training_snapshot_id(result)
    assert validate_graph_training_context(result, symbol="005930", market="KR", as_of=NOW)
    result["features"][0] = 99
    result["source_provenance"].clear()
    again = cache.latest("005930", NOW)
    assert again["features"][0] == 0
    assert again["source_provenance"]


@pytest.mark.parametrize("changes", [
    {"feature_schema_name": "old_schema"}, {"features": (0.0,) * 72},
    {"features": (0.0,)}, {"data_fresh": False}, {"data_fresh": 1},
    {"tradable": False}, {"snapshot_id": ""}, {"feature_snapshot_id": ""},
    {"as_of": NOW.replace(tzinfo=None)}, {"valid_until": NOW.replace(tzinfo=None)},
    {"as_of": NOW+timedelta(microseconds=1)}, {"valid_until": NOW-timedelta(seconds=1)},
    {"as_of": NOW-timedelta(seconds=6)}, {"market": "US"}, {"symbol": "bad/symbol"},
])
def test_invalid_capture_never_enters_training_cache(changes):
    cache = GraphPolicyContextCache(clock=lambda: NOW)
    assert not cache.record(snapshot(**changes))
    assert len(cache) == 0


@pytest.mark.parametrize("value", [float("nan"), float("inf"), -float("inf"), True])
def test_nonfinite_or_boolean_features_are_not_numeric_training_observations(value):
    original = snapshot()
    vector = list(original.features)
    vector[0] = value
    assert not GraphPolicyContextCache(clock=lambda: NOW).record(snapshot(features=vector))


def test_market_flag_and_symbol_must_match_without_cross_market_relabelling():
    cache = GraphPolicyContextCache(clock=lambda: NOW)
    wrong = snapshot("AAPL")
    vector = list(wrong.features)
    vector[context_index("is_krx")] = 1.0
    assert not cache.record(snapshot("AAPL", features=vector))
    assert cache.record(snapshot("AAPL"))
    context = cache.latest("AAPL", NOW)
    assert context["market"] == "US"
    assert not validate_graph_training_context(context, symbol="AAPL", market="KR", as_of=NOW)
    assert not validate_graph_training_context(context, symbol="MSFT", market="US", as_of=NOW)


@pytest.mark.parametrize("flag", [0.0, 1.0])
def test_alphanumeric_krx_code_is_rejected_until_graph_train_serve_classifier_supports_it(flag):
    raw = list(snapshot().features)
    raw[context_index("is_krx")] = flag
    assert not GraphPolicyContextCache(clock=lambda: NOW).record(snapshot("0004T0", features=raw))


def test_capture_must_precede_decision_and_remain_inside_exact_validity_window():
    cache = GraphPolicyContextCache(clock=lambda: NOW+timedelta(seconds=2))
    assert cache.record(snapshot())
    assert cache.latest("005930", NOW+timedelta(seconds=1)) is None
    assert cache.latest("005930", NOW+timedelta(seconds=2)) is not None
    assert cache.latest("005930", NOW+timedelta(seconds=5)) is not None
    assert cache.latest("005930", NOW+timedelta(seconds=5, microseconds=1)) is None
    assert cache.latest("005930", NOW-timedelta(microseconds=1)) is None
    assert cache.latest("005930", NOW.replace(tzinfo=None)) is None


def test_long_declared_validity_cannot_extend_a_live_context_beyond_five_seconds():
    cache = GraphPolicyContextCache(clock=lambda: NOW)
    assert cache.record(snapshot(valid_until=NOW+timedelta(hours=1)))
    assert cache.latest("005930", NOW)["valid_until"] == (NOW+timedelta(seconds=5)).isoformat()
    assert cache.latest("005930", NOW+timedelta(seconds=6)) is None


def test_same_timestamp_conflict_invalidates_context_and_older_data_never_replace_newer():
    clock = [NOW]
    cache = GraphPolicyContextCache(clock=lambda: clock[0])
    assert cache.record(snapshot())
    assert cache.record(snapshot())
    changed = list(snapshot().features)
    changed[0] = 1
    assert not cache.record(snapshot(features=changed))
    assert cache.latest("005930", NOW) is None
    clock[0] = NOW+timedelta(seconds=2)
    assert cache.record(snapshot(as_of=clock[0], valid_until=clock[0]+timedelta(seconds=5)))
    assert not cache.record(snapshot())
    assert cache.latest("005930", clock[0])["as_of"] == clock[0].isoformat()


def test_invalid_new_capture_cannot_leave_older_healthy_context_available():
    cache = GraphPolicyContextCache(clock=lambda: NOW)
    assert cache.record(snapshot())
    assert not cache.record(snapshot(data_fresh=False))
    assert cache.latest("005930", NOW) is None


@pytest.mark.parametrize("key,value", [
    ("features", [0.] * STRATEGY_GRAPH_CONTEXT_DIM), ("schema", "wrong"),
    ("source", "historically_reconstructed"), ("source_provenance", []),
    ("snapshot_id", "same-looking-id"), ("recorded_at", (NOW+timedelta(seconds=1)).isoformat()),
])
def test_label_reader_can_reject_changed_vector_or_provenance_with_unchanged_id(key, value):
    cache = GraphPolicyContextCache(clock=lambda: NOW)
    assert cache.record(snapshot())
    context = cache.latest("005930", NOW)
    context[key] = value
    assert not validate_graph_training_context(context, symbol="005930", market="KR", as_of=NOW)


def test_thread_safe_cache_never_exceeds_256_and_immutable_raw_capture():
    cache = GraphPolicyContextCache(max_entries=1000, clock=lambda: NOW)
    with ThreadPoolExecutor(max_workers=8) as executor:
        outcomes = list(executor.map(lambda number: cache.record(snapshot(f"{number:06d}")), range(400)))
    assert all(outcomes)
    assert len(cache) == 256
    for number in range(400):
        found = cache.latest(f"{number:06d}", NOW)
        if found is not None:
            assert validate_graph_training_context(found, symbol=f"{number:06d}", market="KR", as_of=NOW)
