"""The entry-blockade chain must name the layer that is actually blocking.

The panel exists because a single reason code from whichever layer failed last
named the GNN for 11,614 consecutive cycles while the real constraint was the
market session. These tests pin the equivalent failure one level up: a cycle
that returns early on a closed market publishes no ``live_armed`` flag and no
strategy-session block, and reading those absences as verdicts made the chain
report "라이브 제출이 무장되지 않음" and "서버 재시작 필요" every night and
weekend - two answers that send the operator to fix things that are not broken.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest

web = pytest.importorskip("app.web")


class _Engine:
    def __init__(self, status: dict) -> None:
        self._status = status

    def get_status(self) -> dict:
        return self._status


def _closed_market_status() -> dict:
    """What the engine reports when every scanned market is shut.

    ``run_once`` returns before it stamps live_armed or the session block, so
    the summary carries the reason code and nothing else.
    """
    return {
        "buy_enabled": True,
        "buy_disabled_reason": None,
        "last_summary": {"reason": "MARKET_SESSION_CLOSED", "submitted": 0},
        "strategy_session": {
            "phase": "SCANNING",
            "bandit_selected_arm": "no_trade",
            "bandit_conservative_edge_bps": -3.2,
            "bandit_reason_codes": ["NO_POSITIVE_CONSERVATIVE_EDGE"],
        },
    }


def _chain_for(monkeypatch: pytest.MonkeyPatch, status: dict) -> dict[str, dict]:
    monkeypatch.setattr(web, "_realtime_trading_engine", _Engine(status), raising=False)
    monkeypatch.setattr(
        web, "_realtime_trading_worker", SimpleNamespace(is_alive=lambda: True), raising=False
    )
    return {link["stage"]: link for link in web._entry_blockade_chain()}


def test_closed_market_does_not_report_the_arming_layer(monkeypatch: pytest.MonkeyPatch) -> None:
    chain = _chain_for(monkeypatch, _closed_market_status())

    # The engine is armed; it simply had nothing to do. Falling back to the
    # standing buy_enabled flag keeps the blame on the market session.
    assert chain["live_armed"]["ok"] is True
    assert chain["live_armed"]["data"]["evaluated_this_cycle"] is False
    assert chain["market_session"]["ok"] is False


def test_closed_market_does_not_demand_a_server_restart(monkeypatch: pytest.MonkeyPatch) -> None:
    chain = _chain_for(monkeypatch, _closed_market_status())

    # The bandit fields live on the engine's session object even when the last
    # cycle published no summary block, so "old code" is provably false here.
    assert "재시작" not in chain["strategy_election"]["detail"]
    assert chain["strategy_election"]["data"]["conservative_edge_bps"] == -3.2


def test_a_genuinely_disarmed_engine_is_still_reported(monkeypatch: pytest.MonkeyPatch) -> None:
    status = _closed_market_status()
    status["buy_enabled"] = False
    status["buy_disabled_reason"] = "DAILY_REALIZED_LOSS_BUY_STOP:-1600<=-1500"

    chain = _chain_for(monkeypatch, status)

    assert chain["live_armed"]["ok"] is False
    # The reason a human needs is the stop that fired, not a bare "not armed".
    assert "DAILY_REALIZED_LOSS_BUY_STOP" in chain["live_armed"]["detail"]


def test_live_cycle_arming_flag_wins_over_the_fallback(monkeypatch: pytest.MonkeyPatch) -> None:
    status = _closed_market_status()
    status["buy_enabled"] = True
    # A cycle that ran and disarmed mid-flight is authoritative over the
    # standing flag: the fallback only covers "not evaluated".
    status["last_summary"] = {"reason": "LIQUIDATION", "live_armed": False}

    chain = _chain_for(monkeypatch, status)

    assert chain["live_armed"]["ok"] is False
    assert chain["live_armed"]["data"]["evaluated_this_cycle"] is True


def test_soft_micro_holds_do_not_block_joint_symbol_strategy_election(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    status = _closed_market_status()
    status["last_summary"] = {
        "reason": "NO_POSITIVE_NET_GNN_EDGE",
        "buy_candidate_count": 2,
        "buy_candidate_sample": ["SOFI", "T"],
        "live_armed": True,
        "strategy_session": {
            "phase": "SCANNING",
            "bandit_selected_arm": "no_trade",
            "candidate_diagnostics": [
                {
                    "symbol": "SOFI",
                    "selected_strategy": "hold",
                    "execution_quality": "WEAK",
                    "reason_codes": ["TECHNICAL_EDGE_NON_POSITIVE"],
                },
                {
                    "symbol": "T",
                    "selected_strategy": "hold",
                    "execution_quality": "BLOCKED",
                    "reason_codes": ["LOW_LIQUIDITY_TECHNICAL_BLOCK"],
                },
            ],
        },
    }

    chain = _chain_for(monkeypatch, status)

    assert chain["micro_buy_intents"]["ok"] is True
    assert chain["micro_buy_intents"]["data"]["pair_eligible_count"] == 1
    assert chain["micro_buy_intents"]["data"]["immediate_signal_count"] == 0
    assert chain["micro_buy_intents"]["data"]["hard_blocked_symbols"] == ["T"]


def test_missing_bandit_fields_still_report_stale_code(monkeypatch: pytest.MonkeyPatch) -> None:
    status = _closed_market_status()
    status["strategy_session"] = {"phase": "SCANNING"}

    chain = _chain_for(monkeypatch, status)

    # The original signal must survive: an engine with no bandit fields anywhere
    # really is pre-refactor code.
    assert chain["strategy_election"]["ok"] is False
    assert "재시작" in chain["strategy_election"]["detail"]


def test_candidate_affordability_diagnostic_never_compares_us_ask_to_krw() -> None:
    summary = web._candidate_affordability_summary(
        (("NIO", "USD", 4.50),),
        {"KRW": 490_097.0, "USD": 0.0},
    )

    assert summary["unaffordable"] is True
    assert summary["unaffordable_currencies"] == ["USD"]
    assert summary["cheapest_by_currency"]["USD"] == {
        "symbol": "NIO",
        "ask": 4.5,
    }
    assert summary["orderable_cash_by_currency"]["USD"] == 0.0


def test_candidate_blockade_exposes_same_cycle_filter_reasons(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    status = _closed_market_status()
    status["last_summary"] = {
        "reason": "NO_CANDIDATE_SYMBOLS",
        "buy_candidate_count": 0,
        "buy_candidate_sample": [],
        "live_armed": True,
    }
    status["live_trace"] = {"cycle_id": "2026-08-13T15:00:00+00:00"}
    monkeypatch.setattr(
        web,
        "_realtime_candidate_filter_state",
        {
            "engine_cycle_id": "2026-08-13T15:00:00+00:00",
            "input_count": 6,
            "selected_count": 0,
            "reason_counts": {"INSUFFICIENT_USD_ORDERABLE_CASH": 6},
            "reason_samples": {"INSUFFICIENT_USD_ORDERABLE_CASH": ["F", "SOFI"]},
        },
        raising=False,
    )

    chain = _chain_for(monkeypatch, status)

    data = chain["buy_candidates"]["data"]
    assert data["engine_cycle_id"] == "2026-08-13T15:00:00+00:00"
    assert data["candidate_filter_reason_counts"] == {
        "INSUFFICIENT_USD_ORDERABLE_CASH": 6
    }
    assert data["candidate_filter_reason_samples"] == {
        "INSUFFICIENT_USD_ORDERABLE_CASH": ["F", "SOFI"]
    }


def test_candidate_filter_diagnostics_do_not_add_a_dependency_gate(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class _Store:
        def active_symbols(self, *_args, **_kwargs):
            return ("SOFI",)

    monkeypatch.setattr(web, "RealtimeMarketDataStore", _Store)
    monkeypatch.setattr(web, "_realtime_engine_account_snapshot", lambda: None)
    monkeypatch.setattr(web, "_cached_context_buy_candidates", lambda **_kwargs: ("SOFI",))
    monkeypatch.setattr(web, "_cached_domestic_ranking_symbols", lambda: ())
    monkeypatch.setattr(
        web,
        "_prioritize_realtime_buy_candidates",
        lambda symbols, **_kwargs: tuple(symbols),
    )
    monkeypatch.setattr(web, "_is_live_buy_candidate_symbol", lambda _symbol: True)
    monkeypatch.setattr(web, "_is_open_live_market_ticker", lambda _symbol: True)
    monkeypatch.setattr(web, "_ticker_market_group_for_live_trading", lambda *_args: "US")
    monkeypatch.setattr(web, "_current_live_reasoning_trace", lambda: {"cycle_id": "cycle-1"})
    monkeypatch.setattr(web, "_kis_overseas_realtime_state", {"symbols": ["SOFI"]})
    diagnostic = {}
    monkeypatch.setattr(web, "_realtime_candidate_filter_state", diagnostic)
    monkeypatch.setattr(web, "_us_learning_watchlist_cache", {"symbols": []})

    selected = web._realtime_engine_buy_candidates()

    assert selected == ("SOFI",)
    assert diagnostic["selected_count"] == 1
    assert diagnostic["reason_counts"] == {
        "ACCOUNT_OR_STORE_UNAVAILABLE_NOT_FILTERED": 1
    }


def test_extended_session_is_not_described_as_regular_market(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    status = _closed_market_status()
    status["last_summary"] = {
        "reason": "NO_CANDIDATE_SYMBOLS",
        "buy_candidate_count": 0,
        "buy_candidate_sample": [],
        "live_armed": True,
    }
    monkeypatch.setattr(
        "app.data.market_session.new_entry_session_report",
        lambda: {
            "groups": {
                "KRX": {
                    "phase": "closed",
                    "streaming_phase": "pre",
                    "allows_new_entry": True,
                    "fully_closed": False,
                    "session": "NXT_PRE",
                },
                "US": {"allows_new_entry": False},
            },
            "extended_hours_entry_enabled": True,
        },
    )

    chain = _chain_for(monkeypatch, status)

    assert chain["market_session"]["ok"] is True
    assert "신규 진입 허용" in chain["market_session"]["detail"]
    assert "정규장" not in chain["market_session"]["detail"]
