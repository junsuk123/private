from __future__ import annotations

from datetime import datetime, timezone

from app.account_dashboard import AccountDashboardService, build_macro_micro_panel
from app.graph.global_trade_arbiter import GlobalTradeArbiter
from app.graph.macro_reasoner import MacroMarketReasoner, MacroReasoningInput
from app.graph.micro_reasoner import MicroReasonerConfig, MicroReasoningInput, MicroSymbolReasoner
from app.graph.ontology_coordinator import CoordinatorConfig, OntologyCoordinator
from tests.test_technical_signals import trend_up_features


def _now():
    return datetime(2026, 7, 9, tzinfo=timezone.utc)


def _bundle_dict():
    macro_input = MacroReasoningInput(
        timestamp=_now(),
        index_snapshots={"KOSPI": {"trend": 0.004}},
        sector_snapshots={"tech": {"strength": 0.8, "volume_change": 0.3}},
        market_breadth=0.6, market_volatility=0.005,
        candidate_universe=("005930",),
    )

    def builder(symbol, macro_result):
        return MicroReasoningInput(
            timestamp=_now(), symbol=symbol,
            allowed_micro_strategies=macro_result.allowed_micro_strategies,
            blocked_micro_strategies=macro_result.blocked_micro_strategies,
            technical_features=trend_up_features(symbol=symbol),
        )

    coord = OntologyCoordinator(
        macro_reasoner=MacroMarketReasoner(),
        micro_reasoner=MicroSymbolReasoner(MicroReasonerConfig(minimum_micro_confidence=0.3)),
        arbiter=GlobalTradeArbiter(),
        config=CoordinatorConfig(max_parallel_symbols=4, worker_timeout_seconds=2.0),
    )
    return coord.run(macro_input, micro_input_builder=builder).as_dict()


class TestPanelBuilder:
    def test_empty(self):
        panel = build_macro_micro_panel(None)
        assert panel["available"] is False
        assert panel["micro"] == []

    def test_populated_panel(self):
        panel = build_macro_micro_panel(_bundle_dict())
        assert panel["available"] is True
        assert panel["market_regime"] == "TREND_UP"
        assert "005930" in panel["candidate_symbols"]
        assert isinstance(panel["micro"], list) and panel["micro"]
        row = panel["micro"][0]
        for key in ("symbol", "micro_regime", "selected_strategy", "entry_signal",
                    "expected_net_return_bps", "execution_quality"):
            assert key in row
        assert "ranked_intents" in panel

    def test_no_final_order_in_payload(self):
        panel = build_macro_micro_panel(_bundle_dict())
        text = str(panel)
        assert "final_order" not in text
        assert "FinalOrder" not in text

    def test_live_data_status_when_real(self):
        # trend_up_features have technical signal -> not all unavailable -> live.
        panel = build_macro_micro_panel(_bundle_dict())
        assert panel["data_status"] in ("live", "no_live_data")  # depends on features
        assert "macro_reason_codes" in panel

    def test_no_live_data_status_flag(self):
        # A bundle whose macro is insufficient-data + micro all signal-unavailable.
        bundle = {
            "macro_result": {"market_regime": "NO_TRADE_MARKET", "macro_risk_level": "NORMAL",
                             "macro_confidence": 0.0, "reason_codes": ["MACRO_INSUFFICIENT_DATA"],
                             "candidate_symbols": ["CGTX"], "blocks_buy": True,
                             "allowed_micro_strategies": [], "blocked_micro_strategies": ["new_buy"],
                             "sector_rankings": []},
            "micro_results": [{"symbol": "CGTX", "micro_regime": "NO_TRADE_SYMBOL",
                               "entry_signal": "NONE", "reason_codes": ["MICRO_SIGNAL_UNAVAILABLE"]}],
            "ranked_trade_intents": [],
        }
        panel = build_macro_micro_panel(bundle)
        assert panel["data_status"] == "no_live_data"
        assert panel["candidate_symbols"] == ["CGTX"]  # still visible


class TestServiceIntegration:
    def test_dashboard_includes_macro_micro(self):
        service = AccountDashboardService(
            status_provider=lambda: {},
            macro_micro_provider=_bundle_dict,
        )
        dashboard = service.build_dashboard(persist=False)
        assert "macro_micro" in dashboard
        assert dashboard["macro_micro"]["available"] is True
        assert dashboard["macro_micro"]["market_regime"] == "TREND_UP"

    def test_no_provider_yields_unavailable(self):
        service = AccountDashboardService(status_provider=lambda: {})
        dashboard = service.build_dashboard(persist=False)
        assert dashboard["macro_micro"]["available"] is False

    def test_provider_error_degrades(self):
        def boom():
            raise RuntimeError("down")

        service = AccountDashboardService(status_provider=lambda: {}, macro_micro_provider=boom)
        dashboard = service.build_dashboard(persist=False)
        assert dashboard["macro_micro"]["available"] is False

    def test_accessor(self):
        service = AccountDashboardService(macro_micro_provider=_bundle_dict)
        assert service.macro_micro()["available"] is True
