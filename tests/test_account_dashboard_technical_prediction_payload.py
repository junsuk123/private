from __future__ import annotations

from app.account_dashboard import AccountDashboardService, build_technical_panel


def _buy_approved():
    return {
        "symbol": "005930",
        "approved": True,
        "action": "BUY",
        "reason_codes": [],
        "technical_methodology": "momentum_trend_following",
        "technical_regime": "TREND_UP",
        "technical": {
            "methodology": "momentum_trend_following",
            "regime": "TREND_UP",
            "expected_net_return_bps": 18.0,
            "expected_horizon_seconds": 60,
            "expected_exit_price": 100.2,
            "downside_risk_bps": 40.0,
            "confidence": 0.7,
            "explanation": "momentum in TREND_UP",
        },
        "profitability": {"allowed": True, "net_expected_return": 0.0012, "required_min_net_return": 0.0008},
    }


def _buy_rejected(codes, symbol="000660"):
    return {
        "symbol": symbol,
        "approved": False,
        "action": "BUY",
        "reason_codes": codes,
        "technical": {"regime": "TREND_UP", "methodology": "breakout_trading_range_break"},
        "profitability": {"allowed": False},
    }


class TestPanelBuilder:
    def test_empty(self):
        panel = build_technical_panel([])
        assert panel["available"] is False
        assert panel["count"] == 0
        assert panel["cards"] == []

    def test_buy_approved_categorized(self):
        panel = build_technical_panel([_buy_approved()])
        assert panel["available"] is True
        assert len(panel["buy_approved"]) == 1
        card = panel["buy_approved"][0]
        assert card["symbol"] == "005930"
        assert card["methodology"] == "momentum_trend_following"
        assert card["expected_edge_bps"] == 18.0
        assert card["gate_allowed"] is True

    def test_reject_reason_mapping(self):
        cases = {
            "HIGH_VOLATILITY_TECHNICAL_BLOCK": "high_volatility",
            "LOW_LIQUIDITY_TECHNICAL_BLOCK": "low_liquidity",
            "SPREAD_CONSUMES_TECHNICAL_ALPHA": "spread_consumes_alpha",
            "MODEL_UNAVAILABLE": "model_feature_unavailable",
            "ONTOLOGY_REQUIRED_FOR_MODEL_FALLBACK": "no_ontology_support",
            "PROFITABILITY_GATE_REJECTED": "below_net_edge",
        }
        for code, expected in cases.items():
            panel = build_technical_panel([_buy_rejected([code])])
            assert panel["buy_rejected"][0]["reject_reason"] == expected, code

    def test_sell_and_hold_categories(self):
        decisions = [
            {"symbol": "A", "action": "SELL", "approved": True, "reason_codes": [], "technical": {}},
            {"symbol": "B", "action": "REDUCE", "approved": True, "reason_codes": [], "technical": {}},
            {"symbol": "C", "action": "HOLD", "approved": False, "reason_codes": [], "technical": {}},
        ]
        panel = build_technical_panel(decisions)
        assert len(panel["sell"]) == 1
        assert len(panel["reduce"]) == 1
        assert len(panel["hold"]) == 1

    def test_malformed_entries_ignored(self):
        panel = build_technical_panel([None, "bad", 42, _buy_approved()])
        assert panel["count"] == 1


class TestServiceIntegration:
    def test_dashboard_includes_technical_section(self):
        service = AccountDashboardService(
            status_provider=lambda: {},
            technical_provider=lambda: [_buy_approved()],
        )
        dashboard = service.build_dashboard(persist=False)
        assert "technical" in dashboard
        assert dashboard["technical"]["available"] is True
        assert dashboard["technical"]["buy_approved"][0]["symbol"] == "005930"

    def test_no_provider_yields_empty_panel(self):
        service = AccountDashboardService(status_provider=lambda: {})
        dashboard = service.build_dashboard(persist=False)
        assert dashboard["technical"]["available"] is False

    def test_provider_error_degrades_gracefully(self):
        def boom():
            raise RuntimeError("provider down")

        service = AccountDashboardService(status_provider=lambda: {}, technical_provider=boom)
        dashboard = service.build_dashboard(persist=False)
        assert dashboard["technical"]["available"] is False

    def test_technical_accessor(self):
        service = AccountDashboardService(technical_provider=lambda: [_buy_approved()])
        assert service.technical()["count"] == 1
