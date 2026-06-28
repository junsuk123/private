from __future__ import annotations

import sys
import tempfile
import unittest
from datetime import datetime, timedelta, timezone
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from fastapi.testclient import TestClient

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from app.web import app
from app.risk import PrincipalProtectionEngine, RiskManager
from app.schemas import AccountSnapshot, MarketSnapshot, RiskRules
from app.schemas.domain import (
    OrderAction,
    OrderIntent,
    PrincipalProtectionConfig,
    PrincipalProtectionDecisionAction,
    PrincipalProtectionMode,
    SourceMetadata,
)


class PrincipalProtectionTest(unittest.TestCase):
    def test_protected_floor_equals_initial_principal_without_profit(self) -> None:
        account = AccountSnapshot(cash=1_000_000, holdings=())
        config = PrincipalProtectionConfig(initial_principal=1_000_000)

        state = PrincipalProtectionEngine().compute_state(account, config=config)

        self.assertEqual(state.protected_floor, 1_000_000)
        self.assertEqual(state.cushion, 0)
        self.assertEqual(state.current_mode, PrincipalProtectionMode.PRINCIPAL_LOCKDOWN)

    def test_protected_floor_ratchets_with_high_watermark(self) -> None:
        account = AccountSnapshot(cash=1_150_000, holdings=())
        config = PrincipalProtectionConfig(initial_principal=1_000_000, profit_lockin_ratio=0.3)

        state = PrincipalProtectionEngine().compute_state(account, config=config, high_watermark=1_200_000)

        self.assertEqual(state.high_watermark, 1_200_000)
        self.assertEqual(state.protected_floor, 1_060_000)
        self.assertEqual(state.locked_profit, 60_000)

    def test_buy_is_blocked_when_cushion_is_zero_but_sell_is_allowed(self) -> None:
        account = AccountSnapshot(cash=1_000_000, holdings=())
        market = _market()
        buy_intent = _intent(OrderAction.BUY)
        sell_intent = _intent(OrderAction.SELL)
        config = PrincipalProtectionConfig(initial_principal=1_000_000)
        engine = PrincipalProtectionEngine()

        buy = engine.validate_order(buy_intent, account, (), market, None, config, proposed_quantity=10)
        sell = engine.validate_order(sell_intent, account, (), market, None, config, proposed_quantity=10)

        self.assertEqual(buy.action, PrincipalProtectionDecisionAction.LOCKDOWN)
        self.assertFalse(buy.allowed)
        self.assertEqual(sell.action, PrincipalProtectionDecisionAction.ALLOW)
        self.assertTrue(sell.allowed)

    def test_position_size_is_reduced_when_trade_loss_exceeds_budget(self) -> None:
        account = AccountSnapshot(cash=1_100_000, holdings=(), realized_pnl_today=100_000)
        market = _market(last_price=100)
        intent = _intent(OrderAction.BUY, stop_loss_price=88)
        config = PrincipalProtectionConfig(initial_principal=1_000_000, profit_lockin_ratio=0.0)

        decision = PrincipalProtectionEngine().validate_order(
            intent,
            account,
            (),
            market,
            None,
            config,
            proposed_quantity=1_000,
        )

        self.assertEqual(decision.action, PrincipalProtectionDecisionAction.REDUCE_SIZE)
        self.assertIsNotNone(decision.suggested_quantity)
        self.assertLess(decision.suggested_quantity or 0, 1_000)

    def test_drawdown_violation_triggers_de_risk(self) -> None:
        account = AccountSnapshot(cash=1_100_000, holdings=())
        config = PrincipalProtectionConfig(initial_principal=1_000_000, max_total_drawdown=0.05)

        state = PrincipalProtectionEngine().compute_state(account, config=config, high_watermark=1_200_000)

        self.assertEqual(state.current_mode, PrincipalProtectionMode.DE_RISK)
        self.assertIn("MAX_DRAWDOWN_EXCEEDED", state.reason_codes)

    def test_risk_manager_blocks_ai_buy_when_principal_floor_would_be_breached(self) -> None:
        account = AccountSnapshot(cash=1_000_000, holdings=())
        market = _market(last_price=100)
        intent = _intent(OrderAction.BUY, stop_loss_price=88)
        rules = RiskRules(
            min_average_daily_trading_value=1,
            max_volatility=1.0,
            minimum_cash_reserve=0.0,
            principal_protection=PrincipalProtectionConfig(initial_principal=1_000_000),
        )

        result = RiskManager(rules).validate(intent, account, market)

        self.assertFalse(result.approved)
        self.assertIsNone(result.final_order)
        self.assertIn("PRINCIPAL_LOCKDOWN_BUY_BLOCKED", result.rejection_reasons)
        self.assertIn("principal_protection", result.metadata)

    def test_valid_small_buy_can_pass_existing_risk_and_principal_gate(self) -> None:
        account = AccountSnapshot(cash=1_100_000, holdings=(), realized_pnl_today=100_000)
        market = _market(last_price=100)
        intent = _intent(OrderAction.BUY, suggested_weight=0.01, stop_loss_price=98)
        rules = RiskRules(
            min_average_daily_trading_value=1,
            max_volatility=1.0,
            minimum_cash_reserve=0.0,
            principal_protection=PrincipalProtectionConfig(initial_principal=1_000_000, profit_lockin_ratio=0.0),
        )

        result = RiskManager(rules).validate(intent, account, market)

        self.assertTrue(result.approved, result.rejection_reasons)
        self.assertIsNotNone(result.final_order)
        self.assertEqual(result.checks["principal_protection_gate"], True)

    def test_principal_protection_api_exposes_backend_state(self) -> None:
        account = AccountSnapshot(cash=1_100_000, holdings=(), realized_pnl_today=100_000)
        context = SimpleNamespace(account=account, markets=(_market(last_price=100),))
        with tempfile.TemporaryDirectory() as tmp:
            config_path = Path(tmp) / "principal.json"
            state_path = Path(tmp) / "state.json"
            with (
                patch.dict(
                    "os.environ",
                    {
                        "PRINCIPAL_PROTECTION_CONFIG": str(config_path),
                        "PRINCIPAL_PROTECTION_STATE": str(state_path),
                    },
                    clear=False,
                ),
                patch("app.web._get_or_refresh_live", return_value={"context": context}),
            ):
                client = TestClient(app)
                config_response = client.put(
                    "/api/risk/principal-protection/config",
                    json={"initial_principal": 1_000_000, "profit_lockin_ratio": 0.0},
                )
                self.assertEqual(config_response.status_code, 200)
                state_payload = client.get("/api/risk/principal-protection/state").json()
                preview_payload = client.post(
                    "/api/risk/principal-protection/preview-order",
                    json={
                        "ticker": "005930",
                        "action": "BUY",
                        "suggested_weight": 0.01,
                        "stop_loss_price": 98,
                    },
                ).json()

        self.assertEqual(state_payload["state"]["initial_principal"], 1_000_000)
        self.assertIn("available_growth_capital", state_payload["state"])
        self.assertIn(preview_payload["decision"]["action"], {"ALLOW", "REDUCE_SIZE", "BLOCK", "LOCKDOWN", "SELL_ONLY"})


def _market(last_price: float = 100.0) -> MarketSnapshot:
    return MarketSnapshot(
        ticker="005930",
        market="KR",
        company_name="Samsung Electronics",
        sector="Technology",
        last_price=last_price,
        average_daily_trading_value=10_000_000_000,
        volatility_20d=0.02,
        source=SourceMetadata(
            source_name="unit",
            retrieved_at=datetime.now(timezone.utc),
            source_type="licensed",
            trust_level=5,
            quality_score=1.0,
        ),
    )


def _intent(
    action: OrderAction,
    suggested_weight: float = 0.10,
    stop_loss_price: float = 88.0,
) -> OrderIntent:
    return OrderIntent(
        ticker="005930",
        market="KR",
        action=action,
        suggested_weight=suggested_weight,
        confidence=0.9,
        valid_until=datetime.now(timezone.utc) + timedelta(minutes=5),
        reasoning_summary=("unit",),
        supporting_factors=("unit",),
        contradicting_factors=(),
        source_data_ids=("unit",),
        strategy_family="unit",
        expected_exit_price=120.0,
        target_net_return=0.0,
        strategy_metadata={"stop_loss_price": stop_loss_price},
    )


if __name__ == "__main__":
    unittest.main()
