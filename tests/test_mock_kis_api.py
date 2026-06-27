from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from app.data.sample_collectors import collect_sample_market
from app.execution import KisCredentials, KisDevelopersApiClient, MockKisDevelopersApi, load_kis_env_file
from app.goals import NegotiatedGoal
from app.graph import OntologyReasoner
from app.graph.builders import build_market_graph
from app.indicators import build_sample_indicators
from app.risk import RiskManager
from app.schemas.domain import AccountSnapshot, FinalOrder, OrderSide, OrderType
from app.strategy import build_goal_execution_plan
from app.trading import run_mock_trading_cycle


class RecordingKisTransport:
    def __init__(self) -> None:
        self.calls = []

    def request(self, method, url, headers, body=None, params=None, timeout=10.0):
        self.calls.append(
            {
                "method": method,
                "url": url,
                "headers": dict(headers),
                "body": dict(body or {}),
                "params": dict(params or {}),
            }
        )
        if url.endswith("/oauth2/tokenP"):
            return {"access_token": "paper-token", "expires_in": 86400}
        if url.endswith("/uapi/hashkey"):
            return {"HASH": "paper-hash"}
        if url.endswith("/uapi/domestic-stock/v1/trading/order-cash"):
            return {"rt_cd": "0", "msg1": "accepted", "output": {"ODNO": "0000000001"}}
        if url.endswith("/uapi/domestic-stock/v1/trading/inquire-daily-ccld"):
            return {
                "rt_cd": "0",
                "output1": [
                    {
                        "pdno": "005930",
                        "tot_ccld_qty": "2",
                        "avg_prvs": "70000",
                        "sll_buy_dvsn_cd": "02",
                    }
                ],
            }
        if url.endswith("/uapi/domestic-stock/v1/trading/inquire-balance"):
            return {
                "rt_cd": "0",
                "output1": [
                    {
                        "pdno": "005930",
                        "prdt_name": "Samsung Electronics",
                        "hldg_qty": "2",
                        "pchs_avg_pric": "70000",
                        "prpr": "71000",
                    }
                ],
                "output2": [{"dnca_tot_amt": "1000000"}],
            }
        raise AssertionError(f"unexpected KIS request: {method} {url}")


class MockKisApiTest(unittest.TestCase):
    def test_kis_credentials_can_load_ignored_secret_file(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            env_path = Path(tmp) / "kis_api_keys.env"
            env_path.write_text(
                "\n".join(
                    [
                        "KIS_PAPER_APP_KEY=paper-app",
                        "KIS_PAPER_APP_SECRET=paper-secret",
                        "KIS_PAPER_ACCOUNT_NO=12345678-01",
                    ]
                ),
                encoding="utf-8",
            )
            with patch.dict(
                "os.environ",
                {
                    "KIS_PAPER_APP_KEY": "",
                    "KIS_PAPER_APP_SECRET": "",
                    "KIS_PAPER_ACCOUNT_NO": "",
                },
                clear=False,
            ):
                self.assertTrue(load_kis_env_file(env_path, override=True))
                credentials = KisCredentials.from_env(paper=True)

        self.assertEqual(credentials.app_key, "paper-app")
        self.assertEqual(credentials.app_secret, "paper-secret")
        self.assertEqual(credentials.account_no, "12345678")
        self.assertEqual(credentials.account_product_code, "01")

    def test_mock_kis_limit_order_fills_and_updates_portfolio(self) -> None:
        markets = collect_sample_market()
        market = markets[0]
        account = AccountSnapshot(cash=10_000_000, holdings=())
        indicators = build_sample_indicators(markets)
        graph = build_market_graph(markets, indicators, npu_scores=_supportive_npu_scores(markets))
        plan = build_goal_execution_plan(
            NegotiatedGoal(0.03, 300_000, 60, 70, "unit"),
            account,
            markets,
            indicators,
            graph,
        )
        intent = next(item for item in plan.intents if item.ticker == market.ticker)
        risk_result = RiskManager().validate(intent, account, market)
        broker = MockKisDevelopersApi(
            account,
            {market.ticker: market.last_price},
            {market.ticker: market.sector},
            {market.ticker: market.company_name},
        )

        receipt = broker.place_limit_order(risk_result.final_order)
        execution = broker.get_order_status(receipt.order_id)
        portfolio = broker.get_portfolio()

        self.assertTrue(receipt.accepted)
        self.assertEqual(execution.status, "FILLED")
        self.assertEqual(execution.side, OrderSide.BUY)
        self.assertLess(portfolio.account.cash, account.cash)
        self.assertEqual(portfolio.account.holdings[0].ticker, market.ticker)

    def test_mock_trading_cycle_exposes_replaceable_broker_flow(self) -> None:
        markets = collect_sample_market()
        account = AccountSnapshot(cash=10_000_000, holdings=())
        indicators = build_sample_indicators(markets)
        graph = build_market_graph(markets, indicators, npu_scores=_supportive_npu_scores(markets))
        OntologyReasoner(graph).infer()

        run = run_mock_trading_cycle(
            NegotiatedGoal(0.03, 300_000, 60, 70, "unit"),
            account,
            markets,
            indicators,
            graph,
        )

        self.assertEqual(run.llm_judgment.decision, "PROPOSE_LIMIT_ORDERS")
        self.assertGreater(len(run.ontology_evidence), 0)
        self.assertGreater(len(run.order_intents), 0)
        self.assertGreater(len(run.kis_order_receipts), 0)
        self.assertGreater(len(run.kis_executions), 0)
        self.assertGreater(len(run.portfolio.account.holdings), 0)

    def test_paper_kis_client_uses_real_rest_contract_in_mock_cycle(self) -> None:
        transport = RecordingKisTransport()
        broker = KisDevelopersApiClient(
            app_key="paper-app",
            app_secret="paper-secret",
            account_no="12345678-01",
            paper=True,
            enabled=True,
            transport=transport,
        )
        order = FinalOrder(
            ticker="005930",
            market="KR",
            order_type=OrderType.LIMIT,
            side=OrderSide.BUY,
            quantity=2,
            limit_price=70000,
        )

        receipt = broker.place_limit_order(order)
        execution = broker.get_order_status(receipt.order_id)
        portfolio = broker.get_portfolio()

        order_call = next(call for call in transport.calls if call["url"].endswith("/order-cash"))
        self.assertIn("openapivts.koreainvestment.com:29443", order_call["url"])
        self.assertEqual(order_call["headers"]["tr_id"], "VTTC0012U")
        self.assertEqual(order_call["headers"]["hashkey"], "paper-hash")
        self.assertEqual(order_call["body"]["CANO"], "12345678")
        self.assertEqual(order_call["body"]["ACNT_PRDT_CD"], "01")
        self.assertEqual(order_call["body"]["EXCG_ID_DVSN_CD"], "KRX")
        self.assertEqual(receipt.order_id, "0000000001")
        self.assertEqual(execution.status, "FILLED")
        self.assertEqual(portfolio.account.holdings[0].ticker, "005930")

        status_call = next(call for call in transport.calls if call["url"].endswith("/inquire-daily-ccld"))
        self.assertEqual(status_call["params"]["EXCG_ID_DVSN_CD"], "KRX")


def _supportive_npu_scores(markets):
    return {market.ticker: (0.2, 0.2, 0.0, 0.0, 0.0, 0.4) for market in markets}


if __name__ == "__main__":
    unittest.main()
