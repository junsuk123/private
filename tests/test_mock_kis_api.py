from __future__ import annotations

import json
import sys
import tempfile
import unittest
from datetime import datetime, timedelta, timezone
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
        if url.endswith("/uapi/domestic-stock/v1/trading/inquire-psbl-order"):
            return {"rt_cd": "0", "output": {"ord_psbl_cash": "1000000"}}
        if url.endswith("/uapi/overseas-stock/v1/trading/inquire-present-balance"):
            return {
                "rt_cd": "0",
                "output2": [{"crcy_cd": "USD", "frcr_dncl_amt_2": "12.34", "bass_exrt": "1300"}],
                "output3": {"tot_asst_amt": "9983", "tot_frcr_cblc_smtl": "9983.000000"},
            }
        if url.endswith("/uapi/overseas-stock/v1/trading/inquire-psamount"):
            return {"rt_cd": "0", "output": {"ord_psbl_amt": "12.34"}}
        raise AssertionError(f"unexpected KIS request: {method} {url}")


class MockKisApiTest(unittest.TestCase):
    def test_kis_credentials_can_load_ignored_secret_file(self) -> None:
        # KIS is live-only now, so credentials come from the KIS_APP_KEY/SECRET/ACCOUNT_NO
        # vars. This still verifies that a gitignored secrets .env is loaded and that the
        # already-set process env is not clobbered by the default-file setdefault load in
        # from_env (override=True on the temp file wins).
        with tempfile.TemporaryDirectory() as tmp:
            env_path = Path(tmp) / "kis_api_keys.env"
            env_path.write_text(
                "\n".join(
                    [
                        "KIS_APP_KEY=unit-app",
                        "KIS_APP_SECRET=unit-secret",
                        "KIS_ACCOUNT_NO=12345678-01",
                    ]
                ),
                encoding="utf-8",
            )
            with patch.dict(
                "os.environ",
                {
                    "KIS_APP_KEY": "",
                    "KIS_APP_SECRET": "",
                    "KIS_ACCOUNT_NO": "",
                },
                clear=False,
            ):
                self.assertTrue(load_kis_env_file(env_path, override=True))
                credentials = KisCredentials.from_env()

        self.assertEqual(credentials.app_key, "unit-app")
        self.assertEqual(credentials.app_secret, "unit-secret")
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

    def test_kis_client_uses_real_rest_contract_in_mock_cycle(self) -> None:
        # KIS is live-only now (paper/VTS API removed; for_mode forces paper=False), so the
        # order routes to the LIVE host with the LIVE tr_ids over the same REST contract.
        transport = RecordingKisTransport()
        with tempfile.TemporaryDirectory() as tmp:
            broker = KisDevelopersApiClient(
                app_key="paper-app",
                app_secret="paper-secret",
                account_no="12345678-01",
                paper=True,
                enabled=True,
                transport=transport,
                token_cache_path=Path(tmp) / "kis_access_token.live.json",
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
        self.assertIn("openapi.koreainvestment.com:9443", order_call["url"])
        self.assertEqual(order_call["headers"]["tr_id"], "TTTC0012U")
        self.assertEqual(order_call["headers"]["hashkey"], "paper-hash")
        self.assertEqual(order_call["body"]["CANO"], "12345678")
        self.assertEqual(order_call["body"]["ACNT_PRDT_CD"], "01")
        self.assertEqual(order_call["body"]["EXCG_ID_DVSN_CD"], "KRX")
        self.assertEqual(receipt.order_id, "0000000001")
        self.assertEqual(execution.status, "FILLED")
        self.assertEqual(portfolio.account.holdings[0].ticker, "005930")

        status_call = next(call for call in transport.calls if call["url"].endswith("/inquire-daily-ccld"))
        self.assertEqual(status_call["params"]["EXCG_ID_DVSN_CD"], "KRX")
        orderable_cash_call = next(call for call in transport.calls if call["url"].endswith("/inquire-psbl-order"))
        self.assertEqual(orderable_cash_call["headers"]["tr_id"], "TTTC8908R")
        self.assertEqual(orderable_cash_call["params"]["PDNO"], "")
        self.assertEqual(orderable_cash_call["params"]["ORD_UNPR"], "")
        self.assertEqual(orderable_cash_call["params"]["CMA_EVLU_AMT_ICLD_YN"], "N")
        self.assertEqual(orderable_cash_call["params"]["OVRS_ICLD_YN"], "N")

    def test_kis_access_token_is_reused_from_cache_across_clients(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            token_cache_path = Path(tmp) / "kis_access_token.paper.json"
            transport = RecordingKisTransport()
            first = KisDevelopersApiClient(
                app_key="paper-app",
                app_secret="paper-secret",
                account_no="12345678-01",
                paper=True,
                enabled=True,
                transport=transport,
                token_cache_path=token_cache_path,
            )
            self.assertEqual(first.issue_access_token(), "paper-token")
            cached_payload = json.loads(token_cache_path.read_text(encoding="utf-8"))
            self.assertEqual(cached_payload["access_token"], "paper-token")
            # The KIS client operates in live-only mode now (paper/VTS API was removed;
            # for_mode forces paper=False), so the cache always records mode "live".
            self.assertEqual(cached_payload["mode"], "live")

            second = KisDevelopersApiClient(
                app_key="paper-app",
                app_secret="paper-secret",
                account_no="12345678-01",
                paper=True,
                enabled=True,
                transport=transport,
                token_cache_path=token_cache_path,
            )
            self.assertEqual(second.issue_access_token(), "paper-token")

        token_calls = [call for call in transport.calls if call["url"].endswith("/oauth2/tokenP")]
        self.assertEqual(len(token_calls), 1)

    def test_kis_access_token_cache_is_not_reused_after_credentials_change(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            token_cache_path = Path(tmp) / "kis_access_token.live.json"
            transport = RecordingKisTransport()
            first = KisDevelopersApiClient(
                app_key="first-app",
                app_secret="first-secret",
                account_no="12345678-01",
                enabled=True,
                transport=transport,
                token_cache_path=token_cache_path,
            )
            first.issue_access_token()
            second = KisDevelopersApiClient(
                app_key="second-app",
                app_secret="second-secret",
                account_no="12345678-01",
                enabled=True,
                transport=transport,
                token_cache_path=token_cache_path,
            )

            second.issue_access_token()

        token_calls = [call for call in transport.calls if call["url"].endswith("/oauth2/tokenP")]
        self.assertEqual(len(token_calls), 2)
        self.assertEqual(second.token_source, "issued")

    def test_unwritable_kis_token_cache_blocks_token_request(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            token_cache_path = Path(tmp) / "kis_access_token.paper.json"
            token_cache_path.mkdir()
            transport = RecordingKisTransport()
            client = KisDevelopersApiClient(
                app_key="paper-app",
                app_secret="paper-secret",
                account_no="12345678-01",
                paper=True,
                enabled=True,
                transport=transport,
                token_cache_path=token_cache_path,
            )

            with self.assertRaisesRegex(RuntimeError, "KIS token cache path is a directory"):
                client.issue_access_token()

        token_calls = [call for call in transport.calls if call["url"].endswith("/oauth2/tokenP")]
        self.assertEqual(token_calls, [])

    def test_expired_kis_access_token_cache_is_refreshed(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            token_cache_path = Path(tmp) / "kis_access_token.paper.json"
            token_cache_path.write_text(
                json.dumps(
                    {
                        "access_token": "expired-token",
                        "expires_at": (datetime.now(timezone.utc) - timedelta(minutes=1)).isoformat(),
                        "mode": "paper",
                    }
                ),
                encoding="utf-8",
            )
            transport = RecordingKisTransport()
            client = KisDevelopersApiClient(
                app_key="paper-app",
                app_secret="paper-secret",
                account_no="12345678-01",
                paper=True,
                enabled=True,
                transport=transport,
                token_cache_path=token_cache_path,
            )
            self.assertEqual(client.issue_access_token(), "paper-token")

        token_calls = [call for call in transport.calls if call["url"].endswith("/oauth2/tokenP")]
        self.assertEqual(len(token_calls), 1)

    def test_kis_live_access_token_env_is_reused_without_issuance(self) -> None:
        transport = RecordingKisTransport()
        with tempfile.TemporaryDirectory() as tmp:
            with patch.dict(
                "os.environ",
                {
                    "KIS_LIVE_ACCESS_TOKEN": "manual-live-token",
                    "KIS_LIVE_ACCESS_TOKEN_EXPIRES_AT": (
                        datetime.now(timezone.utc) + timedelta(hours=1)
                    ).isoformat(),
                },
                clear=False,
            ):
                client = KisDevelopersApiClient(
                    app_key="live-app",
                    app_secret="live-secret",
                    account_no="12345678-01",
                    paper=False,
                    enabled=True,
                    transport=transport,
                    token_cache_path=Path(tmp) / "kis_access_token.live.json",
                )
                portfolio = client.get_portfolio()

        token_calls = [call for call in transport.calls if call["url"].endswith("/oauth2/tokenP")]
        balance_call = next(call for call in transport.calls if call["url"].endswith("/inquire-balance"))
        self.assertEqual(token_calls, [])
        self.assertEqual(balance_call["headers"]["authorization"], "Bearer manual-live-token")
        self.assertEqual(client.token_source, "env")
        self.assertEqual(portfolio.account.cash, 1_000_000)
        self.assertEqual(portfolio.account.cash_equivalent_krw, 1_016_042)
        self.assertEqual(portfolio.account.equity, 1_158_042)
        self.assertEqual(portfolio.account.cash_by_currency["KRW"], 1_000_000)
        self.assertEqual(portfolio.account.cash_by_currency["USD"], 12.34)

    def test_kis_portfolio_keeps_domestic_deposit_and_orderable_cash_separate(self) -> None:
        class OrderableCashTransport(RecordingKisTransport):
            def request(self, method, url, headers, body=None, params=None, timeout=10.0):
                self.calls.append({"method": method, "url": url, "headers": dict(headers), "body": dict(body or {}), "params": dict(params or {})})
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
                if url.endswith("/uapi/domestic-stock/v1/trading/inquire-psbl-order"):
                    return {"rt_cd": "0", "output": {"ord_psbl_cash": "750000"}}
                if url.endswith("/uapi/overseas-stock/v1/trading/inquire-present-balance"):
                    return {"rt_cd": "0", "output2": [], "output3": {"tot_asst_amt": "0"}}
                raise AssertionError(f"unexpected KIS request: {method} {url}")

        broker = KisDevelopersApiClient(
            app_key="paper-app",
            app_secret="paper-secret",
            account_no="12345678-01",
            paper=True,
            enabled=True,
            transport=OrderableCashTransport(),
            access_token="token",
        )

        portfolio = broker.get_portfolio()

        self.assertEqual(portfolio.account.cash, 1_000_000)
        self.assertEqual(portfolio.account.cash_by_currency["KRW"], 1_000_000)
        self.assertEqual(portfolio.account.orderable_cash_by_currency["KRW"], 750_000)
        self.assertEqual(portfolio.account.securities_market_value, 142_000)
        self.assertEqual(portfolio.account.equity, 1_142_000)

    def test_kis_portfolio_reads_domestic_balance_continuation_pages(self) -> None:
        class PagedDomesticBalanceTransport(RecordingKisTransport):
            def request(self, method, url, headers, body=None, params=None, timeout=10.0):
                self.calls.append({"method": method, "url": url, "headers": dict(headers), "body": dict(body or {}), "params": dict(params or {})})
                if url.endswith("/uapi/domestic-stock/v1/trading/inquire-balance"):
                    if headers.get("tr_cont") == "N":
                        return {
                            "rt_cd": "0",
                            "ctx_area_fk100": "ctx",
                            "ctx_area_nk100": "",
                            "output1": [
                                {"pdno": "484590", "prdt_name": "Second", "hldg_qty": "3", "pchs_avg_pric": "7000", "prpr": "7340"},
                                {"pdno": "000000", "prdt_name": "Zero", "hldg_qty": "0", "pchs_avg_pric": "0", "prpr": "1"},
                            ],
                            "output2": [{"dnca_tot_amt": "20374", "scts_evlu_amt": "36020", "tot_evlu_amt": "56394"}],
                        }
                    return {
                        "rt_cd": "0",
                        "ctx_area_fk100": "ctx",
                        "ctx_area_nk100": "next",
                        "output1": [
                            {"pdno": "481070", "prdt_name": "First", "hldg_qty": "1", "pchs_avg_pric": "14020", "prpr": "14000"},
                            {"pdno": "006880", "prdt_name": "Zero", "hldg_qty": "0", "pchs_avg_pric": "0", "prpr": "4575"},
                        ],
                        "output2": [{"dnca_tot_amt": "20374", "scts_evlu_amt": "36020", "tot_evlu_amt": "56394"}],
                    }
                if url.endswith("/uapi/domestic-stock/v1/trading/inquire-psbl-order"):
                    return {"rt_cd": "0", "output": {"ord_psbl_cash": "13515"}}
                if url.endswith("/uapi/overseas-stock/v1/trading/inquire-present-balance"):
                    return {"rt_cd": "0", "output2": [], "output3": {"tot_asst_amt": "0"}}
                if url.endswith("/uapi/overseas-stock/v1/trading/inquire-balance"):
                    return {"rt_cd": "0", "output1": []}
                if url.endswith("/uapi/overseas-stock/v1/trading/inquire-psamount"):
                    return {"rt_cd": "0", "output": {"ord_psbl_cash": "0"}}
                raise AssertionError(f"unexpected KIS request: {method} {url}")

        broker = KisDevelopersApiClient(
            app_key="paper-app",
            app_secret="paper-secret",
            account_no="12345678-01",
            paper=True,
            enabled=True,
            transport=PagedDomesticBalanceTransport(),
            access_token="token",
        )

        portfolio = broker.get_portfolio()

        self.assertEqual([(h.ticker, h.quantity) for h in portfolio.account.holdings], [("481070", 1), ("484590", 3)])
        self.assertEqual(portfolio.account.securities_market_value, 36_020)
        self.assertEqual(portfolio.account.cash, 20_374)
        self.assertEqual(portfolio.account.orderable_cash_by_currency["KRW"], 13_515)

    def test_kis_portfolio_uses_overseas_orderable_cash_not_total_usd_balance(self) -> None:
        class OverseasOrderableTransport(RecordingKisTransport):
            def request(self, method, url, headers, body=None, params=None, timeout=10.0):
                self.calls.append({"method": method, "url": url, "headers": dict(headers), "body": dict(body or {}), "params": dict(params or {})})
                if url.endswith("/uapi/domestic-stock/v1/trading/inquire-balance"):
                    return {"rt_cd": "0", "output1": [], "output2": [{"dnca_tot_amt": "2401"}]}
                if url.endswith("/uapi/domestic-stock/v1/trading/inquire-psbl-order"):
                    return {"rt_cd": "0", "output": {"ord_psbl_cash": "2401"}}
                if url.endswith("/uapi/overseas-stock/v1/trading/inquire-balance"):
                    return {"rt_cd": "0", "output1": []}
                if url.endswith("/uapi/overseas-stock/v1/trading/inquire-present-balance"):
                    return {
                        "rt_cd": "0",
                        "output2": [
                            {
                                "crcy_cd": "USD",
                                "frcr_dncl_amt_2": "3.22",
                                "frcr_drwg_psbl_amt_1": "0.49",
                                "bass_exrt": "1545.0",
                            }
                        ],
                        "output3": {"tot_asst_amt": "9974", "tot_frcr_cblc_smtl": "9974"},
                    }
                if url.endswith("/uapi/overseas-stock/v1/trading/inquire-psamount"):
                    return {"rt_cd": "0", "output": {"ord_psbl_amt": "0.49"}}
                raise AssertionError(f"unexpected KIS request: {method} {url}")

        transport = OverseasOrderableTransport()
        broker = KisDevelopersApiClient(
            app_key="paper-app",
            app_secret="paper-secret",
            account_no="12345678-01",
            paper=False,
            enabled=True,
            transport=transport,
            access_token="token",
        )

        portfolio = broker.get_portfolio()

        self.assertEqual(portfolio.account.cash_by_currency["KRW"], 2401)
        self.assertEqual(portfolio.account.cash_by_currency["USD"], 0.49)
        self.assertEqual(portfolio.account.equity, 9974)
        psamount_call = next(call for call in transport.calls if call["url"].endswith("/inquire-psamount"))
        self.assertEqual(psamount_call["params"]["OVRS_ORD_UNPR"], "1")
        self.assertEqual(psamount_call["params"]["ITEM_CD"], "AAPL")

    def test_kis_overseas_balance_uses_foreign_stock_evaluation_amount(self) -> None:
        class OverseasBalanceTransport(RecordingKisTransport):
            def request(self, method, url, headers, body=None, params=None, timeout=10.0):
                self.calls.append({"method": method, "url": url, "headers": dict(headers), "body": dict(body or {}), "params": dict(params or {})})
                if url.endswith("/uapi/domestic-stock/v1/trading/inquire-balance"):
                    return {"rt_cd": "0", "output1": [], "output2": [{"dnca_tot_amt": "2401"}]}
                if url.endswith("/uapi/domestic-stock/v1/trading/inquire-psbl-order"):
                    return {"rt_cd": "0", "output": {"ord_psbl_cash": "2401"}}
                if url.endswith("/uapi/overseas-stock/v1/trading/inquire-balance"):
                    return {
                        "rt_cd": "0",
                        "output1": [
                            {
                                "ovrs_pdno": "TSLA",
                                "ovrs_item_name": "Tesla",
                                "ovrs_cblc_qty": "2",
                                "pchs_avg_pric": "150.00",
                                "now_pric2": "0",
                                "ovrs_stck_evlu_amt": "512.34",
                                "tr_crcy_cd": "USD",
                                "ovrs_excg_cd": "NASD",
                            }
                        ],
                    }
                if url.endswith("/uapi/overseas-stock/v1/trading/inquire-present-balance"):
                    return {
                        "rt_cd": "0",
                        "output2": [{"crcy_cd": "USD", "frcr_dncl_amt_2": "0.49", "bass_exrt": "1500"}],
                        "output3": {"tot_asst_amt": "771586"},
                    }
                if url.endswith("/uapi/overseas-stock/v1/trading/inquire-psamount"):
                    return {"rt_cd": "0", "output": {"ord_psbl_amt": "0.49"}}
                raise AssertionError(f"unexpected KIS request: {method} {url}")

        broker = KisDevelopersApiClient(
            app_key="paper-app",
            app_secret="paper-secret",
            account_no="12345678-01",
            paper=False,
            enabled=True,
            transport=OverseasBalanceTransport(),
            access_token="token",
        )

        portfolio = broker.get_portfolio()

        self.assertEqual(portfolio.account.holdings[0].ticker, "TSLA")
        self.assertAlmostEqual(portfolio.account.holdings[0].last_price, 256.17)
        self.assertAlmostEqual(portfolio.account.holdings[0].market_value, 512.34)

    def test_kis_total_assets_combines_domestic_and_overseas_without_duplicate_krw_cash(self) -> None:
        class SplitTotalTransport(RecordingKisTransport):
            def request(self, method, url, headers, body=None, params=None, timeout=10.0):
                self.calls.append({"method": method, "url": url, "headers": dict(headers), "body": dict(body or {}), "params": dict(params or {})})
                if url.endswith("/uapi/domestic-stock/v1/trading/inquire-balance"):
                    # KRW deposit 20,374 + one domestic holding worth 79,180 = tot_evlu_amt 99,554.
                    return {
                        "rt_cd": "0",
                        "output1": [{"pdno": "005930", "hldg_qty": "1", "ord_psbl_qty": "1", "pchs_avg_pric": "79180", "prpr": "79180"}],
                        "output2": [{"dnca_tot_amt": "20374", "tot_evlu_amt": "99554"}],
                    }
                if url.endswith("/uapi/domestic-stock/v1/trading/inquire-psbl-order"):
                    return {"rt_cd": "0", "output": {"ord_psbl_cash": "20374"}}
                if url.endswith("/uapi/overseas-stock/v1/trading/inquire-balance"):
                    return {"rt_cd": "0", "output1": []}
                if url.endswith("/uapi/overseas-stock/v1/trading/inquire-present-balance"):
                    # tot_asst_amt is the broker's integrated 총자산 (settled KRW deposit +
                    # usable foreign cash), which re-includes the domestic KRW deposit and so
                    # must NOT have KRW cash re-added — only the domestic stock (79,180).
                    return {
                        "rt_cd": "0",
                        "output2": [{"crcy_cd": "USD", "frcr_dncl_amt_2": "38.8", "frcr_drwg_psbl_amt_1": "38.8", "bass_exrt": "1554.1"}],
                        "output3": {"tot_asst_amt": "123518", "tot_dncl_amt": "20374"},
                    }
                if url.endswith("/uapi/overseas-stock/v1/trading/inquire-psamount"):
                    return {"rt_cd": "0", "output": {"ord_psbl_amt": "38.8"}}
                raise AssertionError(f"unexpected KIS request: {method} {url}")

        broker = KisDevelopersApiClient(
            app_key="paper-app",
            app_secret="paper-secret",
            account_no="12345678-01",
            paper=False,
            enabled=True,
            transport=SplitTotalTransport(),
            access_token="token",
        )

        portfolio = broker.get_portfolio()

        self.assertEqual(portfolio.account.equity, 202_698)

    def test_kis_krw_only_account_does_not_render_integrated_cash_as_foreign_cash(self) -> None:
        class KrwOnlyIntegratedMarginTransport(RecordingKisTransport):
            def request(self, method, url, headers, body=None, params=None, timeout=10.0):
                self.calls.append({"method": method, "url": url, "headers": dict(headers), "body": dict(body or {}), "params": dict(params or {})})
                if url.endswith("/uapi/domestic-stock/v1/trading/inquire-balance"):
                    return {
                        "rt_cd": "0",
                        "output1": [],
                        "output2": [{"dnca_tot_amt": "200000", "tot_evlu_amt": "200000"}],
                    }
                if url.endswith("/uapi/domestic-stock/v1/trading/inquire-psbl-order"):
                    return {"rt_cd": "0", "output": {"ord_psbl_cash": "200000"}}
                if url.endswith("/uapi/overseas-stock/v1/trading/inquire-balance"):
                    return {"rt_cd": "0", "output1": []}
                if url.endswith("/uapi/overseas-stock/v1/trading/inquire-present-balance"):
                    # KIS may echo the KRW integrated-margin deposit in this
                    # foreign-looking summary even though no currency row exists.
                    return {
                        "rt_cd": "0",
                        "output1": [],
                        "output2": [],
                        "output3": {
                            "tot_asst_amt": "200000",
                            "tot_frcr_cblc_smtl": "200000",
                        },
                    }
                if url.endswith("/uapi/overseas-stock/v1/trading/inquire-psamount"):
                    return {"rt_cd": "0", "output": {"ord_psbl_amt": "0"}}
                raise AssertionError(f"unexpected KIS request: {method} {url}")

        broker = KisDevelopersApiClient(
            app_key="paper-app",
            app_secret="paper-secret",
            account_no="12345678-01",
            paper=False,
            enabled=True,
            transport=KrwOnlyIntegratedMarginTransport(),
            access_token="token",
        )

        portfolio = broker.get_portfolio()

        self.assertEqual(portfolio.account.equity, 200_000)
        self.assertEqual(portfolio.account.cash, 200_000)
        self.assertEqual(portfolio.account.cash_equivalent_krw, 200_000)
        self.assertEqual(portfolio.account.foreign_cash_krw, 0)
        self.assertEqual(portfolio.account.cash_by_currency, {"KRW": 200_000})

    def test_kis_portfolio_prefers_account_asset_balance_total(self) -> None:
        class AccountAssetBalanceTransport(RecordingKisTransport):
            def request(self, method, url, headers, body=None, params=None, timeout=10.0):
                self.calls.append({"method": method, "url": url, "headers": dict(headers), "body": dict(body or {}), "params": dict(params or {})})
                if url.endswith("/uapi/domestic-stock/v1/trading/inquire-balance"):
                    return {
                        "rt_cd": "0",
                        "output1": [{"pdno": "232680", "hldg_qty": "1", "ord_psbl_qty": "1", "pchs_avg_pric": "8290", "prpr": "8270"}],
                        "output2": [{"dnca_tot_amt": "73992", "tot_evlu_amt": "82262"}],
                    }
                if url.endswith("/uapi/domestic-stock/v1/trading/inquire-account-balance"):
                    return {
                        "rt_cd": "0",
                        "output2": {
                            "tot_asst_amt": "268179",
                            "dncl_amt": "73992",
                            "pchs_amt_smtl": "113220",
                            "evlu_amt_smtl": "109905",
                        },
                    }
                if url.endswith("/uapi/domestic-stock/v1/trading/inquire-psbl-order"):
                    return {"rt_cd": "0", "output": {"ord_psbl_cash": "73992"}}
                if url.endswith("/uapi/overseas-stock/v1/trading/inquire-balance"):
                    return {
                        "rt_cd": "0",
                        "output1": [
                            {
                                "ovrs_pdno": "HBAN",
                                "ovrs_item_name": "HBAN",
                                "ovrs_cblc_qty": "1",
                                "pchs_avg_pric": "17.41",
                                "now_pric2": "17.60",
                                "ovrs_stck_evlu_amt": "17.60",
                                "tr_crcy_cd": "USD",
                                "ovrs_excg_cd": "NASD",
                            }
                        ],
                    }
                if url.endswith("/uapi/overseas-stock/v1/trading/inquire-present-balance"):
                    return {
                        "rt_cd": "0",
                        "output2": [{"crcy_cd": "USD", "frcr_dncl_amt_2": "7.3", "bass_exrt": "1509.9"}],
                        "output3": {"tot_asst_amt": "0"},
                    }
                if url.endswith("/uapi/overseas-stock/v1/trading/inquire-psamount"):
                    return {"rt_cd": "0", "output": {"ord_psbl_amt": "7.3"}}
                raise AssertionError(f"unexpected KIS request: {method} {url}")

        broker = KisDevelopersApiClient(
            app_key="paper-app",
            app_secret="paper-secret",
            account_no="12345678-01",
            paper=False,
            enabled=True,
            transport=AccountAssetBalanceTransport(),
            access_token="token",
        )

        portfolio = broker.get_portfolio()

        self.assertEqual(portfolio.account.equity, 268_179)
        self.assertEqual(portfolio.account.cash_equivalent_krw, 158_274)
        self.assertAlmostEqual(portfolio.account.foreign_cash_krw, 11_022.27)
        self.assertAlmostEqual(portfolio.account.cash_by_currency["KRW"], 147_251.73)
        self.assertEqual(portfolio.account.orderable_cash_by_currency["KRW"], 73_992)

    def test_kis_balance_does_not_treat_total_evaluation_as_cash(self) -> None:
        class BalanceOnlyTransport(RecordingKisTransport):
            def request(self, method, url, headers, body=None, params=None, timeout=10.0):
                self.calls.append({"method": method, "url": url, "headers": dict(headers), "body": dict(body or {}), "params": dict(params or {})})
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
                        "output2": [{"tot_evlu_amt": "1142000", "scts_evlu_amt": "142000"}],
                    }
                if url.endswith("/uapi/domestic-stock/v1/trading/inquire-psbl-order"):
                    return {"rt_cd": "1", "msg1": "temporary orderable cash lookup failure"}
                if url.endswith("/uapi/overseas-stock/v1/trading/inquire-present-balance"):
                    return {"rt_cd": "0", "output2": [], "output3": {"tot_asst_amt": "999999999"}}
                raise AssertionError(f"unexpected KIS request: {method} {url}")

        broker = KisDevelopersApiClient(
            app_key="paper-app",
            app_secret="paper-secret",
            account_no="12345678-01",
            paper=True,
            enabled=True,
            transport=BalanceOnlyTransport(),
            access_token="token",
        )

        portfolio = broker.get_portfolio()

        self.assertEqual(portfolio.account.cash, 1_000_000)
        self.assertEqual(portfolio.account.cash_equivalent_krw, 1_000_000)
        self.assertEqual(portfolio.account.securities_market_value, 142_000)
        self.assertEqual(portfolio.account.equity, 1_142_000)


def _supportive_npu_scores(markets):
    return {market.ticker: (0.2, 0.2, 0.0, 0.0, 0.0, 0.4) for market in markets}


if __name__ == "__main__":
    unittest.main()
