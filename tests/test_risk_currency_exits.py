from dataclasses import replace
from datetime import datetime, timedelta, timezone

from app.audit import AuditLogger
from app.portfolio import valuation_complete
from app.risk import RiskManager
from app.schemas.domain import AccountSnapshot, Holding, MarketSnapshot, OrderAction, OrderIntent, SourceMetadata


def _cover_case(tmp_path):
    now = datetime.now(timezone.utc)
    account = AccountSnapshot(
        cash=100_000, holdings=(Holding("AAPL", "NASDAQ", "Apple", "Technology", 1, 100, 100, direction="SHORT", execution_product="CREDIT_BORROW"),),
        cash_by_currency={"KRW": 100_000, "USD": 200}, total_equity_krw=420_000,
    )
    market = MarketSnapshot("AAPL", "NASDAQ", "Apple", "Technology", 100, 10_000_000_000, .02, SourceMetadata("KIS", now))
    intent = OrderIntent(
        ticker="AAPL", market="NASDAQ", action=OrderAction.BUY, suggested_weight=.01, confidence=.9,
        valid_until=now + timedelta(minutes=5), reasoning_summary=("cover",), supporting_factors=("cover",), contradicting_factors=(),
        source_data_ids=("quote",), strategy_family="short_cover", expected_exit_price=95,
        position_direction="SHORT", position_effect="CLOSE", execution_product="CREDIT_BORROW",
        strategy_metadata={"loan_date": "20260918", "cover_quantity": 1},
    )
    return RiskManager(audit_logger=AuditLogger(tmp_path / "audit.jsonl")), account, market, intent


def test_missing_fx_cannot_block_a_valid_short_cover_by_position_or_sector_weight(tmp_path):
    manager, account, market, intent = _cover_case(tmp_path)
    assert not valuation_complete(account)
    result = manager.validate(intent, account, market)
    assert result.checks["currency_valuation_complete"]
    assert result.checks["max_single_stock_weight"]
    assert result.checks["max_intraday_position_weight"]
    assert result.checks["max_sector_weight"]
    assert result.approved, result.rejection_reasons
    assert result.final_order.quantity == 1
    assert result.final_order.position_effect == "CLOSE"


def test_cover_still_requires_loan_contract_quantity_and_cash_checks(tmp_path):
    manager, account, market, intent = _cover_case(tmp_path)
    missing_loan = manager.validate(replace(intent, strategy_metadata={"cover_quantity": 1}), account, market)
    assert not missing_loan.approved
    assert not missing_loan.checks["short_close_contract_valid"]
    missing_quantity = manager.validate(replace(intent, strategy_metadata={"loan_date": "20260918"}), account, market)
    assert not missing_quantity.approved
    assert "SHORT_COVER_QUANTITY_UNKNOWN" in missing_quantity.rejection_reasons
    no_cash = manager.validate(intent, replace(account, orderable_cash_by_currency={"USD": 0}), market)
    assert not no_cash.approved
    assert not no_cash.checks["cash_available"]


def test_forged_long_buy_close_does_not_gain_exposure_exemptions(tmp_path):
    manager, account, market, intent = _cover_case(tmp_path)
    result = manager.validate(replace(intent, position_direction="LONG", execution_product="CASH"), account, market)
    assert not result.approved
    assert not result.checks["order_contract_complete"]
    assert not result.checks["max_sector_weight"]


def test_long_buy_open_still_requires_complete_currency_valuation(tmp_path):
    manager, account, market, intent = _cover_case(tmp_path)
    result = manager.validate(replace(intent, position_direction="LONG", position_effect="OPEN", execution_product="CASH"), account, market)
    assert not result.approved
    assert not result.checks["currency_valuation_complete"]


def test_fully_invested_long_exit_does_not_require_cash_reserve(tmp_path):
    manager, account, market, intent = _cover_case(tmp_path)
    holding = Holding("AAPL", "NASDAQ", "Apple", "Technology", 2, 100, 100)
    account = replace(account, cash=0, cash_by_currency={"KRW": 0, "USD": 0}, holdings=(holding,))
    intent = replace(intent, action=OrderAction.SELL, position_direction="LONG", execution_product="CASH")
    result = manager.validate(intent, account, market)
    assert not valuation_complete(account)
    assert result.approved, result.rejection_reasons
    assert result.checks["cash_available"]
    assert result.final_order.quantity == 2
    # A missing currency cash bucket is equally irrelevant to a long liquidation.
    result = manager.validate(intent, replace(account, cash_by_currency={}), market)
    assert result.approved, result.rejection_reasons


def test_cover_can_spend_native_cash_without_reserve_but_cannot_overdraw(tmp_path):
    manager, account, market, intent = _cover_case(tmp_path)
    exact_cash = replace(account, orderable_cash_by_currency={"USD": 100})
    result = manager.validate(intent, exact_cash, market)
    assert result.approved, result.rejection_reasons
    assert result.metadata["cover_cash_required"] == 100
    insufficient = manager.validate(intent, replace(account, orderable_cash_by_currency={"USD": 99}), market)
    assert not insufficient.approved
    assert not insufficient.checks["deposit_limit_check"]
    assert not insufficient.checks["cash_available"]


def test_long_exit_never_exceeds_broker_held_or_sellable_quantity(tmp_path):
    manager, account, market, intent = _cover_case(tmp_path)
    holding = Holding("AAPL", "NASDAQ", "Apple", "Technology", 2, 100, 100, sellable_quantity=1)
    account = replace(account, cash=0, cash_by_currency={}, holdings=(holding,))
    intent = replace(intent, action=OrderAction.SELL, position_direction="LONG", execution_product="CASH")
    result = manager.validate(intent, account, replace(market, last_price=40))
    assert result.approved, result.rejection_reasons
    assert result.final_order.quantity == 1
    unavailable = manager.validate(intent, replace(account, holdings=(replace(holding, sellable_quantity=0),)), market)
    assert not unavailable.approved
    absent = manager.validate(intent, replace(account, holdings=()), market)
    assert not absent.approved
    assert "holding_exists" in absent.rejection_reasons


def test_buy_open_keeps_required_cash_reserve(tmp_path):
    manager, _, market, intent = _cover_case(tmp_path)
    manager.rules = replace(manager.rules, minimum_cash_reserve=.95)
    account = AccountSnapshot(cash=1000, holdings=(), base_currency="USD")
    intent = replace(intent, position_direction="LONG", position_effect="OPEN", execution_product="CASH")
    result = manager.validate(intent, account, market)
    assert result.checks["currency_valuation_complete"]
    assert result.checks["deposit_limit_check"]
    assert not result.checks["cash_available"]
    assert not result.approved


def test_malformed_cover_quantity_is_rejected_without_raising(tmp_path):
    manager, account, market, intent = _cover_case(tmp_path)
    for quantity in (float("inf"), float("nan"), -1, .5):
        malformed = replace(intent, strategy_metadata={"loan_date": "20260918", "cover_quantity": quantity})
        result = manager.validate(malformed, account, market)
        assert not result.approved
        assert "SHORT_COVER_QUANTITY_UNKNOWN" in result.rejection_reasons
    result = manager.validate(replace(intent, strategy_metadata={"loan_date": "20260918", "quantity": float("inf")}), account, market)
    assert not result.approved
