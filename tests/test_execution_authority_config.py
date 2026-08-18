"""The authority config documents the architecture and cannot contradict it."""

from __future__ import annotations

import pytest

from app.config.execution_authority import (
    REQUIRED_INVARIANTS,
    ExecutionAuthorityConfigError,
    load_execution_authority_config,
)


def test_the_shipped_config_parses_and_matches_the_architecture() -> None:
    config = load_execution_authority_config()
    assert config.source_path is not None
    assert config.order_type == "LIMIT"
    assert config.max_buy_orders_per_cycle == 1
    assert config.manual_strategy_confirmation is False
    assert config.strict_affordability is True
    assert dict(config.invariants) == dict(REQUIRED_INVARIANTS)


def test_the_declared_live_defaults_are_all_present() -> None:
    invariants = dict(load_execution_authority_config().invariants)
    assert invariants["strategy_owned_execution"] is True
    assert invariants["post_selection_profitability_veto"] is False
    assert invariants["post_selection_position_resizing"] is False
    assert invariants["post_selection_risk_veto"] is False
    assert invariants["execution_guard"] is True
    assert invariants["duplicate_protection"] is True
    assert invariants["idempotency"] is True
    assert invariants["synthetic_live_data"] is False
    assert invariants["unknown_source_live"] is False


@pytest.mark.parametrize(
    "name",
    [
        "post_selection_profitability_veto",
        "post_selection_position_resizing",
        "post_selection_risk_veto",
        "post_selection_ontology_reapproval",
    ],
)
def test_re_enabling_a_removed_veto_is_refused(tmp_path, name: str) -> None:
    """The file cannot describe a system that does not exist."""
    path = tmp_path / "authority.yaml"
    path.write_text(f"invariants:\n  {name}: true\n", encoding="utf-8")
    with pytest.raises(ExecutionAuthorityConfigError) as excinfo:
        load_execution_authority_config(path)
    assert name in str(excinfo.value)


@pytest.mark.parametrize(
    "name", ["strategy_owned_execution", "execution_guard", "idempotency"]
)
def test_disabling_a_required_invariant_is_refused(tmp_path, name: str) -> None:
    path = tmp_path / "authority.yaml"
    path.write_text(f"invariants:\n  {name}: false\n", encoding="utf-8")
    with pytest.raises(ExecutionAuthorityConfigError):
        load_execution_authority_config(path)


def test_an_unknown_invariant_is_refused(tmp_path) -> None:
    path = tmp_path / "authority.yaml"
    path.write_text("invariants:\n  invented_switch: true\n", encoding="utf-8")
    with pytest.raises(ExecutionAuthorityConfigError):
        load_execution_authority_config(path)


def test_a_market_order_default_is_refused(tmp_path) -> None:
    path = tmp_path / "authority.yaml"
    path.write_text("operational:\n  order_type: MARKET\n", encoding="utf-8")
    with pytest.raises(ExecutionAuthorityConfigError):
        load_execution_authority_config(path)


def test_an_absent_file_falls_back_to_the_architecture(tmp_path) -> None:
    config = load_execution_authority_config(tmp_path / "missing.yaml")
    assert config.source_path is None
    assert dict(config.invariants) == dict(REQUIRED_INVARIANTS)


def test_operational_values_are_ordinary_configuration(tmp_path) -> None:
    path = tmp_path / "authority.yaml"
    path.write_text(
        "operational:\n"
        "  order_type: LIMIT\n"
        "  max_buy_orders_per_cycle: 3\n"
        "  trade_plan_ttl_seconds: 120\n"
        "  strict_affordability: false\n",
        encoding="utf-8",
    )
    config = load_execution_authority_config(path)
    assert config.max_buy_orders_per_cycle == 3
    assert config.trade_plan_ttl_seconds == 120.0
    assert config.strict_affordability is False


def test_a_malformed_file_raises_rather_than_defaulting(tmp_path) -> None:
    path = tmp_path / "authority.yaml"
    path.write_text("invariants: not-a-mapping\n", encoding="utf-8")
    with pytest.raises(ExecutionAuthorityConfigError):
        load_execution_authority_config(path)
