from __future__ import annotations

import pytest

from app.config.refactor_flags import RefactorFeatureFlags
from app.config.refactor_profile import (
    RefactorMode,
    RefactorRuntimeProfile,
    load_refactor_profile,
)


def test_example_profile_is_shadow_and_cannot_submit() -> None:
    profile = load_refactor_profile()
    assert profile.mode == RefactorMode.SHADOW
    assert not profile.broker_submission_enabled


def test_paper_cannot_reach_broker() -> None:
    profile = RefactorRuntimeProfile(
        mode=RefactorMode.PAPER,
        broker_submission_enabled=True,
        maximum_order_notional=0,
        allowed_symbols=(),
        flags=RefactorFeatureFlags(),
    )
    with pytest.raises(ValueError, match="cannot submit"):
        profile.validate()


def test_canary_requires_bounded_allowlist_and_strategy_ownership() -> None:
    profile = RefactorRuntimeProfile(
        mode=RefactorMode.CANARY,
        broker_submission_enabled=True,
        maximum_order_notional=100000,
        allowed_symbols=("005930",),
        flags=RefactorFeatureFlags(),
    )
    with pytest.raises(ValueError, match="strategy ownership"):
        profile.validate()
