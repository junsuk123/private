from __future__ import annotations

import json
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path

from app.config.refactor_flags import RefactorFeatureFlags


class RefactorMode(StrEnum):
    RESEARCH = "research"
    REPLAY = "replay"
    PAPER = "paper"
    SHADOW = "shadow"
    CANARY = "canary"
    LIVE = "live"


@dataclass(frozen=True)
class RefactorRuntimeProfile:
    mode: RefactorMode
    broker_submission_enabled: bool
    maximum_order_notional: float
    allowed_symbols: tuple[str, ...]
    flags: RefactorFeatureFlags

    def validate(self) -> None:
        if self.mode in {
            RefactorMode.RESEARCH,
            RefactorMode.REPLAY,
            RefactorMode.PAPER,
            RefactorMode.SHADOW,
        } and self.broker_submission_enabled:
            raise ValueError(f"{self.mode.value} mode cannot submit broker orders")
        if self.mode in {RefactorMode.CANARY, RefactorMode.LIVE}:
            if not self.broker_submission_enabled:
                raise ValueError(f"{self.mode.value} requires explicit broker submission")
            if not self.flags.live_enabled or not self.flags.strategy_owned_execution:
                raise ValueError("canary/live requires refactor live and strategy ownership")
            if self.maximum_order_notional <= 0 or not self.allowed_symbols:
                raise ValueError("canary/live requires bounded notional and symbol allowlist")
        if self.mode == RefactorMode.CANARY and self.flags.npu_inference:
            raise ValueError("NPU is not promoted for canary")


def load_refactor_profile(
    path: str | Path = "config/refactor_profile.json",
    *,
    allow_example: bool = True,
) -> RefactorRuntimeProfile:
    selected = Path(path)
    if not selected.exists() and allow_example:
        selected = selected.with_name(f"{selected.stem}.example{selected.suffix}")
    payload = json.loads(selected.read_text(encoding="utf-8"))
    flags = RefactorFeatureFlags(**dict(payload.get("flags") or {}))
    flags.validate()
    profile = RefactorRuntimeProfile(
        mode=RefactorMode(payload["mode"]),
        broker_submission_enabled=bool(payload["broker_submission_enabled"]),
        maximum_order_notional=float(payload.get("maximum_order_notional", 0)),
        allowed_symbols=tuple(str(value) for value in payload.get("allowed_symbols", ())),
        flags=flags,
    )
    profile.validate()
    return profile
