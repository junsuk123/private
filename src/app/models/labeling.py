from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime

from app.features.schemas import OHLCVBar


@dataclass(frozen=True)
class LabelConfig:
    horizon_bars: int = 5
    profit_taking: float = 0.05
    stop_loss: float = -0.03


@dataclass(frozen=True)
class TripleBarrierLabel:
    ticker: str
    as_of: datetime
    label: int
    touched_at: datetime | None
    return_at_touch: float | None
    horizon_bars: int


def future_return(bars: tuple[OHLCVBar, ...], as_of: datetime, horizon_bars: int) -> float | None:
    ordered = tuple(sorted(bars, key=lambda bar: bar.as_of))
    index = _index_at_or_before(ordered, as_of)
    if index is None or index + horizon_bars >= len(ordered):
        return None
    start = ordered[index].close
    if start == 0:
        return None
    return ordered[index + horizon_bars].close / start - 1


def triple_barrier_label(
    bars: tuple[OHLCVBar, ...],
    as_of: datetime,
    config: LabelConfig | None = None,
) -> TripleBarrierLabel | None:
    cfg = config or LabelConfig()
    ordered = tuple(sorted(bars, key=lambda bar: bar.as_of))
    index = _index_at_or_before(ordered, as_of)
    if index is None or index + cfg.horizon_bars >= len(ordered):
        return None
    entry = ordered[index].close
    if entry == 0:
        return None
    for bar in ordered[index + 1 : index + cfg.horizon_bars + 1]:
        high_return = bar.high / entry - 1
        low_return = bar.low / entry - 1
        if high_return >= cfg.profit_taking:
            return TripleBarrierLabel(ordered[index].ticker, ordered[index].as_of, 1, bar.as_of, high_return, cfg.horizon_bars)
        if low_return <= cfg.stop_loss:
            return TripleBarrierLabel(ordered[index].ticker, ordered[index].as_of, -1, bar.as_of, low_return, cfg.horizon_bars)
    terminal_return = ordered[index + cfg.horizon_bars].close / entry - 1
    label = 1 if terminal_return > 0 else -1 if terminal_return < 0 else 0
    return TripleBarrierLabel(ordered[index].ticker, ordered[index].as_of, label, ordered[index + cfg.horizon_bars].as_of, terminal_return, cfg.horizon_bars)


def _index_at_or_before(bars: tuple[OHLCVBar, ...], as_of: datetime) -> int | None:
    candidates = [index for index, bar in enumerate(bars) if bar.as_of <= as_of]
    return candidates[-1] if candidates else None
