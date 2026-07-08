"""Replay / walk-forward evaluation for the technical prediction layer.

For each decision index i over a time-ordered bar series, features are built
from PAST bars only (``bars[:i+1]``) and labels from FUTURE bars only
(``bars[i+1:]``) — a hard no-look-ahead split. The predicted edge is then
compared against the realized cost-adjusted outcome, aggregated overall and
broken down by methodology and regime, and split walk-forward by time so
regime drift is visible and no segment is evaluated on data used elsewhere.

Reports are plain dicts (JSON-serializable); the CLI wrapper
(`scripts/replay_technical_prediction.py`) persists them under
``data/models/technical_replay_reports/``. This module reads no wall clock and
performs no I/O, so it is deterministic and unit-testable.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from statistics import fmean
from typing import Sequence

from app.features.schemas import OHLCVBar
from app.technical.feature_builder import build_technical_feature_set
from app.technical.labels import LabelBuilder, LabelConfig
from app.technical.prediction import PredictionConfig, TechnicalPredictionEngine


@dataclass(frozen=True)
class ReplayConfig:
    warmup_bars: int = 30
    walk_forward_splits: int = 3
    label: LabelConfig = field(default_factory=LabelConfig)
    prediction: PredictionConfig = field(default_factory=lambda: PredictionConfig(min_confidence=0.5))
    source: str = "realtime_replay"


def _bucket_metrics(rows: list[dict]) -> dict:
    """Aggregate one list of per-decision rows into metrics."""
    n = len(rows)
    tradable = [r for r in rows if r["tradable"]]
    if not rows:
        return {"n": 0}
    # Among tradable BUYs, fraction whose realized net-after-cost label == 1.
    resolved = [r for r in tradable if r["net_label"] is not None]
    wins = [r for r in resolved if r["net_label"] == 1]
    predicted_edges = [r["predicted_net_bps"] for r in tradable if r["predicted_net_bps"] is not None]
    realized_nets = [r["realized_net_bps"] for r in tradable if r["realized_net_bps"] is not None]
    edge_errors = [
        r["predicted_net_bps"] - r["realized_net_bps"]
        for r in tradable
        if r["predicted_net_bps"] is not None and r["realized_net_bps"] is not None
    ]
    mfes = [r["mfe_bps"] for r in rows if r["mfe_bps"] is not None]
    maes = [r["mae_bps"] for r in rows if r["mae_bps"] is not None]
    return {
        "n": n,
        "tradable_count": len(tradable),
        "tradable_rate": round(len(tradable) / n, 4),
        "resolved_count": len(resolved),
        "hit_rate": round(len(wins) / len(resolved), 4) if resolved else None,
        "precision_net_profitable": round(len(wins) / len(tradable), 4) if tradable else None,
        "avg_predicted_net_bps": round(fmean(predicted_edges), 3) if predicted_edges else None,
        "avg_realized_net_bps": round(fmean(realized_nets), 3) if realized_nets else None,
        "avg_edge_error_bps": round(fmean(edge_errors), 3) if edge_errors else None,
        "avg_mfe_bps": round(fmean(mfes), 3) if mfes else None,
        "avg_mae_bps": round(fmean(maes), 3) if maes else None,
        "turnover": len(tradable),  # one round-trip per tradable BUY in this simple model
    }


class TechnicalReplayEvaluator:
    def __init__(self, config: ReplayConfig | None = None, *, cost_engine=None) -> None:
        self.config = config or ReplayConfig()
        self.engine = TechnicalPredictionEngine(config=self.config.prediction)
        self.label_builder = LabelBuilder(cost_engine=cost_engine, config=self.config.label)

    def evaluate(self, symbol: str, bars: Sequence[OHLCVBar]) -> dict:
        rows = self.evaluate_rows(symbol, bars)
        cfg = self.config
        report = {
            "symbol": symbol,
            "bars": len(tuple(bars)),
            "warmup_bars": cfg.warmup_bars,
            "overall": _bucket_metrics(rows),
            "by_methodology": self._group(rows, "methodology"),
            "by_regime": self._group(rows, "regime"),
            "walk_forward": self._walk_forward(rows, cfg.walk_forward_splits),
            "no_lookahead": True,
        }
        return report

    def evaluate_rows(self, symbol: str, bars: Sequence[OHLCVBar]) -> list[dict]:
        """Per-decision rows. Features use PAST only; labels use FUTURE only."""
        bars = tuple(bars)
        cfg = self.config
        rows: list[dict] = []
        last_index = len(bars) - 1
        for i in range(cfg.warmup_bars, last_index):
            past = bars[: i + 1]              # PAST ONLY (inclusive of decision bar)
            future = bars[i + 1 :]            # FUTURE ONLY
            if not future:
                break
            features = build_technical_feature_set(past, symbol=symbol)
            prediction = self.engine.predict(features)
            entry_price = features.price
            decision_at = bars[i].as_of
            future_path = [
                ((b.as_of - decision_at).total_seconds(), float(b.close)) for b in future
            ]
            labels = (
                self.label_builder.build(
                    symbol=symbol,
                    entry_price=entry_price,
                    future_path=future_path,
                    source=cfg.source,
                )
                if entry_price
                else None
            )
            realized_net = labels.metadata.get("net_return_after_cost") if labels else None
            rows.append(
                {
                    "index": i,
                    "decision_at": decision_at.isoformat(),
                    "tradable": bool(prediction.tradable),
                    "methodology": prediction.methodology,
                    "regime": prediction.regime,
                    "predicted_net_bps": prediction.expected_net_return_bps if prediction.tradable else None,
                    "realized_net_bps": (realized_net * 10_000.0) if realized_net is not None else None,
                    "net_label": labels.net_profitable_after_cost_label if labels else None,
                    "mfe_bps": labels.max_favorable_excursion_bps if labels else None,
                    "mae_bps": labels.max_adverse_excursion_bps if labels else None,
                }
            )
        return rows

    @staticmethod
    def _group(rows: list[dict], key: str) -> dict:
        groups: dict[str, list[dict]] = {}
        for r in rows:
            if r["tradable"] or key == "regime":
                groups.setdefault(str(r[key]), []).append(r)
        return {k: _bucket_metrics(v) for k, v in sorted(groups.items())}

    @staticmethod
    def _walk_forward(rows: list[dict], splits: int) -> list[dict]:
        if splits <= 1 or not rows:
            return [{"split": 0, **_bucket_metrics(rows)}]
        size = max(1, len(rows) // splits)
        out = []
        for s in range(splits):
            segment = rows[s * size : (s + 1) * size] if s < splits - 1 else rows[s * size :]
            if segment:
                out.append({"split": s, "start_index": segment[0]["index"], **_bucket_metrics(segment)})
        return out
