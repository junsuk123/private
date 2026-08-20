from __future__ import annotations

from contextlib import closing
from concurrent.futures import Future, ProcessPoolExecutor
from dataclasses import asdict, dataclass
from datetime import datetime, timedelta, timezone
import json
import multiprocessing
import os
from pathlib import Path
import sqlite3
import threading
import time
from typing import Any, Iterable, Iterator

from app.cost import TradingCostEngine
from app.routing.actions import is_actionable_strategy_route
from app.strategy.catalog import is_short_strategy
from app.strategy.exit_geometry import (
    exit_geometry,
    reference_round_trip_cost_bps,
)
from app.trading.directional import (
    PositionDirection,
    gross_return_bps,
    parse_direction,
    trailing_breached,
    trailing_price,
)


@dataclass(frozen=True)
class GnnRealtimeTrust:
    passed: bool
    score: float
    sample_count: int
    minimum_samples: int
    positive_net_rate: float | None
    mean_realized_net_bps: float | None
    brier_score: float | None
    mean_uncertainty: float | None
    net_sign_accuracy: float | None
    mean_net_mae_bps: float | None
    reason_codes: tuple[str, ...]
    evaluated_at: str
    horizon_seconds: int
    strategy_sample_counts: dict[str, int]
    strategy_metrics: dict[str, dict[str, float | int | bool | str | None]]
    strategy_market_metrics: dict[
        str, dict[str, dict[str, float | int | bool | str | None]]
    ]
    calibrated_strategy_ids: tuple[str, ...]
    trusted_strategy_ids: tuple[str, ...]
    trusted_strategy_markets: dict[str, tuple[str, ...]]
    outcome_validation_method: str
    outcome_validation_uses_live_algorithm: bool
    outcome_validation_caveat: str

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


class GnnRealtimeTrustEvaluator:
    """Forward-only validation of GNN strategy elections on live market data.

    A prediction is evaluated only after its horizon has elapsed. Prices come
    from the realtime tick store, and duplicate per-second predictions are
    collapsed into one symbol/strategy/horizon bucket to avoid fake confidence
    from highly correlated samples.
    """

    def __init__(
        self,
        *,
        comparison_path: str | Path = "logs/refactor-shadow-comparison.jsonl",
        database_path: str | Path = "data/store/realtime_market_data.sqlite3",
        checkpoint_metadata_path: str | Path = (
            "data/models/strategy_utility/rgcn_shadow.json"
        ),
        horizon_seconds: int | None = None,
        minimum_samples: int | None = None,
        window_samples: int | None = None,
        cache_seconds: float | None = None,
        allow_checkpoint_history: bool | None = None,
        stale_while_refresh: bool = False,
        background_process: bool = False,
    ) -> None:
        self.comparison_path = Path(comparison_path)
        self.database_path = Path(database_path)
        self.checkpoint_metadata_path = Path(checkpoint_metadata_path)
        self.use_strategy_horizons = horizon_seconds is None
        self.horizon_seconds = max(
            30,
            int(horizon_seconds or os.getenv("GNN_TRUST_HORIZON_SECONDS", "600")),
        )
        self.minimum_samples = max(
            10,
            int(minimum_samples or os.getenv("GNN_TRUST_MIN_SAMPLES", "40")),
        )
        self.window_samples = max(
            self.minimum_samples,
            int(window_samples or os.getenv("GNN_TRUST_WINDOW_SAMPLES", "240")),
        )
        self.cache_seconds = max(
            1.0,
            float(
                cache_seconds
                if cache_seconds is not None
                else os.getenv("GNN_TRUST_CACHE_SECONDS", "30")
            ),
        )
        self.allow_checkpoint_history = (
            os.getenv("GNN_TRUST_ALLOW_CHECKPOINT_HISTORY", "true")
            .strip()
            .lower()
            in {"1", "true", "yes", "on"}
            if allow_checkpoint_history is None
            else bool(allow_checkpoint_history)
        )
        self.stale_while_refresh = bool(stale_while_refresh)
        self.background_process = bool(background_process)
        self.minimum_strategy_samples = max(
            8,
            int(
                os.getenv(
                    "GNN_TRUST_MIN_SAMPLES_PER_STRATEGY",
                    "10",
                )
            ),
        )
        self.cost_engine = TradingCostEngine()
        self._cached_at = 0.0
        self._cached: GnnRealtimeTrust | None = None
        self._evaluation_lock = threading.Lock()
        self._refresh_state_lock = threading.Lock()
        self._refreshing = False
        #: When the in-flight refresh started, so a refresh that never finishes cannot
        #: latch ``_refreshing`` for the process lifetime. See
        #: ``_trust_process_executor`` for the hang this guards.
        self._refresh_started_at: float | None = None
        self.refresh_timeout_seconds = max(
            60.0, float(os.getenv("GNN_TRUST_REFRESH_TIMEOUT_SEC", "600"))
        )
        # Sparse validation journals often need the full bounded scan. Remember
        # that discovered depth so the next refresh does not reparse the same
        # 4.8k, 9.6k, 19.2k... prefixes before reaching it again.
        self._scan_lines_hint = max(2000, self.window_samples * 20)

    def evaluate(self, now: datetime | None = None) -> GnnRealtimeTrust:
        current = now or datetime.now(timezone.utc)
        monotonic_now = time.monotonic()
        if (
            self._cached is not None
            and monotonic_now - self._cached_at < self.cache_seconds
        ):
            return self._cached
        if self.stale_while_refresh:
            self._start_background_refresh(current)
            return self._cached or self._empty(
                current,
                ("GNN_TRUST_REFRESH_PENDING",),
            )
        # Engine routing, the operations dashboard and the trust endpoint can
        # all arrive as the cache expires. Without single-flight protection each
        # caller independently scans the large tick database, starving the API
        # for tens of seconds. One caller refreshes; all others reuse its result.
        with self._evaluation_lock:
            monotonic_now = time.monotonic()
            if (
                self._cached is not None
                and monotonic_now - self._cached_at < self.cache_seconds
            ):
                return self._cached
            result = self._evaluate_uncached(current)
            self._cached = result
            self._cached_at = time.monotonic()
            return result

    def _start_background_refresh(self, current: datetime) -> None:
        with self._refresh_state_lock:
            if self._refreshing:
                elapsed = time.monotonic() - (self._refresh_started_at or 0.0)
                if elapsed < self.refresh_timeout_seconds:
                    return
                # The in-flight refresh has outlived any legitimate scan, so treat it
                # as lost rather than waiting on it forever. A queued task cannot be
                # rescued: the pool holds one worker, so a new submit would simply
                # queue behind the stuck one. Discarding the executor is what lets the
                # next attempt make progress -- either on a fresh pool, or on the
                # in-process daemon fallback if the pool itself is broken.
                _discard_trust_process_executor()
            self._refreshing = True
            self._refresh_started_at = time.monotonic()
        if self.background_process:
            try:
                future = _trust_process_executor().submit(
                    _evaluate_trust_in_worker,
                    {
                        "comparison_path": str(self.comparison_path),
                        "database_path": str(self.database_path),
                        "checkpoint_metadata_path": str(
                            self.checkpoint_metadata_path
                        ),
                        "horizon_seconds": (
                            None if self.use_strategy_horizons else self.horizon_seconds
                        ),
                        "minimum_samples": self.minimum_samples,
                        "window_samples": self.window_samples,
                        "allow_checkpoint_history": self.allow_checkpoint_history,
                    },
                    current.isoformat(),
                )
                # Bind the token this refresh started with. A refresh abandoned by the
                # timeout above can still complete afterwards, and without the token it
                # would overwrite the cache a newer refresh had already filled and clear
                # ``_refreshing`` out from under it.
                token = self._refresh_started_at
                future.add_done_callback(
                    lambda done: self._accept_process_refresh(done, token)
                )
                return
            except Exception:  # noqa: BLE001 - fall back to the safe daemon path.
                pass
        threading.Thread(
            target=self._refresh_in_background,
            args=(current,),
            name="gnn-realtime-trust-refresh",
            daemon=True,
        ).start()

    def _accept_process_refresh(
        self, future: Future[GnnRealtimeTrust], token: float | None = None
    ) -> None:
        try:
            result = future.result()
        except Exception:  # noqa: BLE001 - validation failure remains fail-closed.
            result = self._empty(
                datetime.now(timezone.utc),
                ("GNN_TRUST_BACKGROUND_REFRESH_FAILED",),
            )
        with self._refresh_state_lock:
            # A superseded refresh must not publish. Its result is stale by definition
            # and its completion says nothing about the refresh now in flight.
            if token is not None and self._refresh_started_at != token:
                return
            self._refreshing = False
            self._refresh_started_at = None
        with self._evaluation_lock:
            self._cached = result
            self._cached_at = time.monotonic()

    def _refresh_in_background(self, current: datetime) -> None:
        try:
            with self._evaluation_lock:
                try:
                    result = self._evaluate_uncached(current)
                except Exception:  # noqa: BLE001 - refresh must not kill its daemon.
                    result = self._empty(
                        current,
                        ("GNN_TRUST_BACKGROUND_REFRESH_FAILED",),
                    )
                self._cached = result
                self._cached_at = time.monotonic()
        finally:
            with self._refresh_state_lock:
                self._refreshing = False
                self._refresh_started_at = None

    def _evaluate_uncached(self, now: datetime) -> GnnRealtimeTrust:
        reasons: list[str] = []
        if not self.comparison_path.exists():
            return self._empty(now, ("GNN_TRUST_LOG_MISSING",))
        if not self.database_path.exists():
            return self._empty(now, ("GNN_TRUST_MARKET_DATA_MISSING",))

        candidates = self._prediction_candidates(now)
        try:
            outcomes = self._query_outcomes(candidates)
        except sqlite3.Error:
            return self._empty(now, ("GNN_TRUST_MARKET_DATA_QUERY_FAILED",))

        sample_count = len(outcomes)
        if sample_count < self.minimum_samples:
            reasons.append("GNN_TRUST_INSUFFICIENT_REALTIME_SAMPLES")
        if not outcomes:
            return self._empty(now, tuple(reasons or ("GNN_TRUST_NO_MATURE_OUTCOMES",)))

        trade_outcomes = [item for item in outcomes if item[4] > 0.0]
        all_realized = [item[0] for item in outcomes]
        expected_nets = [item[4] for item in outcomes]
        realized = [item[0] for item in trade_outcomes]
        probabilities = [item[1] for item in outcomes]
        uncertainties = [item[2] for item in outcomes]
        positive_rate = (
            sum(value > 0 for value in realized) / len(realized)
            if realized
            else 0.0
        )
        mean_net = sum(realized) / len(realized) if realized else 0.0
        sign_accuracy = sum(
            (expected > 0.0) == (actual > 0.0)
            for actual, expected in zip(all_realized, expected_nets)
        ) / sample_count
        net_mae = sum(
            abs(actual - expected)
            for actual, expected in zip(all_realized, expected_nets)
        ) / sample_count
        brier = sum(
            (probability - (1.0 if net > 0 else 0.0)) ** 2
            for net, probability in zip(all_realized, probabilities)
        ) / sample_count
        mean_uncertainty = sum(uncertainties) / sample_count

        minimum_positive_rate = float(
            os.getenv("GNN_TRUST_MIN_POSITIVE_NET_RATE", "0.52")
        )
        maximum_brier = float(os.getenv("GNN_TRUST_MAX_BRIER", "0.25"))
        maximum_uncertainty = float(
            os.getenv("GNN_TRUST_MAX_MEAN_UNCERTAINTY", "1.0")
        )
        minimum_sign_accuracy = float(
            os.getenv("GNN_TRUST_MIN_NET_SIGN_ACCURACY", "0.60")
        )
        maximum_net_mae_bps = float(
            os.getenv("GNN_TRUST_MAX_NET_MAE_BPS", "80.0")
        )
        minimum_score = float(os.getenv("GNN_TRUST_MIN_SCORE", "0.68"))
        if brier > maximum_brier:
            reasons.append("GNN_TRUST_CALIBRATION_ERROR_TOO_HIGH")
        if mean_uncertainty > maximum_uncertainty:
            reasons.append("GNN_TRUST_UNCERTAINTY_TOO_HIGH")
        if sign_accuracy < minimum_sign_accuracy:
            reasons.append("GNN_TRUST_NET_SIGN_ACCURACY_TOO_LOW")
        if net_mae > maximum_net_mae_bps:
            reasons.append("GNN_TRUST_NET_MAE_TOO_HIGH")

        sample_score = min(1.0, sample_count / self.minimum_samples)
        sign_score = max(0.0, min(1.0, sign_accuracy))
        net_error_score = max(
            0.0,
            min(1.0, 1.0 - net_mae / max(maximum_net_mae_bps, 1e-9)),
        )
        uncertainty_score = max(
            0.0,
            min(1.0, 1.0 - mean_uncertainty / max(maximum_uncertainty, 1e-9)),
        )
        score = (
            0.20 * sample_score
            + 0.25 * max(0.0, 1.0 - brier)
            + 0.25 * sign_score
            + 0.15 * net_error_score
            + 0.15 * uncertainty_score
        )
        if score < minimum_score:
            reasons.append("GNN_TRUST_SCORE_BELOW_THRESHOLD")
        grouped: dict[
            str, list[tuple[float, float, float, str, float, str]]
        ] = {}
        for outcome in outcomes:
            grouped.setdefault(outcome[3], []).append(outcome)
        strategy_metrics = {
            strategy_id: _strategy_metrics(
                strategy_outcomes,
                horizon_seconds=self._strategy_horizon(strategy_id),
                minimum_samples=self.minimum_strategy_samples,
                minimum_positive_rate=minimum_positive_rate,
                maximum_brier=maximum_brier,
                maximum_uncertainty=maximum_uncertainty,
                minimum_sign_accuracy=minimum_sign_accuracy,
                maximum_net_mae_bps=maximum_net_mae_bps,
                minimum_score=minimum_score,
            )
            for strategy_id, strategy_outcomes in sorted(grouped.items())
        }
        grouped_by_market: dict[
            tuple[str, str], list[tuple[float, float, float, str, float, str]]
        ] = {}
        for outcome in outcomes:
            grouped_by_market.setdefault((outcome[3], outcome[5]), []).append(outcome)
        strategy_market_metrics: dict[
            str, dict[str, dict[str, float | int | bool | str | None]]
        ] = {}
        for (strategy_id, market), market_outcomes in sorted(
            grouped_by_market.items()
        ):
            strategy_market_metrics.setdefault(strategy_id, {})[market] = (
                _strategy_metrics(
                    market_outcomes,
                    horizon_seconds=self._strategy_horizon(strategy_id),
                    minimum_samples=self.minimum_strategy_samples,
                    minimum_positive_rate=minimum_positive_rate,
                    maximum_brier=maximum_brier,
                    maximum_uncertainty=maximum_uncertainty,
                    minimum_sign_accuracy=minimum_sign_accuracy,
                    maximum_net_mae_bps=maximum_net_mae_bps,
                    minimum_score=minimum_score,
                )
            )
        calibrated_strategy_ids = tuple(
            strategy_id
            for strategy_id, metrics in strategy_metrics.items()
            if metrics["calibration_passed"] is True
        )
        trusted_strategy_markets = {
            strategy_id: tuple(
                market
                for market, metrics in sorted(markets.items())
                if metrics["entry_authorized"] is True
            )
            for strategy_id, markets in strategy_market_metrics.items()
        }
        trusted_strategy_markets = {
            strategy_id: markets
            for strategy_id, markets in trusted_strategy_markets.items()
            if markets
        }
        trusted_strategy_ids = tuple(sorted(trusted_strategy_markets))
        if not calibrated_strategy_ids:
            reasons.append("GNN_TRUST_NO_STRATEGY_PASSED")
        if not trusted_strategy_ids:
            reasons.append("GNN_TRUST_NO_POSITIVE_EDGE_VALIDATED_STRATEGY")
        return GnnRealtimeTrust(
            # Model calibration and live entry authority are deliberately
            # separate. A model may be reliable at rejecting bad trades before
            # any strategy has enough profitable positive forecasts.
            passed=bool(calibrated_strategy_ids),
            score=round(score, 6),
            sample_count=sample_count,
            minimum_samples=self.minimum_samples,
            positive_net_rate=round(positive_rate, 6) if realized else None,
            mean_realized_net_bps=round(mean_net, 6) if realized else None,
            brier_score=round(brier, 6),
            mean_uncertainty=round(mean_uncertainty, 6),
            net_sign_accuracy=round(sign_accuracy, 6),
            mean_net_mae_bps=round(net_mae, 6),
            reason_codes=tuple(dict.fromkeys(reasons)),
            evaluated_at=now.isoformat(),
            horizon_seconds=self.horizon_seconds,
            strategy_sample_counts={
                strategy_id: int(metrics["sample_count"])
                for strategy_id, metrics in strategy_metrics.items()
            },
            strategy_metrics=strategy_metrics,
            strategy_market_metrics=strategy_market_metrics,
            calibrated_strategy_ids=calibrated_strategy_ids,
            trusted_strategy_ids=trusted_strategy_ids,
            trusted_strategy_markets=trusted_strategy_markets,
            outcome_validation_method="directional_strategy_policy_replay_v2",
            outcome_validation_uses_live_algorithm=True,
            outcome_validation_caveat=(
                "entry=first tick after an admissible forecast; "
                "exit=direction-aware target/stop/trailing/max-holding policy; "
                "contextual supervisor halts are not reconstructed from "
                "price-only history"
            ),
        )
    def _query_outcomes(
        self,
        candidates: list[dict[str, Any]],
    ) -> list[tuple[float, float, float, str, float, str]]:
        """Read the live WAL without competing with the realtime writer.

        The market database is several gigabytes and is written continuously.
        A short default SQLite timeout made one transient writer/checkpoint
        collision cache a total GNN trust failure.  A read-only query-only
        connection plus bounded retries keeps execution fail-closed while
        allowing the validation state to recover automatically.
        """
        database_uri = f"file:{self.database_path.resolve().as_posix()}?mode=ro"
        last_error: sqlite3.Error | None = None
        for attempt in range(3):
            try:
                with closing(
                    sqlite3.connect(database_uri, uri=True, timeout=30.0)
                ) as connection:
                    connection.execute("pragma query_only = on")
                    connection.execute("pragma busy_timeout = 30000")
                    return self._outcomes(connection, candidates)
            except sqlite3.Error as exc:
                last_error = exc
                if attempt < 2:
                    time.sleep(0.1 * (attempt + 1))
        assert last_error is not None
        raise last_error

    def _prediction_candidates(
        self, evaluation_time: datetime
    ) -> list[dict[str, Any]]:
        active_checkpoint_hash = self._active_checkpoint_hash()
        scan_lines = max(
            max(2000, self.window_samples * 20),
            int(self._scan_lines_hint),
        )
        max_scan_lines = max(
            scan_lines,
            int(
                os.getenv(
                    "GNN_TRUST_MAX_SCAN_LINES",
                    str(max(20_000, self.window_samples * 200)),
                )
            ),
        )
        while True:
            rows, has_earlier_rows = _tail_text_lines_across_rotations(
                self.comparison_path,
                max_lines=scan_lines,
            )
            candidates = self._prediction_candidates_from_rows(
                rows,
                evaluation_time=evaluation_time,
                active_checkpoint_hash=active_checkpoint_hash,
            )
            # Raw comparison rows are not trust samples. A mismatched checkpoint
            # can emit thousands of NO_TRADE rows with no validation candidate
            # and push every valid forecast out of a fixed physical-line tail.
            # Expand only until the bounded *valid* sample window is filled.
            if (
                len(candidates) >= self.window_samples
                or not has_earlier_rows
                or scan_lines >= max_scan_lines
            ):
                self._scan_lines_hint = scan_lines
                return self._bounded_per_strategy(candidates)
            scan_lines = min(max_scan_lines, scan_lines * 2)

    def _bounded_per_strategy(
        self, candidates: list[dict[str, Any]]
    ) -> list[dict[str, Any]]:
        """Bound the trust window per strategy rather than globally.

        Every threshold downstream is per strategy — ``minimum_strategy_samples``
        and, for entry authority, ``minimum_positive_prediction_samples``. This
        window used to keep the newest ``window_samples`` across ALL strategies.
        With five strategies that is ~48 samples each, and since a positive
        net-edge forecast is a small minority of deduplicated samples, the global
        cut discarded precisely the evidence ``entry_authorized`` consumes.

        Measured at 2026-08-06T05:30Z (KRX regular session): the expansion loop
        collected 666 candidates holding 51 positive-edge forecasts, and the
        global cut left 20 — below the required 5 per strategy for three of five
        strategies. Those reported ``CALIBRATED_AWAITING_POSITIVE_EDGE``, i.e.
        "still gathering evidence", when the honest verdict on the full pool was
        ``POSITIVE_EDGE_VALIDATION_FAILED``: realized net was negative on the
        forecasts the model called positive. A gate that hides a failed
        validation behind a "warming up" label is the worst of both outcomes.

        Discarding was also pure waste: the loop above already paid for the scan
        (~2.1s of JSON parsing), while retaining a candidate costs ~0.4ms in the
        outcome query. The per-strategy cap keeps the window bounded for
        pathological inputs without throwing away work already done.
        """
        per_strategy: dict[tuple[str, str], list[dict[str, Any]]] = {}
        for candidate in candidates:
            per_strategy.setdefault(
                (
                    str(candidate["strategy_id"]),
                    str(candidate.get("position_direction") or "LONG"),
                ),
                [],
            ).append(candidate)
        kept: list[dict[str, Any]] = []
        for strategy_candidates in per_strategy.values():
            # ``candidates`` arrives sorted by as_of, so the tail is the newest.
            kept.extend(strategy_candidates[-self.window_samples :])
        kept.sort(key=lambda item: item["as_of"])
        return kept

    def _prediction_candidates_from_rows(
        self,
        rows: Iterable[str],
        *,
        evaluation_time: datetime,
        active_checkpoint_hash: str | None,
    ) -> list[dict[str, Any]]:
        deduplicated: dict[tuple[str, str, str, int], dict[str, Any]] = {}
        for row_index, raw in enumerate(rows):
            # This runs in a daemon thread under stale-while-refresh. JSON
            # decoding is CPU-bound and otherwise monopolises the GIL long
            # enough to make tiny chart/API reads wait several seconds. Yield
            # cooperatively without changing which rows or outcomes are used.
            if row_index and row_index % 256 == 0:
                time.sleep(0)
            try:
                payload = json.loads(raw)
                observed = datetime.fromisoformat(
                    str(payload.get("as_of") or "").replace("Z", "+00:00")
                )
            except (ValueError, TypeError, json.JSONDecodeError):
                continue
            if observed.tzinfo is None:
                observed = observed.replace(tzinfo=timezone.utc)
            validation_rows = payload.get("validation_candidates") or ()
            # Backward compatibility for logs written before validation-only
            # forecasts were separated from executable route decisions.
            if not validation_rows:
                validation_rows = tuple(
                    item
                    for item in payload.get("decisions", ())
                    if item.get("path") == "cpu_gnn"
                    and is_actionable_strategy_route(item.get("action"))
                    and float(item.get("expected_net_return_bps") or 0.0) > 0.0
                )
            for decision in validation_rows:
                strategy_id = (
                    decision.get("validation_strategy_id")
                    or decision.get("strategy_id")
                )
                if (
                    not strategy_id
                    or strategy_id != decision.get("strategy_id")
                    or float(decision.get("ontology_compatibility") or 0.0) <= 0.0
                    or decision.get("probability_success") is None
                ):
                    continue
                strategy_horizon = self._strategy_horizon(str(strategy_id))
                if observed + timedelta(seconds=strategy_horizon) > evaluation_time:
                    continue
                decision_checkpoint = str(
                    decision.get("checkpoint_hash") or ""
                ).strip()
                if not decision_checkpoint:
                    continue
                if (
                    active_checkpoint_hash
                    and decision_checkpoint != active_checkpoint_hash
                    and not self.allow_checkpoint_history
                ):
                    continue
                bucket = int(observed.timestamp()) // strategy_horizon
                direction = str(
                    decision.get("position_direction")
                    or ("SHORT" if is_short_strategy(str(strategy_id)) else "LONG")
                ).upper()
                if direction not in {"LONG", "SHORT"}:
                    continue
                key = (
                    str(payload.get("symbol") or "").upper(),
                    str(strategy_id),
                    direction,
                    bucket,
                )
                candidate = {
                    "symbol": key[0],
                    "as_of": observed,
                    "probability": float(decision["probability_success"]),
                    "uncertainty": max(
                        0.0, float(decision.get("total_uncertainty") or 0.0)
                    ),
                    "cost_bps": max(
                        0.0, float(decision.get("expected_cost_bps") or 0.0)
                    ),
                    "expected_net_return_bps": float(
                        decision.get("expected_net_return_bps") or 0.0
                    ),
                    "strategy_id": str(strategy_id),
                    "position_direction": direction,
                    "horizon_seconds": strategy_horizon,
                }
                existing = deduplicated.get(key)
                # The bucket represents the first decision the live router
                # would actually have acted on.  Keeping the first forecast
                # unconditionally discarded a later actionable positive
                # election whenever an earlier scan in the same horizon was
                # negative.  Preserve the first positive forecast; otherwise
                # retain the first negative forecast for calibration.
                if existing is None or (
                    float(existing["expected_net_return_bps"]) <= 0.0
                    < float(candidate["expected_net_return_bps"])
                ):
                    deduplicated[key] = candidate
        return sorted(
            deduplicated.values(),
            key=lambda item: item["as_of"],
        )

    def _strategy_horizon(self, strategy_id: str) -> int:
        if not self.use_strategy_horizons:
            return self.horizon_seconds
        return max(30, exit_geometry(strategy_id).max_holding_seconds)

    def _active_checkpoint_hash(self) -> str | None:
        try:
            metadata = json.loads(
                self.checkpoint_metadata_path.read_text(encoding="utf-8")
            )
        except (OSError, ValueError, json.JSONDecodeError):
            return None
        value = str(metadata.get("checkpoint_hash") or "").strip()
        return value or None

    def _outcomes(
        self,
        connection: sqlite3.Connection,
        candidates: list[dict[str, Any]],
    ) -> list[tuple[float, float, float, str, float, str]]:
        """Resolve bounded outcomes through the symbol/received_at covering index."""
        outcomes: list[tuple[float, float, float, str, float]] = []
        for item in candidates:
            outcome = self._outcome(connection, item)
            if outcome is not None:
                outcomes.append(
                    (
                        *outcome,
                        str(item["strategy_id"]),
                        float(item.get("expected_net_return_bps") or 0.0),
                        _market_key(str(item["symbol"])),
                    )
                )
        return outcomes

    def _outcome(
        self,
        connection: sqlite3.Connection,
        item: dict[str, Any],
    ) -> tuple[float, float, float] | None:
        start = item["as_of"].isoformat()
        end_target = (
            item["as_of"]
            + timedelta(
                seconds=int(
                    item.get("horizon_seconds") or self.horizon_seconds
                )
            )
        ).isoformat()
        entry = connection.execute(
            """
            select price, received_at from realtime_ticks
            where symbol = ? and received_at >= ?
            order by received_at asc limit 1
            """,
            (item["symbol"], start),
        ).fetchone()
        exit_ = connection.execute(
            """
            select price, received_at from realtime_ticks
            where symbol = ? and received_at >= ?
            order by received_at asc limit 1
            """,
            (item["symbol"], end_target),
        ).fetchone()
        if not entry or not exit_:
            return None
        entry_price = float(entry[0] or 0.0)
        horizon_exit_price = float(exit_[0] or 0.0)
        if entry_price <= 0 or horizon_exit_price <= 0:
            return None
        symbol = str(item["symbol"])
        is_krx = symbol.isdigit() and len(symbol) == 6
        market = "KR" if is_krx else "US"
        venue = "KRX" if is_krx else "NASD"
        instrument_type = "domestic_stock" if is_krx else "overseas_stock"
        baseline_cost_bps = (
            self.cost_engine.estimate(
                symbol=symbol,
                market=market,
                venue=venue,
                instrument_type=instrument_type,
                entry_price=entry_price,
                expected_exit_price=entry_price,
                quantity=1,
            ).total_cost_rate
            * 10_000.0
        )
        strategy_id = str(item.get("strategy_id") or "")
        direction = parse_direction(
            item.get("position_direction"),
            PositionDirection.SHORT
            if is_short_strategy(strategy_id)
            else PositionDirection.LONG,
        )
        geometry = exit_geometry(strategy_id)
        stop_bps = geometry.stop_loss_bps
        configured_target_bps = geometry.take_profit_bps
        # Short replay must include at least the same borrow-aware reference cost
        # that sized its live exit geometry.  The forecast's own cost may be higher
        # and remains authoritative in that case.
        directional_cost_floor_bps = reference_round_trip_cost_bps(strategy_id)
        baseline_cost_bps = max(baseline_cost_bps, directional_cost_floor_bps)
        target_bps = max(
            configured_target_bps,
            baseline_cost_bps + max(25.0, baseline_cost_bps),
        )
        path = connection.execute(
            """
            select price from realtime_ticks
            where symbol = ? and received_at > ? and received_at <= ?
            order by received_at asc
            """,
            (symbol, str(entry[1]), str(exit_[1])),
        ).fetchall()
        exit_price = horizon_exit_price
        favourable_watermark = entry_price
        for (raw_price,) in path:
            price = float(raw_price or 0.0)
            if price <= 0.0:
                continue
            move_bps = gross_return_bps(entry_price, price, direction)
            if move_bps >= target_bps or move_bps <= -stop_bps:
                exit_price = price
                break
            favourable_watermark = (
                max(favourable_watermark, price)
                if direction is PositionDirection.LONG
                else min(favourable_watermark, price)
            )
            resolved_trailing = trailing_price(
                favourable_watermark,
                geometry.trailing_bps / 10_000.0,
                direction,
            )
            if trailing_breached(
                price,
                resolved_trailing,
                entry_price,
                direction,
            ):
                exit_price = price
                break
        realized_cost_bps = (
            self.cost_engine.estimate(
                symbol=symbol,
                market=market,
                venue=venue,
                instrument_type=instrument_type,
                entry_price=entry_price,
                expected_exit_price=exit_price,
                quantity=1,
            ).total_cost_rate
            * 10_000.0
        )
        cost_bps = max(
            float(item["cost_bps"]),
            realized_cost_bps,
            directional_cost_floor_bps,
        )
        net_bps = gross_return_bps(entry_price, exit_price, direction) - cost_bps
        return net_bps, item["probability"], item["uncertainty"]

    def _empty(
        self,
        now: datetime,
        reasons: tuple[str, ...],
    ) -> GnnRealtimeTrust:
        return GnnRealtimeTrust(
            passed=False,
            score=0.0,
            sample_count=0,
            minimum_samples=self.minimum_samples,
            positive_net_rate=None,
            mean_realized_net_bps=None,
            brier_score=None,
            mean_uncertainty=None,
            net_sign_accuracy=None,
            mean_net_mae_bps=None,
            reason_codes=reasons,
            evaluated_at=now.isoformat(),
            horizon_seconds=self.horizon_seconds,
            strategy_sample_counts={},
            strategy_metrics={},
            strategy_market_metrics={},
            calibrated_strategy_ids=(),
            trusted_strategy_ids=(),
            trusted_strategy_markets={},
            outcome_validation_method="directional_strategy_policy_replay_v2",
            outcome_validation_uses_live_algorithm=True,
            outcome_validation_caveat=(
                "entry=first tick after an admissible forecast; "
                "exit=direction-aware target/stop/trailing/max-holding policy; "
                "contextual supervisor halts are not reconstructed from "
                "price-only history"
            ),
        )


_default_evaluator: GnnRealtimeTrustEvaluator | None = None
_default_evaluator_lock = threading.Lock()


def default_gnn_realtime_trust_evaluator() -> GnnRealtimeTrustEvaluator:
    """One process-wide evaluator for every live ingestion and UI consumer.

    KRX ingestion, US ingestion, the strategy session, and the dashboard all
    validate the same checkpoint against the same journals. Separate evaluator
    instances each launched their own large-log refresh thread, tripling CPU and
    starving small API reads. A single cache also guarantees that every consumer
    sees the same evaluated timestamp and verdict.
    """
    global _default_evaluator
    if _default_evaluator is not None:
        return _default_evaluator
    with _default_evaluator_lock:
        if _default_evaluator is None:
            _default_evaluator = GnnRealtimeTrustEvaluator(
                stale_while_refresh=True,
                background_process=True,
            )
        return _default_evaluator


_trust_executor: ProcessPoolExecutor | None = None
_trust_executor_lock = threading.Lock()
_worker_evaluators: dict[tuple[Any, ...], GnnRealtimeTrustEvaluator] = {}


def _trust_process_executor() -> ProcessPoolExecutor:
    """One background worker for trust refreshes, started with ``spawn``.

    The start method is the whole point of this function. ``ProcessPoolExecutor``
    defaults to ``fork`` on Linux, and this pool is created from the app server --
    a process running the trading engine thread, the collectors and uvicorn's own
    workers. Forking a multi-threaded process copies the mutex STATE but only the
    calling thread, so any lock another thread happened to hold at that instant is
    inherited already-locked with no owner left to release it. The child then
    blocks on it forever.

    That is not hypothetical here: it was measured on 2026-08-19. The forked worker
    sat in ``futex_wait_queue`` for 4h51m having used 5 seconds of CPU, and because
    ``max_workers=1`` every later refresh queued behind it. ``_refreshing`` stayed
    latched True, ``_cached`` stayed None, and the endpoint returned
    ``GNN_TRUST_REFRESH_PENDING`` indefinitely while the engine reported
    ``GNN_NOT_LIVE_AUTHORIZED`` -- so the trust gate, which live entry
    authorisation requires, was silently dead with no traceback and no error count.
    It is restart-dependent (it survived the previous boot), which is what makes it
    a latent fault rather than an obvious one.

    ``spawn`` starts a fresh interpreter that inherits no locks. It costs a re-import
    per refresh, which is irrelevant for a periodic background job and is the correct
    trade against an unrecoverable hang.
    """
    global _trust_executor
    if _trust_executor is not None:
        return _trust_executor
    with _trust_executor_lock:
        if _trust_executor is None:
            _trust_executor = ProcessPoolExecutor(
                max_workers=1,
                mp_context=multiprocessing.get_context("spawn"),
            )
        return _trust_executor


def _discard_trust_process_executor() -> None:
    """Drop the shared pool so the next refresh builds a new one.

    Called when a refresh has outlived its timeout. ``cancel_futures`` clears the
    queue and ``wait=False`` matters: the stuck worker may never return, and blocking
    on it here would move the hang from the pool into the caller.
    """
    global _trust_executor
    with _trust_executor_lock:
        executor, _trust_executor = _trust_executor, None
    if executor is not None:
        try:
            executor.shutdown(wait=False, cancel_futures=True)
        except Exception:  # noqa: BLE001 - a failed teardown must not block the retry.
            pass


def _evaluate_trust_in_worker(
    config: dict[str, Any], evaluated_at: str
) -> GnnRealtimeTrust:
    """CPU-heavy journal parsing in one reusable worker process."""
    key = (
        config["comparison_path"],
        config["database_path"],
        config["checkpoint_metadata_path"],
        config["horizon_seconds"],
        config["minimum_samples"],
        config["window_samples"],
        config["allow_checkpoint_history"],
    )
    evaluator = _worker_evaluators.get(key)
    if evaluator is None:
        evaluator = GnnRealtimeTrustEvaluator(
            comparison_path=config["comparison_path"],
            database_path=config["database_path"],
            checkpoint_metadata_path=config["checkpoint_metadata_path"],
            horizon_seconds=config["horizon_seconds"],
            minimum_samples=config["minimum_samples"],
            window_samples=config["window_samples"],
            allow_checkpoint_history=config["allow_checkpoint_history"],
            stale_while_refresh=False,
            background_process=False,
        )
        _worker_evaluators[key] = evaluator
    moment = datetime.fromisoformat(evaluated_at.replace("Z", "+00:00"))
    return evaluator._evaluate_uncached(moment)


@dataclass(frozen=True)
class _TailTextWindow:
    path: Path
    start: int
    size: int
    line_count: int
    inode: int


def _tail_text_window(path: Path, *, max_lines: int) -> tuple[_TailTextWindow | None, bool]:
    """Locate a bounded tail while keeping the file closed between operations."""
    limit = max(1, int(max_lines))
    block_size = 256 * 1024
    try:
        handle = path.open("rb")
    except FileNotFoundError:
        return None, False
    with handle:
        stat = os.fstat(handle.fileno())
        handle.seek(0, 2)
        size = handle.tell()
        position = size
        ends_with_newline = False
        if size:
            handle.seek(size - 1)
            ends_with_newline = handle.read(1) == b"\n"
        target_newlines = limit + (1 if ends_with_newline else 0)
        newline_count = 0
        start = 0
        has_earlier_rows = False
        while position > 0:
            read_size = min(block_size, position)
            position -= read_size
            handle.seek(position)
            chunk = handle.read(read_size)
            chunk_newlines = chunk.count(b"\n")
            if newline_count + chunk_newlines >= target_newlines:
                needed_in_chunk = target_newlines - newline_count
                index = len(chunk)
                for _ in range(needed_in_chunk):
                    index = chunk.rfind(b"\n", 0, index)
                start = position + index + 1
                has_earlier_rows = True
                break
            newline_count += chunk_newlines

    line_count = (
        limit
        if has_earlier_rows
        else newline_count + (1 if size and not ends_with_newline else 0)
    )
    if line_count <= 0:
        return None, has_earlier_rows
    return (
        _TailTextWindow(
            path=path,
            start=start,
            size=size,
            line_count=line_count,
            inode=int(getattr(stat, "st_ino", 0) or 0),
        ),
        has_earlier_rows,
    )


def _window_rows(window: _TailTextWindow) -> Iterator[str]:
    try:
        handle = window.path.open("rb")
    except FileNotFoundError:
        return
    with handle:
        current_inode = int(getattr(os.fstat(handle.fileno()), "st_ino", 0) or 0)
        # Rotation may occur between locating the byte boundary and reopening the
        # path.  Never apply an offset from the old segment to a new active file.
        if window.inode and current_inode and current_inode != window.inode:
            return
        handle.seek(window.start)
        for _ in range(window.line_count):
            raw = handle.readline()
            if not raw:
                break
            yield raw.decode("utf-8", errors="ignore").rstrip("\r\n")


def _tail_text_lines(
    path: Path, *, max_lines: int
) -> tuple[Iterator[str], bool]:
    """Stream a bounded physical tail without retaining its large JSON rows.

    Comparison rows can exceed 100 KiB.  The old deque implementation joined and
    decoded thousands of them at once, briefly allocating hundreds of megabytes
    on every trust refresh.  Find the byte boundary in reverse, then parse forward
    one line at a time. ``has_earlier_rows`` preserves the evaluator's adaptive
    expansion behaviour without materialising the window just to call ``len``.
    """
    window, has_earlier_rows = _tail_text_window(path, max_lines=max_lines)
    return (_window_rows(window) if window is not None else iter(())), has_earlier_rows


def _comparison_history_paths(path: Path) -> tuple[Path, ...]:
    """Current journal followed by immutable/legacy rotations, newest first."""
    prefix = f"{path.name}."
    rotated: list[Path] = []
    try:
        candidates = tuple(path.parent.glob(f"{path.name}.*"))
    except OSError:
        candidates = ()
    for candidate in candidates:
        suffix = candidate.name[len(prefix) :]
        if candidate.is_file() and (suffix.isdigit() or suffix.startswith("r")):
            rotated.append(candidate)
    rotated.sort(
        key=lambda item: item.stat().st_mtime_ns if item.exists() else 0,
        reverse=True,
    )
    maximum_rotations = max(
        1,
        int(os.getenv("GNN_TRUST_MAX_ROTATED_LOGS", "3")),
    )
    current = (path,) if path.exists() else ()
    return (*current, *rotated[:maximum_rotations])


def _tail_text_lines_across_rotations(
    path: Path, *, max_lines: int
) -> tuple[Iterator[str], bool]:
    """Read one chronological physical tail across journal rotation boundaries."""
    limit = max(1, int(max_lines))
    paths = _comparison_history_paths(path)
    if not paths:
        return iter(()), False

    remaining = limit
    windows: list[_TailTextWindow] = []
    has_earlier_rows = False
    for index, history_path in enumerate(paths):
        window, earlier_in_file = _tail_text_window(
            history_path,
            max_lines=remaining,
        )
        if window is not None:
            windows.append(window)
            remaining -= window.line_count
        if remaining <= 0:
            has_earlier_rows = earlier_in_file or index < len(paths) - 1
            break
    else:
        has_earlier_rows = False

    def rows() -> Iterator[str]:
        # Windows were discovered newest -> oldest; emit oldest -> newest so the
        # evaluator's first-positive-in-horizon rule remains chronological.
        for window in reversed(windows):
            yield from _window_rows(window)

    return rows(), has_earlier_rows


def _market_key(symbol: str) -> str:
    normalized = str(symbol or "").upper().strip()
    return "KRX" if normalized.isdigit() and len(normalized) == 6 else "US"


def _strategy_metrics(
    outcomes: list[tuple[float, float, float, str, float, str]],
    *,
    horizon_seconds: int,
    minimum_samples: int,
    minimum_positive_rate: float,
    maximum_brier: float,
    maximum_uncertainty: float,
    minimum_sign_accuracy: float,
    maximum_net_mae_bps: float,
    minimum_score: float,
) -> dict[str, float | int | bool | str | None]:
    sample_count = len(outcomes)
    trade_outcomes = [item for item in outcomes if item[4] > 0.0]
    all_realized = [item[0] for item in outcomes]
    expected_nets = [item[4] for item in outcomes]
    realized = [item[0] for item in trade_outcomes]
    probabilities = [item[1] for item in outcomes]
    uncertainties = [item[2] for item in outcomes]
    trade_sample_count = len(trade_outcomes)
    positive_rate = (
        sum(value > 0 for value in realized) / trade_sample_count
        if trade_sample_count
        else 0.0
    )
    mean_net = sum(realized) / trade_sample_count if trade_sample_count else 0.0
    sign_accuracy = sum(
        (expected > 0.0) == (actual > 0.0)
        for actual, expected in zip(all_realized, expected_nets)
    ) / sample_count
    net_mae = sum(
        abs(actual - expected)
        for actual, expected in zip(all_realized, expected_nets)
    ) / sample_count
    brier = sum(
        (probability - (1.0 if net > 0 else 0.0)) ** 2
        for net, probability in zip(all_realized, probabilities)
    ) / sample_count
    mean_uncertainty = sum(uncertainties) / sample_count
    sample_score = min(1.0, sample_count / minimum_samples)
    sign_score = max(0.0, min(1.0, sign_accuracy))
    net_error_score = max(
        0.0,
        min(1.0, 1.0 - net_mae / max(maximum_net_mae_bps, 1e-9)),
    )
    uncertainty_score = max(
        0.0,
        min(1.0, 1.0 - mean_uncertainty / max(maximum_uncertainty, 1e-9)),
    )
    score = (
        0.20 * sample_score
        + 0.25 * max(0.0, 1.0 - brier)
        + 0.25 * sign_score
        + 0.15 * net_error_score
        + 0.15 * uncertainty_score
    )
    minimum_trade_samples = max(
        3,
        int(os.getenv("GNN_TRUST_MIN_POSITIVE_PREDICTION_SAMPLES", "5")),
    )
    calibration_passed = (
        sample_count >= minimum_samples
        and brier <= maximum_brier
        and mean_uncertainty <= maximum_uncertainty
        and sign_accuracy >= minimum_sign_accuracy
        and net_mae <= maximum_net_mae_bps
        and score >= minimum_score
    )
    entry_authorized = (
        calibration_passed
        and trade_sample_count >= minimum_trade_samples
        and positive_rate >= minimum_positive_rate
        and mean_net
        >= max(0.0, float(os.getenv("GNN_TRUST_MIN_MEAN_NET_BPS", "5.0")))
    )
    return {
        "sample_count": sample_count,
        "trade_sample_count": trade_sample_count,
        "minimum_samples": minimum_samples,
        "minimum_positive_prediction_samples": minimum_trade_samples,
        "horizon_seconds": horizon_seconds,
        # ``0.0`` here used to mean two different things -- "every profitable
        # forecast lost" and "no profitable forecast was ever made" -- and on this
        # account it was always the second. Reporting None for the empty case
        # matches ``mean_realized_net_bps`` and stops the dashboard reading an
        # untested strategy as a failed one. The gate below still uses the
        # numeric ``positive_rate``; only the reported field changes.
        "positive_net_rate": (
            round(positive_rate, 6) if trade_sample_count else None
        ),
        "mean_realized_net_bps": (
            round(mean_net, 6) if trade_sample_count else None
        ),
        "brier_score": round(brier, 6),
        "mean_uncertainty": round(mean_uncertainty, 6),
        "net_sign_accuracy": round(sign_accuracy, 6),
        "mean_net_mae_bps": round(net_mae, 6),
        "execution_validation_stage": (
            "ENTRY_AUTHORIZED"
            if entry_authorized
            else (
                "POSITIVE_EDGE_VALIDATION_FAILED"
                if trade_sample_count >= minimum_trade_samples
                else "CALIBRATED_AWAITING_POSITIVE_EDGE"
            )
        ),
        "score": round(score, 6),
        "calibration_passed": calibration_passed,
        "entry_authorized": entry_authorized,
        # Backward-compatible UI field: "passed" means model calibration.
        "passed": calibration_passed,
        # Entry admissibility was already frozen into the validation candidate.
        # From that point onward replay uses the same directional target, stop,
        # trailing and max-holding geometry as StrategySessionManager.  Contextual
        # supervisor halts cannot be reconstructed from price ticks and therefore
        # remain outside this metric.
        "outcome_validation_method": "directional_strategy_policy_replay_v2",
        "outcome_validation_uses_live_algorithm": True,
        "outcome_validation_caveat": (
            "entry=first tick after an admissible forecast; exit=direction-aware "
            "target/stop/trailing/max-holding policy; contextual supervisor halts "
            "are not reconstructed from price-only history"
        ),
    }
