from __future__ import annotations

import importlib
import importlib.metadata
from dataclasses import dataclass
from typing import Any, Callable, Sequence

import numpy as np

REFERENCE_VERSION = "2.1.3"


@dataclass(frozen=True)
class ParityResult:
    metric: str
    available: bool
    passed: bool
    local_value: float | None
    reference_value: float | None
    absolute_tolerance: float
    relative_tolerance: float
    unavailable_reason: str | None = None


class GSQuantReferenceAdapter:
    """Explicit offline-only local-function validator.

    This module imports only ``gs_quant.timeseries`` submodules. It never constructs a
    session, pricing context, API client, dataset, asset or portfolio service.
    """

    def __init__(self) -> None:
        self._statistics: Any = None
        self._technicals: Any = None
        self._error: str | None = None

    def health(self) -> dict[str, Any]:
        try:
            installed = importlib.metadata.version("gs-quant")
        except importlib.metadata.PackageNotFoundError:
            return {"available": False, "version": None, "network_required": False, "reason": "gs_quant_not_installed"}
        return {
            "available": installed == REFERENCE_VERSION,
            "version": installed,
            "network_required": False,
            "reason": None if installed == REFERENCE_VERSION else "reference_version_mismatch",
        }

    def validate(
        self,
        metric: str,
        values: Sequence[float],
        local_value: float,
        *,
        window: int,
        atol: float = 1e-10,
        rtol: float = 1e-8,
    ) -> ParityResult:
        health = self.health()
        if not health["available"]:
            return ParityResult(metric, False, False, local_value, None, atol, rtol, str(health["reason"]))
        try:
            import pandas as pd
            function, kwargs = self._resolve(metric, window)
            series = pd.Series(tuple(float(value) for value in values), dtype="float64")
            output = function(series, **kwargs)
            reference = float(output.dropna().iloc[-1])
            passed = bool(np.isclose(local_value, reference, atol=atol, rtol=rtol))
            return ParityResult(metric, True, passed, local_value, reference, atol, rtol, None)
        except Exception as exc:
            return ParityResult(metric, False, False, local_value, None, atol, rtol, f"reference_error:{type(exc).__name__}:{exc}")

    def _resolve(self, metric: str, window: int) -> tuple[Callable[..., Any], dict[str, Any]]:
        if self._statistics is None:
            self._statistics = importlib.import_module("gs_quant.timeseries.statistics")
        if self._technicals is None:
            self._technicals = importlib.import_module("gs_quant.timeseries.technicals")
        mapping = {
            "rolling_mean": (self._statistics.mean, {"w": window}),
            "rolling_std": (self._statistics.std, {"w": window}),
            "zscore": (self._statistics.zscores, {"w": window}),
            "moving_average": (self._technicals.moving_average, {"w": window}),
            "exponential_moving_average": (
                self._technicals.exponential_moving_average,
                {"beta": (window - 1.0) / (window + 1.0)},
            ),
            "smoothed_moving_average": (self._technicals.smoothed_moving_average, {"w": window}),
            "rsi": (self._technicals.relative_strength_index, {"w": window}),
        }
        if metric not in mapping:
            raise ValueError(f"unsupported reference metric: {metric}")
        return mapping[metric]
