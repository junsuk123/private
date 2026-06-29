from __future__ import annotations


class KisConfigurationError(RuntimeError):
    pass


class KisModeMismatchError(KisConfigurationError):
    pass


class KisReadinessError(RuntimeError):
    def __init__(self, failed_gates: dict[str, str]) -> None:
        super().__init__("KIS readiness failed: " + ", ".join(sorted(failed_gates)))
        self.failed_gates = dict(failed_gates)


class LiveExecutionBlocked(RuntimeError):
    def __init__(self, reason_codes: tuple[str, ...]) -> None:
        super().__init__("live execution blocked: " + ", ".join(reason_codes))
        self.reason_codes = reason_codes
