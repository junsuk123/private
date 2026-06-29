from __future__ import annotations

import threading
import time


class KisRateLimiter:
    """Small process-local pacer for KIS calls.

    It is intentionally conservative and dependency-free. It does not attempt to
    model every KIS product quota; it prevents accidental tight retry loops.
    """

    def __init__(self, min_interval_seconds: float = 0.12) -> None:
        self.min_interval_seconds = max(0.0, float(min_interval_seconds))
        self._lock = threading.Lock()
        self._last_call = 0.0

    def wait(self) -> None:
        with self._lock:
            now = time.monotonic()
            sleep_for = self.min_interval_seconds - (now - self._last_call)
            if sleep_for > 0:
                time.sleep(sleep_for)
            self._last_call = time.monotonic()
