from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from app.trading.live_runtime_guard import disarm


def main() -> int:
    disarm()
    print("disarmed: new live orders are blocked until re-armed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
