from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from app.config import load_live_trading_safety_config
from app.trading.live_runtime_guard import create_arming_file


def main() -> int:
    parser = argparse.ArgumentParser(description="Create a short-lived live arming file.")
    parser.add_argument("--ttl-seconds", type=int, default=None)
    args = parser.parse_args()

    config = load_live_trading_safety_config(allow_example=False)
    ttl = args.ttl_seconds or config.arming_ttl_seconds
    path = create_arming_file(ttl_seconds=ttl)
    print(f"armed: {path} ttl_seconds={ttl}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
