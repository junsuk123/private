from __future__ import annotations

import argparse


def main() -> int:
    parser = argparse.ArgumentParser(description="Fail-closed placeholder for guarded live loop startup.")
    parser.add_argument("--symbols", required=True)
    parser.parse_args()
    print(
        "START_BLOCKED: live strategy loop requires fresh realtime pipeline, live-eligible model, "
        "readiness report, and manual arming before order execution."
    )
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
