from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from app.execution import KisApiError, KisDevelopersApiClient, load_kis_env_file


def main() -> int:
    parser = argparse.ArgumentParser(description="Read-only KIS Open API connection check.")
    parser.add_argument(
        "--account",
        action="store_true",
        help="Also call inquire-balance. This is read-only and never places orders.",
    )
    args = parser.parse_args()

    loaded = load_kis_env_file()
    client = KisDevelopersApiClient(enabled=args.account)
    mode = "paper" if client.paper else "live"
    print(f"KIS secrets file loaded: {loaded}")
    print(f"KIS mode: {mode}")
    print(f"KIS base URL: {client.endpoints.base_url}")
    print(f"KIS account suffix: ...{client.credentials.account_no[-2:]}")

    try:
        token = client.issue_access_token()
        print(f"Token issued: yes, length={len(token)}")
        if args.account:
            portfolio = client.get_portfolio()
            print(f"Balance lookup: yes, holdings={len(portfolio.account.holdings)}")
            print(f"Cash field parsed: {portfolio.account.cash:.0f}")
        else:
            print("Balance lookup: skipped; pass --account for read-only account lookup.")
    except (KisApiError, RuntimeError, OSError) as exc:
        print(f"KIS check failed: {exc}")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
