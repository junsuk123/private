from __future__ import annotations

import argparse
import csv
import io
import os
import tempfile
import urllib.request
from datetime import datetime, timezone
from pathlib import Path


SOURCES = (
    ("NASDAQ", "https://www.nasdaqtrader.com/dynamic/SymDir/nasdaqlisted.txt"),
    ("OTHER", "https://www.nasdaqtrader.com/dynamic/SymDir/otherlisted.txt"),
)


def _download(url: str) -> str:
    request = urllib.request.Request(url, headers={"User-Agent": "OBAITS/1.0"})
    with urllib.request.urlopen(request, timeout=30) as response:  # noqa: S310 - fixed official URLs
        return response.read().decode("utf-8", errors="replace")


def collect() -> list[dict[str, str]]:
    output: dict[str, dict[str, str]] = {}
    for source, url in SOURCES:
        reader = csv.DictReader(io.StringIO(_download(url)), delimiter="|")
        for row in reader:
            symbol = str(row.get("Symbol") or row.get("ACT Symbol") or "").strip().upper()
            if not symbol or symbol.startswith("FILE CREATION TIME"):
                continue
            output[symbol] = {
                "symbol": symbol,
                "security_name": str(row.get("Security Name") or "").strip(),
                "exchange": str(row.get("Market Category") or row.get("Exchange") or source).strip(),
                "is_etf": "Y" if str(row.get("ETF") or "").strip().upper() == "Y" else "N",
                "source": url,
                "collected_at": datetime.now(timezone.utc).isoformat(),
            }
    return [output[key] for key in sorted(output)]


def write(rows: list[dict[str, str]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(
                handle,
                fieldnames=("symbol", "security_name", "exchange", "is_etf", "source", "collected_at"),
            )
            writer.writeheader()
            writer.writerows(rows)
        os.replace(temporary, path)
    except Exception:
        try:
            os.unlink(temporary)
        except OSError:
            pass
        raise


def main() -> int:
    parser = argparse.ArgumentParser(description="Collect official U.S. ETF classification; no orders.")
    parser.add_argument("--output", type=Path, default=Path("data/universe/us_instrument_types.csv"))
    args = parser.parse_args()
    rows = collect()
    if not rows:
        raise RuntimeError("official U.S. instrument catalogue was empty")
    write(rows, args.output)
    print({"output": str(args.output), "rows": len(rows), "etfs": sum(r["is_etf"] == "Y" for r in rows), "no_orders": True})
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

