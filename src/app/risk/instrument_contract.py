"""Orderable instrument restrictions shared by legacy and ontology adapters."""


def is_non_common_equity_ticker(ticker: str) -> bool:
    """US warrant/unit/right suffixes; ordinary and numeric KR symbols survive."""
    symbol = str(ticker or "").upper().strip()
    if not symbol:
        return False
    suffixes = (".WS", "-WT", ".WT", "/WS", "-WS", "+",
                ".U", "-UN", ".UN", "-U", "/U", ".RT", "-RT", ".RTS", "-RTS", "/R")
    return any(symbol.endswith(suffix) for suffix in suffixes) or (
        symbol.isalpha() and len(symbol) == 5 and symbol[-1] in {"W", "U", "R"}
    )
