from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))


def _install_thread_dump_signal() -> None:
    """Dump every thread's Python stack to stderr on SIGUSR1.

    This process runs ~90 threads plus several asyncio loops, and its worst
    failures are silent stalls rather than exceptions: on 2026-08-21 the KIS
    market-data socket stayed ESTABLISHED with the kernel receive queue climbing
    while nothing raised, and the only way to find out which await was wedged was
    to read the stacks. ``py-spy`` cannot attach here because
    ``kernel.yama.ptrace_scope`` is 1, so the process has to be able to tell us
    itself. Read-only, costs nothing until the signal arrives.

    Usage: ``kill -USR1 <pid>`` then read ``logs/run-server.err.log``.
    """
    import faulthandler
    import signal

    handler = getattr(signal, "SIGUSR1", None)
    if handler is None:  # Windows has no SIGUSR1.
        return
    try:
        faulthandler.register(handler, all_threads=True, chain=False)
    except Exception:  # noqa: BLE001 - diagnostics must never block startup.
        pass


def _require_runtime_dependencies() -> None:
    """Fail before starting workers when a partial Python environment is used.

    The web process can otherwise boot with FastAPI installed and fail minutes
    later when the live ontology graph imports rdflib.  That leaves trading
    marked running while the live cache silently stops refreshing.
    """

    required = ("rdflib",)
    missing = [name for name in required if importlib.util.find_spec(name) is None]
    if not missing:
        return
    venv_python = ROOT / ".venv" / "Scripts" / "python.exe"
    hint = f' Use "{venv_python}" run.py.' if venv_python.exists() else " Install requirements.txt."
    raise SystemExit(
        f"Missing runtime dependencies: {', '.join(missing)}."
        f" Refusing to start a partially functional live server.{hint}"
    )


_install_thread_dump_signal()
_require_runtime_dependencies()

from app.run import main


if __name__ == "__main__":
    main()
