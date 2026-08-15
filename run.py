from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))


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


_require_runtime_dependencies()

from app.run import main


if __name__ == "__main__":
    main()
