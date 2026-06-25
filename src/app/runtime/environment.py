from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

DataMode = Literal["realtime", "custom"]


@dataclass(frozen=True)
class DataEnvironment:
    mode: DataMode
    root: Path

    @property
    def store_dir(self) -> Path:
        return self.root / "store"

    @property
    def raw_dir(self) -> Path:
        return self.root / "raw"

    @property
    def model_dir(self) -> Path:
        return self.root / "models"

    @property
    def synthetic_dir(self) -> Path:
        return self.root / "synthetic_disabled"

    @property
    def report_dir(self) -> Path:
        return self.root / "reports"

    def ensure_layout(self) -> None:
        for path in (self.store_dir, self.raw_dir, self.model_dir, self.synthetic_dir, self.report_dir):
            path.mkdir(parents=True, exist_ok=True)

    @classmethod
    def realtime(cls, root: Path = Path("data")) -> "DataEnvironment":
        env = cls(mode="realtime", root=root)
        env.ensure_layout()
        return env

    @classmethod
    def live(cls, root: Path = Path("data")) -> "DataEnvironment":
        return cls.realtime(root)

    @classmethod
    def simulation(cls, root: Path = Path("data")) -> "DataEnvironment":
        return cls.realtime(root)

    @classmethod
    def custom(cls, root: Path) -> "DataEnvironment":
        env = cls(mode="custom", root=root)
        env.ensure_layout()
        return env


def default_environment() -> DataEnvironment:
    root = Path(os.getenv("DATA_ROOT", "data"))
    mode = os.getenv("DATA_ENV", "realtime").strip().lower()
    if mode in {"custom"}:
        return DataEnvironment.custom(root)
    return DataEnvironment.realtime(root)
