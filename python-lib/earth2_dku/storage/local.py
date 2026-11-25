"""Filesystem-backed storage for tests and notebooks."""
from __future__ import annotations

from pathlib import Path

from ..errors import ConfigError
from .base import Storage, normalize_path


class LocalStorage(Storage):
    def __init__(self, root):
        self.root = Path(root).resolve()
        self.root.mkdir(parents=True, exist_ok=True)

    def _full(self, path: str) -> Path:
        full = (self.root / normalize_path(path)).resolve()
        if full != self.root and self.root not in full.parents:
            raise ConfigError(f"Path '{path}' escapes the storage root {self.root}.")
        return full

    def exists(self, path: str) -> bool:
        return self._full(path).is_file()

    def read_bytes(self, path: str) -> bytes:
        return self._full(path).read_bytes()

    def write_bytes(self, path: str, data: bytes) -> None:
        full = self._full(path)
        full.parent.mkdir(parents=True, exist_ok=True)
        tmp = full.with_name(full.name + ".tmp")
        tmp.write_bytes(data)
        tmp.replace(full)

    def list(self, prefix: str = "") -> list[str]:
        prefix = normalize_path(prefix)
        out = []
        for p in self.root.rglob("*"):
            if p.is_file() and not p.name.endswith(".tmp"):
                rel = p.relative_to(self.root).as_posix()
                if rel.startswith(prefix):
                    out.append(rel)
        return sorted(out)
