"""Storage abstraction used by all core code (local filesystem in tests, DSS folders in recipes)."""
from __future__ import annotations

from abc import ABC, abstractmethod


def normalize_path(path: str) -> str:
    """Canonical form used across the plugin: forward slashes, no leading slash."""
    return str(path).replace("\\", "/").lstrip("/")


class Storage(ABC):
    @abstractmethod
    def exists(self, path: str) -> bool: ...

    @abstractmethod
    def read_bytes(self, path: str) -> bytes: ...

    @abstractmethod
    def write_bytes(self, path: str, data: bytes) -> None: ...

    @abstractmethod
    def list(self, prefix: str = "") -> list[str]:
        """Return normalized paths (no leading slash) starting with ``prefix``."""


class PrefixedStorage(Storage):
    """View of another storage restricted to a sub-folder (paths are relative to it)."""

    def __init__(self, inner: Storage, prefix: str):
        self.inner = inner
        self.prefix = normalize_path(prefix).rstrip("/") + "/"

    def _full(self, path: str) -> str:
        return self.prefix + normalize_path(path)

    def exists(self, path: str) -> bool:
        return self.inner.exists(self._full(path))

    def read_bytes(self, path: str) -> bytes:
        return self.inner.read_bytes(self._full(path))

    def write_bytes(self, path: str, data: bytes) -> None:
        self.inner.write_bytes(self._full(path), data)

    def list(self, prefix: str = "") -> list[str]:
        return [p[len(self.prefix):] for p in self.inner.list(self._full(prefix))]
