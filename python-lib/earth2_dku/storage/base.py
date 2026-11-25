"""Storage abstraction used by all core code (local filesystem in tests, DSS folders in recipes)."""
from __future__ import annotations

import json
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

    def read_json(self, path: str):
        return json.loads(self.read_bytes(path).decode("utf-8"))

    def write_json(self, path: str, obj) -> None:
        self.write_bytes(path, json.dumps(obj, indent=2).encode("utf-8"))
