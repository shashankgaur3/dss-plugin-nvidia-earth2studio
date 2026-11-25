"""Storage backed by a Dataiku managed folder. One of only two modules allowed to import dataiku."""
from __future__ import annotations

import dataiku

from .base import Storage, normalize_path


def _dss_path(path: str) -> str:
    # DSS folder APIs use paths with a leading slash.
    return "/" + normalize_path(path)


class DkuFolderStorage(Storage):
    def __init__(self, folder):
        self.folder = folder

    @classmethod
    def from_name(cls, name: str) -> "DkuFolderStorage":
        return cls(dataiku.Folder(name))

    def exists(self, path: str) -> bool:
        try:
            details = self.folder.get_path_details(_dss_path(path))
            return bool(details.get("exists")) and not details.get("directory", False)
        except Exception:  # older DSS versions / connections without path details
            return normalize_path(path) in set(self.list(path))

    def read_bytes(self, path: str) -> bytes:
        with self.folder.get_download_stream(_dss_path(path)) as stream:
            return stream.read()

    def write_bytes(self, path: str, data: bytes) -> None:
        with self.folder.get_writer(_dss_path(path)) as writer:
            writer.write(data)

    def list(self, prefix: str = "") -> list[str]:
        prefix = normalize_path(prefix)
        paths = (normalize_path(p) for p in self.folder.list_paths_in_partition())
        return sorted(p for p in paths if p.startswith(prefix))
