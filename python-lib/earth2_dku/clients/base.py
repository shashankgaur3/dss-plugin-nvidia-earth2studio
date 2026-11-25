"""NIM client interface."""
from __future__ import annotations

import posixpath
from abc import ABC, abstractmethod
from dataclasses import dataclass
from datetime import datetime
from typing import Iterator


@dataclass
class HealthStatus:
    ready: bool
    detail: str


@dataclass
class ForecastStep:
    lead_hours: int
    batch_idx: int
    filename: str  # e.g. "006_000.npy"
    raw_bytes: bytes  # the .npy bytes, unchanged


def parse_step_filename(name: str) -> tuple[int, int]:
    """'006_000.npy' -> (6, 0). Raises ValueError for other names."""
    base = posixpath.basename(name)
    if not base.endswith(".npy"):
        raise ValueError(f"not a step file: {name}")
    parts = base[: -len(".npy")].split("_")
    if len(parts) != 2:
        raise ValueError(f"unexpected step file name: {name}")
    return int(parts[0]), int(parts[1])


class NimClient(ABC):
    #: Variables actually sent with the last accepted request; None means the NIM
    #: returned all model variables (no `variables` field sent). Set once the
    #: request has been accepted, i.e. before the first step is yielded.
    last_variables_sent: list[str] | None = None

    @abstractmethod
    def health(self) -> HealthStatus: ...

    @abstractmethod
    def infer_stream(
        self,
        input_array_bytes: bytes,
        init_time: datetime,
        simulation_length: int,
        variables: list[str] | None,
    ) -> Iterator[ForecastStep]:
        """Yield forecast steps as they arrive."""
