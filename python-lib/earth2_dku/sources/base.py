"""Initial-condition source interface and shape validation."""
from __future__ import annotations

from abc import ABC, abstractmethod
from datetime import datetime

import numpy as np

from ..errors import ConfigError, PreflightError

N_LAT, N_LON = 721, 1440


def validate_state_array(arr: np.ndarray, model) -> None:
    expected = (1, 1, model.n_channels, N_LAT, N_LON)
    if arr.shape != expected or arr.dtype != np.float32:
        raise PreflightError(
            f"Initial condition has shape {arr.shape} / dtype {arr.dtype}; {model.model_id} needs "
            f"{expected} float32. A wrong channel count makes the NIM fail mid-stream "
            "('size of tensor a (73) must match ... (72)'). Re-fetch initial conditions for this model."
        )


def earth2studio_missing(exc: Exception) -> ConfigError:
    return ConfigError(
        f"Earth2Studio is required to fetch initial conditions but could not be imported ({exc}). "
        "Rebuild the plugin code environment (earth2studio>=0.3.0, Python 3.11/3.12)."
    )


class InitialConditionSource(ABC):
    name: str = ""
    earliest_time: datetime | None = None

    @abstractmethod
    def get_state(self, time: datetime, variables: list[str]) -> np.ndarray:
        """float32 array of shape (1, 1, n_vars, 721, 1440)."""

    @abstractmethod
    def latest_available_time(self) -> datetime: ...
