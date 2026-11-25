"""Initial-condition sources. Earth2Studio is imported lazily inside get_state() only."""
from __future__ import annotations

from ..errors import ConfigError
from .base import InitialConditionSource, validate_state_array
from .era5_arco import Era5ArcoSource
from .gfs import GfsSource

SOURCES = {"era5_arco": Era5ArcoSource, "gfs": GfsSource}


def build_source(name: str, **kwargs) -> InitialConditionSource:
    if name not in SOURCES:
        raise ConfigError(f"Unknown initial-condition source '{name}'. Use one of: {', '.join(SOURCES)}.")
    return SOURCES[name](**kwargs)


__all__ = ["SOURCES", "Era5ArcoSource", "GfsSource", "InitialConditionSource", "build_source", "validate_state_array"]
