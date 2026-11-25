"""Registry of derived variables computed from extracted model variables."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np
import pandas as pd

from ..errors import ConfigError


@dataclass(frozen=True)
class DerivedVariable:
    name: str
    requires: tuple
    func: Callable
    description: str = ""


_REGISTRY: dict[str, DerivedVariable] = {}


def derived(name: str, requires, description: str = ""):
    def decorator(func):
        _REGISTRY[name] = DerivedVariable(name, tuple(requires), func, description)
        return func

    return decorator


def available_derived() -> list[str]:
    return list(_REGISTRY)


def get_derived(name: str) -> DerivedVariable:
    if name not in _REGISTRY:
        raise ConfigError(f"Unknown derived variable '{name}'. Available: {', '.join(_REGISTRY)}.")
    return _REGISTRY[name]


def required_inputs(names) -> list[str]:
    out: list[str] = []
    for name in names or []:
        for r in get_derived(name).requires:
            if r not in out:
                out.append(r)
    return out


def add_derived(wide: pd.DataFrame, names) -> pd.DataFrame:
    for name in names or []:
        spec = get_derived(name)
        missing = [r for r in spec.requires if r not in wide.columns]
        if missing:
            raise ConfigError(
                f"Derived variable '{name}' needs {list(spec.requires)} but {missing} are not in the "
                "forecast output. Re-run the forecast requesting these variables."
            )
        wide[name] = spec.func(*[wide[r].to_numpy(dtype="float64") for r in spec.requires])
    return wide


@derived("wind_speed_10m", requires=["u10m", "v10m"], description="10 m wind speed (m/s)")
def _wind_speed_10m(u, v):
    return np.hypot(u, v)


@derived("wind_speed_100m", requires=["u100m", "v100m"], description="100 m wind speed (m/s)")
def _wind_speed_100m(u, v):
    return np.hypot(u, v)


@derived("t2m_celsius", requires=["t2m"], description="2 m temperature (deg C)")
def _t2m_celsius(t2m):
    return t2m - 273.15
