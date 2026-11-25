"""NOAA GFS analysis (via Earth2Studio, imported lazily). Cycles every 6 hours."""
from __future__ import annotations

from datetime import datetime, timedelta

import numpy as np

from ..times import floor_to_cycle, parse_time, utcnow
from .base import InitialConditionSource, earth2studio_missing

DEFAULT_SAFETY_MARGIN_HOURS = 6


class GfsSource(InitialConditionSource):
    name = "gfs"

    def __init__(self, safety_margin_hours: float = DEFAULT_SAFETY_MARGIN_HOURS, now=None, cache: bool = False):
        self.safety_margin_hours = float(safety_margin_hours)
        self._now = now or utcnow
        self.cache = cache

    def latest_available_time(self) -> datetime:
        return floor_to_cycle(self._now() - timedelta(hours=self.safety_margin_hours))

    def get_state(self, time, variables) -> np.ndarray:
        try:
            from earth2studio.data import GFS
        except ImportError as exc:
            raise earth2studio_missing(exc) from exc
        t = parse_time(time).replace(tzinfo=None)
        data = GFS(cache=self.cache)(time=[t], variable=list(variables))
        return data.to_numpy()[None].astype("float32")
