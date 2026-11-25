"""ERA5 reanalysis from the ARCO Zarr store on Google Cloud (via Earth2Studio, imported lazily)."""
from __future__ import annotations

from datetime import datetime, timedelta

import numpy as np

from ..times import UTC, floor_to_cycle, parse_time, utcnow
from .base import InitialConditionSource, earth2studio_missing

DEFAULT_LAG_DAYS = 6  # preliminary ERA5 lags ~5 days; keep a margin


class Era5ArcoSource(InitialConditionSource):
    name = "era5_arco"
    earliest_time = datetime(1940, 1, 1, tzinfo=UTC)

    def __init__(self, lag_days: float = DEFAULT_LAG_DAYS, now=None, cache: bool = False):
        self.lag_days = float(lag_days)
        self._now = now or utcnow
        self.cache = cache

    def latest_available_time(self) -> datetime:
        return floor_to_cycle(self._now() - timedelta(days=self.lag_days))

    def get_state(self, time, variables) -> np.ndarray:
        try:
            from earth2studio.data import ARCO
        except ImportError as exc:
            raise earth2studio_missing(exc) from exc
        t = parse_time(time).replace(tzinfo=None)
        data = ARCO(cache=self.cache)(time=[t], variable=list(variables))
        return data.to_numpy()[None].astype("float32")
