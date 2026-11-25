"""Caching of initial conditions in a storage, keyed {source}/{model_id}/{YYYYmmddTHH}.npy."""
from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import datetime

from ..arrays import load_npy, npy_bytes
from ..errors import ConfigError, PreflightError
from ..times import floor_to_cycle, parse_time
from .base import InitialConditionSource, validate_state_array

logger = logging.getLogger(__name__)

def ic_key(source_name: str, model_id: str, time) -> str:
    return f"{source_name}/{model_id}/{parse_time(time):%Y%m%dT%H}.npy"


@dataclass(frozen=True)
class ICRef:
    source: str
    model_id: str
    time: datetime
    path: str


class CachedSource(InitialConditionSource):
    """Wraps a source; reuses arrays already present in storage."""

    def __init__(self, source: InitialConditionSource, storage, model_id: str, validator=None):
        self.source = source
        self.storage = storage
        self.model_id = model_id
        self.validator = validator
        self.last_hit = False

    @property
    def name(self) -> str:
        return self.source.name

    def latest_available_time(self) -> datetime:
        return self.source.latest_available_time()

    def key(self, time) -> str:
        return ic_key(self.source.name, self.model_id, time)

    def get_state(self, time, variables):
        key = self.key(time)
        if self.storage.exists(key):
            arr = load_npy(self.storage.read_bytes(key))
            try:
                if self.validator:
                    self.validator(arr)
                self.last_hit = True
                return arr
            except PreflightError as exc:
                logger.warning("Cached %s is invalid (%s); fetching again.", key, exc)
        self.last_hit = False
        arr = self.source.get_state(parse_time(time), variables)
        if self.validator:
            self.validator(arr)  # never cache a bad array
        self.storage.write_bytes(key, npy_bytes(arr))
        return arr


def fetch_initial_condition(source, storage, model, time) -> tuple[ICRef, bool]:
    """Fetch (or reuse) the initial condition for ``model`` at ``time`` (rounded down to a 6 h cycle).

    Returns (reference, was_cached).
    """
    t = floor_to_cycle(time)
    if t != parse_time(time):
        logger.info("Init time %s rounded down to the %s cycle.", parse_time(time), t)
    earliest = getattr(source, "earliest_time", None)
    if earliest and t < earliest:
        raise ConfigError(f"{source.name} has no data before {earliest:%Y-%m-%d}. Choose a later init time.")
    latest = source.latest_available_time()
    if t > latest:
        raise ConfigError(
            f"{t:%Y-%m-%dT%H}Z is not yet available from {source.name} (latest: {latest:%Y-%m-%dT%H}Z). "
            "Choose an earlier init time or use time mode 'latest_available'."
        )
    cached = CachedSource(source, storage, model.model_id, validator=lambda a: validate_state_array(a, model))
    cached.get_state(t, model.input_variables())
    return ICRef(source.name, model.model_id, t, cached.key(t)), cached.last_hit
