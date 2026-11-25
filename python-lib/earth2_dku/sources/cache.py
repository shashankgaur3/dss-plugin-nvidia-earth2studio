"""Caching of initial conditions in a storage, keyed {source}/{model_id}/{YYYYmmddTHH}.npy."""
from __future__ import annotations

import logging
import re
from dataclasses import dataclass
from datetime import datetime

from ..arrays import load_npy, npy_bytes
from ..errors import ConfigError, PreflightError
from ..times import UTC, floor_to_cycle, parse_time
from .base import InitialConditionSource, validate_state_array

logger = logging.getLogger(__name__)

_KEY_RE = re.compile(r"^(?P<source>[^/]+)/(?P<model>[^/]+)/(?P<time>\d{8}T\d{2})\.npy$")


def ic_key(source_name: str, model_id: str, time) -> str:
    return f"{source_name}/{model_id}/{parse_time(time):%Y%m%dT%H}.npy"


@dataclass(frozen=True)
class ICRef:
    source: str
    model_id: str
    time: datetime
    path: str


def parse_ic_key(path: str) -> ICRef | None:
    m = _KEY_RE.match(path.lstrip("/"))
    if not m:
        return None
    t = datetime.strptime(m["time"], "%Y%m%dT%H").replace(tzinfo=UTC)
    return ICRef(m["source"], m["model"], t, path.lstrip("/"))


def list_initial_conditions(storage, model_id: str | None = None) -> list[ICRef]:
    refs = [r for r in (parse_ic_key(p) for p in storage.list()) if r]
    if model_id:
        refs = [r for r in refs if r.model_id == model_id]
    return sorted(refs, key=lambda r: (r.time, r.source))


def find_initial_condition(storage, model_id: str, init_time=None, source: str | None = None) -> ICRef:
    """Pick the initial condition to run: the given init time, or the latest one present."""
    refs = list_initial_conditions(storage, model_id)
    if source:
        refs = [r for r in refs if r.source == source]
    if not refs:
        raise ConfigError(
            f"No initial conditions for model {model_id} in the input folder. Run the 'Fetch initial "
            "conditions' recipe with the same connection preset (model) first."
        )
    if init_time is None:
        latest = refs[-1].time
        candidates = [r for r in refs if r.time == latest]
    else:
        wanted = parse_time(init_time)
        candidates = [r for r in refs if r.time == wanted]
        if not candidates:
            raise ConfigError(
                f"No initial condition for {wanted:%Y-%m-%dT%H}Z and model {model_id}. Available: "
                f"{', '.join(sorted({f'{r.time:%Y-%m-%dT%H}' for r in refs}))}."
            )
    if len(candidates) > 1:
        logger.warning("Several sources for %s; using %s.", candidates[0].time, candidates[0].source)
    return candidates[0]


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

    @property
    def earliest_time(self):
        return self.source.earliest_time

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
