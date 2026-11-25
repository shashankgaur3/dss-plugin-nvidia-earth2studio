import sys

import numpy as np
import pytest

from earth2_dku.arrays import npy_bytes
from earth2_dku.errors import ConfigError, PreflightError
from earth2_dku.sources import build_source
from earth2_dku.sources.base import validate_state_array
from earth2_dku.sources.cache import (
    CachedSource, fetch_initial_condition, find_initial_condition, ic_key, list_initial_conditions, parse_ic_key,
)
from earth2_dku.sources.era5_arco import Era5ArcoSource
from earth2_dku.sources.gfs import GfsSource
from earth2_dku.times import parse_time, to_rfc3339
from fakes import FakeSource, make_model

NOW = lambda: parse_time("2024-06-10T09:30")  # noqa: E731


def test_importing_sources_does_not_import_earth2studio():
    import earth2_dku.sources  # noqa: F401

    assert "earth2studio" not in sys.modules


def test_latest_available_times():
    assert to_rfc3339(Era5ArcoSource(now=NOW).latest_available_time()) == "2024-06-04T06:00:00Z"
    assert to_rfc3339(GfsSource(now=NOW).latest_available_time()) == "2024-06-10T00:00:00Z"
    assert to_rfc3339(GfsSource(safety_margin_hours=12, now=NOW).latest_available_time()) == "2024-06-09T18:00:00Z"


def test_build_source():
    assert build_source("gfs").name == "gfs"
    with pytest.raises(ConfigError, match="era5_arco"):
        build_source("nope")


def test_get_state_without_earth2studio_gives_clear_error(monkeypatch):
    monkeypatch.setitem(sys.modules, "earth2studio", None)
    with pytest.raises(ConfigError, match="code environment"):
        Era5ArcoSource().get_state(parse_time("2023-01-01"), ["u10m"])


def test_keys():
    k = ic_key("era5_arco", "fourcastnet-1.1.0", parse_time("2023-01-01T06"))
    assert k == "era5_arco/fourcastnet-1.1.0/20230101T06.npy"
    ref = parse_ic_key("/" + k)
    assert ref.source == "era5_arco" and ref.time == parse_time("2023-01-01T06")
    assert parse_ic_key("other/file.txt") is None


def test_validate_state_array():
    model = make_model(3)
    validate_state_array(np.zeros((1, 1, 3, 721, 1440), dtype="float32"), model)
    with pytest.raises(PreflightError, match="73"):
        validate_state_array(np.zeros((1, 1, 2, 721, 1440), dtype="float32"), model)
    with pytest.raises(PreflightError):
        validate_state_array(np.zeros((1, 1, 3, 721, 1440), dtype="float64"), model)


def test_cached_source_reuses(storage):
    src = FakeSource()
    cached = CachedSource(src, storage, "m1")
    cached.get_state(parse_time("2023-01-01"), ["a", "b"])
    assert not cached.last_hit and storage.exists("fake/m1/20230101T00.npy")
    arr = cached.get_state(parse_time("2023-01-01"), ["a", "b"])
    assert cached.last_hit and arr.shape == (1, 1, 2, 721, 1440)
    assert len(src.calls) == 1


def test_cached_source_never_caches_bad_arrays(storage):
    model = make_model(3)
    src = FakeSource(shape_override=(1, 1, 2, 721, 1440))
    cached = CachedSource(src, storage, "m1", validator=lambda a: validate_state_array(a, model))
    with pytest.raises(PreflightError):
        cached.get_state(parse_time("2023-01-01"), ["a", "b", "c"])
    assert not storage.exists("fake/m1/20230101T00.npy")


def test_cached_source_refetches_invalid_cache(storage):
    model = make_model(2)
    storage.write_bytes("fake/m1/20230101T00.npy", npy_bytes(np.zeros((1, 1, 5, 721, 1440), dtype="float32")))
    src = FakeSource()
    cached = CachedSource(src, storage, "m1", validator=lambda a: validate_state_array(a, model))
    assert cached.get_state(parse_time("2023-01-01"), ["a", "b"]).shape[2] == 2
    assert len(src.calls) == 1


def test_fetch_initial_condition(storage, monkeypatch):
    model = make_model(2)
    monkeypatch.setattr(type(model), "input_variables", lambda self: ["a", "b"])
    src = FakeSource()
    ref, cached = fetch_initial_condition(src, storage, model, "2023-01-01T08:15")
    assert ref.path == "fake/fake-model/20230101T06.npy" and not cached
    _, cached = fetch_initial_condition(src, storage, model, "2023-01-01T06")
    assert cached
    with pytest.raises(ConfigError, match="not yet available"):
        fetch_initial_condition(src, storage, model, "2030-01-01")


def test_find_initial_condition(storage):
    for key in ("era5_arco/m1/20230101T00.npy", "era5_arco/m1/20230101T06.npy", "gfs/m2/20230102T00.npy", "notes.txt"):
        storage.write_bytes(key, b"x")
    assert [r.path for r in list_initial_conditions(storage, "m1")] == [
        "era5_arco/m1/20230101T00.npy", "era5_arco/m1/20230101T06.npy"]
    assert find_initial_condition(storage, "m1").path.endswith("T06.npy")
    assert find_initial_condition(storage, "m1", parse_time("2023-01-01T00")).path.endswith("T00.npy")
    with pytest.raises(ConfigError, match="Available"):
        find_initial_condition(storage, "m1", parse_time("2023-02-01"))
    with pytest.raises(ConfigError, match="Fetch initial"):
        find_initial_condition(storage, "m3")
