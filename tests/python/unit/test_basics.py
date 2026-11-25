import sys
import types

import numpy as np
import pytest

from earth2_dku.arrays import load_npy, npy_bytes
from earth2_dku.errors import ConfigError, NimStreamBrokenError
from earth2_dku.models.registry import get_model, list_models, load_registry
from earth2_dku.times import floor_to_cycle, parse_time, to_rfc3339
from fakes import make_model


# ---------------------------------------------------------------- errors / times
def test_stream_broken_error_carries_count_and_advice():
    err = NimStreamBrokenError(steps_received=3, detail="x")
    assert err.steps_received == 3
    assert "pod logs" in str(err)


@pytest.mark.parametrize("text", ["2023-01-01T00:00:00Z", "2023-01-01T00", "2023-01-01 00:00", "2023-01-01", "20230101T00"])
def test_parse_time_formats(text):
    assert to_rfc3339(parse_time(text)) == "2023-01-01T00:00:00Z"


def test_rfc3339_has_no_fractional_seconds():
    assert to_rfc3339(parse_time("2023-01-01T06:00:00.123456+00:00")) == "2023-01-01T06:00:00Z"


def test_parse_time_converts_offsets_to_utc():
    assert to_rfc3339(parse_time("2023-01-01T02:00:00+02:00")) == "2023-01-01T00:00:00Z"


def test_parse_time_rejects_garbage():
    with pytest.raises(ConfigError):
        parse_time("yesterday")
    with pytest.raises(ConfigError):
        parse_time("")


def test_floor_to_cycle():
    assert to_rfc3339(floor_to_cycle(parse_time("2023-01-01T11:59"))) == "2023-01-01T06:00:00Z"
    assert to_rfc3339(floor_to_cycle(parse_time("2023-01-01T18:00"))) == "2023-01-01T18:00:00Z"


# ------------------------------------------------------------------------ arrays
def test_npy_roundtrip():
    arr = np.arange(24, dtype="float32").reshape(1, 2, 3, 4)
    np.testing.assert_array_equal(load_npy(npy_bytes(arr)), arr)


# ---------------------------------------------------------------------- storage
def test_local_storage(storage):
    storage.write_bytes("/a/b.bin", b"x")
    assert storage.exists("a/b.bin") and storage.exists("/a/b.bin")
    assert not storage.exists("a/c.bin")
    assert storage.read_bytes("a/b.bin") == b"x"
    storage.write_bytes("a/m.json", b"{}")
    assert storage.list("a/") == ["a/b.bin", "a/m.json"]
    assert storage.list("/a/b") == ["a/b.bin"]


def test_prefixed_storage(storage):
    from earth2_dku.storage.base import PrefixedStorage

    view = PrefixedStorage(storage, "/inputs")
    view.write_bytes("era5/x.npy", b"1")
    storage.write_bytes("runs/m.json", b"2")
    assert storage.exists("inputs/era5/x.npy") and view.exists("/era5/x.npy")
    assert view.read_bytes("era5/x.npy") == b"1"
    assert view.list() == ["era5/x.npy"] and view.list("era5/") == ["era5/x.npy"]


def test_local_storage_blocks_escape(storage):
    with pytest.raises(ConfigError):
        storage.write_bytes("../escape", b"x")


# --------------------------------------------------------------------- registry
def test_registry_entries():
    ids = [m.model_id for m in list_models()]
    assert ids == ["fourcastnet-1.1.0", "fourcastnet-2.0.0"]
    m1, m2 = get_model("fourcastnet-1.1.0"), get_model("fourcastnet-2.0.0")
    assert (m1.n_channels, m1.min_gpu_mem_gb, m1.supports_variables_param, m1.nim_version) == (73, 40, "unverified", "1.1.0")
    assert (m2.n_channels, m2.min_gpu_mem_gb, m2.supports_variables_param) == (72, 60, "true")


def test_unknown_model_lists_available():
    with pytest.raises(ConfigError, match="fourcastnet-1.1.0"):
        get_model("nope")


def test_registry_rejects_bad_support_value(tmp_path):
    p = tmp_path / "m.yaml"
    p.write_text(
        "x:\n  label: X\n  nim_image: i:1\n  input_variables_from: a:B\n  n_channels: 1\n  step_hours: 6\n"
        "  min_gpu_mem_gb: 1\n  supports_variables_param: maybe\n  grid: era5_0p25\n"
    )
    with pytest.raises(ConfigError):
        load_registry(p)


def _install_fake_module(monkeypatch, name, variables):
    mod = types.ModuleType(name)
    mod.VARIABLES = variables
    monkeypatch.setitem(sys.modules, name, mod)


def test_input_variables_lazy_import(monkeypatch):
    _install_fake_module(monkeypatch, "fake_e2s_ok", ["a", "b", "c"])
    assert make_model(3, variables_from="fake_e2s_ok:VARIABLES").input_variables() == ["a", "b", "c"]


def test_input_variables_count_mismatch(monkeypatch):
    _install_fake_module(monkeypatch, "fake_e2s_bad", ["a", "b"])
    with pytest.raises(ConfigError, match="expects 3 input channels"):
        make_model(3, variables_from="fake_e2s_bad:VARIABLES").input_variables()


def test_input_variables_missing_module():
    with pytest.raises(ConfigError, match="code environment"):
        make_model(3, variables_from="does_not_exist_xyz:VARIABLES").input_variables()
