import sys
import types

import numpy as np
import pytest

from earth2_dku.arrays import load_npy
from earth2_dku.clients.self_hosted import SelfHostedNimClient
from earth2_dku.errors import ConfigError, NimRequestError, NimStreamBrokenError, PreflightError
from earth2_dku.forecast.manifest import read_manifest
from earth2_dku.forecast.runner import INPUTS_PREFIX, run_forecast, run_forecasts
from earth2_dku.storage.base import PrefixedStorage
from earth2_dku.times import parse_time
from fakes import SMALL_STEP_SHAPE, FakeNimClient, FakeNimServer, FakeSource, make_model, small_steps

RUN_DIR = "runs/init=2023-01-01T00"
INIT = parse_time("2023-01-01T00")


def _run(client, storage, model=None, variables=("u10m", "v10m"), n=2):
    return run_forecast(client, model or make_model(), storage, RUN_DIR, b"in", INIT, n,
                        list(variables) if variables else None, source="era5_arco")


def test_complete_run(storage):
    files = small_steps(3)
    m = _run(FakeNimClient(files), storage)
    assert m.status == "complete" and m.steps_received == 3
    assert [s.file for s in m.steps] == ["steps/000_000.npy", "steps/006_000.npy", "steps/012_000.npy"]
    assert m.output_variables == ["u10m", "v10m"] and m.source == "era5_arco"
    on_disk = read_manifest(storage, RUN_DIR)
    assert on_disk.status == "complete" and on_disk.steps == m.steps
    assert storage.read_bytes(f"{RUN_DIR}/steps/012_000.npy") == files[2][1]


def test_partial_run_preserved(storage):
    with pytest.raises(NimStreamBrokenError) as exc:
        _run(FakeNimClient(small_steps(3), break_after=2), storage)
    assert exc.value.steps_received == 2
    m = read_manifest(storage, RUN_DIR)
    assert m.status == "partial" and m.steps_received == 2 and len(m.steps) == 2
    assert "pod logs" in m.message
    assert storage.exists(f"{RUN_DIR}/steps/006_000.npy")


def test_break_before_any_step_is_failed(storage):
    with pytest.raises(NimStreamBrokenError):
        _run(FakeNimClient(small_steps(3), break_after=0), storage)
    assert read_manifest(storage, RUN_DIR).status == "failed"


def test_other_errors_recorded_and_reraised(storage):
    class Boom(FakeNimClient):
        def infer_stream(self, *a, **k):
            raise NimRequestError("nope")
            yield  # pragma: no cover

    with pytest.raises(NimRequestError):
        _run(Boom([]), storage)
    m = read_manifest(storage, RUN_DIR)
    assert m.status == "failed" and "nope" in m.message


def test_dropped_variables_falls_back_to_model_list(storage, monkeypatch):
    mod = types.ModuleType("fake_e2s_run")
    mod.VARIABLES = ["a", "b"]
    monkeypatch.setitem(sys.modules, "fake_e2s_run", mod)
    model = make_model(2, variables_from="fake_e2s_run:VARIABLES")
    m = _run(FakeNimClient(small_steps(3), variables_dropped=True), storage, model=model)
    assert m.output_variables == ["a", "b"]


def test_end_to_end_over_http_with_break(storage):
    with FakeNimServer(small_steps(4), break_after=3) as srv:
        with pytest.raises(NimStreamBrokenError):
            _run(SelfHostedNimClient(srv.base_url), storage, n=3)
    m = read_manifest(storage, RUN_DIR)
    assert m.status == "partial" and m.steps_received == 3
    np.testing.assert_array_equal(load_npy(storage.read_bytes(f"{RUN_DIR}/steps/012_000.npy")),
                                  np.full(SMALL_STEP_SHAPE, 2, dtype="float32"))


# ------------------------------------------------------------- fetch + run (recipe core)
@pytest.fixture
def two_var_model(monkeypatch):
    model = make_model(2)
    monkeypatch.setattr(type(model), "input_variables", lambda self: ["u10m", "v10m"])
    return model


def _run_many(client, storage, model, times, source=None, **kw):
    inputs = PrefixedStorage(storage, INPUTS_PREFIX)
    return run_forecasts(client, model, source or FakeSource(), inputs, storage, [parse_time(t) for t in times],
                         2, ["u10m", "v10m"], **kw)


def test_run_forecasts_fetches_inputs_into_forecast_folder(storage, two_var_model):
    src = FakeSource()
    ms = _run_many(FakeNimClient(small_steps(3)), storage, two_var_model, ["2023-01-01T00", "2023-01-02T07"], src)
    assert [m.init_time for m in ms] == ["2023-01-01T00:00:00Z", "2023-01-02T06:00:00Z"]
    assert all(m.status == "complete" and m.source == "fake" for m in ms)
    assert storage.exists("inputs/fake/fake-model/20230102T06.npy")
    assert storage.exists("runs/init=2023-01-02T06/steps/012_000.npy")
    # second run reuses the cached inputs
    _run_many(FakeNimClient(small_steps(3)), storage, two_var_model, ["2023-01-01T00"], src)
    assert len(src.calls) == 2


def test_run_forecasts_checks_nim_before_downloading(storage, two_var_model):
    src = FakeSource()
    with pytest.raises(PreflightError, match="not ready: fake endpoint down"):
        _run_many(FakeNimClient([], ready=False), storage, two_var_model, ["2023-01-01T00"], src)
    assert src.calls == []


def test_run_forecasts_rejects_unavailable_time(storage, two_var_model):
    with pytest.raises(ConfigError, match="not yet available"):
        _run_many(FakeNimClient(small_steps(3)), storage, two_var_model, ["2030-01-01T00"])


def test_run_forecasts_partial_policy(storage, two_var_model):
    ms = _run_many(FakeNimClient(small_steps(3), break_after=2), storage, two_var_model, ["2023-01-01T00"])
    assert ms[0].status == "partial" and ms[0].steps_received == 2
    with pytest.raises(NimStreamBrokenError):
        _run_many(FakeNimClient(small_steps(3), break_after=2), storage, two_var_model, ["2023-01-01T00"],
                  fail_on_partial=True)
    with pytest.raises(NimStreamBrokenError):
        _run_many(FakeNimClient(small_steps(3), break_after=0), storage, two_var_model, ["2023-01-01T00"])
