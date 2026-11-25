import pytest

from earth2_dku.arrays import npy_bytes
from earth2_dku.errors import PreflightError
from earth2_dku.models.registry import get_model
from earth2_dku.validation.preflight import (
    check_gpu_memory, check_init_time, check_input_array, check_input_variables, ensure_ready_for_forecast,
    render_html, run_preflight,
)
from fakes import FakeNimClient, FakeSource, make_model

import numpy as np


def _by_name(results):
    return {r.name: r for r in results}


def test_endpoint_down_is_flagged():
    res = _by_name(run_preflight("fourcastnet-1.1.0", FakeNimClient([], ready=False)))
    assert not res["NIM endpoint health"].ok and "down" in res["NIM endpoint health"].message
    assert res["Model in registry"].ok


def test_unknown_model():
    res = run_preflight("nope", FakeNimClient([]))
    assert not res[1].ok and "fourcastnet-1.1.0" in res[1].message
    assert not res[2].ok and "Skipped" in res[2].message


def test_gpu_memory_2_0_0_on_48gb():
    r = check_gpu_memory(get_model("fourcastnet-2.0.0"), 48)
    assert not r.ok
    assert "FourCastNet NIM 2.0.0 needs a single GPU with 60 GB" in r.message
    assert "1.1.0" in r.message and "cannot be split" in r.message
    assert check_gpu_memory(get_model("fourcastnet-1.1.0"), 48).ok
    assert check_gpu_memory(get_model("fourcastnet-2.0.0"), 80).ok
    assert check_gpu_memory(get_model("fourcastnet-2.0.0"), None).ok


def test_input_variables_check_reports_missing_earth2studio():
    r = check_input_variables(make_model(3, variables_from="no_such_mod:VARIABLES"))
    assert not r.ok and "code environment" in r.message


def test_init_time_checks():
    src = FakeSource(now="2024-06-10T09:30")
    assert check_init_time(src, None).ok
    assert check_init_time(src, "2024-06-01T00").ok
    r = check_init_time(src, "2024-06-01T05")
    assert r.ok and "rounded down" in r.message
    assert not check_init_time(src, "2024-06-11T00").ok
    assert not check_init_time(src, "garbage").ok


def test_input_array_check_and_ensure_ready():
    model = make_model(2)
    good = npy_bytes(np.zeros((1, 1, 2, 721, 1440), dtype="float32"))
    bad = npy_bytes(np.zeros((1, 1, 3, 721, 1440), dtype="float32"))
    assert check_input_array(good, model).ok
    assert not check_input_array(bad, model).ok
    assert not check_input_array(b"junk", model).ok
    ensure_ready_for_forecast(FakeNimClient([]), model, good)
    with pytest.raises(PreflightError, match="Input array shape"):
        ensure_ready_for_forecast(FakeNimClient([]), model, bad)
    with pytest.raises(PreflightError, match="health"):
        ensure_ready_for_forecast(FakeNimClient([], ready=False), model, good)


def test_render_html_escapes():
    from earth2_dku.validation.preflight import CheckResult

    out = render_html([CheckResult("a<b", False, "x & y"), CheckResult("c", True, "fine")])
    assert "a&lt;b" in out and "x &amp; y" in out and "1 check(s) failed" in out
