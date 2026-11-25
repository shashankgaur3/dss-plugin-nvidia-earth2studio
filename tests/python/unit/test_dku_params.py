"""dku/params.py parsing helpers, with a stub `dataiku` module (DSS is not installed in unit tests)."""
import sys
import types

import pandas as pd
import pytest

from earth2_dku.errors import ConfigError


@pytest.fixture
def params(monkeypatch):
    dku = types.ModuleType("dataiku")
    dku.get_custom_variables = lambda: {"init": "2023-01-01T07:00"}
    cr = types.ModuleType("dataiku.customrecipe")
    cr.get_input_names_for_role = lambda role: []
    cr.get_output_names_for_role = lambda role: []
    monkeypatch.setitem(sys.modules, "dataiku", dku)
    monkeypatch.setitem(sys.modules, "dataiku.customrecipe", cr)
    from earth2_dku.dku import params as p

    return p


PRESET = {"connection": {"deployment": "self_hosted", "base_url": " http://nim:8000/ ", "model_id": "fourcastnet-2.0.0",
                         "timeout_seconds": 600.0, "gpu_memory_gb": 48}}


def test_connection_and_client(params):
    conn = params.connection_from_config(PRESET)
    assert (conn.base_url, conn.model_id, conn.timeout_seconds, conn.gpu_memory_gb) == ("http://nim:8000/", "fourcastnet-2.0.0", 600, 48.0)
    model = params.build_model(conn)
    client = params.build_client(conn, model)
    assert client.base_url == "http://nim:8000" and client.supports_variables_param == "true"
    with pytest.raises(ConfigError, match="preset"):
        params.connection_from_config({})
    hosted = params.connection_from_config({"connection": {"deployment": "nvcf_hosted", "api_key": "k"}})
    assert type(params.build_client(hosted)).__name__ == "NvcfHostedClient"


def test_parse_list(params):
    assert params.parse_list(" u10m , v10m,") == ["u10m", "v10m"]
    assert params.parse_list("") is None and params.parse_list(None) is None


def test_resolve_init_times(params):
    from earth2_dku.times import to_rfc3339
    from fakes import FakeSource

    src = FakeSource(now="2024-06-10T09:30")
    fixed = params.resolve_init_times({"time_mode": "fixed", "init_time": "2023-01-01T11:00"}, src)
    assert [to_rfc3339(t) for t in fixed] == ["2023-01-01T06:00:00Z"]
    latest = params.resolve_init_times({"time_mode": "latest_available"}, src)
    assert [to_rfc3339(t) for t in latest] == ["2024-06-10T00:00:00Z"]
    pv = params.resolve_init_times({"time_mode": "project_variable", "project_variable_name": "init"}, src)
    assert [to_rfc3339(t) for t in pv] == ["2023-01-01T06:00:00Z"]
    with pytest.raises(ConfigError, match="not defined"):
        params.resolve_init_times({"time_mode": "project_variable", "project_variable_name": "zzz"}, src)


def test_extract_options(params):
    o = params.extract_options({"mode": "region", "lat_min": 1, "lat_max": 2, "lon_min": -1, "lon_max": 1,
                                "variables": "u10m", "derived": ["wind_speed_10m"], "layout": "wide"})
    assert o.bbox.lon_min == -1 and o.variables == ["u10m"] and o.layout == "wide"
    with pytest.raises(ConfigError, match="lat_min"):
        params.extract_options({"mode": "region"})
    with pytest.raises(ConfigError, match="Unknown derived"):
        params.extract_options({"derived": ["x"]})


def test_sites_frame(params):
    df = pd.DataFrame({"id": [1], "lat": [10.0], "lon": [20.0]})
    out = params.sites_frame(df, {"site_id_column": "id", "lat_column": "lat", "lon_column": "lon"})
    assert list(out.columns) == ["site_id", "latitude", "longitude"] and out.site_id[0] == "1"
    with pytest.raises(ConfigError, match="not found"):
        params.sites_frame(df, {})
