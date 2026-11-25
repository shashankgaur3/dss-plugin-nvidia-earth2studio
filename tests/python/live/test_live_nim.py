import numpy as np
import pandas as pd
import pytest

from earth2_dku.arrays import load_npy
from earth2_dku.storage.local import LocalStorage

pytestmark = pytest.mark.nim


def _fields(client, input_bytes, init_time, variables):
    steps = list(client.infer_stream(input_bytes, init_time, 1, variables))
    return steps, client.last_variables_sent


def test_health(client):
    status = client.health()
    assert status.ready, status.detail


def test_simulation_length_1_returns_2_steps(client, input_bytes, init_time, model):
    steps, sent = _fields(client, input_bytes, init_time, ["u10m", "v10m"])
    assert [s.lead_hours for s in steps] == [0, 6]
    expected_fields = 2 if sent else model.n_channels
    for s in steps:
        arr = load_npy(s.raw_bytes)
        assert arr.shape[-2:] == (721, 1440)
        assert int(np.prod(arr.shape[:-2])) == expected_fields


def test_variable_order_follows_request(client, input_bytes, init_time, model):
    """Record the outcome in models.yaml (supports_variables_param) and in the README."""
    steps_uv, sent_uv = _fields(client, input_bytes, init_time, ["u10m", "v10m"])
    steps_vu, sent_vu = _fields(client, input_bytes, init_time, ["v10m", "u10m"])
    if sent_uv is None or sent_vu is None:
        pytest.fail(
            f"The NIM for {model.model_id} rejected the `variables` parameter: set "
            "supports_variables_param: false in models.yaml."
        )
    uv = load_npy(steps_uv[1].raw_bytes).reshape(-1, 721, 1440)
    vu = load_npy(steps_vu[1].raw_bytes).reshape(-1, 721, 1440)
    assert uv.shape[0] == vu.shape[0] == 2, "the NIM ignored `variables` (returned all channels)"
    np.testing.assert_allclose(uv[0], vu[1], rtol=1e-5, atol=1e-5)
    np.testing.assert_allclose(uv[1], vu[0], rtol=1e-5, atol=1e-5)
    assert not np.allclose(uv[0], uv[1])


def test_point_extraction_is_plausible(client, input_bytes, init_time, model, tmp_path):
    from pathlib import Path

    from earth2_dku.extract.points import extract_points
    from earth2_dku.forecast.manifest import run_dir_for
    from earth2_dku.forecast.runner import run_forecast

    storage = LocalStorage(tmp_path)
    manifest = run_forecast(client, model, storage, run_dir_for(init_time), input_bytes, init_time, 2,
                            ["u10m", "v10m"], source="era5_arco")
    assert manifest.status == "complete" and manifest.steps_received == 3
    sites = pd.read_csv(Path(__file__).resolve().parents[3] / "resource/sample_sites.csv")
    df = extract_points(manifest, storage, sites, ["u10m", "v10m"], "bilinear")
    assert len(df) == 3 * len(sites)
    assert (df[["u10m", "v10m"]].abs() < 100).all().all()
