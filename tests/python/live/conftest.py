"""Live integration tests against a real FourCastNet NIM and/or real ERA5/GFS data.

    EARTH2_NIM_URL=http://localhost:8000 pytest tests/python/live -m nim
    pytest tests/python/live -m network

Env vars:
    EARTH2_NIM_URL      base URL of the NIM (required for `nim` tests)
    EARTH2_MODEL_ID     registry model id served by the NIM (default fourcastnet-1.1.0)
    EARTH2_NIM_API_KEY  optional API key for a protected NIM endpoint
    EARTH2_NIM_API_KEY_HEADER  header carrying the key (default X-API-Key)
    EARTH2_INPUT_NPY    optional pre-fetched initial condition (.npy); otherwise ERA5 2023-01-01T00 is
                        fetched with Earth2Studio (needs network)
"""
import os
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "python-lib"))


def pytest_configure(config):
    config.addinivalue_line("markers", "nim: needs a running FourCastNet NIM (EARTH2_NIM_URL)")
    config.addinivalue_line("markers", "network: needs outbound access to ERA5 (ARCO) / GFS and Earth2Studio")


def pytest_collection_modifyitems(config, items):
    skip_nim = pytest.mark.skip(reason="set EARTH2_NIM_URL to run NIM integration tests")
    for item in items:
        if "nim" in item.keywords and not os.environ.get("EARTH2_NIM_URL"):
            item.add_marker(skip_nim)


@pytest.fixture(scope="session")
def model():
    from earth2_dku.models.registry import get_model

    return get_model(os.environ.get("EARTH2_MODEL_ID", "fourcastnet-1.1.0"))


@pytest.fixture(scope="session")
def client(model):
    from earth2_dku.clients.self_hosted import SelfHostedNimClient

    key = os.environ.get("EARTH2_NIM_API_KEY")
    auth = (os.environ.get("EARTH2_NIM_API_KEY_HEADER", "X-API-Key"), key) if key else None
    return SelfHostedNimClient(os.environ["EARTH2_NIM_URL"], model.supports_variables_param, auth_header=auth)


@pytest.fixture(scope="session")
def init_time():
    from earth2_dku.times import parse_time

    return parse_time("2023-01-01T00")


@pytest.fixture(scope="session")
def input_bytes(model, init_time, tmp_path_factory):
    path = os.environ.get("EARTH2_INPUT_NPY")
    if path:
        return Path(path).read_bytes()
    from earth2_dku.arrays import npy_bytes
    from earth2_dku.sources.base import validate_state_array
    from earth2_dku.sources.era5_arco import Era5ArcoSource

    arr = Era5ArcoSource().get_state(init_time, model.input_variables())
    validate_state_array(arr, model)
    return npy_bytes(arr)
