import pytest

from earth2_dku.sources.base import validate_state_array
from earth2_dku.sources.era5_arco import Era5ArcoSource
from earth2_dku.sources.gfs import GfsSource

pytestmark = pytest.mark.network


def test_era5_state_shape(model, init_time):
    arr = Era5ArcoSource().get_state(init_time, model.input_variables())
    validate_state_array(arr, model)


def test_gfs_latest_state_shape(model):
    src = GfsSource()
    arr = src.get_state(src.latest_available_time(), model.input_variables())
    validate_state_array(arr, model)
