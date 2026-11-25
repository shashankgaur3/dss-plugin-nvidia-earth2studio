import logging

import numpy as np
import pandas as pd
import pytest

from earth2_dku.arrays import npy_bytes
from earth2_dku.errors import ConfigError, ManifestError
from earth2_dku.extract.derived import add_derived, available_derived, required_inputs
from earth2_dku.extract.grid import ERA5_0P25, get_grid, to_display_lon, to_grid_lon
from earth2_dku.extract.pipeline import LONG_COLUMNS, ExtractOptions, extract_forecast, output_schema, select_runs
from earth2_dku.extract.points import extract_points
from earth2_dku.extract.region import BBox, extract_region, region_indices
from fakes import write_synthetic_run

SITES = pd.DataFrame({
    "site_id": ["exact", "between", "west", "wrap_neg", "wrap_pos", "north_pole"],
    "latitude": [50.0, 50.1, 32.45, 10.0, 10.0, 90.0],
    "longitude": [10.0, 10.1, -101.85, -0.1, 359.9, 0.0],
})


# --------------------------------------------------------------------------- grid
def test_grid_shape_and_conventions():
    g = get_grid("era5_0p25")
    assert g.lats[0] == 90.0 and g.lats[-1] == -90.0 and len(g.lats) == 721
    assert g.lons[0] == 0.0 and g.lons[-1] == 359.75 and len(g.lons) == 1440
    np.testing.assert_allclose(to_grid_lon([-180, -0.25, 0, 179.75]), [180, 359.75, 0, 179.75])
    np.testing.assert_allclose(to_display_lon([0, 180, 359.75]), [0, -180, -0.25])
    assert tuple(map(int, g.nearest_index(-90.0, -0.1))) == (720, 0)
    assert tuple(map(int, g.nearest_index(0.0, -0.25))) == (360, 1439)
    with pytest.raises(ConfigError):
        get_grid("nope")


# ------------------------------------------------------------------------- points
def test_nearest_points(storage):
    m = write_synthetic_run(storage)
    df = extract_points(m, storage, SITES, ["lat", "lon", "lonc"], "nearest")
    first = df[df.lead_hours == 0].set_index("site_id")
    assert len(df) == len(SITES) * 2
    assert first.loc["exact", "lat"] == 50.0 and first.loc["exact", "lon"] == 10.0
    assert first.loc["between", "lat"] == 50.0 and first.loc["between", "lon"] == 10.0
    assert first.loc["west", "lat"] == 32.5 and first.loc["west", "lon"] == pytest.approx(258.25)
    assert first.loc["wrap_neg", "lon"] == 0.0 and first.loc["wrap_pos", "lon"] == 0.0
    # original site coordinates are kept
    assert first.loc["wrap_neg", "longitude"] == -0.1


def test_bilinear_points_are_exact_on_linear_fields(storage):
    m = write_synthetic_run(storage)
    df = extract_points(m, storage, SITES, ["lat", "lon", "lonc"], "bilinear").set_index("site_id")
    df = df[df.lead_hours == 0]
    assert df.loc["between", "lat"] == pytest.approx(50.1)
    assert df.loc["between", "lon"] == pytest.approx(10.1)
    assert df.loc["west", "lat"] == pytest.approx(32.45)
    assert df.loc["west", "lon"] == pytest.approx(258.15)
    assert df.loc["north_pole", "lat"] == pytest.approx(90.0)
    # wraparound 359.75 -> 0: (lon+180)%360 is continuous across 0 deg, so bilinear must be exact
    assert df.loc["wrap_neg", "lonc"] == pytest.approx(179.9)
    assert df.loc["wrap_pos", "lonc"] == pytest.approx(179.9)


def test_points_keys(storage):
    m = write_synthetic_run(storage, variables=("lat", "lead"), leads=(0, 6, 12))
    df = extract_points(m, storage, SITES.head(1), None, "nearest")
    assert list(df.lead_hours) == [0, 6, 12]
    assert list(df["lead"]) == [0.0, 6.0, 12.0]
    assert (df.valid_time - df.run_init_time).dt.total_seconds().tolist() == [0, 21600, 43200]
    assert str(df.run_init_time.dt.tz) == "UTC"
    assert set(df.model_id) == {"fake-model"} and set(df.source) == {"era5_arco"}


def test_points_errors(storage):
    m = write_synthetic_run(storage, leads=(0,))
    with pytest.raises(ConfigError, match="not in the forecast run"):
        extract_points(m, storage, SITES, ["u10m"])
    with pytest.raises(ConfigError, match="latitude"):
        extract_points(m, storage, SITES.drop(columns="latitude"), ["lat"])
    bad = SITES.copy()
    bad.loc[0, "latitude"] = 95
    with pytest.raises(ConfigError, match="within"):
        extract_points(m, storage, bad, ["lat"])


def test_field_count_mismatch_is_manifest_error(storage):
    m = write_synthetic_run(storage, leads=(0,))
    m.output_variables = ["lat", "lon"]  # but the step holds 3 fields
    with pytest.raises(ManifestError, match="holds 3 fields"):
        extract_points(m, storage, SITES, ["lat"])


def test_missing_step_file(storage):
    m = write_synthetic_run(storage, leads=(0,))
    m.steps[0].file = "steps/999_000.npy"
    with pytest.raises(ManifestError, match="missing"):
        extract_points(m, storage, SITES, ["lat"])


def test_multi_dim_leading_shape_is_reshaped(storage):
    """Variable order comes from the manifest; leading dims are flattened, not hardcoded."""
    m = write_synthetic_run(storage, variables=("lat", "lon"), leads=(0,))
    from fakes import encoded_fields

    arr = encoded_fields(["lat", "lon"])[0]  # (1, 2, 721, 1440): drop the batch dim
    storage.write_bytes(f"{m.location}/{m.steps[0].file}", npy_bytes(arr))
    df = extract_points(m, storage, SITES.head(1), ["lon", "lat"])
    assert df.loc[0, "lat"] == 50.0 and df.loc[0, "lon"] == 10.0


# ------------------------------------------------------------------------- region
def test_region_simple():
    rows, cols = region_indices(ERA5_0P25, BBox(50, 51, 10, 11))
    assert len(rows) == 5 and len(cols) == 5


def test_region_crossing_zero_meridian(storage):
    m = write_synthetic_run(storage, leads=(0,))
    df = extract_region(m, storage, BBox(50, 51, -1, 1), ["lat", "lon"])
    assert len(df) == 5 * 9
    assert sorted(set(df.longitude)) == [-1, -0.75, -0.5, -0.25, 0, 0.25, 0.5, 0.75, 1]
    np.testing.assert_allclose(df.lon, to_grid_lon(df.longitude))
    np.testing.assert_allclose(df.lat, df.latitude)
    assert df.site_id.iloc[0] == "51.00_-1.00"


def test_region_crossing_antimeridian():
    rows, cols = region_indices(ERA5_0P25, BBox(0, 0, 179.5, -179.5))
    np.testing.assert_allclose(to_display_lon(ERA5_0P25.lons[cols]), [179.5, 179.75, -180, -179.75, -179.5])


def test_region_full_globe():
    rows, cols = region_indices(ERA5_0P25, BBox(-90, 90, -180, 180))
    assert len(rows) == 721 and len(cols) == 1440


def test_region_errors_and_warning(storage, caplog):
    with pytest.raises(ConfigError):
        BBox(10, 0, 0, 1)
    with pytest.raises(ConfigError, match="no grid cell"):
        region_indices(ERA5_0P25, BBox(50.1, 50.2, 10.1, 10.2))
    m = write_synthetic_run(storage, leads=(0,))
    with caplog.at_level(logging.WARNING):
        extract_region(m, storage, BBox(50, 51, 10, 11), ["lat"], max_cells=10)
    assert "25 grid cells" in caplog.text


# ------------------------------------------------------------------------ derived
def test_derived_registry():
    assert available_derived() == ["wind_speed_10m", "wind_speed_100m", "t2m_celsius"]
    assert required_inputs(["wind_speed_10m", "t2m_celsius"]) == ["u10m", "v10m", "t2m"]
    df = add_derived(pd.DataFrame({"u10m": [3.0], "v10m": [4.0], "t2m": [273.15]}), ["wind_speed_10m", "t2m_celsius"])
    assert df.wind_speed_10m[0] == 5.0 and df.t2m_celsius[0] == pytest.approx(0.0)
    with pytest.raises(ConfigError, match="wind_speed_10m"):
        required_inputs(["nope"])
    with pytest.raises(ConfigError, match="needs"):
        add_derived(pd.DataFrame({"u10m": [1.0]}), ["wind_speed_10m"])


# ----------------------------------------------------------------------- pipeline
def test_pipeline_long_with_derived(storage):
    write_synthetic_run(storage, variables=("u10m", "v10m"), leads=(0, 6))
    opts = ExtractOptions(variables=None, derived=["wind_speed_10m"])
    df = extract_forecast(storage, opts, SITES.head(2))
    assert list(df.columns) == LONG_COLUMNS
    assert set(df.variable) == {"u10m", "v10m", "wind_speed_10m"}
    assert len(df) == 2 * 2 * 3
    w = df.pivot_table(index=["site_id", "lead_hours"], columns="variable", values="value")
    np.testing.assert_allclose(w.wind_speed_10m, np.hypot(w.u10m, w.v10m))
    assert [c["type"] for c in output_schema(df)][:3] == ["date", "date", "bigint"]


def test_pipeline_derived_only_adds_requirements_silently(storage):
    write_synthetic_run(storage, variables=("u10m", "v10m", "t2m"), leads=(0,))
    df = extract_forecast(storage, ExtractOptions(variables=["t2m"], derived=["wind_speed_10m"], layout="wide"),
                          SITES.head(1))
    assert list(df.columns)[-2:] == ["t2m", "wind_speed_10m"]
    assert "u10m" not in df.columns


def test_pipeline_run_selection(storage):
    write_synthetic_run(storage, "2023-01-01T00", variables=("lat",), leads=(0,))
    write_synthetic_run(storage, "2023-01-02T00", variables=("lat",), leads=(0,))
    write_synthetic_run(storage, "2023-01-03T00", variables=("lat",), leads=(0,), status="failed")
    assert [m.init_time for m in select_runs(storage, "latest")] == ["2023-01-02T00:00:00Z"]
    assert len(select_runs(storage, "all")) == 2
    assert select_runs(storage, "specific", "2023-01-01T00")[0].init_time == "2023-01-01T00:00:00Z"
    with pytest.raises(ConfigError, match="Available"):
        select_runs(storage, "specific", "2023-01-03T00")
    df = extract_forecast(storage, ExtractOptions(run_selection="all"), SITES.head(1))
    assert df.run_init_time.nunique() == 2


def test_pipeline_region_mode(storage):
    write_synthetic_run(storage, variables=("lat",), leads=(0,))
    df = extract_forecast(storage, ExtractOptions(mode="region", bbox=BBox(50, 50.5, 0, 0.5)))
    assert len(df) == 3 * 3


def test_pipeline_errors(storage):
    with pytest.raises(ManifestError, match="No usable forecast run"):
        extract_forecast(storage, ExtractOptions(), SITES)
    write_synthetic_run(storage, variables=("lat",), leads=(0,))
    with pytest.raises(ConfigError, match="sites"):
        extract_forecast(storage, ExtractOptions(), None)
    with pytest.raises(ConfigError, match="lat_min"):
        extract_forecast(storage, ExtractOptions(mode="region"))
    with pytest.raises(ConfigError, match="layout"):
        extract_forecast(storage, ExtractOptions(layout="tall"), SITES)
    with pytest.raises(ConfigError, match="Unknown derived"):
        extract_forecast(storage, ExtractOptions(derived=["x"]), SITES)
    with pytest.raises(ConfigError, match="interpolation"):
        extract_forecast(storage, ExtractOptions(interpolation="cubic"), SITES)
