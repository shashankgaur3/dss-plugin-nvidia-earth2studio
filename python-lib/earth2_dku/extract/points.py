"""Point (site) extraction with nearest or bilinear interpolation."""
from __future__ import annotations

import numpy as np
import pandas as pd

from ..errors import ConfigError
from .grid import Grid, validate_lat_lon
from .reader import KEY_COLUMNS, iter_step_fields, resolve_variables, step_keys

METHODS = ("nearest", "bilinear")


def bilinear_indices(grid: Grid, lats, lons):
    fi = grid.lat_index_float(lats)
    i0 = np.clip(np.floor(fi), 0, grid.n_lat - 2).astype(int)
    di = np.clip(fi - i0, 0.0, 1.0)
    fj = grid.lon_index_float(lons)
    j0f = np.floor(fj)
    dj = fj - j0f
    j0 = np.mod(j0f, grid.n_lon).astype(int)
    j1 = np.mod(j0 + 1, grid.n_lon)  # wraps 359.75 -> 0
    return i0, i0 + 1, j0, j1, di, dj


def sample(field: np.ndarray, grid: Grid, lats, lons, method: str = "nearest") -> np.ndarray:
    if method == "nearest":
        i, j = grid.nearest_index(lats, lons)
        return field[i, j].astype("float64")
    if method == "bilinear":
        i0, i1, j0, j1, di, dj = bilinear_indices(grid, lats, lons)
        f = field.astype("float64", copy=False)
        return (
            (1 - di) * (1 - dj) * f[i0, j0]
            + (1 - di) * dj * f[i0, j1]
            + di * (1 - dj) * f[i1, j0]
            + di * dj * f[i1, j1]
        )
    raise ConfigError(f"Unknown interpolation '{method}'. Use one of {METHODS}.")


def extract_points(manifest, storage, sites_df: pd.DataFrame, variables, method: str = "nearest") -> pd.DataFrame:
    """Wide DataFrame: KEY_COLUMNS + one column per variable, one row per (step, site).

    ``sites_df`` needs columns site_id, latitude, longitude (longitude in -180..180 or 0..360).
    """
    for col in ("site_id", "latitude", "longitude"):
        if col not in sites_df.columns:
            raise ConfigError(f"The sites table needs a '{col}' column. Map it in the recipe settings.")
    if sites_df.empty:
        raise ConfigError("The sites dataset is empty. Add at least one site.")
    variables = resolve_variables(manifest, variables)
    grid = Grid.from_dict(manifest.grid)
    lats = sites_df["latitude"].to_numpy(dtype="float64")
    lons = sites_df["longitude"].to_numpy(dtype="float64")
    validate_lat_lon(lats, lons)
    site_ids = sites_df["site_id"].astype(str).to_numpy()

    frames = []
    for step, fields in iter_step_fields(manifest, storage, variables):
        df = pd.DataFrame({"site_id": site_ids, "latitude": lats, "longitude": lons})
        for k, v in step_keys(manifest, step).items():
            df[k] = v
        for var in variables:
            df[var] = sample(fields[var], grid, lats, lons, method)
        frames.append(df)
    if not frames:
        return pd.DataFrame(columns=KEY_COLUMNS + variables)
    return pd.concat(frames, ignore_index=True)[KEY_COLUMNS + variables]
