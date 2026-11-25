"""Point (site) extraction with nearest or bilinear interpolation."""
from __future__ import annotations

import numpy as np
import pandas as pd

from ..errors import ConfigError
from .grid import Grid, validate_lat_lon
from .reader import extract_table

METHODS = ("nearest", "bilinear")


def make_sampler(grid: Grid, lats, lons, method: str = "nearest"):
    """Return ``sample(field2d) -> values at (lats, lons)``; indices/weights are computed once."""
    if method == "nearest":
        i, j = grid.nearest_index(lats, lons)
        return lambda f: f[i, j].astype("float64")
    if method == "bilinear":
        fi = grid.lat_index_float(lats)
        i0 = np.clip(np.floor(fi), 0, grid.n_lat - 2).astype(int)
        i1, di = i0 + 1, np.clip(fi - i0, 0.0, 1.0)
        fj = grid.lon_index_float(lons)
        j0 = np.mod(np.floor(fj), grid.n_lon).astype(int)
        j1, dj = np.mod(j0 + 1, grid.n_lon), fj - np.floor(fj)  # j1 wraps 359.75 -> 0
        return lambda f: ((1 - di) * (1 - dj) * f[i0, j0] + (1 - di) * dj * f[i0, j1]
                          + di * (1 - dj) * f[i1, j0] + di * dj * f[i1, j1])
    raise ConfigError(f"Unknown interpolation '{method}'. Use one of {METHODS}.")


def extract_points(manifest, storage, sites_df: pd.DataFrame, variables, method: str = "nearest") -> pd.DataFrame:
    """Wide DataFrame, one row per (step, site). ``sites_df`` needs site_id, latitude, longitude."""
    missing = [c for c in ("site_id", "latitude", "longitude") if c not in sites_df.columns]
    if missing:
        raise ConfigError(f"The sites table needs columns {missing}. Map them in the recipe settings.")
    if sites_df.empty:
        raise ConfigError("The sites dataset is empty. Add at least one site.")
    cells = pd.DataFrame({
        "site_id": sites_df["site_id"].astype(str).to_numpy(),
        "latitude": sites_df["latitude"].to_numpy(dtype="float64"),
        "longitude": sites_df["longitude"].to_numpy(dtype="float64"),
    })
    validate_lat_lon(cells["latitude"], cells["longitude"])
    sampler = make_sampler(Grid.from_dict(manifest.grid), cells["latitude"], cells["longitude"], method)
    return extract_table(manifest, storage, variables, cells, sampler)
