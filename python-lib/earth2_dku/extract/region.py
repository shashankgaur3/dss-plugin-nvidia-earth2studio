"""Bounding-box extraction on the native grid."""
from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np
import pandas as pd

from ..errors import ConfigError
from .grid import Grid, to_display_lon, to_grid_lon
from .reader import extract_table

logger = logging.getLogger(__name__)

DEFAULT_MAX_CELLS = 50_000
_EPS = 1e-9


@dataclass(frozen=True)
class BBox:
    lat_min: float
    lat_max: float
    lon_min: float
    lon_max: float

    def __post_init__(self):
        if self.lat_min > self.lat_max:
            raise ConfigError(f"lat_min ({self.lat_min}) must be <= lat_max ({self.lat_max}).")
        if max(abs(self.lat_min), abs(self.lat_max)) > 90:
            raise ConfigError("Bounding box latitudes must be within [-90, 90].")


def region_indices(grid: Grid, bbox: BBox) -> tuple[np.ndarray, np.ndarray]:
    """Row and column indices inside the box. A box with lon_min > lon_max (after mapping
    to 0..360) crosses the 0 deg meridian, e.g. -10..10 -> 350..360 + 0..10."""
    lats = grid.lats
    rows = np.where((lats >= bbox.lat_min - _EPS) & (lats <= bbox.lat_max + _EPS))[0]
    lons = to_grid_lon(grid.lons)
    if bbox.lon_max - bbox.lon_min >= 360 - _EPS:
        cols = np.arange(grid.n_lon)
    else:
        a, b = float(to_grid_lon(bbox.lon_min)), float(to_grid_lon(bbox.lon_max))
        lo, hi = lons >= a - _EPS, lons <= b + _EPS
        cols = np.where(lo & hi if a <= b else lo | hi)[0]
        cols = cols[np.argsort(np.mod(lons[cols] - a, 360.0), kind="stable")]  # west -> east from lon_min
    if rows.size == 0 or cols.size == 0:
        raise ConfigError(
            f"The bounding box {bbox} contains no grid cell (grid step {abs(grid.lat_step)} deg). Enlarge it."
        )
    return rows, cols


def extract_region(manifest, storage, bbox: BBox, variables, max_cells: int = DEFAULT_MAX_CELLS) -> pd.DataFrame:
    """Wide DataFrame, one row per (step, grid cell). site_id is "lat_lon"; longitudes in -180..180."""
    grid = Grid.from_dict(manifest.grid)
    rows, cols = region_indices(grid, bbox)
    if max_cells and rows.size * cols.size > max_cells:
        logger.warning(
            "The bounding box covers %d grid cells per step (limit %d). Output will be large; "
            "consider a smaller box or point extraction.", rows.size * cols.size, max_cells,
        )
    ii, jj = (a.ravel() for a in np.meshgrid(rows, cols, indexing="ij"))
    lat, lon = grid.lats[ii], to_display_lon(grid.lons[jj])
    cells = pd.DataFrame({"site_id": [f"{a:.2f}_{b:.2f}" for a, b in zip(lat, lon)], "latitude": lat, "longitude": lon})
    return extract_table(manifest, storage, variables, cells, lambda f: f[ii, jj].astype("float64"))
