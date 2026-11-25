"""Bounding-box extraction on the native grid."""
from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np
import pandas as pd

from ..errors import ConfigError
from .grid import Grid, to_display_lon, to_grid_lon
from .reader import KEY_COLUMNS, iter_step_fields, resolve_variables, step_keys

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
        for v in (self.lat_min, self.lat_max):
            if abs(v) > 90:
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
        if a <= b:
            mask = (lons >= a - _EPS) & (lons <= b + _EPS)
        else:
            mask = (lons >= a - _EPS) | (lons <= b + _EPS)
        cols = np.where(mask)[0]
        cols = cols[np.argsort(np.mod(lons[cols] - a, 360.0), kind="stable")]  # west -> east from lon_min
    if rows.size == 0 or cols.size == 0:
        raise ConfigError(
            f"The bounding box {bbox} contains no grid cell (grid step {abs(grid.lat_step)} deg). Enlarge it."
        )
    return rows, cols


def extract_region(manifest, storage, bbox: BBox, variables, max_cells: int = DEFAULT_MAX_CELLS) -> pd.DataFrame:
    """Wide DataFrame (KEY_COLUMNS + variables), one row per (step, grid cell).

    site_id is a cell identifier "lat_lon"; longitudes are reported in -180..180.
    """
    variables = resolve_variables(manifest, variables)
    grid = Grid.from_dict(manifest.grid)
    rows, cols = region_indices(grid, bbox)
    n_cells = rows.size * cols.size
    if max_cells and n_cells > max_cells:
        logger.warning(
            "The bounding box covers %d grid cells per step (limit %d). Output will be large; "
            "consider a smaller box or point extraction.", n_cells, max_cells,
        )
    ii, jj = np.meshgrid(rows, cols, indexing="ij")
    ii, jj = ii.ravel(), jj.ravel()
    cell_lat = grid.lats[ii]
    cell_lon = to_display_lon(grid.lons[jj])
    cell_ids = np.array([f"{la:.2f}_{lo:.2f}" for la, lo in zip(cell_lat, cell_lon)])

    frames = []
    for step, fields in iter_step_fields(manifest, storage, variables):
        df = pd.DataFrame({"site_id": cell_ids, "latitude": cell_lat, "longitude": cell_lon})
        for k, v in step_keys(manifest, step).items():
            df[k] = v
        for var in variables:
            df[var] = fields[var][ii, jj].astype("float64")
        frames.append(df)
    if not frames:
        return pd.DataFrame(columns=KEY_COLUMNS + variables)
    return pd.concat(frames, ignore_index=True)[KEY_COLUMNS + variables]
