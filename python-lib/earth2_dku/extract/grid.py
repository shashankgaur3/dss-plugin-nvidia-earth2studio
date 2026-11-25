"""Lat/lon grid helpers. ERA5 0.25 deg: lat 90 -> -90 (721), lon 0 -> 359.75 (1440)."""
from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np

from ..errors import ConfigError


@dataclass(frozen=True)
class Grid:
    name: str
    n_lat: int
    n_lon: int
    lat_start: float
    lat_step: float
    lon_start: float
    lon_step: float

    @classmethod
    def from_dict(cls, d: dict) -> "Grid":
        return cls(
            name=str(d["name"]),
            n_lat=int(d["n_lat"]),
            n_lon=int(d["n_lon"]),
            lat_start=float(d["lat_start"]),
            lat_step=float(d["lat_step"]),
            lon_start=float(d["lon_start"]),
            lon_step=float(d["lon_step"]),
        )

    def to_dict(self) -> dict:
        return asdict(self)

    @property
    def lats(self) -> np.ndarray:
        return self.lat_start + self.lat_step * np.arange(self.n_lat)

    @property
    def lons(self) -> np.ndarray:
        """Grid longitudes in the grid's own convention (0..360 for ERA5)."""
        return self.lon_start + self.lon_step * np.arange(self.n_lon)

    def lat_index_float(self, lat):
        return (np.asarray(lat, dtype="float64") - self.lat_start) / self.lat_step

    def lon_index_float(self, lon):
        """Fractional column index in [0, n_lon), with longitude wraparound."""
        offset = np.mod(to_grid_lon(lon) - self.lon_start, 360.0)
        return np.mod(offset / self.lon_step, self.n_lon)

    def nearest_index(self, lat, lon):
        i = np.clip(np.rint(self.lat_index_float(lat)), 0, self.n_lat - 1).astype(int)
        j = np.mod(np.rint(self.lon_index_float(lon)), self.n_lon).astype(int)
        return i, j


ERA5_0P25 = Grid("era5_0p25", 721, 1440, 90.0, -0.25, 0.0, 0.25)
GRIDS = {ERA5_0P25.name: ERA5_0P25}


def get_grid(name: str) -> Grid:
    if name not in GRIDS:
        raise ConfigError(f"Unknown grid '{name}'. Known grids: {', '.join(GRIDS)}. Fix models.yaml.")
    return GRIDS[name]


def to_grid_lon(lon):
    """-180..180 (or any) longitude -> 0..360."""
    return np.mod(np.asarray(lon, dtype="float64"), 360.0)


def to_display_lon(lon):
    """0..360 longitude -> -180..180."""
    return np.mod(np.asarray(lon, dtype="float64") + 180.0, 360.0) - 180.0


def validate_lat_lon(lat, lon, what="site"):
    lat = np.asarray(lat, dtype="float64")
    lon = np.asarray(lon, dtype="float64")
    if np.isnan(lat).any() or np.isnan(lon).any():
        raise ConfigError(f"Some {what} latitudes/longitudes are empty. Fill or remove those rows.")
    if (np.abs(lat) > 90).any():
        raise ConfigError(f"{what.capitalize()} latitudes must be within [-90, 90].")
    if ((lon < -180) | (lon > 360)).any():
        raise ConfigError(f"{what.capitalize()} longitudes must be within [-180, 360].")
