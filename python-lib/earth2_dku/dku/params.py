"""Translate DSS recipe/preset configs into core objects.

All Dataiku-specific config parsing lives here. ``dataiku`` is imported lazily so
the pure parsing helpers stay unit-testable.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass

from ..errors import ConfigError
from ..extract.pipeline import ExtractOptions
from ..extract.region import DEFAULT_MAX_CELLS, BBox
from ..models.registry import get_model
from ..sources import build_source as _build_source
from ..times import floor_to_cycle, parse_time

logger = logging.getLogger(__name__)

DEFAULT_TIMEOUT = 1200


@dataclass
class ConnectionConfig:
    base_url: str = ""
    model_id: str = "fourcastnet-1.1.0"
    timeout_seconds: int = DEFAULT_TIMEOUT


def _blank(v) -> bool:
    return v is None or (isinstance(v, str) and not v.strip())


def as_int(v, default=None):
    return default if _blank(v) else int(float(v))


def parse_list(value) -> list[str] | None:
    """'u10m, v10m' or ['u10m','v10m'] -> list; blank -> None."""
    if _blank(value):
        return None
    items = value if isinstance(value, (list, tuple)) else str(value).split(",")
    out = [str(v).strip() for v in items if str(v).strip()]
    return out or None


def connection_from_config(config: dict, key: str = "connection") -> ConnectionConfig:
    preset = (config or {}).get(key)
    if not isinstance(preset, dict) or not preset:
        raise ConfigError("Select an 'Earth-2 FourCastNet NIM connection' preset in the settings.")
    return ConnectionConfig(
        base_url=(preset.get("base_url") or "").strip(),
        model_id=preset.get("model_id") or "fourcastnet-1.1.0",
        timeout_seconds=as_int(preset.get("timeout_seconds"), DEFAULT_TIMEOUT),
    )


def build_model(conn: ConnectionConfig):
    return get_model(conn.model_id)


def build_client(conn: ConnectionConfig, model):
    from ..clients.self_hosted import SelfHostedNimClient

    return SelfHostedNimClient(
        conn.base_url, supports_variables_param=model.supports_variables_param, timeout_seconds=conn.timeout_seconds
    )


def build_source(config: dict):
    name = config.get("source") or "era5_arco"
    kwargs = {}
    if name == "gfs" and not _blank(config.get("gfs_safety_margin_hours")):
        kwargs["safety_margin_hours"] = float(config["gfs_safety_margin_hours"])
    return _build_source(name, **kwargs)


def optional_time(config: dict, key: str = "init_time"):
    value = (config or {}).get(key)
    return None if _blank(value) else parse_time(value)


def resolve_init_times(config: dict, source) -> list:
    """Init times to forecast, each rounded down to a 6-hour cycle."""
    dates_input = _role_names("dates", output=False)
    if dates_input:
        column = config.get("dates_column")
        if _blank(column):
            raise ConfigError("A 'dates' dataset is connected: choose its timestamp column in the settings.")
        import dataiku

        df = dataiku.Dataset(dates_input[0]).get_dataframe(columns=[column])
        times = sorted({floor_to_cycle(parse_time(v.to_pydatetime() if hasattr(v, "to_pydatetime") else v))
                        for v in df[column].dropna()})
        if not times:
            raise ConfigError(f"The 'dates' dataset has no values in column '{column}'.")
        return times
    mode = config.get("time_mode") or "fixed"
    if mode == "fixed":
        return [floor_to_cycle(parse_time(config.get("init_time")))]
    if mode == "latest_available":
        return [source.latest_available_time()]
    if mode == "project_variable":
        name = config.get("project_variable_name")
        if _blank(name):
            raise ConfigError("Set the project variable name that holds the init time.")
        import dataiku

        variables = dataiku.get_custom_variables()
        if name not in variables:
            raise ConfigError(f"Project variable '{name}' is not defined. Add it in the project variables.")
        return [floor_to_cycle(parse_time(variables[name]))]
    raise ConfigError(f"Unknown time mode '{mode}'. Use fixed, latest_available or project_variable.")


def _role_names(role: str, output: bool) -> list[str]:
    from dataiku.customrecipe import get_input_names_for_role, get_output_names_for_role

    return (get_output_names_for_role if output else get_input_names_for_role)(role)


def folder_storage(role: str, output: bool = False, required: bool = True):
    """Storage for the folder bound to ``role``; None if optional and not connected."""
    from ..storage.dku_folder import DkuFolderStorage

    names = _role_names(role, output)
    if not names:
        if not required:
            return None
        raise ConfigError(f"Connect a managed folder to the '{role}' role.")
    return DkuFolderStorage.from_name(names[0])


def read_sites(config: dict):
    """The optional 'sites' dataset, renamed to site_id / latitude / longitude. None if not connected."""
    names = _role_names("sites", output=False)
    if not names:
        return None
    import dataiku

    df = dataiku.Dataset(names[0]).get_dataframe()
    return sites_frame(df, config)


def sites_frame(df, config: dict):
    mapping = {
        "site_id": config.get("site_id_column") or "site_id",
        "latitude": config.get("lat_column") or "latitude",
        "longitude": config.get("lon_column") or "longitude",
    }
    missing = [src for src in mapping.values() if src not in df.columns]
    if missing:
        raise ConfigError(f"Columns {missing} not found in the sites dataset. Map the site columns in the settings.")
    out = df[[mapping["site_id"], mapping["latitude"], mapping["longitude"]]].copy()
    out.columns = ["site_id", "latitude", "longitude"]
    out["site_id"] = out["site_id"].astype(str)
    return out


def extract_options(config: dict) -> ExtractOptions:
    mode = config.get("mode") or "points"
    bbox = None
    if mode == "region":
        values = [config.get(k) for k in ("lat_min", "lat_max", "lon_min", "lon_max")]
        if any(_blank(v) for v in values):
            raise ConfigError("Region mode needs lat_min, lat_max, lon_min and lon_max.")
        bbox = BBox(*[float(v) for v in values])
    return ExtractOptions(
        run_selection=config.get("run_selection") or "latest",
        init_time=config.get("init_time") or None,
        mode=mode,
        variables=parse_list(config.get("variables")),
        derived=list(config.get("derived") or []),
        interpolation=config.get("interpolation") or "nearest",
        layout=config.get("layout") or "long",
        bbox=bbox,
        max_cells=as_int(config.get("max_cells"), DEFAULT_MAX_CELLS),
    )
