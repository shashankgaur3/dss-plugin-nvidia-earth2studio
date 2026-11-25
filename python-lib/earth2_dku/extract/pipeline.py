"""End-to-end extraction: select runs, extract, add derived variables, shape the output table."""
from __future__ import annotations

import logging
from dataclasses import dataclass, field

import pandas as pd

from ..errors import ConfigError, ManifestError
from ..forecast.manifest import list_manifests
from ..times import parse_time
from .derived import add_derived, required_inputs
from .points import extract_points
from .reader import KEY_COLUMNS, resolve_variables
from .region import DEFAULT_MAX_CELLS, BBox, extract_region

logger = logging.getLogger(__name__)

LONG_COLUMNS = [
    "run_init_time", "valid_time", "lead_hours", "site_id", "latitude", "longitude",
    "variable", "value", "model_id", "source",
]
_DSS_TYPES = {
    "run_init_time": "date", "valid_time": "date", "lead_hours": "bigint", "site_id": "string",
    "latitude": "double", "longitude": "double", "variable": "string", "value": "double",
    "model_id": "string", "source": "string",
}


@dataclass
class ExtractOptions:
    run_selection: str = "latest"  # latest | all | specific
    init_time: str | None = None
    mode: str = "points"  # points | region
    variables: list | None = None  # None = all variables in the manifest
    derived: list = field(default_factory=list)
    interpolation: str = "nearest"
    layout: str = "long"  # long | wide
    bbox: BBox | None = None
    max_cells: int = DEFAULT_MAX_CELLS


def select_runs(storage, selection: str = "latest", init_time=None) -> list:
    manifests = list_manifests(storage)
    usable = []
    for m in manifests:
        if m.status == "complete":
            usable.append(m)
        elif m.status == "partial" and m.steps:
            logger.warning("Run %s is partial (%d steps); extracting the steps received.", m.init_time, len(m.steps))
            usable.append(m)
        else:
            logger.warning("Skipping run %s with status '%s'.", m.init_time, m.status)
    if not usable:
        raise ManifestError("No usable forecast run found in the forecast folder. Run the 'Run forecast' recipe first.")
    if selection == "latest":
        return [usable[-1]]
    if selection == "all":
        return usable
    if selection == "specific":
        wanted = parse_time(init_time)
        match = [m for m in usable if m.init_datetime == wanted]
        if not match:
            raise ConfigError(
                f"No usable run for init time {wanted:%Y-%m-%dT%H:%M}Z. Available: "
                f"{', '.join(m.init_time for m in usable)}."
            )
        return match
    raise ConfigError(f"Unknown run selection '{selection}'. Use latest, all or specific.")


def _extract_one(manifest, storage, options: ExtractOptions, sites_df):
    requested = resolve_variables(manifest, options.variables)
    to_extract = list(requested)
    for r in required_inputs(options.derived):
        if r not in to_extract:
            to_extract.append(r)
    if options.mode == "points":
        if sites_df is None:
            raise ConfigError(
                "Point mode needs a 'sites' input dataset (site_id, latitude, longitude). "
                "Add one (e.g. the plugin's sample_sites.csv), or switch to region mode."
            )
        wide = extract_points(manifest, storage, sites_df, to_extract, options.interpolation)
    elif options.mode == "region":
        if options.bbox is None:
            raise ConfigError("Region mode needs lat_min, lat_max, lon_min and lon_max.")
        wide = extract_region(manifest, storage, options.bbox, to_extract, options.max_cells)
    else:
        raise ConfigError(f"Unknown mode '{options.mode}'. Use points or region.")
    wide = add_derived(wide, options.derived)
    value_cols = requested + [d for d in options.derived if d not in requested]
    return wide[KEY_COLUMNS + value_cols], value_cols


def to_long(wide: pd.DataFrame, value_cols) -> pd.DataFrame:
    long = wide.melt(id_vars=KEY_COLUMNS, value_vars=list(value_cols), var_name="variable", value_name="value")
    return long[LONG_COLUMNS]


def extract_forecast(storage, options: ExtractOptions, sites_df: pd.DataFrame | None = None) -> pd.DataFrame:
    if options.layout not in ("long", "wide"):
        raise ConfigError(f"Unknown layout '{options.layout}'. Use long or wide.")
    frames, all_cols = [], []
    for manifest in select_runs(storage, options.run_selection, options.init_time):
        wide, value_cols = _extract_one(manifest, storage, options, sites_df)
        frames.append(to_long(wide, value_cols) if options.layout == "long" else wide)
        all_cols += [c for c in value_cols if c not in all_cols]
    df = pd.concat(frames, ignore_index=True)
    if options.layout == "long":
        df = df.sort_values(["run_init_time", "site_id", "variable", "lead_hours"], kind="stable")
        df = df[LONG_COLUMNS]
    else:
        df = df.sort_values(["run_init_time", "site_id", "lead_hours"], kind="stable")
        df = df[KEY_COLUMNS + all_cols]
    return df.reset_index(drop=True)


def output_schema(df: pd.DataFrame) -> list[dict]:
    """Explicit DSS schema for the output dataset."""
    return [{"name": c, "type": _DSS_TYPES.get(c, "double")} for c in df.columns]
