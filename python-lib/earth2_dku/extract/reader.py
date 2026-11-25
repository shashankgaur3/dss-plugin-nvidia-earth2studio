"""Read forecast step arrays using only what the manifest says (no model assumptions)."""
from __future__ import annotations

import posixpath
from datetime import timedelta
from typing import Iterator

import numpy as np
import pandas as pd

from ..arrays import load_npy
from ..errors import ConfigError, ManifestError
from .grid import Grid

KEY_COLUMNS = ["run_init_time", "valid_time", "lead_hours", "site_id", "latitude", "longitude", "model_id", "source"]


def resolve_variables(manifest, variables) -> list[str]:
    available = list(manifest.output_variables)
    if not variables:
        return available
    unknown = [v for v in variables if v not in available]
    if unknown:
        raise ConfigError(
            f"Variables {unknown} are not in the forecast run {manifest.init_time} "
            f"(available: {', '.join(available)}). Change the variable list, or re-run the forecast "
            "requesting these variables."
        )
    return list(variables)


def iter_step_fields(manifest, storage, variables) -> Iterator[tuple[object, dict]]:
    """Yield (StepRecord, {variable: 2D field}) for every step listed in the manifest."""
    grid = Grid.from_dict(manifest.grid)
    n_vars = len(manifest.output_variables)
    index = {v: manifest.output_variables.index(v) for v in variables}
    for step in sorted(manifest.steps, key=lambda s: (s.lead_hours, s.batch_idx)):
        path = posixpath.join(manifest.location, step.file)
        if not storage.exists(path):
            raise ManifestError(f"Step file {path} listed in the manifest is missing. Re-run the forecast.")
        arr = load_npy(storage.read_bytes(path))
        if arr.ndim < 2 or tuple(arr.shape[-2:]) != (grid.n_lat, grid.n_lon):
            raise ManifestError(
                f"Step {path} has shape {arr.shape}; expected trailing dims ({grid.n_lat}, {grid.n_lon}). "
                "Re-run the forecast."
            )
        n_fields = int(np.prod(arr.shape[:-2]))
        if n_fields != n_vars:
            raise ManifestError(
                f"Step {path} holds {n_fields} fields (shape {arr.shape}) but the manifest lists "
                f"{n_vars} output variables. The NIM did not honour the requested variables; re-run "
                "the forecast and check supports_variables_param in models.yaml."
            )
        flat = arr.reshape(-1, grid.n_lat, grid.n_lon)
        yield step, {v: flat[i] for v, i in index.items()}


def step_keys(manifest, step) -> dict:
    init = manifest.init_datetime
    return {
        "run_init_time": init,
        "valid_time": init + timedelta(hours=step.lead_hours),
        "lead_hours": int(step.lead_hours),
        "model_id": manifest.model_id,
        "source": manifest.source,
    }


def extract_table(manifest, storage, variables, cells: pd.DataFrame, sample) -> pd.DataFrame:
    """Wide table: one row per (step, cell), KEY_COLUMNS + one column per variable.

    ``cells`` has site_id/latitude/longitude; ``sample(field2d)`` returns one value per cell.
    """
    variables = resolve_variables(manifest, variables)
    frames = [
        cells.assign(**step_keys(manifest, step), **{v: sample(fields[v]) for v in variables})
        for step, fields in iter_step_fields(manifest, storage, variables)
    ]
    if not frames:
        return pd.DataFrame(columns=KEY_COLUMNS + variables)
    return pd.concat(frames, ignore_index=True)[KEY_COLUMNS + variables]
