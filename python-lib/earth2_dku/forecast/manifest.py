"""Run manifest: the only contract between the run-forecast and extract recipes (spec section 6)."""
from __future__ import annotations

import json
import logging
import posixpath
from dataclasses import dataclass, field
from datetime import datetime

from .. import __version__
from ..errors import ManifestError
from ..times import parse_time, to_rfc3339, utcnow

logger = logging.getLogger(__name__)

SCHEMA_VERSION = 1
MANIFEST_FILE = "manifest.json"
RUNS_PREFIX = "runs/"
STATUSES = ("running", "complete", "partial", "failed")
GRID_KEYS = ("name", "n_lat", "n_lon", "lat_start", "lat_step", "lon_start", "lon_step")
_REQUIRED = (
    "schema_version", "model_id", "nim_image", "source", "init_time", "step_hours",
    "simulation_length", "output_variables", "grid", "steps", "status", "steps_received",
    "created_at", "plugin_version",
)


def run_dir_for(init_time) -> str:
    return f"runs/init={parse_time(init_time):%Y-%m-%dT%H}"


def manifest_path(run_dir: str) -> str:
    return posixpath.join(run_dir, MANIFEST_FILE)


@dataclass
class StepRecord:
    file: str
    lead_hours: int
    batch_idx: int = 0

    def to_dict(self) -> dict:
        return {"file": self.file, "lead_hours": self.lead_hours, "batch_idx": self.batch_idx}


@dataclass
class Manifest:
    model_id: str
    nim_image: str
    source: str
    init_time: str
    step_hours: int
    simulation_length: int
    output_variables: list
    grid: dict
    steps: list = field(default_factory=list)
    status: str = "running"
    steps_received: int = 0
    created_at: str = ""
    plugin_version: str = __version__
    schema_version: int = SCHEMA_VERSION
    message: str | None = None
    # Where the manifest lives in its storage. Not serialized.
    run_dir: str = field(default="", compare=False, repr=False)

    @property
    def init_datetime(self) -> datetime:
        return parse_time(self.init_time)

    @property
    def location(self) -> str:
        return self.run_dir or run_dir_for(self.init_time)

    @classmethod
    def new(cls, model, source, init_time, simulation_length, output_variables, grid, now=None):
        return cls(
            model_id=model.model_id,
            nim_image=model.nim_image,
            source=source,
            init_time=to_rfc3339(init_time),
            step_hours=model.step_hours,
            simulation_length=int(simulation_length),
            output_variables=list(output_variables),
            grid=dict(grid),
            created_at=to_rfc3339(now or utcnow()),
        )

    def to_dict(self) -> dict:
        d = {
            "schema_version": self.schema_version,
            "model_id": self.model_id,
            "nim_image": self.nim_image,
            "source": self.source,
            "init_time": self.init_time,
            "step_hours": self.step_hours,
            "simulation_length": self.simulation_length,
            "output_variables": list(self.output_variables),
            "grid": dict(self.grid),
            "steps": [s.to_dict() for s in self.steps],
            "status": self.status,
            "steps_received": self.steps_received,
            "created_at": self.created_at,
            "plugin_version": self.plugin_version,
        }
        if self.message:
            d["message"] = self.message
        return d

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), indent=2)

    @classmethod
    def from_dict(cls, d: dict, run_dir: str = "") -> "Manifest":
        where = f" at {run_dir}" if run_dir else ""
        hint = " Re-run the 'Run forecast' recipe for this init time."

        def bad(msg):
            return ManifestError(f"Invalid manifest{where}: {msg}.{hint}")

        if not isinstance(d, dict):
            raise bad("not a JSON object")
        missing = [k for k in _REQUIRED if k not in d]
        if missing:
            raise bad(f"missing keys {missing}")
        if d["schema_version"] != SCHEMA_VERSION:
            raise ManifestError(
                f"Manifest{where} has schema_version {d['schema_version']}, this plugin reads "
                f"{SCHEMA_VERSION}. Update the plugin or re-run the forecast."
            )
        if d["status"] not in STATUSES:
            raise bad(f"status '{d['status']}' not in {STATUSES}")
        vars_ = d["output_variables"]
        if not isinstance(vars_, list) or not all(isinstance(v, str) for v in vars_):
            raise bad("output_variables must be a list of strings")
        grid = d["grid"]
        if not isinstance(grid, dict) or any(k not in grid for k in GRID_KEYS):
            raise bad(f"grid must contain {GRID_KEYS}")
        try:
            parse_time(d["init_time"])
            steps = [
                StepRecord(file=str(s["file"]), lead_hours=int(s["lead_hours"]), batch_idx=int(s.get("batch_idx", 0)))
                for s in d["steps"]
            ]
            return cls(
                model_id=str(d["model_id"]),
                nim_image=str(d["nim_image"]),
                source=str(d["source"]),
                init_time=str(d["init_time"]),
                step_hours=int(d["step_hours"]),
                simulation_length=int(d["simulation_length"]),
                output_variables=list(vars_),
                grid=dict(grid),
                steps=steps,
                status=d["status"],
                steps_received=int(d["steps_received"]),
                created_at=str(d["created_at"]),
                plugin_version=str(d["plugin_version"]),
                schema_version=int(d["schema_version"]),
                message=d.get("message"),
                run_dir=run_dir,
            )
        except ManifestError:
            raise
        except Exception as exc:
            raise bad(str(exc)) from exc

    @classmethod
    def from_json(cls, text, run_dir: str = "") -> "Manifest":
        try:
            d = json.loads(text)
        except ValueError as exc:
            raise ManifestError(f"Manifest at {run_dir or '?'} is not valid JSON ({exc}). Re-run the forecast.") from exc
        return cls.from_dict(d, run_dir=run_dir)


def write_manifest(storage, manifest: Manifest, run_dir: str | None = None) -> None:
    run_dir = run_dir or manifest.location
    manifest.run_dir = run_dir
    storage.write_bytes(manifest_path(run_dir), manifest.to_json().encode("utf-8"))


def read_manifest(storage, run_dir: str) -> Manifest:
    path = manifest_path(run_dir)
    if not storage.exists(path):
        raise ManifestError(f"No manifest at {path}. Run the 'Run forecast' recipe first.")
    return Manifest.from_json(storage.read_bytes(path).decode("utf-8"), run_dir=run_dir)


def list_manifests(storage) -> list[Manifest]:
    """All readable manifests under runs/. Corrupt ones are skipped with a warning."""
    out = []
    for path in storage.list(RUNS_PREFIX):
        if posixpath.basename(path) != MANIFEST_FILE:
            continue
        run_dir = posixpath.dirname(path)
        try:
            out.append(read_manifest(storage, run_dir))
        except ManifestError as exc:
            logger.warning("Skipping %s: %s", run_dir, exc)
    return sorted(out, key=lambda m: m.init_datetime)
