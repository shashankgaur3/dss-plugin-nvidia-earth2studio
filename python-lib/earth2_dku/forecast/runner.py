"""Run a forecast against a NIM and persist steps + manifest to storage."""
from __future__ import annotations

import logging
import posixpath

from ..errors import NimStreamBrokenError
from ..extract.grid import get_grid
from .manifest import Manifest, StepRecord, write_manifest

logger = logging.getLogger(__name__)

STEPS_DIR = "steps"


def run_forecast(
    client,
    model,
    storage,
    run_dir: str,
    input_bytes: bytes,
    init_time,
    simulation_length: int,
    variables: list[str] | None,
    source: str = "unknown",
    now=None,
) -> Manifest:
    """Stream a forecast into ``run_dir``.

    Writes manifest.json with status "running" first, each step under steps/ as it
    arrives, then the final status ("complete", or "partial"/"failed" with
    steps_received). Errors are re-raised after the state has been recorded, and
    partial results are kept.
    """
    variables = [v for v in (variables or []) if v] or None
    manifest = Manifest.new(
        model=model,
        source=source,
        init_time=init_time,
        simulation_length=simulation_length,
        output_variables=variables or [],
        grid=get_grid(model.grid).to_dict(),
        now=now,
    )
    write_manifest(storage, manifest, run_dir)

    try:
        for step in client.infer_stream(input_bytes, init_time, simulation_length, variables):
            if not manifest.steps:
                # The client knows which variables the NIM actually accepted (it may have
                # dropped the `variables` field); None means the full model variable list.
                sent = client.last_variables_sent
                manifest.output_variables = list(sent) if sent else model.input_variables()
            rel = posixpath.join(STEPS_DIR, step.filename)
            storage.write_bytes(posixpath.join(run_dir, rel), step.raw_bytes)
            manifest.steps.append(StepRecord(file=rel, lead_hours=step.lead_hours, batch_idx=step.batch_idx))
            manifest.steps_received = len(manifest.steps)
            write_manifest(storage, manifest, run_dir)
            logger.info("Stored step %s (lead %sh)", rel, step.lead_hours)
    except NimStreamBrokenError as exc:
        manifest.status = "partial" if manifest.steps else "failed"
        manifest.message = str(exc)
        write_manifest(storage, manifest, run_dir)
        raise
    except BaseException as exc:
        manifest.status = "failed"
        manifest.message = f"{type(exc).__name__}: {exc}"
        write_manifest(storage, manifest, run_dir)
        raise

    manifest.status = "complete"
    manifest.steps_received = len(manifest.steps)
    write_manifest(storage, manifest, run_dir)
    return manifest
