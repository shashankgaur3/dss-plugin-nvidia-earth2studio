"""Fetch initial conditions, run forecasts against a NIM, persist steps + manifest to storage."""
from __future__ import annotations

import logging
import posixpath

from ..errors import NimStreamBrokenError, PreflightError
from ..extract.grid import get_grid
from ..sources.cache import fetch_initial_condition
from .manifest import Manifest, StepRecord, read_manifest, run_dir_for, write_manifest

logger = logging.getLogger(__name__)

STEPS_DIR = "steps"
INPUTS_PREFIX = "inputs/"  # where initial conditions go when no separate folder is given


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
    write_manifest(storage, manifest, run_dir)
    return manifest


def run_forecasts(
    client,
    model,
    source,
    ic_storage,
    forecast_storage,
    init_times,
    simulation_length: int,
    variables: list[str] | None,
    fail_on_partial: bool = False,
) -> list[Manifest]:
    """For each init time: fetch (or reuse) the initial condition, then run and store the forecast.

    NIM health is checked once, before anything is downloaded. A stream that breaks after at
    least one step keeps the partial run and moves on, unless ``fail_on_partial``.
    """
    health = client.health()
    if not health.ready:
        raise PreflightError(f"The FourCastNet NIM is not ready: {health.detail}")
    manifests = []
    for init_time in init_times:
        ref, cached = fetch_initial_condition(source, ic_storage, model, init_time)
        logger.info("%s initial condition %s", "Reusing" if cached else "Fetched", ref.path)
        run_dir = run_dir_for(ref.time)
        try:
            manifests.append(run_forecast(
                client, model, forecast_storage, run_dir, ic_storage.read_bytes(ref.path), ref.time,
                simulation_length, variables, source=ref.source,
            ))
        except NimStreamBrokenError as exc:
            if fail_on_partial or exc.steps_received == 0:
                raise
            logger.warning("Keeping partial forecast %s (%d steps): %s", run_dir, exc.steps_received, exc)
            manifests.append(read_manifest(forecast_storage, run_dir))
    return manifests
