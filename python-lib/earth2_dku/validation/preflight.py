"""Preflight checks. Each check returns CheckResult(name, ok, message) and never raises."""
from __future__ import annotations

import html
from typing import NamedTuple

from ..arrays import read_npy_header
from ..errors import Earth2Error, PreflightError
from ..models.registry import get_model, list_models
from ..sources.base import validate_state_shape
from ..times import floor_to_cycle, parse_time


class CheckResult(NamedTuple):
    name: str
    ok: bool
    message: str


def check_health(client) -> CheckResult:
    name = "NIM endpoint health"
    if client is None:
        return CheckResult(name, False, "No client could be built from the preset; fix the preset settings.")
    try:
        status = client.health()
    except Exception as exc:  # defensive: health() should not raise
        return CheckResult(name, False, f"Health check failed: {exc}")
    return CheckResult(name, status.ready, status.detail)


def check_model_exists(model_id: str) -> CheckResult:
    name = "Model in registry"
    try:
        model = get_model(model_id)
    except Earth2Error as exc:
        return CheckResult(name, False, str(exc))
    return CheckResult(name, True, f"{model.model_id}: {model.label} ({model.nim_image}).")


def check_input_variables(model) -> CheckResult:
    name = "Input variables vs channels"
    if model is None:
        return CheckResult(name, False, "Skipped: unknown model.")
    try:
        variables = model.input_variables()
    except Earth2Error as exc:
        return CheckResult(name, False, str(exc))
    return CheckResult(name, True, f"Earth2Studio lists {len(variables)} input variables = n_channels.")


def check_gpu_memory(model, gpu_memory_gb) -> CheckResult:
    name = "GPU memory"
    if model is None:
        return CheckResult(name, False, "Skipped: unknown model.")
    if gpu_memory_gb in (None, "", 0):
        return CheckResult(
            name, True, f"Not checked (no GPU memory in preset). {model.model_id} needs >= {model.min_gpu_mem_gb} GB."
        )
    gpu = float(gpu_memory_gb)
    if gpu >= model.min_gpu_mem_gb:
        return CheckResult(name, True, f"{gpu:g} GB >= {model.min_gpu_mem_gb} GB required by {model.model_id}.")
    fitting = [m.nim_version for m in list_models() if m.min_gpu_mem_gb <= gpu]
    alt = f"Use FourCastNet NIM {' or '.join(fitting)} on this GPU, or " if fitting else "Use "
    return CheckResult(
        name,
        False,
        f"FourCastNet NIM {model.nim_version} needs a single GPU with {model.min_gpu_mem_gb} GB; a "
        f"{gpu:g} GB GPU (e.g. an L40S has 48 GB) will run out of memory. {alt}a GPU with at least "
        f"{model.min_gpu_mem_gb} GB (e.g. 80 GB). The model cannot be split across GPUs.",
    )


def check_init_time(source, init_time=None) -> CheckResult:
    name = "Init time"
    if source is None:
        return CheckResult(name, True, "Not checked (no source selected).")
    try:
        latest = source.latest_available_time()
        if init_time in (None, ""):
            return CheckResult(name, True, f"Latest available {source.name} cycle: {latest:%Y-%m-%dT%H}Z.")
        t = parse_time(init_time)
    except Earth2Error as exc:
        return CheckResult(name, False, str(exc))
    cycle = floor_to_cycle(t)
    note = "" if cycle == t else f" (will be rounded down to {cycle:%Y-%m-%dT%H}Z)"
    earliest = getattr(source, "earliest_time", None)
    if earliest and cycle < earliest:
        return CheckResult(name, False, f"{source.name} has no data before {earliest:%Y-%m-%d}.")
    if cycle > latest:
        return CheckResult(
            name, False,
            f"{cycle:%Y-%m-%dT%H}Z is not yet available from {source.name}; latest is {latest:%Y-%m-%dT%H}Z.",
        )
    return CheckResult(name, True, f"{cycle:%Y-%m-%dT%H}Z is available from {source.name}{note}.")


def check_input_array(input_bytes: bytes, model) -> CheckResult:
    name = "Input array shape"
    try:
        shape, dtype = read_npy_header(input_bytes)
        validate_state_shape(shape, dtype, model)
    except PreflightError as exc:
        return CheckResult(name, False, str(exc))
    except Exception as exc:
        return CheckResult(name, False, f"Input is not a readable .npy file ({exc}). Re-fetch initial conditions.")
    return CheckResult(name, True, f"Shape {shape}, {dtype}.")


def run_preflight(model_id, client=None, gpu_memory_gb=None, source=None, init_time=None) -> list[CheckResult]:
    results = [check_health(client), check_model_exists(model_id)]
    model = get_model(model_id) if results[1].ok else None
    results += [
        check_input_variables(model),
        check_gpu_memory(model, gpu_memory_gb),
        check_init_time(source, init_time),
    ]
    return results


def ensure_ready_for_forecast(client, model, input_bytes: bytes) -> None:
    """Health + shape checks before sending a forecast request. Raises PreflightError."""
    failed = [r for r in (check_health(client), check_input_array(input_bytes, model)) if not r.ok]
    if failed:
        raise PreflightError(" | ".join(f"{r.name}: {r.message}" for r in failed))


def render_html(results) -> str:
    rows = "".join(
        "<tr><td>{}</td><td style='color:{};font-weight:bold'>{}</td><td>{}</td></tr>".format(
            html.escape(r.name), "#2e7d32" if r.ok else "#c62828", "OK" if r.ok else "FAILED", html.escape(r.message)
        )
        for r in results
    )
    n_failed = sum(not r.ok for r in results)
    summary = "All checks passed." if n_failed == 0 else f"{n_failed} check(s) failed."
    return (
        f"<p><b>{summary}</b></p><table class='table table-striped'>"
        "<thead><tr><th>Check</th><th>Status</th><th>Message</th></tr></thead>"
        f"<tbody>{rows}</tbody></table>"
    )
