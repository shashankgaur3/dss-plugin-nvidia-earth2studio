"""Plugin error hierarchy. Every message tells the user what to do next."""
from __future__ import annotations


class Earth2Error(Exception):
    """Base class for all plugin errors."""


class ConfigError(Earth2Error):
    """Invalid or missing configuration (preset, recipe settings, registry)."""


class PreflightError(Earth2Error):
    """A check that must pass before calling the NIM failed."""


class NimRequestError(Earth2Error):
    """The NIM rejected the request or could not be reached."""


class NimStreamBrokenError(Earth2Error):
    """The NIM started streaming (HTTP 200) but the stream ended early.

    The NIM returns HTTP 200 and starts streaming before inference finishes, so
    server-side failures (wrong channel count, CUDA out of memory) only show up
    as a truncated stream on the client side.
    """

    def __init__(self, steps_received: int, detail: str = ""):
        self.steps_received = int(steps_received)
        self.detail = detail
        msg = (
            f"The FourCastNet NIM stream broke after {self.steps_received} step(s)"
            + (f" ({detail})" if detail else "")
            + ". The NIM answers HTTP 200 before inference finishes, so server-side failures "
            "(wrong input channel count, CUDA out of memory) only appear as a truncated stream. "
            "Check the NIM pod logs (e.g. `kubectl logs <fourcastnet-pod>`) for the real error. "
            "If you use a port-forward or load balancer, an idle timeout can also cut long streams; "
            "prefer the in-cluster service URL."
        )
        super().__init__(msg)


class HostedLargeAssetError(Earth2Error):
    """The NVIDIA hosted API answered 'Large asset written' without a retrievable result."""

    def __init__(self, detail: str = ""):
        msg = (
            "The NVIDIA hosted API reported 'Large asset written' but the result cannot be "
            "retrieved (known issue with large hosted outputs"
            + (f"; {detail}" if detail else "")
            + "). Reduce simulation_length and/or the number of requested variables, "
            "or use a self-hosted FourCastNet NIM."
        )
        super().__init__(msg)


class ManifestError(Earth2Error):
    """A forecast run manifest is missing, corrupt or inconsistent with its step files."""
