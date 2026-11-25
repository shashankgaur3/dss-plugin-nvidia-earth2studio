"""Client for a self-hosted FourCastNet NIM (POST /v1/infer, streamed tar response)."""
from __future__ import annotations

import http.client
import logging
import posixpath
import tarfile
from typing import Iterator

import requests
import urllib3

from ..errors import ConfigError, NimRequestError, NimStreamBrokenError
from ..times import to_rfc3339
from .base import ForecastStep, HealthStatus, NimClient, parse_step_filename

logger = logging.getLogger(__name__)

DEFAULT_TIMEOUT_SECONDS = 1200
CONNECT_TIMEOUT_SECONDS = 30

# Broken/truncated streams: requests and urllib3 errors (ChunkedEncodingError, ProtocolError,
# ReadTimeoutError...), http.client.IncompleteRead, and tarfile.ReadError on a cut archive.
_STREAM_ERRORS = (
    requests.RequestException,
    urllib3.exceptions.HTTPError,
    http.client.HTTPException,
    tarfile.TarError,
    EOFError,
)


class _DeferredErrorReader:
    """File-like wrapper around ``resp.raw`` that keeps already-received bytes when the stream breaks.

    urllib3 discards the data buffered inside a ``read(n)`` call that hits a broken
    connection, and tarfile reads ahead, so a step fully received just before the
    break could be lost. Reading in pieces and deferring the error to the next call
    keeps every complete step.
    """

    PIECE = 64 * 1024

    def __init__(self, raw):
        self.raw = raw
        self.error: BaseException | None = None

    def read(self, n: int = -1) -> bytes:
        if self.error is not None:
            raise self.error
        out = bytearray()
        while n is None or n < 0 or len(out) < n:
            want = self.PIECE if n is None or n < 0 else min(self.PIECE, n - len(out))
            try:
                data = self.raw.read(want)
            except _STREAM_ERRORS as exc:
                self.error = exc
                break
            if not data:
                break
            out += data
        if not out and self.error is not None:
            raise self.error
        return bytes(out)


def _body_excerpt(resp, limit: int = 500) -> str:
    try:
        return resp.text[:limit]
    except Exception:
        return ""


class SelfHostedNimClient(NimClient):
    def __init__(
        self,
        base_url: str,
        supports_variables_param: str = "unverified",
        timeout_seconds: float = DEFAULT_TIMEOUT_SECONDS,
        session: requests.Session | None = None,
        auth_header: tuple[str, str] | None = None,
    ):
        """``auth_header``: optional (name, value) sent with every request, e.g. ("X-API-Key", "...")."""
        if not base_url or not str(base_url).strip():
            raise ConfigError(
                "The NIM base URL is empty. Set it in the connection preset, e.g. "
                "http://fourcastnet.nim-services.svc.cluster.local:8000"
            )
        self.base_url = str(base_url).strip().rstrip("/")
        self.supports_variables_param = supports_variables_param
        self.timeout_seconds = float(timeout_seconds or DEFAULT_TIMEOUT_SECONDS)
        self.session = session or requests.Session()
        if auth_header:
            self.session.headers[auth_header[0]] = auth_header[1]
        self.last_variables_sent = None

    # ------------------------------------------------------------------ health
    def health(self) -> HealthStatus:
        url = f"{self.base_url}/v1/health/ready"
        try:
            resp = self.session.get(url, timeout=min(self.timeout_seconds, CONNECT_TIMEOUT_SECONDS))
        except requests.RequestException as exc:
            return HealthStatus(
                False,
                f"Cannot reach {url} ({exc}). Check the base URL in the preset and that DSS can reach the NIM.",
            )
        if resp.status_code == 200:
            return HealthStatus(True, f"{url} returned 200: NIM is ready.")
        if resp.status_code in (401, 403):
            return HealthStatus(
                False,
                f"{url} returned HTTP {resp.status_code}: the NIM requires authentication. Check the API key "
                "header name and value in the connection preset.",
            )
        return HealthStatus(
            False,
            f"{url} returned HTTP {resp.status_code}. The NIM may still be loading the model; "
            "check the NIM pod status and logs.",
        )

    # ------------------------------------------------------------------- infer
    def _post(self, input_array_bytes, init_time, simulation_length, variables):
        data = {"input_time": to_rfc3339(init_time), "simulation_length": str(int(simulation_length))}
        if variables:
            data["variables"] = ",".join(variables)
        files = {"input_array": ("input_array.npy", input_array_bytes, "application/octet-stream")}
        try:
            return self.session.post(
                f"{self.base_url}/v1/infer",
                files=files,
                data=data,
                headers={"accept": "application/x-tar"},
                stream=True,
                timeout=(CONNECT_TIMEOUT_SECONDS, self.timeout_seconds),
            )
        except requests.RequestException as exc:
            raise NimRequestError(
                f"Could not send the inference request to {self.base_url}/v1/infer ({exc}). "
                "Check the base URL, network access from DSS and the NIM pod status."
            ) from exc

    def infer_stream(self, input_array_bytes, init_time, simulation_length, variables) -> Iterator[ForecastStep]:
        variables = [v for v in (variables or []) if v] or None
        send_vars = variables
        if variables and self.supports_variables_param == "false":
            logger.warning(
                "This model's NIM does not support the `variables` parameter; requesting all variables "
                "(responses will be large)."
            )
            send_vars = None

        resp = self._post(input_array_bytes, init_time, simulation_length, send_vars)
        if (
            resp.status_code in (400, 422)
            and send_vars
            and self.supports_variables_param == "unverified"
            and "variables" in _body_excerpt(resp, 5000).lower()
        ):
            logger.warning(
                "The NIM rejected the `variables` parameter (HTTP %s: %s). Retrying once without it; "
                "all model variables will be returned. Consider setting supports_variables_param: false "
                "in models.yaml.",
                resp.status_code,
                _body_excerpt(resp, 200),
            )
            resp.close()
            send_vars = None
            resp = self._post(input_array_bytes, init_time, simulation_length, None)

        if resp.status_code != 200:
            body = _body_excerpt(resp)
            resp.close()
            raise NimRequestError(
                f"The NIM answered HTTP {resp.status_code} on /v1/infer: {body!r}. Check that the input "
                "array has shape (1, 1, n_channels, 721, 1440) for the configured model, that input_time "
                "is RFC 3339 without fractional seconds, and the NIM pod logs."
            )

        self.last_variables_sent = list(send_vars) if send_vars else None
        yield from self._iter_tar(resp, expected_steps=int(simulation_length) + 1)

    @staticmethod
    def _iter_tar(resp, expected_steps: int) -> Iterator[ForecastStep]:
        received = 0
        try:
            resp.raw.decode_content = True
            with tarfile.open(fileobj=_DeferredErrorReader(resp.raw), mode="r|") as tar:
                for member in tar:
                    if not member.isfile():
                        continue
                    name = posixpath.basename(member.name)
                    try:
                        lead, batch = parse_step_filename(name)
                    except ValueError:
                        logger.warning("Ignoring unexpected file in NIM response: %s", member.name)
                        continue
                    data = tar.extractfile(member).read()
                    if len(data) != member.size:
                        raise tarfile.ReadError(f"truncated member {name}")
                    received += 1
                    yield ForecastStep(lead_hours=lead, batch_idx=batch, filename=name, raw_bytes=data)
        except _STREAM_ERRORS as exc:
            raise NimStreamBrokenError(steps_received=received, detail=f"{type(exc).__name__}: {exc}") from exc
        finally:
            resp.close()
        if received < expected_steps:
            raise NimStreamBrokenError(
                steps_received=received,
                detail=f"expected {expected_steps} step files, the stream ended cleanly after {received}",
            )
