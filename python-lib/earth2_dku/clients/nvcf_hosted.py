"""EXPERIMENTAL client for the NVIDIA hosted FourCastNet API (NVCF).

Flow: upload the input .npy as an NVCF asset, POST the inference request, then
follow 202 (poll the status endpoint) / 302 (download the result *without* the
auth header) until a zip/tar archive arrives, and unpack it into ForecastSteps.

Known issue: large results may come back as HTTP 200 with body
{"message": "Large asset written"} and no retrievable URL (status endpoint 410).
That case raises HostedLargeAssetError and is never written to disk as output.
"""
from __future__ import annotations

import io
import json
import logging
import tarfile
import time
import zipfile
from typing import Iterator

import requests

from ..errors import ConfigError, HostedLargeAssetError, NimRequestError
from ..times import to_rfc3339
from .base import ForecastStep, HealthStatus, NimClient, parse_step_filename

logger = logging.getLogger(__name__)

INFER_URL = "https://climate.api.nvidia.com/v1/nvidia/fourcastnet"
STATUS_URL = "https://api.nvcf.nvidia.com/v2/nvcf/pexec/status/{req_id}"
ASSETS_URL = "https://api.nvcf.nvidia.com/v2/nvcf/assets"
POLL_SECONDS = 300
LARGE_ASSET_MARKER = "large asset written"


def is_large_asset_body(body: bytes) -> bool:
    if not body or len(body) > 4096 or not body.lstrip().startswith(b"{"):
        return False
    try:
        msg = json.loads(body.decode("utf-8")).get("message", "")
    except (ValueError, AttributeError, UnicodeDecodeError):
        return False
    return LARGE_ASSET_MARKER in str(msg).lower()


def sniff_archive(body: bytes) -> str:
    if body[:4] == b"PK\x03\x04":
        return "zip"
    if body[:2] == b"\x1f\x8b":
        return "gzip"
    if len(body) > 262 and body[257:262] == b"ustar":
        return "tar"
    return "unknown"


def unpack_archive(body: bytes) -> list[ForecastStep]:
    kind = sniff_archive(body)
    files: list[tuple[str, bytes]] = []
    if kind == "zip":
        with zipfile.ZipFile(io.BytesIO(body)) as zf:
            files = [(n, zf.read(n)) for n in zf.namelist() if not n.endswith("/")]
    elif kind in ("gzip", "tar"):
        with tarfile.open(fileobj=io.BytesIO(body), mode="r:gz" if kind == "gzip" else "r:") as tf:
            files = [(m.name, tf.extractfile(m).read()) for m in tf.getmembers() if m.isfile()]
    else:
        raise NimRequestError(
            f"The hosted API returned an unrecognised payload (first bytes {body[:16]!r}); expected a zip or tar "
            "archive. Retry later, or use a self-hosted NIM."
        )
    steps = []
    for name, data in files:
        try:
            lead, batch = parse_step_filename(name)
        except ValueError:
            logger.warning("Ignoring unexpected file in hosted response: %s", name)
            continue
        steps.append(ForecastStep(lead, batch, name.rsplit("/", 1)[-1], data))
    if not steps:
        raise NimRequestError("The hosted API archive contains no step .npy files. Use a self-hosted NIM.")
    return sorted(steps, key=lambda s: (s.lead_hours, s.batch_idx))


class NvcfHostedClient(NimClient):
    experimental = True

    def __init__(self, api_key: str, timeout_seconds: float = 1200, session=None, sleep=time.sleep,
                 max_polls: int = 100):
        if not api_key:
            raise ConfigError("The NVIDIA hosted API needs an API key. Set it in the connection preset.")
        self.api_key = api_key
        self.timeout_seconds = float(timeout_seconds or 1200)
        self.session = session or requests.Session()
        self.sleep = sleep
        self.max_polls = max_polls
        self.last_variables_sent = None

    def _auth(self) -> dict:
        return {"Authorization": f"Bearer {self.api_key}", "NVCF-POLL-SECONDS": str(POLL_SECONDS)}

    def health(self) -> HealthStatus:
        return HealthStatus(
            True,
            "NVIDIA hosted API (experimental): no health endpoint; API key is set. "
            "Availability is only verified when a forecast runs.",
        )

    def _upload_asset(self, data: bytes) -> str:
        desc = "earth2-dku fourcastnet input"
        resp = self.session.post(
            ASSETS_URL, headers={**self._auth(), "Content-Type": "application/json", "accept": "application/json"},
            json={"contentType": "application/octet-stream", "description": desc}, timeout=60,
        )
        if resp.status_code != 200:
            raise NimRequestError(f"NVCF asset creation failed (HTTP {resp.status_code}): {resp.text[:300]!r}. Check the API key.")
        info = resp.json()
        put = self.session.put(
            info["uploadUrl"], data=data,
            headers={"Content-Type": "application/octet-stream", "x-amz-meta-nvcf-asset-description": desc},
            timeout=self.timeout_seconds,
        )
        if put.status_code not in (200, 201):
            raise NimRequestError(f"Uploading the input array to NVCF failed (HTTP {put.status_code}).")
        return info["assetId"]

    def _resolve(self, resp) -> bytes:
        for _ in range(self.max_polls):
            code = resp.status_code
            if code == 202:
                req_id = resp.headers.get("NVCF-REQID")
                if not req_id:
                    raise NimRequestError("Hosted API answered 202 without NVCF-REQID. Retry later.")
                self.sleep(1)
                resp = self.session.get(STATUS_URL.format(req_id=req_id), headers=self._auth(),
                                        allow_redirects=False, timeout=self.timeout_seconds)
            elif code == 302:
                location = resp.headers.get("Location")
                if not location:
                    raise NimRequestError("Hosted API answered 302 without a Location header. Retry later.")
                # Pre-signed URL: must be fetched WITHOUT the Authorization header.
                resp = self.session.get(location, allow_redirects=False, timeout=self.timeout_seconds)
            elif code == 410:
                raise HostedLargeAssetError("status endpoint returned HTTP 410 Gone")
            elif code == 200:
                body = resp.content
                if is_large_asset_body(body):
                    raise HostedLargeAssetError()
                return body
            else:
                raise NimRequestError(
                    f"Hosted API error HTTP {code}: {resp.text[:300]!r}. Check the API key, the request "
                    "size (simulation_length, variables), or use a self-hosted NIM."
                )
        raise NimRequestError(f"Hosted API result not ready after {self.max_polls} polls. Retry later.")

    def infer_stream(self, input_array_bytes, init_time, simulation_length, variables) -> Iterator[ForecastStep]:
        variables = [v for v in (variables or []) if v] or None
        asset_id = self._upload_asset(input_array_bytes)
        body = {
            "input_id": asset_id,
            "input_time": to_rfc3339(init_time),
            "simulation_length": int(simulation_length),
            "ensemble_size": 1,
            "noise_amplitude": 0.0,
        }
        if variables:
            body["variables"] = ",".join(variables)
        try:
            resp = self.session.post(
                INFER_URL, json=body, allow_redirects=False, timeout=self.timeout_seconds,
                headers={**self._auth(), "NVCF-INPUT-ASSET-REFERENCES": asset_id, "accept": "application/json"},
            )
        except requests.RequestException as exc:
            raise NimRequestError(f"Could not reach the NVIDIA hosted API ({exc}). Check outbound network access.") from exc
        payload = self._resolve(resp)
        self.last_variables_sent = variables
        yield from unpack_archive(payload)
